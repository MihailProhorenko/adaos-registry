"""AdaOS media indexer skill handlers."""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import re
import sys
import time
from typing import Any, Dict, List

import yaml

from adaos.sdk.core.decorators import subscribe, tool
from adaos.sdk.data import ctx_subnet
from adaos.sdk.data import skill_memory
from adaos.sdk.data.context import clear_current_skill, set_current_skill
from adaos.sdk.data.skill_env import skill_env_path
from adaos.sdk.io.out import stream_variable_publish
from adaos.services.agent_context import get_ctx

_SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(_SKILL_ROOT))

from lib.enrichment import EnrichmentService
from lib.extractor import TechnicalMetadataExtractor
from lib.ner_predictor import NERPredictor, model_weights_status
from lib.scanner import DirectoryScanner
from lib.vector_db import VectorDatabase

logger = logging.getLogger(__name__)

REQUIRES_DATA_PROJECTIONS = True
SCORE_THRESHOLD = 25.0
DEFAULT_QUERY = "test"
SETTINGS_KEY = "media_indexer.settings"
INDEX_META_KEY = "media_indexer.index"
OPERATION_RECEIVER = "media_indexer.operations"
MAX_RESULTS = 20

_state: Dict[str, Any] = {
    "scanner": None,
    "extractor": None,
    "ner": None,
    "enricher": None,
    "vector_db": None,
    "indexed_directory": None,
    "selected_directory": "",
    "selected_query": DEFAULT_QUERY,
    "index_loaded": False,
    "last_operation": None,
}


def _event_payload(evt: Any) -> Dict[str, Any]:
    payload = getattr(evt, "payload", None) if hasattr(evt, "payload") else evt
    return payload if isinstance(payload, dict) else {}


def _safe_memory_get(key: str, default: Any = None) -> Any:
    try:
        return skill_memory.get(key, default)
    except Exception:
        return default


def _safe_memory_set(key: str, value: Any) -> None:
    try:
        skill_memory.set(key, value)
    except Exception:
        logger.debug("failed to write skill memory key=%s", key, exc_info=True)


def _internal_data_dir() -> pathlib.Path:
    override = os.getenv("MEDIA_INDEXER_DATA_DIR")
    if override:
        path = pathlib.Path(override)
    else:
        try:
            env_path = skill_env_path()
            data_root = env_path.parents[1] if env_path.parent.name == "db" else env_path.parent
            path = data_root / "internal" / "media_indexer"
        except Exception:
            path = _SKILL_ROOT / ".skill_state"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _index_dir() -> pathlib.Path:
    path = _internal_data_dir() / "faiss"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _has_persisted_index() -> bool:
    path = _index_dir()
    return (path / "metadata.json").exists() and (path / "text.index").exists() and (path / "image.index").exists()


def _settings() -> Dict[str, Any]:
    stored = _safe_memory_get(SETTINGS_KEY, {})
    settings = dict(stored) if isinstance(stored, dict) else {}
    settings.setdefault("default_directory", "")
    settings.setdefault("selected_directory", "")
    settings.setdefault("selected_query", DEFAULT_QUERY)
    settings.setdefault("k", 5)
    return settings


def _save_settings(**updates: Any) -> Dict[str, Any]:
    settings = _settings()
    for key, value in updates.items():
        if value is not None:
            settings[key] = value
    _safe_memory_set(SETTINGS_KEY, settings)
    return settings


def _target_context(payload: Dict[str, Any]) -> tuple[bool, str | None]:
    meta = payload.get("_meta") if isinstance(payload.get("_meta"), dict) else {}
    target_node_id = str(
        payload.get("target_node_id")
        or payload.get("node_id")
        or meta.get("target_node_id")
        or meta.get("node_target_id")
        or ""
    ).strip()
    try:
        local_node_id = str(getattr(get_ctx().config, "node_id", "") or "").strip()
    except Exception:
        local_node_id = ""
    if target_node_id and local_node_id and target_node_id != local_node_id:
        return False, None
    raw_ws = payload.get("webspace_id") or payload.get("workspace_id") or meta.get("webspace_id") or meta.get("workspace_id")
    return True, str(raw_ws).strip() if raw_ws else None


def _load_skill_data_projections() -> None:
    try:
        ctx = get_ctx()
        try:
            existing = ctx.projections.resolve("subnet", "media_indexer.snapshot")
        except Exception:
            existing = []
        if existing:
            return
        skills_root = ctx.paths.skills_workspace_dir()
        skills_root = skills_root() if callable(skills_root) else skills_root
        manifest_path = pathlib.Path(skills_root) / "media_indexer_skill" / "skill.yaml"
        if not manifest_path.exists():
            return
        spec = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        entries = spec.get("data_projections") or []
        if isinstance(entries, list) and entries:
            ctx.projections.load_entries(entries)
    except Exception:
        logger.debug("failed to load media_indexer_skill data_projections", exc_info=True)


def _current_form(directory: str | None = None, query: str | None = None, k: int | None = None) -> Dict[str, Any]:
    settings = _settings()
    selected_directory = directory if directory is not None else (
        _state.get("selected_directory") or settings.get("selected_directory") or settings.get("default_directory") or ""
    )
    selected_query = query if query is not None else (_state.get("selected_query") or settings.get("selected_query") or DEFAULT_QUERY)
    return {"directory": selected_directory, "query": selected_query, "k": int(k or settings.get("k") or 5)}


def _resolve_directory(payload: Dict[str, Any]) -> str:
    raw = str(payload.get("directory") or "").strip()
    if raw.startswith("$"):
        raw = ""
    if not raw:
        raw = str(_current_form().get("directory") or "").strip()
    return raw


def _resolve_query(payload: Dict[str, Any]) -> str:
    raw = str(payload.get("query") or "").strip()
    if raw.startswith("$"):
        raw = ""
    return raw or str(_current_form().get("query") or DEFAULT_QUERY).strip() or DEFAULT_QUERY


def _status_payload(
    *,
    value: str,
    subtitle: str,
    description: str,
    error: str = "",
    indexed_count: int | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "value": value,
        "label": "Media Indexer",
        "subtitle": subtitle,
        "description": description,
        "error": error,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if indexed_count is not None:
        payload["indexed_count"] = int(indexed_count)
    return payload


def _snapshot_payload(
    *,
    status: Dict[str, Any],
    form: Dict[str, Any] | None = None,
    results: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    return {"status": status, "form": form or _current_form(), "results": list(results or [])[:MAX_RESULTS]}


def _project_snapshot(snapshot: Dict[str, Any], *, webspace_id: str | None = None) -> None:
    pushed = False
    try:
        _load_skill_data_projections()
        pushed = set_current_skill("media_indexer_skill")
        ctx_subnet.set("media_indexer.snapshot", snapshot, webspace_id=webspace_id)
    except Exception:
        logger.warning("failed to project media_indexer.snapshot", exc_info=True)
    finally:
        if pushed:
            clear_current_skill()


def _publish_operation(value: Dict[str, Any], *, webspace_id: str | None = None) -> None:
    payload = {"label": "Media Indexer", **value, "updated_at": time.time()}
    _state["last_operation"] = payload
    try:
        stream_variable_publish(
            OPERATION_RECEIVER,
            payload,
            var_id="media_indexer.operation",
            ttl_ms=10 * 60 * 1000,
            _meta={"webspace_id": webspace_id} if webspace_id else None,
        )
    except Exception:
        logger.debug("failed to publish media indexer operation stream", exc_info=True)


def _ensure_initialized(*, load_index: bool = False) -> None:
    if _state["vector_db"] is None:
        logger.info("Initializing media_indexer_skill ML components")
        _state["scanner"] = None
        _state["extractor"] = TechnicalMetadataExtractor()
        _state["ner"] = NERPredictor()
        _state["enricher"] = EnrichmentService()
        _state["vector_db"] = VectorDatabase()
        _state["index_loaded"] = False
    if load_index and not _state.get("index_loaded"):
        loaded = _state["vector_db"].load(_index_dir())
        _state["index_loaded"] = bool(loaded.get("loaded"))
        if loaded.get("loaded"):
            logger.info("Loaded persisted media index: %s", loaded)


def _persist_index(directory: str, indexed_count: int) -> Dict[str, Any]:
    vector_db = _state.get("vector_db")
    if vector_db is None:
        return {"saved": False, "reason": "not_initialized"}
    metadata = vector_db.save(_index_dir())
    payload = {
        "indexed_directory": directory,
        "indexed_count": indexed_count,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "index_dir": str(_index_dir()),
        **metadata,
    }
    _safe_memory_set(INDEX_META_KEY, payload)
    _state["index_loaded"] = True
    return payload


def has_cyrillic(text: str) -> bool:
    return bool(text) and bool(re.search(r"[\u0400-\u04FF]", text))


def _flatten_inventory(inventory: Dict[str, List[Any]]) -> List[tuple[Any, str]]:
    all_files: List[tuple[Any, str]] = []
    for m_type, m_list in inventory.items():
        all_files.extend((media, m_type) for media in m_list)
    return all_files


def _build_display_title(stem: str, title: str, artist: str) -> str:
    if artist and title:
        return f"{artist} - {title}"
    if title:
        return title
    return stem


@tool("scan_and_index")
def scan_and_index(directory: str) -> Dict[str, Any]:
    if not str(directory or "").strip():
        return {"status": "error", "indexed_count": 0, "errors": ["Directory is empty. Set a directory first."]}

    _ensure_initialized(load_index=False)

    path = pathlib.Path(directory).expanduser()
    if not path.exists() or not path.is_dir():
        return {"status": "error", "indexed_count": 0, "errors": [f"Directory not found or not a directory: {directory}"]}

    try:
        scanner = DirectoryScanner(str(path), compute_hashes=False)
        _state["scanner"] = scanner
        inventory = scanner.scan()
    except Exception as exc:
        logger.exception("Failed to scan directory %s", path)
        return {"status": "error", "indexed_count": 0, "errors": [str(exc)]}

    all_files = _flatten_inventory(inventory)
    if not all_files:
        _state["indexed_directory"] = str(path)
        _save_settings(selected_directory=str(path), default_directory=str(path))
        _persist_index(str(path), 0)
        return {"status": "ok", "indexed_count": 0, "errors": []}

    extractor = _state["extractor"]
    ner = _state["ner"]
    enricher = _state["enricher"]
    vector_db = _state["vector_db"]

    errors: List[str] = []
    indexed = 0

    for media, ftype in all_files:
        try:
            logger.info("Indexing media file: %s", media.name)
            extractor.extract(media.full_path, ftype)

            ner_result = ner.extract_entities(media.name)
            title = ner_result.get("title") or ""
            year = ner_result.get("year") or "---"
            quality = ner_result.get("quality") or "---"
            artist = ner_result.get("artist") or ""

            enriched = enricher.enrich(media.full_path, ftype)
            if ftype == "video" and title:
                enriched.update(enricher.enrich_video(title))

            stem = pathlib.Path(media.name).stem
            display_title = _build_display_title(stem, title, artist)
            payload = {
                "real_file_name": media.name,
                "display_title": display_title,
                "full_path": media.full_path,
                "type": ftype,
                "ftype": ftype,
                "title": display_title,
                "ner_title": title,
                "year": year,
                "quality": quality,
                "artist": artist,
                "enriched": enriched,
            }

            if ftype == "image":
                vector_db.add_image(media.full_path, payload)
                vector_db.add_text(" ".join(["photo image изображение фотография", stem]), payload)
            elif ftype == "audio":
                parts = ["music audio song track музыка аудио песня трек"]
                if artist:
                    parts.append(f"artist {artist} исполнитель {artist}")
                if title:
                    parts.append(f"title {title} название {title}")
                if quality and quality != "---":
                    parts.append(quality)
                if enriched.get("shazam_title"):
                    parts.append(f"shazam {enriched['shazam_title']}")
                if enriched.get("shazam_subtitle"):
                    parts.append(f"shazam artist {enriched['shazam_subtitle']}")
                if enriched.get("shazam_genre"):
                    parts.append(f"genre {enriched['shazam_genre']}")
                if has_cyrillic(stem):
                    parts.append("русская русский на русском")
                parts.append(stem)
                vector_db.add_text(" ".join(filter(bool, parts)), payload)
            elif ftype == "video":
                parts = ["video movie film видео фильм кино"]
                if title:
                    parts.append(f"title {title} название {title}")
                if year != "---":
                    parts.append(f"year {year} год {year}")
                if quality != "---":
                    parts.append(quality)
                plot = (enriched.get("imdb") or {}).get("plot", "")
                if plot:
                    parts.append(plot)
                if has_cyrillic(stem):
                    parts.append("русское кино русский фильм")
                parts.append(stem)
                vector_db.add_text(" ".join(filter(bool, parts)), payload)
            else:
                continue

            indexed += 1
        except Exception as exc:
            logger.exception("Failed to index %s", getattr(media, "name", "unknown"))
            errors.append(f"{getattr(media, 'name', 'unknown')}: {exc}")

    _state["indexed_directory"] = str(path)
    _save_settings(selected_directory=str(path), default_directory=str(path))
    index_meta = _persist_index(str(path), indexed)
    return {"status": "ok", "indexed_count": indexed, "errors": errors, "index": index_meta}


@tool("search_media")
def search_media(query: str, k: int = 5) -> Dict[str, Any]:
    if not query or not query.strip():
        return {"status": "ok", "results": []}

    if _state["vector_db"] is None and not _has_persisted_index():
        return {"status": "error", "results": [], "message": "Index is empty. Call scan_and_index first."}

    _ensure_initialized(load_index=True)
    if not _state.get("index_loaded") and not ((_state.get("vector_db") or {}).text_docs if hasattr(_state.get("vector_db"), "text_docs") else False):
        return {"status": "error", "results": [], "message": "Index is empty. Call scan_and_index first."}

    try:
        limit = max(1, min(MAX_RESULTS, int(k or 5)))
    except (TypeError, ValueError):
        limit = 5

    _save_settings(selected_query=query.strip(), k=limit)
    raw_results = _state["vector_db"].search(query.strip(), k=limit)
    valid_results = [result for result in raw_results if result.get("score", 0) >= SCORE_THRESHOLD]
    formatted = [
        {
            "score": float(result.get("score", 0.0)),
            "path": result.get("payload", {}).get("full_path", ""),
            "payload": result.get("payload", {}),
        }
        for result in valid_results[:MAX_RESULTS]
    ]
    return {"status": "ok", "results": formatted}


@tool("get_settings")
def get_settings() -> Dict[str, Any]:
    return {
        "status": "ok",
        "settings": _settings(),
        "index": _safe_memory_get(INDEX_META_KEY, {}),
        "model_weights": model_weights_status(),
    }


@tool("rehydrate")
def rehydrate() -> Dict[str, Any]:
    settings = _settings()
    _state["selected_directory"] = settings.get("selected_directory") or settings.get("default_directory") or ""
    _state["selected_query"] = settings.get("selected_query") or DEFAULT_QUERY
    return {"status": "ok", "settings": settings, "index": _safe_memory_get(INDEX_META_KEY, {})}


@tool("dispose")
def dispose() -> Dict[str, Any]:
    _state.update(
        {
            "scanner": None,
            "extractor": None,
            "ner": None,
            "enricher": None,
            "vector_db": None,
            "index_loaded": False,
        }
    )
    return {"status": "ok"}


@subscribe("media_indexer.action")
async def on_media_indexer_action(evt: Any) -> None:
    payload = _event_payload(evt)
    allowed, webspace_id = _target_context(payload)
    if not allowed:
        return

    action_id = str(payload.get("id") or payload.get("action") or "").strip().lower()
    directory = _resolve_directory(payload)
    query = _resolve_query(payload)
    try:
        k = max(1, min(MAX_RESULTS, int(payload.get("k") or 5)))
    except (TypeError, ValueError):
        k = 5

    _state["selected_directory"] = directory
    _state["selected_query"] = query
    _save_settings(selected_directory=directory, selected_query=query, k=k)
    form = _current_form(directory=directory, query=query, k=k)

    if action_id in {"set_directory", "directory"}:
        _project_snapshot(
            _snapshot_payload(
                status=_status_payload(value="ready", subtitle="directory selected", description=f"Selected {directory or '(empty)'}"),
                form=form,
            ),
            webspace_id=webspace_id,
        )
        return

    if action_id in {"set_query", "query"}:
        _project_snapshot(
            _snapshot_payload(
                status=_status_payload(value="ready", subtitle="query selected", description=f"Query: {query}"),
                form=form,
            ),
            webspace_id=webspace_id,
        )
        return

    if action_id == "scan":
        status = _status_payload(value="scanning", subtitle="indexing media files", description=f"Scanning {directory or '(empty)'}")
        _project_snapshot(_snapshot_payload(status=status, form=form), webspace_id=webspace_id)
        _publish_operation({"value": "scanning", "subtitle": "indexing media files", "description": status["description"]}, webspace_id=webspace_id)
        result = await asyncio.to_thread(scan_and_index, directory)
        errors = list(result.get("errors") or [])
        ok = str(result.get("status") or "").lower() == "ok"
        indexed_count = int(result.get("indexed_count") or 0)
        final_status = _status_payload(
            value="indexed" if ok else "error",
            subtitle=f"{indexed_count} files indexed" if ok else "scan failed",
            description="Index is ready for semantic search." if ok else "; ".join(errors[:3]),
            error="" if ok else "; ".join(errors[:3]),
            indexed_count=indexed_count,
        )
        _project_snapshot(_snapshot_payload(status=final_status, form=_current_form(k=k)), webspace_id=webspace_id)
        _publish_operation(
            {
                "value": final_status["value"],
                "subtitle": final_status["subtitle"],
                "description": final_status["description"],
                "indexed_count": indexed_count,
            },
            webspace_id=webspace_id,
        )
        return

    if action_id == "search":
        status = _status_payload(value="searching", subtitle="semantic search", description=f"Searching for: {query}")
        _project_snapshot(_snapshot_payload(status=status, form=form), webspace_id=webspace_id)
        _publish_operation({"value": "searching", "subtitle": "semantic search", "description": status["description"]}, webspace_id=webspace_id)
        result = await asyncio.to_thread(search_media, query, k=k)
        results = list(result.get("results") or [])
        error = str(result.get("message") or "") if str(result.get("status") or "").lower() != "ok" else ""
        final_status = _status_payload(
            value="done" if not error else "error",
            subtitle=f"{len(results)} results",
            description=f"Query: {query}" if not error else error,
            error=error,
        )
        _project_snapshot(_snapshot_payload(status=final_status, form=_current_form(k=k), results=results), webspace_id=webspace_id)
        _publish_operation(
            {
                "value": final_status["value"],
                "subtitle": final_status["subtitle"],
                "description": final_status["description"],
                "result_count": len(results),
            },
            webspace_id=webspace_id,
        )


@subscribe("webio.stream.snapshot.requested")
async def on_stream_snapshot_requested(evt: Any) -> None:
    payload = _event_payload(evt)
    receiver = str(payload.get("receiver") or "").strip()
    if receiver != OPERATION_RECEIVER:
        return
    allowed, webspace_id = _target_context(payload)
    if not allowed:
        return
    _publish_operation(
        _state.get("last_operation")
        or {
            "value": "ready",
            "subtitle": "waiting for action",
            "description": "Set a directory, build an index, then search.",
        },
        webspace_id=webspace_id,
    )
