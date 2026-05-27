from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable


_FALSE_VALUES = {"0", "false", "no", "off", "disable", "disabled", "none"}
_SERVICE_NAME = "neuro_nlu_lite_skill"
_SERVICE_VERSION = "0.1.0"


@dataclass(frozen=True)
class Example:
    intent: str
    text: str
    masked: str


@dataclass
class MaskResult:
    normalized: str
    masked: str
    slots: dict[str, str] = field(default_factory=dict)


@dataclass
class VectorizerConfig:
    dim: int = 4096
    char_ngrams: tuple[int, ...] = (3, 4, 5)
    word_ngrams: tuple[int, ...] = (1, 2)
    min_similarity: float = 0.16
    min_margin: float = 0.05


_DEFAULT_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("voice.timer.start", "поставь таймер на {duration}"),
    ("voice.timer.start", "запусти таймер на {duration}"),
    ("voice.timer.start", "установи таймер на {duration}"),
    ("voice.timer.start", "set timer for {duration}"),
    ("voice.time.now", "сколько времени"),
    ("voice.time.now", "который час"),
    ("voice.time.now", "what time is it"),
    ("desktop.open_marketplace", "открой маркетплейс"),
    ("desktop.open_marketplace", "покажи маркетплейс"),
    ("desktop.open_marketplace", "open marketplace"),
    ("desktop.open_weather", "открой погоду"),
    ("desktop.open_weather", "покажи погоду"),
    ("weather.show", "какая погода в {city}"),
    ("weather.show", "погода в {city}"),
    ("weather.show", "weather in {city}"),
)

_DURATION_RE = re.compile(
    r"\b(?P<value>\d+\s*(?:секунд(?:у|ы)?|сек|минут(?:у|ы)?|мин|час(?:а|ов)?|seconds?|secs?|minutes?|mins?|hours?))\b",
    re.IGNORECASE | re.UNICODE,
)
_WEATHER_CITY_RE = re.compile(
    r"\b(?:какая\s+погода|погода|прогноз|температура)\b(?:\s+(?:в|для)\s+(?P<city>[^?.!,;:]+))?",
    re.IGNORECASE | re.UNICODE,
)
_WEATHER_CITY_EN_RE = re.compile(
    r"\b(?:weather|forecast|temperature)\b(?:\s+(?:in|for)\s+(?P<city>[^?.!,;:]+))?",
    re.IGNORECASE | re.UNICODE,
)


def _env_flag(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default).strip().lower()
    return raw not in _FALSE_VALUES if raw else default.strip().lower() not in _FALSE_VALUES


def _log(event: str, level: str = "INFO", **fields: Any) -> None:
    payload: dict[str, Any] = {
        "ts": time.time(),
        "level": level,
        "event": event,
        "skill": _SERVICE_NAME,
        "pid": os.getpid(),
    }
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _artifact_root() -> Path:
    explicit = os.getenv("ADAOS_NEURO_LITE_ARTIFACT_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()

    data_dir = os.getenv("ADAOS_SKILL_DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir).expanduser().resolve() / "files" / "nlu" / "neuro_lite"

    base = Path(os.getenv("ADAOS_BASE_DIR", "~/.adaos")).expanduser().resolve()
    return base / "workspace" / "skills" / ".runtime" / _SERVICE_NAME / "v0.1" / "data" / "files" / "nlu" / "neuro_lite"


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    try:
        length = int(handler.headers.get("Content-Length") or "0")
    except Exception:
        length = 0
    raw = handler.rfile.read(max(length, 0)) if length > 0 else b"{}"
    try:
        payload = json.loads(raw.decode("utf-8", errors="ignore"))
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _normalize(text: str) -> str:
    token = str(text or "").lower().replace("ё", "е")
    token = re.sub(r"[\"'`«»]", " ", token)
    token = re.sub(r"\s+", " ", token, flags=re.UNICODE)
    return token.strip()


def _clean_slot(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip().strip(" \t\r\n'\"()[]{}")
    return token if token else None


def _mask_text(text: str) -> MaskResult:
    normalized = _normalize(text)
    masked = normalized
    slots: dict[str, str] = {}

    match = _DURATION_RE.search(masked)
    if match:
        value = _clean_slot(match.group("value"))
        if value:
            slots["duration"] = value
            slots["duration_canon"] = value
            masked = masked[: match.start()] + "{duration}" + masked[match.end() :]

    city: str | None = None
    for pattern in (_WEATHER_CITY_RE, _WEATHER_CITY_EN_RE):
        match = pattern.search(masked)
        if match:
            city = _clean_slot(match.groupdict().get("city"))
            break
    if city:
        slots["city"] = city
        slots["city_canon"] = city
        masked = masked.replace(city, "{city}", 1)

    masked = re.sub(r"(?<!\w)\d+(?:[.,]\d+)?(?!\w)", "{number}", masked)
    masked = re.sub(r"\s+", " ", masked).strip()
    return MaskResult(normalized=normalized, masked=masked, slots=slots)


def _stable_hash(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _ngrams(tokens: list[str], sizes: Iterable[int]) -> Iterable[str]:
    for size in sizes:
        if size <= 0 or len(tokens) < size:
            continue
        for index in range(0, len(tokens) - size + 1):
            yield " ".join(tokens[index : index + size])


class NeuroLiteRuntime:
    def __init__(self) -> None:
        self.root = _artifact_root()
        self.config = self._load_config()
        self.examples: list[Example] = []
        self.intent_examples: dict[str, list[Example]] = {}
        self.prototype_vectors: dict[str, dict[int, float]] = {}
        self.example_vectors: list[dict[int, float]] = []
        self.model_id = ""
        self.reload()

    def _load_config(self) -> VectorizerConfig:
        path = _artifact_root() / "vectorizer.json"
        if not path.exists():
            return VectorizerConfig()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return VectorizerConfig()
        return VectorizerConfig(
            dim=max(128, int(data.get("dim") or 4096)),
            char_ngrams=tuple(int(x) for x in data.get("char_ngrams", [3, 4, 5])),
            word_ngrams=tuple(int(x) for x in data.get("word_ngrams", [1, 2])),
            min_similarity=float(data.get("min_similarity", 0.16)),
            min_margin=float(data.get("min_margin", 0.05)),
        )

    def _example_paths(self) -> list[Path]:
        explicit = os.getenv("ADAOS_NEURO_LITE_EXAMPLES_PATHS", "").strip()
        paths: list[Path] = []
        if explicit:
            for item in re.split(r"[;,]", explicit):
                token = item.strip()
                if token:
                    paths.append(Path(token).expanduser().resolve())
        paths.append(self.root / "examples_manifest.jsonl")
        return paths

    def _load_examples(self) -> list[Example]:
        rows: list[Example] = []
        seen: set[tuple[str, str]] = set()
        for path in self._example_paths():
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if not isinstance(item, dict):
                    continue
                intent = str(item.get("intent") or item.get("skill") or "").strip()
                text = str(item.get("text") or "").strip()
                if not intent or not text:
                    continue
                masked = str(item.get("masked") or _mask_text(text).masked).strip()
                key = (intent, masked)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(Example(intent=intent, text=text, masked=masked))
        if rows:
            return rows
        return [Example(intent=intent, text=text, masked=_mask_text(text).masked) for intent, text in _DEFAULT_EXAMPLES]

    def reload(self) -> dict[str, Any]:
        self.config = self._load_config()
        self.examples = self._load_examples()
        grouped: dict[str, list[Example]] = defaultdict(list)
        for example in self.examples:
            grouped[example.intent].append(example)
        self.intent_examples = dict(grouped)
        self.example_vectors = [self.vectorize(example.masked) for example in self.examples]
        self.prototype_vectors = {
            intent: self._average_vectors([self.vectorize(example.masked) for example in items])
            for intent, items in self.intent_examples.items()
        }
        digest = hashlib.sha256()
        for example in self.examples:
            digest.update(example.intent.encode("utf-8"))
            digest.update(b"\0")
            digest.update(example.masked.encode("utf-8"))
            digest.update(b"\0")
        self.model_id = f"neuro-lite-{digest.hexdigest()[:12]}"
        return self.health()

    def vectorize(self, text: str) -> dict[int, float]:
        masked = _normalize(text)
        values: dict[int, float] = defaultdict(float)
        compact = f"^{masked}$"
        chars = list(compact)
        for gram in _ngrams(chars, self.config.char_ngrams):
            index = _stable_hash("c:" + gram) % self.config.dim
            values[index] += 1.0
        words = re.findall(r"[\w{}]+", masked, flags=re.UNICODE)
        for gram in _ngrams(words, self.config.word_ngrams):
            index = _stable_hash("w:" + gram) % self.config.dim
            values[index] += 1.25
        return self._normalize_vector(values)

    @staticmethod
    def _normalize_vector(values: dict[int, float]) -> dict[int, float]:
        norm = math.sqrt(sum(value * value for value in values.values()))
        if norm <= 0.0:
            return {}
        return {key: value / norm for key, value in values.items() if value}

    def _average_vectors(self, vectors: list[dict[int, float]]) -> dict[int, float]:
        if not vectors:
            return {}
        merged: dict[int, float] = defaultdict(float)
        for vector in vectors:
            for key, value in vector.items():
                merged[key] += value
        scale = float(len(vectors))
        return self._normalize_vector({key: value / scale for key, value in merged.items()})

    @staticmethod
    def _dot(left: dict[int, float], right: dict[int, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(key, 0.0) for key, value in left.items())

    def _nearest_examples(self, query: dict[int, float], *, intent: str | None = None, limit: int = 3) -> list[dict[str, Any]]:
        matches: list[tuple[float, Example]] = []
        for example, vector in zip(self.examples, self.example_vectors):
            if intent and example.intent != intent:
                continue
            matches.append((self._dot(query, vector), example))
        matches.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "intent": example.intent,
                "similarity": round(score, 6),
                "matched_example": example.masked,
                "raw_example": example.text,
            }
            for score, example in matches[:limit]
        ]

    def detect(self, text: str, *, webspace_id: str | None = None, locale: str | None = None) -> dict[str, Any]:
        masked = _mask_text(text)
        query = self.vectorize(masked.masked)
        scored = [
            (intent, self._dot(query, vector))
            for intent, vector in self.prototype_vectors.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        top_intent, top_score = scored[0] if scored else ("", 0.0)
        second_intent, second_score = scored[1] if len(scored) > 1 else ("", 0.0)
        margin = top_score - second_score
        accepted = bool(top_intent and top_score >= self.config.min_similarity and margin >= self.config.min_margin)
        confidence = max(0.0, min(0.99, 0.55 * top_score + 0.45 * max(0.0, min(1.0, margin * 2.0))))
        nearest_positive = self._nearest_examples(query, intent=top_intent, limit=3) if top_intent else []
        nearest_negative = self._nearest_examples(query, limit=5)
        nearest_negative = [item for item in nearest_negative if item["intent"] != top_intent][:3]
        alternatives = [
            {"intent": intent, "confidence": round(max(0.0, min(0.99, score)), 6)}
            for intent, score in scored[1:5]
        ]
        reason = ""
        if not accepted:
            reason = "below_similarity_threshold" if top_score < self.config.min_similarity else "below_margin_threshold"
        return {
            "top_intent": top_intent if accepted else None,
            "intent": top_intent if accepted else None,
            "confidence": round(confidence if accepted else 0.0, 6),
            "accepted": accepted,
            "alternatives": alternatives,
            "slots": masked.slots,
            "via": "neuro_lite",
            "model_id": self.model_id,
            "evidence": {
                "backend": "hash_ngram_prototypes",
                "reason": reason,
                "webspace_id": webspace_id,
                "locale": locale,
                "normalized_text": masked.normalized,
                "masked_text": masked.masked,
                "positive_similarity": round(top_score, 6),
                "negative_similarity": round(second_score, 6),
                "positive_negative_margin": round(margin, 6),
                "second_intent": second_intent,
                "matched_examples": [item["matched_example"] for item in nearest_positive],
                "nearest_positive_examples": nearest_positive,
                "nearest_negative_examples": nearest_negative,
                "thresholds": {
                    "min_similarity": self.config.min_similarity,
                    "min_margin": self.config.min_margin,
                },
            },
        }

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": _SERVICE_NAME,
            "version": _SERVICE_VERSION,
            "backend": "hash_ngram_prototypes",
            "model_loaded": bool(self.examples and self.prototype_vectors),
            "model_id": self.model_id,
            "examples_total": len(self.examples),
            "intents_total": len(self.prototype_vectors),
            "artifact_root": str(self.root),
            "dependencies": [],
            "runtime": "stdlib",
        }


_RUNTIME = NeuroLiteRuntime()


class Handler(BaseHTTPRequestHandler):
    server_version = "AdaOSNeuroNLULite/0.1"

    def log_message(self, _format: str, *args: Any) -> None:  # noqa: A003
        if _env_flag("ADAOS_NEURO_LITE_HTTP_LOG"):
            super().log_message(_format, *args)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            _json_response(self, 200, _RUNTIME.health())
            return
        _json_response(self, 404, {"ok": False, "error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        started = time.perf_counter()
        if self.path == "/rebuild":
            try:
                result = _RUNTIME.reload()
            except Exception as exc:
                _log("rebuild.error", level="ERROR", error=repr(exc), traceback=traceback.format_exc(limit=10))
                _json_response(self, 500, {"ok": False, "error": "rebuild_failed", "message": str(exc)})
                return
            _log("rebuild", latency_ms=round((time.perf_counter() - started) * 1000.0, 3), **result)
            _json_response(self, 200, result)
            return

        if self.path != "/parse":
            _json_response(self, 404, {"ok": False, "error": "not_found"})
            return

        payload = _read_json(self)
        text = payload.get("text") or payload.get("utterance")
        if not isinstance(text, str) or not text.strip():
            _json_response(self, 400, {"ok": False, "error": "text_required"})
            return
        try:
            result = _RUNTIME.detect(
                text.strip(),
                webspace_id=payload.get("webspace_id") if isinstance(payload.get("webspace_id"), str) else None,
                locale=payload.get("locale") if isinstance(payload.get("locale"), str) else None,
            )
        except Exception as exc:
            _log("parse.error", level="ERROR", error=repr(exc), traceback=traceback.format_exc(limit=10))
            _json_response(self, 500, {"ok": False, "error": "parse_failed", "message": str(exc)})
            return
        _log(
            "parse",
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            accepted=result.get("accepted"),
            top_intent=result.get("top_intent"),
            confidence=result.get("confidence"),
            model_id=result.get("model_id"),
        )
        _json_response(self, 200, {"ok": True, **result, "result": result})


if __name__ == "__main__":
    host = os.getenv("ADAOS_SERVICE_HOST", "127.0.0.1")
    try:
        port = int(os.getenv("ADAOS_SERVICE_PORT", "18093") or "18093")
    except Exception:
        port = 18093
    _log("service.start", host=host, port=port, **_RUNTIME.health())
    ThreadingHTTPServer((host, port), Handler).serve_forever()
