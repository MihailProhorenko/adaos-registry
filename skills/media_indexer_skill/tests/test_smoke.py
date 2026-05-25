from __future__ import annotations

import importlib
import json
import pathlib
import sys

import yaml


SKILL_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))


def test_manifest_declares_runtime_contracts() -> None:
    manifest = yaml.safe_load((SKILL_ROOT / "skill.yaml").read_text(encoding="utf-8"))

    assert manifest["name"] == "media_indexer_skill"
    assert "requirements.txt" not in {path.name for path in SKILL_ROOT.iterdir()}
    assert "faiss-cpu==1.13.2" in manifest["dependencies"]
    assert "torch==2.10.0" in manifest["dependencies"]
    assert "media_indexer.action" in manifest["events"]["subscribe"]
    assert "webio.stream.snapshot.requested" in manifest["events"]["subscribe"]
    assert any(route["route"] == "stream" and route["receiver"] == "media_indexer.operations" for route in manifest["data_routes"])
    assert manifest["lifecycle"]["rehydrate"] == "rehydrate"
    assert manifest["lifecycle"]["dispose"] == "dispose"


def test_webui_declares_compact_yjs_and_stream_receiver() -> None:
    webui = json.loads((SKILL_ROOT / "webui.json").read_text(encoding="utf-8"))

    assert webui["ydoc_defaults"]["data/media_indexer"]["form"]["directory"] == ""
    assert "D:\\diploma_final\\demo_media" not in json.dumps(webui)
    receiver = webui["webio"]["receivers"]["media_indexer.operations"]
    assert receiver["mode"] == "replace"
    assert receiver["snapshotPolicy"] == "on_subscribe"


def test_scanner_finds_supported_media_without_hashing(tmp_path: pathlib.Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "clip.mp4").write_bytes(b"not a real mp4")
    (media_dir / "notes.txt").write_text("ignored", encoding="utf-8")

    from lib.scanner import DirectoryScanner

    inventory = DirectoryScanner(str(media_dir), compute_hashes=False).scan()

    assert [item.name for item in inventory["video"]] == ["clip.mp4"]
    assert "audio" not in inventory


def test_handler_import_is_passive_and_search_without_index_does_not_load_models(monkeypatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setenv("ADAOS_SKILL_ENV_PATH", str(tmp_path / "skill_env.json"))
    monkeypatch.setenv("MEDIA_INDEXER_DATA_DIR", str(tmp_path / "data"))

    main = importlib.import_module("handlers.main")
    main.dispose()

    result = main.search_media("anything")

    assert result["status"] == "error"
    assert result["results"] == []
    assert main._state["vector_db"] is None
