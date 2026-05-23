from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from uuid import uuid4

import yaml


if "y_py" not in sys.modules:
    sys.modules["y_py"] = types.SimpleNamespace(
        YDoc=type("YDoc", (), {}),
        encode_state_vector=lambda *args, **kwargs: b"",
        encode_state_as_update=lambda *args, **kwargs: b"",
        apply_update=lambda *args, **kwargs: None,
    )
if "ypy_websocket.ystore" not in sys.modules:
    ystore_module = types.ModuleType("ypy_websocket.ystore")
    ystore_module.BaseYStore = type("BaseYStore", (), {})
    ystore_module.YDocNotFound = type("YDocNotFound", (Exception,), {})
    sys.modules["ypy_websocket.ystore"] = ystore_module
if "ypy_websocket" not in sys.modules:
    pkg = types.ModuleType("ypy_websocket")
    pkg.ystore = sys.modules["ypy_websocket.ystore"]
    sys.modules["ypy_websocket"] = pkg


def _load_voice_chat_module():
    path = Path(__file__).resolve().parents[1] / "handlers" / "main.py"
    module_name = f"test_voice_chat_skill_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_voice_chat_get_snapshot_uses_projected_state_without_yjs(monkeypatch):
    mod = _load_voice_chat_module()
    assert not hasattr(mod, "get_ydoc")

    projected: list[tuple[str, str | None, object]] = []
    streamed: list[tuple[str, object, dict[str, object]]] = []
    monkeypatch.setattr(
        mod.ctx_subnet,
        "set",
        lambda slot, value, *, webspace_id=None: projected.append((slot, webspace_id, value)),
    )
    monkeypatch.setattr(
        mod._STREAM_RUNTIME,
        "publish_snapshot",
        lambda receiver, data, **kwargs: streamed.append((receiver, data, kwargs)),
    )

    state = mod._state_for("desktop", "member-01")
    state["messages"] = [
        {"id": "m-1", "from": "user", "text": "weather in Berlin"},
    ]
    state["last_refresh_ts"] = 123.0

    snapshot = mod.get_snapshot(webspace_id="desktop", target_node_id="member-01")

    assert snapshot["voice_chat"]["messages"][0]["text"] == "weather in Berlin"
    assert snapshot["messages"][0]["text"] == "weather in Berlin"
    assert snapshot["last_refresh_ts"] == 123.0
    assert projected and projected[0][0] == "voice_chat.state"
    assert streamed and streamed[0][0] == "voice_chat.messages"
    assert streamed[0][2]["webspace_id"] == "desktop"
    assert streamed[0][2]["force"] is True


def test_voice_chat_skill_yaml_exports_get_snapshot():
    manifest = (
        Path(__file__).resolve().parents[1]
        / "skill.yaml"
    )
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))

    tools = payload.get("tools") or []
    assert any((item or {}).get("name") == "get_snapshot" for item in tools)
