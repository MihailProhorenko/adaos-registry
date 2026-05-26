from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_main_module():
    path = Path(__file__).resolve().parents[1] / "handlers" / "main.py"
    spec = importlib.util.spec_from_file_location("_rasa_nlu_service_main_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Runtime:
    def __init__(self, token: str) -> None:
        self.token = token
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_load_runtime_reloads_when_model_file_changes(tmp_path, monkeypatch):
    module = _load_main_module()
    model_path = tmp_path / "interpreter_latest.tar.gz"
    model_path.write_text("first", encoding="utf-8")
    calls: list[Path] = []

    def load_model(path: Path):
        calls.append(path)
        return _Runtime(f"runtime-{len(calls)}")

    monkeypatch.setitem(sys.modules, "adaos_rasa_nlu", SimpleNamespace(load_model=load_model))

    first = module._load_runtime(model_path)
    cached = module._load_runtime(model_path)
    assert cached is first

    new_ns = model_path.stat().st_mtime_ns + 1_000_000_000
    model_path.write_text("second", encoding="utf-8")
    os.utime(model_path, ns=(new_ns, new_ns))

    second = module._load_runtime(model_path)
    assert second is not first
    assert first.closed is True
    assert [runtime_path for runtime_path in calls] == [model_path, model_path]


def test_train_forces_reload_even_when_path_is_unchanged(tmp_path, monkeypatch):
    module = _load_main_module()
    project_dir = tmp_path / "project"
    out_dir = tmp_path / "models"
    project_dir.mkdir()
    out_dir.mkdir()
    model_path = out_dir / "interpreter_latest.tar.gz"
    model_path.write_text("model", encoding="utf-8")
    calls: list[Path] = []

    def load_model(path: Path):
        calls.append(path)
        return _Runtime(f"runtime-{len(calls)}")

    def train_nlu(_project_dir: Path, _out_dir: Path, *, fixed_model_name: str):
        assert _project_dir == project_dir
        assert _out_dir == out_dir
        assert fixed_model_name == "interpreter_latest"
        return SimpleNamespace(model_path=model_path)

    monkeypatch.setitem(
        sys.modules,
        "adaos_rasa_nlu",
        SimpleNamespace(load_model=load_model, train_nlu=train_nlu),
    )

    module._load_runtime(model_path)
    result = module._train({"project_dir": str(project_dir), "out_dir": str(out_dir)})

    assert result == {"ok": True, "model_path": str(model_path.resolve())}
    assert calls == [model_path, model_path.resolve()]
