from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from uuid import uuid4

import yaml


SKILL_ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = SKILL_ROOT / "handlers" / "main.py"
    module_name = f"test_ai_event_analysis_skill_{uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_declares_measurable_tools_and_stream_wakeup() -> None:
    manifest = yaml.safe_load((SKILL_ROOT / "skill.yaml").read_text(encoding="utf-8"))

    assert manifest["name"] == "ai_event_analysis_skill"
    assert "webio.stream.snapshot.requested" in manifest["events"]["subscribe"]
    assert "ai_event_analysis.evaluate_requested" in manifest["events"]["subscribe"]
    assert {tool["name"] for tool in manifest["tools"]} == {
        "get_lab_snapshot",
        "run_demo_evaluation",
        "evaluate_windows",
        "import_local_logs",
        "build_event_windows",
        "export_event_windows_jsonl",
    }


def test_webui_declares_app_widget_and_results_receiver() -> None:
    webui = json.loads((SKILL_ROOT / "webui.json").read_text(encoding="utf-8"))

    assert webui["apps"][0]["id"] == "ai_event_analysis_app"
    assert webui["widgets"][0]["id"] == "ai_event_analysis_widget"
    assert webui["webio"]["receivers"]["ai_event_analysis.results"]["snapshotPolicy"] == "on_subscribe"
    widgets = webui["registry"]["modals"]["ai_event_analysis_modal"]["schema"]["widgets"]
    assert any(widget["type"] == "visual.metricChart" for widget in widgets)
    assert any(widget["type"] == "ui.table" for widget in widgets)
    assert any(widget["type"] == "ui.list" for widget in widgets)
    tabs = next(widget for widget in widgets if widget["id"] == "ai-event-analysis-tabs")
    assert any(button["id"] == "windows" for button in tabs["inputs"]["buttons"])


def test_rule_baseline_returns_required_measurement_fields() -> None:
    mod = _load_module()

    result = mod.run_demo_evaluation({"webspace_id": "test"})["result"]

    assert result["model"] == "rule_baseline_v1"
    assert result["window_count"] >= 10
    assert result["macro_f1"] >= 0.75
    assert result["critical_recall"] >= 0.85
    assert result["false_positive_rate"] <= 0.15
    assert result["top_reason_hit_rate"] > 0
    assert result["per_class"]


def test_custom_window_evaluation_reports_false_positive_rate() -> None:
    mod = _load_module()

    windows = [
        {
            "window_id": "normal-1",
            "features": {
                "event_total": 10,
                "error_total": 0,
                "drop_total": 0,
                "projection_refresh_total": 2,
                "same_projection_refresh_max": 1,
                "yjs_write_total": 1,
            },
            "label": {"incident": False, "incident_type": "normal", "severity": "info", "reasons": []},
        }
    ]
    result = mod.evaluate_windows({"windows": windows})["result"]

    assert result["accuracy"] == 1.0
    assert result["false_positive_rate"] == 0.0


def test_local_log_import_builds_redacted_event_windows(tmp_path: Path) -> None:
    mod = _load_module()
    log_path = tmp_path / "runtime.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-05-29T09:00:01Z INFO runtime ready token=abc123",
                "2026-05-29T09:00:02Z WARN projection refresh repeated for status-card",
                "2026-05-29T09:00:03Z ERROR yjs write pressure at C:\\Users\\secret\\node.yaml",
            ]
        ),
        encoding="utf-8",
    )

    imported = mod.import_local_logs({"path": str(log_path), "max_lines": 10})
    assert imported["summary"]["record_count"] == 3
    assert "<redacted>" in imported["records"][0]["message"]
    assert "<path>" in imported["records"][2]["message"]

    built = mod.build_event_windows({"records": imported["records"], "window_seconds": 60})
    windows = built["result"]["windows"]
    assert built["result"]["window_count"] == 1
    assert windows[0]["features"]["event_total"] == 3
    assert windows[0]["features"]["projection_refresh_total"] == 1
    assert windows[0]["features"]["yjs_write_total"] == 1


def test_event_window_export_writes_jsonl(tmp_path: Path) -> None:
    mod = _load_module()
    out = tmp_path / "windows.jsonl"
    windows = [
        {
            "window_id": "w1",
            "features": {"event_total": 1},
            "label": {"incident": False, "incident_type": "normal", "severity": "info", "reasons": []},
        }
    ]

    result = mod.export_event_windows_jsonl({"windows": windows, "path": str(out)})

    assert result["ok"] is True
    assert result["export"]["count"] == 1
    assert json.loads(out.read_text(encoding="utf-8"))["window_id"] == "w1"
