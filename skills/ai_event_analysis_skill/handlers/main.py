from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from adaos.sdk.core.decorators import subscribe, tool
from adaos.sdk.data.context import clear_current_skill, set_current_skill
from adaos.sdk.data.events import publish as publish_event
from adaos.sdk.io.out import stream_publish

_RESULTS_RECEIVER = "ai_event_analysis.results"
_SKILL_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _SKILL_ROOT / "data"
_DEFAULT_EXPORT_PATH = _DATA_DIR / "event_windows.jsonl"
_MAX_AUTO_LOG_SOURCES = 4
_MAX_DEFAULT_LOG_LINES = 120
_MAX_EXPLICIT_LOG_LINES = 1000
_MAX_TOOL_WINDOWS = 64
_MAX_STREAM_ROWS = 12
_MAX_STREAM_POINTS = 24
_MAX_EVIDENCE_PER_WINDOW = 4
_TS_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
)

_CLASSES = [
    "normal",
    "eventbus_backpressure",
    "projection_refresh_storm",
    "yjs_write_pressure",
    "browser_session_instability",
    "member_node_disconnect",
    "runtime_rebuild_churn",
]
_PROJECTION_FINGERPRINTS: dict[str, str] = {}


class _LazyCtxSubnet:
    def set(self, *args: Any, **kwargs: Any) -> Any:
        from adaos.sdk.data import ctx_subnet as real_ctx_subnet

        return real_ctx_subnet.set(*args, **kwargs)


ctx_subnet = _LazyCtxSubnet()


def _webspace_id_from_payload(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return "desktop"
    raw = payload.get("webspace_id") or payload.get("workspace_id")
    if isinstance(raw, str) and raw.strip() and not raw.strip().startswith("$"):
        return raw.strip()
    meta = payload.get("_meta")
    if isinstance(meta, Mapping):
        nested = meta.get("webspace_id") or meta.get("workspace_id")
        if isinstance(nested, str) and nested.strip() and not nested.strip().startswith("$"):
            return nested.strip()
    return "desktop"


def _fingerprint(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except Exception:
        return repr(value)


def _set_projection_if_changed(slot: str, value: Any, *, webspace_id: str, force: bool = False) -> bool:
    key = f"{webspace_id}:{slot}"
    fingerprint = _fingerprint(value)
    if not force and _PROJECTION_FINGERPRINTS.get(key) == fingerprint:
        return False
    ctx_subnet.set(slot, value, webspace_id=webspace_id)
    _PROJECTION_FINGERPRINTS[key] = fingerprint
    return True


def _project_sections(sections: Mapping[str, Any], *, webspace_id: str, force: bool = False) -> dict[str, Any]:
    slot_by_section = {
        "summary": "ai_event_analysis.summary",
        "task": "ai_event_analysis.task",
        "dataset": "ai_event_analysis.dataset",
        "windows": "ai_event_analysis.windows",
        "metrics": "ai_event_analysis.metrics",
        "per_class": "ai_event_analysis.per_class",
        "chart": "ai_event_analysis.chart",
        "event_volume_chart": "ai_event_analysis.event_volume_chart",
        "class_distribution_chart": "ai_event_analysis.class_distribution_chart",
        "experiments": "ai_event_analysis.experiments",
    }
    pushed = False
    written: list[str] = []
    try:
        pushed = bool(set_current_skill("ai_event_analysis_skill"))
    except Exception:
        pushed = False
    try:
        for section, slot in slot_by_section.items():
            if section not in sections:
                continue
            try:
                if _set_projection_if_changed(slot, sections[section], webspace_id=webspace_id, force=force):
                    written.append(section)
            except Exception:
                continue
    finally:
        if pushed:
            try:
                clear_current_skill()
            except Exception:
                pass
    return {"ok": True, "written": written, "webspace_id": webspace_id}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_ts(raw: str) -> float | None:
    match = _TS_RE.search(raw or "")
    if not match:
        return None
    value = match.group("ts").replace(",", ".").replace(" ", "T")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    if re.search(r"[+-]\d{4}$", value):
        value = value[:-5] + value[-5:-2] + ":" + value[-2:]
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return None


def _severity_from_text(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("critical", "fatal", "traceback", "exception")):
        return "critical"
    if any(token in lowered for token in ("error", "failed", "failure")):
        return "error"
    if any(token in lowered for token in ("warning", "warn", "degraded", "retry")):
        return "warning"
    return "info"


def _topic_from_text(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("drop", "supersede", "backpressure", "queue")):
        return "eventbus.pressure"
    if any(token in lowered for token in ("projection", "refresh", "materializ")):
        return "projection.lifecycle"
    if any(token in lowered for token in ("yjs", "ydoc", "syncchannel")):
        return "yjs.sync"
    if any(token in lowered for token in ("browser", "session", "websocket", "/ws", "/yws")):
        return "browser.session"
    if any(token in lowered for token in ("member", "subnet", "node disconnect", "offline")):
        return "member.connectivity"
    if any(token in lowered for token in ("rebuild", "runtime", "supervisor", "core update")):
        return "runtime.lifecycle"
    return "runtime.log"


def _redact_line(text: str, *, max_len: int = 240) -> str:
    value = re.sub(r"(?i)(token|secret|password|authorization)=\S+", r"\1=<redacted>", text or "")
    value = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1<redacted>", value)
    value = re.sub(r"[A-Za-z]:\\[^\s]+", "<path>", value)
    value = re.sub(r"/(?:[\w.\-]+/){2,}[\w.\-]+", "<path>", value)
    return value[:max_len]


def _log_candidates() -> list[Path]:
    roots = [
        Path.cwd() / ".adaos" / "state",
        Path.cwd() / ".adaos" / "runtime",
        Path.cwd() / ".adaos" / "logs",
        _SKILL_ROOT.parent / "infrastate_skill",
    ]
    out: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*.log", "*.jsonl", "*.txt"):
            out.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted({path.resolve() for path in out})[:64]


def _read_log_records(path: Path, *, max_lines: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists() or not path.is_file():
        return records
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return records
    base_ts = time.time()
    bounded_lines = max(1, min(int(max_lines or _MAX_DEFAULT_LOG_LINES), _MAX_EXPLICIT_LOG_LINES))
    for index, line in enumerate(lines[-bounded_lines:]):
        if not line.strip():
            continue
        ts = _parse_ts(line)
        if ts is None:
            ts = base_ts + index * 0.001
        severity = _severity_from_text(line)
        topic = _topic_from_text(line)
        records.append(
            {
                "id": f"{path.name}:{index}",
                "ts": ts,
                "ts_iso": datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z"),
                "source": "local_log",
                "source_path": str(path),
                "topic": topic,
                "severity": severity,
                "message": _redact_line(line),
            }
        )
    return records


def _records_to_windows(
    records: list[Mapping[str, Any]],
    *,
    window_seconds: int = 60,
    node_id: str = "local",
    subnet_id: str = "local",
    webspace_id: str = "desktop",
) -> list[dict[str, Any]]:
    buckets: dict[int, list[Mapping[str, Any]]] = {}
    size = max(1, int(window_seconds or 60))
    for record in records:
        ts = _value(record, "ts")
        bucket = int(ts // size) * size
        buckets.setdefault(bucket, []).append(record)

    windows: list[dict[str, Any]] = []
    for bucket_start in sorted(buckets):
        items = buckets[bucket_start]
        topic_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}
        for item in items:
            topic = str(item.get("topic") or "runtime.log")
            severity = str(item.get("severity") or "info")
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        features = {
            "event_total": len(items),
            "error_total": severity_counts.get("error", 0) + severity_counts.get("critical", 0),
            "critical_total": severity_counts.get("critical", 0),
            "drop_total": topic_counts.get("eventbus.pressure", 0),
            "supersede_total": sum(1 for item in items if "supersede" in str(item.get("message", "")).lower()),
            "projection_refresh_total": topic_counts.get("projection.lifecycle", 0),
            "same_projection_refresh_max": topic_counts.get("projection.lifecycle", 0),
            "yjs_write_total": topic_counts.get("yjs.sync", 0),
            "browser_reconnect_total": topic_counts.get("browser.session", 0),
            "member_disconnect_total": topic_counts.get("member.connectivity", 0),
            "runtime_rebuild_total": topic_counts.get("runtime.lifecycle", 0),
        }
        prediction = _rule_predict({"features": features})
        windows.append(
            {
                "window_id": f"{node_id}:{bucket_start}-{bucket_start + size}",
                "scope": {
                    "node_id": node_id,
                    "subnet_id": subnet_id,
                    "webspace_id": webspace_id,
                },
                "time": {
                    "start_ts": bucket_start,
                    "end_ts": bucket_start + size,
                    "start": datetime.fromtimestamp(bucket_start, timezone.utc).isoformat().replace("+00:00", "Z"),
                    "end": datetime.fromtimestamp(bucket_start + size, timezone.utc).isoformat().replace("+00:00", "Z"),
                    "window_seconds": size,
                },
                "features": features,
                "evidence": [
                    {
                        "id": item.get("id"),
                        "topic": item.get("topic"),
                        "severity": item.get("severity"),
                        "message": item.get("message"),
                    }
                    for item in items[:_MAX_EVIDENCE_PER_WINDOW]
                ],
                "label": {
                    "incident": False,
                    "incident_type": "normal",
                    "severity": "unlabeled",
                    "reasons": [],
                    "source": "unlabeled_import",
                },
                "baseline_prediction": prediction,
            }
        )
    return windows


def _class_distribution_points(windows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for window in windows:
        prediction = window.get("baseline_prediction") if isinstance(window.get("baseline_prediction"), Mapping) else {}
        label = window.get("label") if isinstance(window.get("label"), Mapping) else {}
        key = str(prediction.get("incident_type") or label.get("incident_type") or "normal")
        counts[key] = counts.get(key, 0) + 1
    return [{"ts": key, "value": value} for key, value in sorted(counts.items())]


def _event_volume_points(windows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for index, window in enumerate(windows[:48]):
        features = window.get("features") if isinstance(window.get("features"), Mapping) else {}
        time_info = window.get("time") if isinstance(window.get("time"), Mapping) else {}
        label = str(time_info.get("start") or index)
        points.append({"ts": label[-13:-4] if len(label) > 12 else label, "value": _value(features, "event_total")})
    return points


def _window_rows(windows: list[Mapping[str, Any]], *, limit: int = 32) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in windows[:limit]:
        features = window.get("features") if isinstance(window.get("features"), Mapping) else {}
        prediction = window.get("baseline_prediction") if isinstance(window.get("baseline_prediction"), Mapping) else _rule_predict(window)
        rows.append(
            {
                "id": window.get("window_id"),
                "window_id": window.get("window_id"),
                "events": int(_value(features, "event_total")),
                "errors": int(_value(features, "error_total")),
                "prediction": prediction.get("incident_type"),
                "severity": prediction.get("severity"),
                "confidence": prediction.get("confidence"),
            }
        )
    return rows


def _compact_evaluation_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": result.get("model"),
        "window_count": result.get("window_count"),
        "accuracy": result.get("accuracy"),
        "macro_f1": result.get("macro_f1"),
        "false_positive_rate": result.get("false_positive_rate"),
        "critical_recall": result.get("critical_recall"),
        "avg_detection_delay_s": result.get("avg_detection_delay_s"),
        "top_reason_hit_rate": result.get("top_reason_hit_rate"),
        "per_class": list(result.get("per_class") or []),
        "evaluated_at": result.get("evaluated_at"),
    }


def _compact_dataset_result(result: Mapping[str, Any]) -> dict[str, Any]:
    def compact_chart(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        chart = dict(value)
        points = chart.get("points")
        if isinstance(points, list):
            chart["points"] = points[:_MAX_STREAM_POINTS]
            chart["truncated"] = len(points) > _MAX_STREAM_POINTS
        return chart

    return {
        "window_count": result.get("window_count"),
        "record_count": result.get("record_count"),
        "window_seconds": result.get("window_seconds"),
        "rows": list(result.get("rows") or [])[:_MAX_STREAM_ROWS],
        "event_volume_chart": compact_chart(result.get("event_volume_chart")),
        "class_distribution_chart": compact_chart(result.get("class_distribution_chart")),
        "built_at": result.get("built_at"),
    }


def _export_jsonl(windows: list[Mapping[str, Any]], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for window in windows:
            fh.write(json.dumps(dict(window), ensure_ascii=False, sort_keys=True) + "\n")
    return {"path": str(path), "count": len(windows), "bytes": path.stat().st_size}


def _window(
    window_id: str,
    incident_type: str,
    severity: str,
    features: Mapping[str, float],
    *,
    first_symptom_s: float = 0.0,
    labeled_at_s: float = 60.0,
) -> dict[str, Any]:
    return {
        "window_id": window_id,
        "features": dict(features),
        "label": {
            "incident": incident_type != "normal",
            "incident_type": incident_type,
            "severity": severity,
            "reasons": _top_feature_names(features),
        },
        "timing": {
            "first_symptom_s": first_symptom_s,
            "labeled_at_s": labeled_at_s,
        },
    }


def _synthetic_windows() -> list[dict[str, Any]]:
    return [
        _window("demo-001", "normal", "info", {"event_total": 18, "error_total": 0, "drop_total": 0, "projection_refresh_total": 4, "same_projection_refresh_max": 2, "yjs_write_total": 3, "browser_reconnect_total": 0, "member_disconnect_total": 0, "runtime_rebuild_total": 0}),
        _window("demo-002", "normal", "info", {"event_total": 31, "error_total": 1, "drop_total": 0, "projection_refresh_total": 6, "same_projection_refresh_max": 3, "yjs_write_total": 5, "browser_reconnect_total": 1, "member_disconnect_total": 0, "runtime_rebuild_total": 0}),
        _window("demo-003", "eventbus_backpressure", "warning", {"event_total": 220, "error_total": 7, "drop_total": 18, "supersede_total": 42, "projection_refresh_total": 19, "same_projection_refresh_max": 7, "yjs_write_total": 8, "browser_reconnect_total": 1, "member_disconnect_total": 0, "runtime_rebuild_total": 0}, first_symptom_s=12),
        _window("demo-004", "projection_refresh_storm", "warning", {"event_total": 140, "error_total": 3, "drop_total": 2, "supersede_total": 9, "projection_refresh_total": 96, "same_projection_refresh_max": 61, "yjs_write_total": 19, "browser_reconnect_total": 0, "member_disconnect_total": 0, "runtime_rebuild_total": 0}, first_symptom_s=18),
        _window("demo-005", "yjs_write_pressure", "critical", {"event_total": 88, "error_total": 4, "drop_total": 1, "supersede_total": 4, "projection_refresh_total": 34, "same_projection_refresh_max": 15, "yjs_write_total": 168, "browser_reconnect_total": 2, "member_disconnect_total": 0, "runtime_rebuild_total": 0}, first_symptom_s=9),
        _window("demo-006", "browser_session_instability", "warning", {"event_total": 64, "error_total": 2, "drop_total": 0, "supersede_total": 2, "projection_refresh_total": 12, "same_projection_refresh_max": 5, "yjs_write_total": 10, "browser_reconnect_total": 12, "member_disconnect_total": 0, "runtime_rebuild_total": 0}, first_symptom_s=16),
        _window("demo-007", "member_node_disconnect", "critical", {"event_total": 52, "error_total": 5, "drop_total": 1, "supersede_total": 0, "projection_refresh_total": 8, "same_projection_refresh_max": 3, "yjs_write_total": 7, "browser_reconnect_total": 0, "member_disconnect_total": 3, "runtime_rebuild_total": 0}, first_symptom_s=5),
        _window("demo-008", "runtime_rebuild_churn", "warning", {"event_total": 104, "error_total": 3, "drop_total": 0, "supersede_total": 8, "projection_refresh_total": 28, "same_projection_refresh_max": 10, "yjs_write_total": 25, "browser_reconnect_total": 1, "member_disconnect_total": 0, "runtime_rebuild_total": 5}, first_symptom_s=21),
        _window("demo-009", "projection_refresh_storm", "critical", {"event_total": 260, "error_total": 9, "drop_total": 7, "supersede_total": 34, "projection_refresh_total": 180, "same_projection_refresh_max": 124, "yjs_write_total": 52, "browser_reconnect_total": 2, "member_disconnect_total": 0, "runtime_rebuild_total": 1}, first_symptom_s=8),
        _window("demo-010", "eventbus_backpressure", "critical", {"event_total": 420, "error_total": 21, "drop_total": 55, "supersede_total": 90, "projection_refresh_total": 44, "same_projection_refresh_max": 13, "yjs_write_total": 28, "browser_reconnect_total": 2, "member_disconnect_total": 1, "runtime_rebuild_total": 0}, first_symptom_s=4),
        _window("demo-011", "normal", "info", {"event_total": 46, "error_total": 0, "drop_total": 0, "supersede_total": 1, "projection_refresh_total": 9, "same_projection_refresh_max": 3, "yjs_write_total": 7, "browser_reconnect_total": 1, "member_disconnect_total": 0, "runtime_rebuild_total": 1}),
        _window("demo-012", "browser_session_instability", "critical", {"event_total": 94, "error_total": 8, "drop_total": 2, "supersede_total": 4, "projection_refresh_total": 18, "same_projection_refresh_max": 7, "yjs_write_total": 12, "browser_reconnect_total": 23, "member_disconnect_total": 0, "runtime_rebuild_total": 0}, first_symptom_s=7),
    ]


def _value(features: Mapping[str, Any], key: str) -> float:
    raw = features.get(key, 0)
    try:
        return float(raw)
    except Exception:
        return 0.0


def _top_feature_names(features: Mapping[str, Any], limit: int = 3) -> list[str]:
    scored = sorted(
        ((key, abs(_value(features, key))) for key in features),
        key=lambda item: item[1],
        reverse=True,
    )
    return [key for key, value in scored[:limit] if value > 0]


def _rule_predict(window: Mapping[str, Any]) -> dict[str, Any]:
    features = window.get("features") if isinstance(window.get("features"), Mapping) else {}
    assert isinstance(features, Mapping)
    reasons: list[str] = []

    def mark(*names: str) -> list[str]:
        reasons.clear()
        reasons.extend(names)
        return list(reasons)

    if _value(features, "drop_total") >= 15 or _value(features, "supersede_total") >= 40:
        incident_type = "eventbus_backpressure"
        reason_codes = mark("drop_total", "supersede_total", "event_total")
    elif _value(features, "same_projection_refresh_max") >= 30 or _value(features, "projection_refresh_total") >= 80:
        incident_type = "projection_refresh_storm"
        reason_codes = mark("same_projection_refresh_max", "projection_refresh_total")
    elif _value(features, "yjs_write_total") >= 90:
        incident_type = "yjs_write_pressure"
        reason_codes = mark("yjs_write_total", "projection_refresh_total")
    elif _value(features, "browser_reconnect_total") >= 8:
        incident_type = "browser_session_instability"
        reason_codes = mark("browser_reconnect_total")
    elif _value(features, "member_disconnect_total") >= 2:
        incident_type = "member_node_disconnect"
        reason_codes = mark("member_disconnect_total")
    elif _value(features, "runtime_rebuild_total") >= 3:
        incident_type = "runtime_rebuild_churn"
        reason_codes = mark("runtime_rebuild_total", "event_total")
    else:
        incident_type = "normal"
        reason_codes = _top_feature_names(features, limit=2)

    severity = "info"
    if incident_type != "normal":
        severity = "critical" if _value(features, "error_total") >= 8 or _value(features, "drop_total") >= 40 or _value(features, "yjs_write_total") >= 140 else "warning"
    confidence = 0.95 if incident_type != "normal" and len(reason_codes) > 1 else 0.72 if incident_type != "normal" else 0.68
    return {
        "incident": incident_type != "normal",
        "incident_type": incident_type,
        "severity": severity,
        "confidence": confidence,
        "reasons": reason_codes,
    }


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _evaluate(windows: list[Mapping[str, Any]]) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    labels: list[str] = []
    predicted: list[str] = []
    for window in windows:
        label = window.get("label") if isinstance(window.get("label"), Mapping) else {}
        assert isinstance(label, Mapping)
        actual_type = str(label.get("incident_type") or "normal")
        pred = _rule_predict(window)
        predictions.append({"window_id": window.get("window_id"), "actual": actual_type, "predicted": pred})
        labels.append(actual_type)
        predicted.append(str(pred["incident_type"]))

    classes = sorted(set(_CLASSES) | set(labels) | set(predicted))
    confusion = {actual: {pred_class: 0 for pred_class in classes} for actual in classes}
    for actual, pred_class in zip(labels, predicted):
        confusion[actual][pred_class] += 1

    per_class = []
    for class_name in classes:
        tp = confusion[class_name][class_name]
        fp = sum(confusion[other][class_name] for other in classes if other != class_name)
        fn = sum(confusion[class_name][other] for other in classes if other != class_name)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class.append(
            {
                "class": class_name,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(_f1(precision, recall), 3),
                "support": labels.count(class_name),
            }
        )

    correct = sum(1 for actual, pred_class in zip(labels, predicted) if actual == pred_class)
    normal_total = sum(1 for actual in labels if actual == "normal")
    normal_fp = sum(1 for actual, pred_class in zip(labels, predicted) if actual == "normal" and pred_class != "normal")
    critical_total = 0
    critical_found = 0
    delays: list[float] = []
    reason_hits = 0
    reason_total = 0
    for window, pred in zip(windows, predictions):
        label = window.get("label") if isinstance(window.get("label"), Mapping) else {}
        timing = window.get("timing") if isinstance(window.get("timing"), Mapping) else {}
        actual_severity = str(label.get("severity") or "")
        if actual_severity == "critical":
            critical_total += 1
            if pred["predicted"]["incident_type"] == label.get("incident_type"):
                critical_found += 1
        if pred["predicted"]["incident"]:
            first_symptom = _value(timing, "first_symptom_s")
            delays.append(max(0.0, 60.0 - first_symptom))
        expected_reasons = set(label.get("reasons") or [])
        predicted_reasons = set(pred["predicted"].get("reasons") or [])
        if expected_reasons:
            reason_total += 1
            if expected_reasons & predicted_reasons:
                reason_hits += 1

    macro_f1 = sum(row["f1"] for row in per_class) / len(per_class) if per_class else 0.0
    return {
        "model": "rule_baseline_v1",
        "window_count": len(windows),
        "accuracy": round(correct / len(windows), 3) if windows else 0.0,
        "macro_f1": round(macro_f1, 3),
        "false_positive_rate": round(normal_fp / normal_total, 3) if normal_total else 0.0,
        "critical_recall": round(critical_found / critical_total, 3) if critical_total else 0.0,
        "avg_detection_delay_s": round(sum(delays) / len(delays), 1) if delays else 0.0,
        "top_reason_hit_rate": round(reason_hits / reason_total, 3) if reason_total else 0.0,
        "per_class": per_class,
        "confusion": confusion,
        "predictions": predictions,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _metric_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": "accuracy", "metric": "Accuracy", "value": result.get("accuracy"), "target": "sanity only", "status": "info"},
        {"id": "macro_f1", "metric": "Macro-F1", "value": result.get("macro_f1"), "target": ">= 0.75", "status": "ok" if _value(result, "macro_f1") >= 0.75 else "warning"},
        {"id": "critical_recall", "metric": "Critical recall", "value": result.get("critical_recall"), "target": ">= 0.85", "status": "ok" if _value(result, "critical_recall") >= 0.85 else "warning"},
        {"id": "false_positive_rate", "metric": "Normal false positive rate", "value": result.get("false_positive_rate"), "target": "<= 0.15", "status": "ok" if _value(result, "false_positive_rate") <= 0.15 else "warning"},
        {"id": "avg_detection_delay_s", "metric": "Avg detection delay", "value": result.get("avg_detection_delay_s"), "target": "minimize", "status": "info"},
        {"id": "top_reason_hit_rate", "metric": "Top reason hit rate", "value": result.get("top_reason_hit_rate"), "target": "maximize", "status": "info"},
    ]


def _snapshot() -> dict[str, Any]:
    demo_windows = _synthetic_windows()
    demo_result = _evaluate(_synthetic_windows())
    return {
        "summary": {
            "label": "AI Event Analysis",
            "value": f"{demo_result['macro_f1']:.3f}",
            "subtitle": "rule baseline macro-F1",
            "description": "Measurable research task for operational event-window incident classification.",
            "buttons": [
                {"id": "open", "label": "Open"},
                {"id": "run_demo", "label": "Run demo evaluation"},
            ],
        },
        "task": {
            "items": [
                {
                    "id": "objective",
                    "title": "Objective",
                    "description": "Classify fixed operational event windows into normal or incident classes and return top contributing signals.",
                },
                {
                    "id": "dataset",
                    "title": "Dataset contract",
                    "description": "Use compact event-window rows with numeric features, labels, scope, timing, and redacted evidence references.",
                },
                {
                    "id": "measurement",
                    "title": "Measurement",
                    "description": "Track macro-F1, critical recall, normal false-positive rate, detection delay, explanation hit rate, and per-class scores.",
                },
            ]
        },
        "dataset": {
            "items": [
                {"id": "windows", "name": "Labeled windows", "current": len(demo_windows), "target": "500-1000+", "notes": "One row per fixed event window."},
                {"id": "classes", "name": "Classes", "current": len(_CLASSES), "target": "6+", "notes": "normal plus incident classes."},
                {"id": "features", "name": "Feature families", "current": 9, "target": "eventbus/projection/Yjs/device/runtime", "notes": "Aggregated numeric features for baseline and ML models."},
                {"id": "logs", "name": "Local log import", "current": len(_log_candidates()), "target": "explicit paths plus local candidates", "notes": "Core remains unchanged; this skill only reads local files when asked."},
            ]
        },
        "windows": {"items": _window_rows(demo_windows)},
        "metrics": {"items": _metric_rows(demo_result)},
        "per_class": {"items": demo_result["per_class"]},
        "chart": {
            "title": "Baseline quality",
            "unit": "score",
            "points": [
                {"ts": "accuracy", "value": demo_result["accuracy"]},
                {"ts": "macro-F1", "value": demo_result["macro_f1"]},
                {"ts": "critical recall", "value": demo_result["critical_recall"]},
                {"ts": "reason hit", "value": demo_result["top_reason_hit_rate"]},
            ],
        },
        "event_volume_chart": {
            "title": "Event volume by window",
            "unit": "events",
            "points": _event_volume_points(demo_windows),
        },
        "class_distribution_chart": {
            "title": "Baseline class distribution",
            "unit": "windows",
            "points": _class_distribution_points(demo_windows),
        },
        "experiments": {
            "items": [
                {"id": "rule_baseline", "model": "Rule baseline", "status": "implemented", "macro_f1": demo_result["macro_f1"], "next_step": "Use as baseline for all future models."},
                {"id": "classical_ml", "model": "Classical ML", "status": "planned", "macro_f1": "", "next_step": "Train logistic regression/random forest on imported windows."},
                {"id": "neural_window_model", "model": "Neural window model", "status": "planned", "macro_f1": "", "next_step": "Evaluate MLP/GRU/Transformer against the same split."},
            ]
        },
        "details": {
            "result": demo_result,
            "success_criteria": {
                "macro_f1": ">= 0.75",
                "critical_recall": ">= 0.85",
                "false_positive_rate": "<= 0.15",
                "top_reasons": "required for every prediction",
            },
        },
    }


def _project_lab_snapshot(*, webspace_id: str = "desktop", force: bool = False) -> dict[str, Any]:
    return _project_sections(_snapshot(), webspace_id=webspace_id, force=force)


def _project_evaluation_result(result: Mapping[str, Any], *, webspace_id: str) -> dict[str, Any]:
    sections = {
        "summary": {
            "label": "AI Event Analysis",
            "value": f"{_value(result, 'macro_f1'):.3f}",
            "subtitle": "rule baseline macro-F1",
            "description": (
                f"accuracy={result.get('accuracy')} critical_recall={result.get('critical_recall')} "
                f"normal_fpr={result.get('false_positive_rate')}"
            ),
            "buttons": [
                {"id": "open", "label": "Open"},
                {"id": "run_demo", "label": "Run demo evaluation"},
            ],
        },
        "metrics": {"items": _metric_rows(result)},
        "per_class": {"items": list(result.get("per_class") or [])},
        "chart": {
            "title": "Baseline quality",
            "unit": "score",
            "points": [
                {"ts": "accuracy", "value": result.get("accuracy")},
                {"ts": "macro-F1", "value": result.get("macro_f1")},
                {"ts": "critical recall", "value": result.get("critical_recall")},
                {"ts": "reason hit", "value": result.get("top_reason_hit_rate")},
            ],
        },
    }
    return _project_sections(sections, webspace_id=webspace_id)


def _project_windows_result(result: Mapping[str, Any], *, webspace_id: str) -> dict[str, Any]:
    sections = {
        "windows": {"items": list(result.get("rows") or [])},
        "event_volume_chart": result.get("event_volume_chart") or {"title": "Event volume by window", "unit": "events", "points": []},
        "class_distribution_chart": result.get("class_distribution_chart") or {"title": "Baseline class distribution", "unit": "windows", "points": []},
        "dataset": {
            "items": [
                {"id": "windows", "name": "Event windows", "current": result.get("window_count"), "target": "500-1000+", "notes": "Built from local logs or imported records."},
                {"id": "records", "name": "Evidence records", "current": result.get("record_count"), "target": "redacted operational evidence", "notes": "Raw logs stay out of Yjs."},
                {"id": "window_seconds", "name": "Window size", "current": result.get("window_seconds"), "target": "60/300/900 seconds", "notes": "Tune per experiment."},
            ]
        },
    }
    return _project_sections(sections, webspace_id=webspace_id)


def _publish_result(result: Mapping[str, Any], *, webspace_id: str) -> None:
    compact = _compact_evaluation_result(result)
    payload = [
        {
            "id": "summary",
            "title": f"Rule baseline: macro-F1 {result.get('macro_f1')}",
            "description": (
                f"accuracy={result.get('accuracy')} critical_recall={result.get('critical_recall')} "
                f"normal_fpr={result.get('false_positive_rate')} windows={result.get('window_count')}"
            ),
            "content": compact,
        },
        {
            "id": "criteria",
            "title": "Research success gates",
            "description": "Macro-F1 >= 0.75, critical recall >= 0.85, normal false positive rate <= 0.15.",
            "content": {"metrics": _metric_rows(result), "per_class": result.get("per_class")},
        },
    ]
    try:
        stream_publish(_RESULTS_RECEIVER, payload, _meta={"webspace_id": webspace_id})
    except Exception:
        # Tool calls should remain usable in validation and tests where the
        # AdaOS runtime context is intentionally not bootstrapped.
        pass


def _publish_dataset_result(result: Mapping[str, Any], *, webspace_id: str) -> None:
    compact = _compact_dataset_result(result)
    payload = [
        {
            "id": "dataset-windows",
            "title": f"Built {result.get('window_count')} event windows",
            "description": (
                f"records={result.get('record_count')} window_seconds={result.get('window_seconds')} "
                "baseline predictions are advisory until manually labeled"
            ),
            "content": compact,
        }
    ]
    try:
        stream_publish(_RESULTS_RECEIVER, payload, _meta={"webspace_id": webspace_id})
    except Exception:
        pass


@tool("get_lab_snapshot")
def get_lab_snapshot(payload: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    body = payload if isinstance(payload, Mapping) else {}
    if bool(body.get("project")):
        _project_lab_snapshot(webspace_id=_webspace_id_from_payload(body), force=bool(body.get("force")))
    return {"ok": True, "snapshot": _snapshot()}


@tool("refresh_snapshot")
def refresh_snapshot(payload: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    body = payload if isinstance(payload, Mapping) else {}
    webspace_id = _webspace_id_from_payload(body)
    projected = _project_lab_snapshot(webspace_id=webspace_id, force=True)
    return {"ok": True, "projected": projected}


@tool("rehydrate")
def rehydrate(payload: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    return refresh_snapshot(payload if isinstance(payload, Mapping) else {})


@tool("run_demo_evaluation")
def run_demo_evaluation(payload: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    webspace_id = _webspace_id_from_payload(payload)
    result = _evaluate(_synthetic_windows())
    _project_evaluation_result(result, webspace_id=webspace_id)
    _publish_result(result, webspace_id=webspace_id)
    try:
        publish_event(
            "ai_event_analysis.evaluation.completed",
            {"model": result["model"], "macro_f1": result["macro_f1"], "webspace_id": webspace_id},
            source="ai_event_analysis_skill",
        )
    except Exception:
        pass
    return {"ok": True, "result": result}


@tool("evaluate_windows")
def evaluate_windows(payload: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    body = payload if isinstance(payload, Mapping) else {}
    raw_windows = body.get("windows")
    windows = [item for item in raw_windows if isinstance(item, Mapping)] if isinstance(raw_windows, list) else _synthetic_windows()
    result = _evaluate(windows)
    _project_evaluation_result(result, webspace_id=_webspace_id_from_payload(body))
    _publish_result(result, webspace_id=_webspace_id_from_payload(body))
    return {"ok": True, "result": result}


@tool("import_local_logs")
def import_local_logs(payload: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    body = payload if isinstance(payload, Mapping) else {}
    max_lines = int(_value(body, "max_lines") or _MAX_DEFAULT_LOG_LINES)
    raw_path = str(body.get("path") or "").strip()
    paths = [Path(raw_path)] if raw_path else _log_candidates()[:_MAX_AUTO_LOG_SOURCES]
    records: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in paths:
        rows = _read_log_records(path, max_lines=max_lines)
        if rows:
            records.extend(rows)
            sources.append({"path": str(path), "records": len(rows)})
    return {
        "ok": True,
        "records": records,
        "summary": {
            "source_count": len(sources),
            "record_count": len(records),
            "sources": sources,
            "imported_at": _now_iso(),
        },
    }


@tool("build_event_windows")
def build_event_windows(payload: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    body = payload if isinstance(payload, Mapping) else {}
    raw_records = body.get("records")
    if isinstance(raw_records, list):
        records = [item for item in raw_records if isinstance(item, Mapping)]
    else:
        imported = import_local_logs(body)
        records = [item for item in imported.get("records", []) if isinstance(item, Mapping)]
    window_seconds = int(_value(body, "window_seconds") or 60)
    windows = _records_to_windows(
        records,
        window_seconds=window_seconds,
        node_id=str(body.get("node_id") or "local"),
        subnet_id=str(body.get("subnet_id") or "local"),
        webspace_id=_webspace_id_from_payload(body),
    )
    include_windows = bool(body.get("include_windows")) or isinstance(raw_records, list)
    result_windows = windows[:_MAX_TOOL_WINDOWS] if include_windows else []
    result = {
        "window_count": len(windows),
        "record_count": len(records),
        "window_seconds": window_seconds,
        "windows": result_windows,
        "windows_truncated": include_windows and len(windows) > len(result_windows),
        "rows": _window_rows(windows),
        "event_volume_chart": {
            "title": "Event volume by window",
            "unit": "events",
            "points": _event_volume_points(windows),
        },
        "class_distribution_chart": {
            "title": "Baseline class distribution",
            "unit": "windows",
            "points": _class_distribution_points(windows),
        },
        "built_at": _now_iso(),
    }
    _project_windows_result(result, webspace_id=_webspace_id_from_payload(body))
    _publish_dataset_result(result, webspace_id=_webspace_id_from_payload(body))
    return {"ok": True, "result": result}


@tool("export_event_windows_jsonl")
def export_event_windows_jsonl(payload: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    body = payload if isinstance(payload, Mapping) else {}
    raw_windows = body.get("windows")
    if isinstance(raw_windows, list):
        windows = [item for item in raw_windows if isinstance(item, Mapping)]
    else:
        imported = import_local_logs(body)
        records = [item for item in imported.get("records", []) if isinstance(item, Mapping)]
        windows = _records_to_windows(
            records,
            window_seconds=int(_value(body, "window_seconds") or 60),
            node_id=str(body.get("node_id") or "local"),
            subnet_id=str(body.get("subnet_id") or "local"),
            webspace_id=_webspace_id_from_payload(body),
        )
    raw_path = str(body.get("path") or "").strip()
    path = Path(raw_path) if raw_path else _DEFAULT_EXPORT_PATH
    export = _export_jsonl(windows, path)
    return {"ok": True, "export": export}


@subscribe("webio.stream.snapshot.requested")
def on_webio_stream_snapshot_requested(evt: Any) -> None:
    payload = getattr(evt, "payload", evt)
    if not isinstance(payload, Mapping):
        return
    receiver = str(payload.get("receiver") or "").strip()
    if receiver != _RESULTS_RECEIVER:
        return
    webspace_id = _webspace_id_from_payload(payload)
    _publish_result(_evaluate(_synthetic_windows()), webspace_id=webspace_id)


@subscribe("ai_event_analysis.evaluate_requested")
def on_evaluate_requested(evt: Any) -> None:
    payload = getattr(evt, "payload", evt)
    run_demo_evaluation(payload if isinstance(payload, Mapping) else {})
