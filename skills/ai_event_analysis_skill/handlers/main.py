from __future__ import annotations

import time
from typing import Any, Mapping

from adaos.sdk.core.decorators import subscribe, tool
from adaos.sdk.data.events import publish as publish_event
from adaos.sdk.io.out import stream_publish

_RESULTS_RECEIVER = "ai_event_analysis.results"

_CLASSES = [
    "normal",
    "eventbus_backpressure",
    "projection_refresh_storm",
    "yjs_write_pressure",
    "browser_session_instability",
    "member_node_disconnect",
    "runtime_rebuild_churn",
]


def _webspace_id_from_payload(payload: Mapping[str, Any] | None) -> str:
    if not isinstance(payload, Mapping):
        return "desktop"
    raw = payload.get("webspace_id") or payload.get("workspace_id")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    meta = payload.get("_meta")
    if isinstance(meta, Mapping):
        nested = meta.get("webspace_id") or meta.get("workspace_id")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return "desktop"


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
        "dataset": {
            "items": [
                {"id": "windows", "name": "Labeled windows", "current": len(_synthetic_windows()), "target": "500-1000+", "notes": "One row per fixed event window."},
                {"id": "classes", "name": "Classes", "current": len(_CLASSES), "target": "6+", "notes": "normal plus incident classes."},
                {"id": "features", "name": "Feature families", "current": 9, "target": "eventbus/projection/Yjs/device/runtime", "notes": "Aggregated numeric features for baseline and ML models."},
            ]
        },
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


def _publish_result(result: Mapping[str, Any], *, webspace_id: str) -> None:
    payload = [
        {
            "id": "summary",
            "title": f"Rule baseline: macro-F1 {result.get('macro_f1')}",
            "description": (
                f"accuracy={result.get('accuracy')} critical_recall={result.get('critical_recall')} "
                f"normal_fpr={result.get('false_positive_rate')} windows={result.get('window_count')}"
            ),
            "content": result,
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


@tool("get_lab_snapshot")
def get_lab_snapshot(payload: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    _ = payload
    return {"ok": True, "snapshot": _snapshot()}


@tool("run_demo_evaluation")
def run_demo_evaluation(payload: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
    webspace_id = _webspace_id_from_payload(payload)
    result = _evaluate(_synthetic_windows())
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
    _publish_result(result, webspace_id=_webspace_id_from_payload(body))
    return {"ok": True, "result": result}


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
