# AI Event Analysis Skill

`ai_event_analysis_skill` is a student-facing research and prototype skill for
measuring whether AI/ML methods improve operational event analysis in AdaOS.

The first iteration does not change AdaOS core. It uses the existing declarative
Web UI ABI, synthetic event-window samples, and a deterministic rule-based
baseline so the research task has a measurable starting point.

## Problem Statement

AdaOS is moving toward an operational event model where runtime, platform,
skill, browser, projection, and diagnostic signals are emitted as explicit
events and materialized through demanded projections.

The research task is to build and evaluate a model that analyzes a window of
operational events and predicts:

- whether the window contains an incident;
- the incident class;
- severity;
- confidence;
- the most important contributing signals.

The model must be evaluated against simple baselines instead of being judged by
subjective usefulness alone.

## Student Assignment

Title:

> Machine learning methods for anomaly detection and incident classification in
> the AdaOS operational event model.

Goal:

> Build a prototype that classifies operational event windows, compare a neural
> or classical ML approach with a rule-based baseline, and evaluate quality with
> reproducible metrics.

Input unit:

```json
{
  "window_id": "run-001:120-180s",
  "features": {
    "event_total": 128,
    "error_total": 8,
    "drop_total": 3,
    "projection_refresh_total": 42,
    "same_projection_refresh_max": 31,
    "yjs_write_total": 12,
    "browser_reconnect_total": 1,
    "member_disconnect_total": 0
  },
  "label": {
    "incident": true,
    "incident_type": "projection_refresh_storm",
    "severity": "warning",
    "reasons": ["same_projection_refresh_max", "projection_refresh_total"]
  }
}
```

Output unit:

```json
{
  "incident": true,
  "incident_type": "projection_refresh_storm",
  "severity": "warning",
  "confidence": 0.86,
  "reasons": ["same_projection_refresh_max", "projection_refresh_total"]
}
```

## Implementation Plan

- [x] Create a standalone AdaOS skill with no core changes.
- [x] Add a Web UI surface for task statement, dataset shape, baseline metrics,
  and experiment tracking.
- [x] Add a deterministic synthetic dataset for the first iteration.
- [x] Implement a rule-based baseline classifier.
- [x] Compute accuracy, macro-F1, per-class precision/recall/F1, false positive
  rate, critical recall, and detection delay.
- [x] Publish evaluation results through an existing stream receiver.
- [ ] Add import/export of real event-window datasets.
- [ ] Add a classical ML baseline, for example logistic regression, random
  forest, or gradient boosting.
- [ ] Add a neural sequence/window model, for example MLP, GRU/LSTM, or a small
  Transformer encoder.
- [ ] Add operator-reviewed explanation scoring.
- [ ] Connect the skill to real AdaOS event/projection snapshots after the
  operational-event MVP gate accepts the canonical runtime path.

## Success Criteria

Minimum first research milestone:

- dataset has at least 500 labeled windows;
- at least 5 incident classes plus `normal`;
- rule-based baseline is implemented and reproducible;
- ML/NN model is compared against the rule baseline;
- macro-F1 is at least `0.75` on a held-out test split;
- recall for critical incidents is at least `0.85`;
- false positive rate for normal windows is at most `0.15`;
- every prediction returns top contributing signals.

Stretch target:

- improve macro-F1 by at least `10%` relative to the rule baseline, or reduce
  average detection delay by at least `20%` at the same false-positive rate.

## Current Prototype

The skill currently ships:

- a synthetic benchmark dataset generator;
- a deterministic rule baseline;
- metric computation utilities;
- a Web UI app named `AI Event Analysis`;
- a live stream receiver for demo evaluation results.

The prototype intentionally keeps model training out of the first iteration so
the measurement contract can stabilize before dependencies and runtime costs
are introduced.
