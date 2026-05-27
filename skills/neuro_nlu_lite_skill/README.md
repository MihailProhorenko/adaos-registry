# neuro_nlu_lite_skill

Experimental AdaOS NLU backend for weak devices.

The skill is intentionally separate from `neural_nlu_service_skill`. Its goal is
to validate a lighter approach before changing the production Neural NLU
pipeline.

## Runtime Goal

Target steady-state runtime:

- stdlib-only first baseline;
- optional `numpy` later for matrix search;
- no `torch`;
- no `faiss-cpu`;
- no Rasa dependencies.

The first implementation keeps all math in Python lists so the skill can start
on constrained nodes without installing large wheels.

## Pipeline

```text
text
  -> normalize
  -> deterministic slot masking
  -> stable hash n-gram vectorizer
  -> intent prototypes
  -> nearest-positive / nearest-negative margin
  -> accept | abstain | fallback
```

The neural part is deliberately not implemented yet. The baseline measures how
far prototype clustering can go before adding a tiny encoder.

## HTTP API

- `GET /health`
- `POST /parse`
- `POST /rebuild`

`/parse` accepts:

```json
{
  "text": "поставь таймер на 10 минут",
  "webspace_id": "desktop",
  "locale": "ru"
}
```

It returns:

```json
{
  "ok": true,
  "top_intent": "voice.timer.start",
  "confidence": 0.82,
  "slots": {
    "duration": "10 минут",
    "duration_canon": "10 минут"
  },
  "via": "neuro_lite",
  "evidence": {
    "backend": "hash_ngram_prototypes",
    "matched_examples": ["поставь таймер на {duration}"],
    "positive_similarity": 0.91,
    "negative_similarity": 0.54,
    "positive_negative_margin": 0.37
  }
}
```

## Artifacts

Preferred artifact root:

```text
<skill-runtime-data>/files/nlu/neuro_lite
```

Optional files:

- `examples_manifest.jsonl`
- `thresholds.json`
- `vectorizer.json`
- `golden_cases.jsonl`
- `golden_report.json`

If `examples_manifest.jsonl` is absent, the service uses a small built-in smoke
dataset for timer, time, marketplace, and weather commands.

Example row:

```json
{"intent": "voice.timer.start", "text": "поставь таймер на {duration}"}
```

## Attention Layer

Do not add attention before the prototype baseline is measured.

A small attention module can be useful later, but only as an embedding enhancer:

```text
masked tokens -> tiny token encoder -> attention pooling -> vector -> prototypes
```

The decision layer should still stay prototype/margin based. Attention should
not become a sequence-to-intent black box.

Recommended constraints if we add it:

- token count capped at 32;
- embedding dim 32-64;
- single additive attention head;
- no transformer block;
- no self-attention over all character positions;
- inference exportable to `numpy`.

Self-attention/transformer-style layers are not a good fit for the target
device class until the simple baseline fails on a real hard-negative suite.
