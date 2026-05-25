# media_indexer_skill

AdaOS skill for local semantic search over media files.

The skill scans a user-selected directory, extracts lightweight technical
metadata, enriches media metadata where possible, and builds a FAISS index for
natural-language search across videos, audio files, and images.

## Data Routes

- `data/media_indexer/status` and `data/media_indexer/form` are compact Yjs
  projection state for first paint and reconnect recovery.
- `data/media_indexer/results` is a bounded Yjs result list. It is capped by the
  handler and should not be used for large diagnostics.
- `media_indexer.operations` is a replace-mode WebIO stream receiver for live
  operation status. The skill republishes the last operation on
  `webio.stream.snapshot.requested`.
- Private durable settings live in skill memory under `media_indexer.settings`.
- The FAISS index is a derived cache stored under the skill runtime data
  directory, with metadata mirrored in skill memory under `media_indexer.index`.

## Model Storage

The current NER weights source is temporary: Google Drive file
`19YBXzTYLoizbm8RF8gigUQ0fApZVmpoZ`. On first ML initialization the skill
downloads it to:

```text
ml/weights/model2.pt
```

The planned replacement is a dedicated model repository. Keep that migration
localized to `lib/ner_predictor.py`.

## Tools

- `scan_and_index(directory)` scans and indexes a directory.
- `search_media(query, k=5)` searches the persisted/in-memory index.
- `get_settings()` returns persisted settings, index metadata, and model status.
- `rehydrate()` restores lightweight settings without loading ML models.
- `dispose()` releases in-memory models and indexes.

## Runtime Notes

Importing the handler is passive. Heavy packages and models are loaded lazily
only when indexing or searching requires them. Smoke tests intentionally avoid
loading the ML stack.
