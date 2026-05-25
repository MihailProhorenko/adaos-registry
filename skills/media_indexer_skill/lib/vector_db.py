"""Vector search storage for media_indexer_skill.

The module is intentionally import-light. FAISS, Pillow, and sentence-transformer
models are loaded only when VectorDatabase is instantiated, so smoke imports do
not download or allocate ML resources.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Multimodal FAISS index with text and image channels."""

    TEXT_MODEL_NAME = "distiluse-base-multilingual-cased-v2"
    CLIP_TEXT_MODEL_NAME = "clip-ViT-B-32-multilingual-v1"
    CLIP_VISION_MODEL_NAME = "clip-ViT-B-32"

    TEXT_MIN_SIMILARITY = 0.10
    IMAGE_MIN_SIMILARITY = 0.22
    INDEX_DIMENSIONS = 512
    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self.faiss = faiss
        logger.info("Loading text embedding model: %s", self.TEXT_MODEL_NAME)
        self.text_model = SentenceTransformer(self.TEXT_MODEL_NAME)

        logger.info("Loading CLIP models: %s / %s", self.CLIP_TEXT_MODEL_NAME, self.CLIP_VISION_MODEL_NAME)
        self.clip_text = SentenceTransformer(self.CLIP_TEXT_MODEL_NAME)
        self.clip_vision = SentenceTransformer(self.CLIP_VISION_MODEL_NAME)

        self.text_index = self.faiss.IndexFlatIP(self.INDEX_DIMENSIONS)
        self.image_index = self.faiss.IndexFlatIP(self.INDEX_DIMENSIONS)
        self.text_docs: List[Dict[str, Any]] = []
        self.image_docs: List[Dict[str, Any]] = []

    @property
    def counts(self) -> Dict[str, int]:
        return {
            "text_count": len(self.text_docs),
            "image_count": len(self.image_docs),
            "total_count": len(self.text_docs) + len(self.image_docs),
        }

    def add_text(self, text: str, payload: Dict[str, Any]) -> None:
        import numpy as np

        if not text.strip():
            return
        emb = self.text_model.encode(text, normalize_embeddings=True).astype("float32")
        self.text_index.add(np.array([emb]))
        self.text_docs.append({"text": text, "payload": payload})

    def add_image(self, image_path: str, payload: Dict[str, Any]) -> None:
        try:
            import numpy as np
            from PIL import Image

            with Image.open(image_path) as img:
                emb = self.clip_vision.encode(img, normalize_embeddings=True).astype("float32")
            self.image_index.add(np.array([emb]))
            self.image_docs.append(
                {
                    "text": f"[VISUAL] {Path(image_path).name}",
                    "payload": payload,
                }
            )
        except Exception as exc:
            logger.warning("CLIP failed to read %s: %s", image_path, exc)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        import numpy as np

        results: List[Dict[str, Any]] = []

        if self.text_docs:
            q_text_emb = self.text_model.encode(query, normalize_embeddings=True).astype("float32")
            distances, indexes = self.text_index.search(np.array([q_text_emb]), k)
            for idx, similarity in zip(indexes[0], distances[0]):
                if idx == -1 or idx >= len(self.text_docs):
                    continue
                if similarity >= self.TEXT_MIN_SIMILARITY:
                    result = self.text_docs[idx].copy()
                    result["score"] = round(float(similarity) * 100, 1)
                    result["type"] = "media/text"
                    results.append(result)

        if self.image_docs:
            q_img_emb = self.clip_text.encode(query, normalize_embeddings=True).astype("float32")
            distances, indexes = self.image_index.search(np.array([q_img_emb]), k)
            for idx, similarity in zip(indexes[0], distances[0]):
                if idx == -1 or idx >= len(self.image_docs):
                    continue
                raw_similarity = float(similarity)
                if raw_similarity >= self.IMAGE_MIN_SIMILARITY:
                    result = self.image_docs[idx].copy()
                    result["score"] = round(raw_similarity * 100, 1)
                    result["type"] = "image"
                    results.append(result)

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:k]

    def save(self, directory: str | Path) -> Dict[str, Any]:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.text_index, str(target / "text.index"))
        self.faiss.write_index(self.image_index, str(target / "image.index"))
        metadata = {
            "schema": self.SCHEMA_VERSION,
            "models": {
                "text": self.TEXT_MODEL_NAME,
                "clip_text": self.CLIP_TEXT_MODEL_NAME,
                "clip_vision": self.CLIP_VISION_MODEL_NAME,
            },
            "text_docs": self.text_docs,
            "image_docs": self.image_docs,
            **self.counts,
        }
        (target / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    def load(self, directory: str | Path) -> Dict[str, Any]:
        source = Path(directory)
        metadata_path = source / "metadata.json"
        text_index_path = source / "text.index"
        image_index_path = source / "image.index"
        if not metadata_path.exists() or not text_index_path.exists() or not image_index_path.exists():
            return {"loaded": False, "reason": "missing_index_files"}

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("schema") or 0) != self.SCHEMA_VERSION:
            return {"loaded": False, "reason": "schema_mismatch"}

        self.text_index = self.faiss.read_index(str(text_index_path))
        self.image_index = self.faiss.read_index(str(image_index_path))
        self.text_docs = list(metadata.get("text_docs") or [])
        self.image_docs = list(metadata.get("image_docs") or [])
        return {"loaded": True, **self.counts}
