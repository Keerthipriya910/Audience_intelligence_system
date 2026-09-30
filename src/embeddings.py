from __future__ import annotations

import numpy as np
from .demo_models import demo_embedding

class EmbeddingEngine:
    def __init__(self, model_name: str, demo_mode: bool = False):
        self.model_name = model_name
        self.demo_mode = demo_mode
        self._model = None

    def _load(self):
        if self.demo_mode or self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.demo_mode:
            return np.vstack([demo_embedding(t) for t in texts]).astype("float32")
        self._load()
        arr = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=16,
        )
        return np.asarray(arr, dtype="float32")
