from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .demo_models import demo_sentiment

LABEL_ALIASES = {
    "label_0": "negative", "label_1": "neutral", "label_2": "positive",
    "negative": "negative", "neutral": "neutral", "positive": "positive",
    "neg": "negative", "neu": "neutral", "pos": "positive",
}

@dataclass
class SentimentPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]

class SentimentEngine:
    def __init__(self, model_name: str, demo_mode: bool = False):
        self.model_name = model_name
        self.demo_mode = demo_mode
        self._pipe = None

    def _load(self):
        if self.demo_mode or self._pipe is not None:
            return
        from transformers import pipeline
        self._pipe = pipeline(
            "text-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            top_k=None,
            device=-1,
            truncation=True,
            max_length=256,
        )

    @staticmethod
    def _normalise_label(label: str) -> str:
        key = str(label).strip().lower()
        return LABEL_ALIASES.get(key, key)

    def predict(self, text: str) -> SentimentPrediction:
        if self.demo_mode:
            label, confidence, probs = demo_sentiment(text)
            return SentimentPrediction(label, float(confidence), probs)
        self._load()
        outputs = self._pipe(text)
        if outputs and isinstance(outputs[0], list):
            outputs = outputs[0]
        probs = {self._normalise_label(x["label"]): float(x["score"]) for x in outputs}
        label = max(probs, key=probs.get)
        return SentimentPrediction(label, probs[label], probs)
