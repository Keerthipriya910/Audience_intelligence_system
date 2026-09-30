from __future__ import annotations

from dataclasses import dataclass
from .demo_models import demo_emotion

@dataclass
class EmotionPrediction:
    label: str
    confidence: float

class EmotionEngine:
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

    def predict(self, text: str) -> EmotionPrediction:
        if self.demo_mode:
            label, conf = demo_emotion(text)
            return EmotionPrediction(label, float(conf))
        self._load()
        out = self._pipe(text)
        if out and isinstance(out[0], list):
            out = out[0]
        best = max(out, key=lambda x: float(x["score"]))
        return EmotionPrediction(str(best["label"]).lower(), float(best["score"]))
