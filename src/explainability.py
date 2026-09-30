from __future__ import annotations

import re
from .demo_models import POS, NEG

class ExplainabilityEngine:
    """Lightweight word-attribution surface with optional Captum extension point.

    The dashboard always has an explanation. In demo mode it is explicitly marked as
    lexical evidence. In research mode, transformer confidence remains the model result;
    this module returns human-readable token cues and can be replaced by Captum IG for
    deeper experiments without changing the pipeline schema.
    """
    def __init__(self, demo_mode: bool = False):
        self.demo_mode = demo_mode

    def explain(self, text: str, label: str, top_n: int = 8) -> list[dict]:
        toks = re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)
        rows = []
        for t in toks:
            score = 0.0
            if t in POS: score = 1.0
            elif t in NEG: score = -1.0
            elif len(t) >= 7: score = 0.12 if label == "positive" else (-0.12 if label == "negative" else 0.05)
            if score:
                rows.append({"token": t, "contribution": score})
        rows.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return rows[:top_n]
