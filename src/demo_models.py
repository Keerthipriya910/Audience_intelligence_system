from __future__ import annotations

import hashlib
import math
import re
import numpy as np

POS = {"good","great","love","awesome","amazing","excellent","nice","helpful","best","super","fantastic","clear","useful","perfect","wow","బాగుంది","చాలా బాగుంది","अच्छा","बहुत अच्छा","पसंद"}
NEG = {"bad","worst","hate","terrible","poor","boring","awful","broken","problem","issue","slow","noise","low","can't","cannot","waste","చెడు","బాగోలేదు","खराब","बेकार"}
EMOTION_WORDS = {
    "joy": {"love","amazing","happy","great","awesome","excellent","బాగుంది","अच्छा"},
    "anger": {"hate","angry","worst","terrible","awful","बेकार"},
    "sadness": {"sad","disappointed","miss","poor"},
    "fear": {"afraid","scared","worried"},
    "surprise": {"wow","surprised","unexpected"},
    "disgust": {"disgusting","gross"},
    "love": {"love","lovely","heart"},
}

def _tokens(text: str):
    return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)


def demo_sentiment(text: str):
    toks = _tokens(text)
    p = sum(t in POS for t in toks)
    n = sum(t in NEG for t in toks)
    raw = p - n
    if raw > 0:
        label = "positive"
    elif raw < 0:
        label = "negative"
    else:
        label = "neutral"
    confidence = min(0.94, 0.50 + 0.12 * abs(raw) + 0.02 * min(len(toks), 8))
    probs = {"negative": 0.15, "neutral": 0.15, "positive": 0.15}
    probs[label] = confidence
    rem = 1.0 - confidence
    others = [k for k in probs if k != label]
    probs[others[0]] = rem / 2
    probs[others[1]] = rem / 2
    return label, confidence, probs


def demo_emotion(text: str):
    toks = _tokens(text)
    scores = {k: sum(t in words for t in toks) for k, words in EMOTION_WORDS.items()}
    if max(scores.values(), default=0) == 0:
        return "neutral", 0.55
    label = max(scores, key=scores.get)
    return label, min(0.95, 0.6 + 0.1 * scores[label])


def demo_embedding(text: str, dim: int = 96):
    # Deterministic feature-hashing vector for demo/offline execution only.
    v = np.zeros(dim, dtype="float32")
    for tok in _tokens(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        v[idx] += sign
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v
