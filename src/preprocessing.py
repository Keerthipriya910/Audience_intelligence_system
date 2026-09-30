from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
except Exception:  # optional fallback
    detect = None

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
SPACE_RE = re.compile(r"\s+")
REPEAT_RE = re.compile(r"(.)\1{3,}", re.UNICODE)

@dataclass
class ProcessedText:
    original: str
    clean: str
    language: str


def clean_text(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" <URL> ", text)
    text = MENTION_RE.sub(" <USER> ", text)
    text = REPEAT_RE.sub(r"\1\1\1", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def detect_language(text: str) -> str:
    # Script-aware first pass so Telugu/Hindi are not misread by short-text detectors.
    if re.search(r"[\u0C00-\u0C7F]", text):
        if re.search(r"[A-Za-z]", text):
            return "code-mixed-te-en"
        return "te"
    if re.search(r"[\u0900-\u097F]", text):
        if re.search(r"[A-Za-z]", text):
            return "code-mixed-hi-en"
        return "hi"
    if not text.strip():
        return "unknown"
    if detect is None:
        return "en"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def preprocess(text: str) -> ProcessedText:
    clean = clean_text(text)
    return ProcessedText(original=str(text or ""), clean=clean, language=detect_language(clean))
