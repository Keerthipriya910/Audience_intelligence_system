from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Settings:
    app_name: str = "PrismPulse AI"
    data_dir: Path = ROOT / "data"
    export_dir: Path = ROOT / "exports"
    sentiment_model: str = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-xlm-roberta-base-sentiment")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    emotion_model: str = os.getenv("EMOTION_MODEL", "amaan00z/emotion_xlmr3")
    grok_model: str = os.getenv("GROK_MODEL", "grok-4.7")
    youtube_api_key: str = os.getenv("YOUTUBE_API_KEY", "")
    xai_api_key: str = os.getenv("XAI_API_KEY", "")
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.72"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    root_topic_similarity: float = float(os.getenv("ROOT_TOPIC_SIMILARITY", "0.58"))
    max_comments: int = int(os.getenv("MAX_COMMENTS", "400"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    demo_mode: bool = os.getenv("DEMO_MODE", "false").lower() == "true"
    sentiment_labels: list[str] = field(default_factory=lambda: ["negative", "neutral", "positive"])

settings = Settings()
settings.export_dir.mkdir(parents=True, exist_ok=True)
