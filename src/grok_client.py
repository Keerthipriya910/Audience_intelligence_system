from __future__ import annotations

import os
import pandas as pd

SYSTEM = """You are the conversational analyst inside an evidence-backed YouTube audience intelligence system.
Answer only from the supplied analysis context. Distinguish measured metrics from model-generated interpretation.
When discussing a complaint, mention the supporting topic/root-cause evidence. Never invent comments, metrics, or causes.
If the context does not support an answer, say that the available analysis does not establish it."""


def build_context(df: pd.DataFrame, ahs: float, components: dict, root_causes: list[dict]) -> str:
    sent=df["final_sentiment"].value_counts(normalize=True).mul(100).round(1).to_dict() if not df.empty else {}
    emo=df["emotion"].value_counts(normalize=True).mul(100).round(1).head(8).to_dict() if not df.empty else {}
    topics=df["topic_name"].value_counts().head(10).to_dict() if not df.empty else {}
    examples=df.sort_values("like_count",ascending=False).head(12)[["text","final_sentiment","final_confidence","topic_name"]].to_dict("records") if not df.empty else []
    return f"AHS={ahs:.2f}\nComponents={components}\nSentiment%={sent}\nEmotion%={emo}\nTopics={topics}\nRootCauses={root_causes[:8]}\nRepresentativeComments={examples}"


def ask_grok(question: str, context: str, api_key: str, model: str) -> str:
    if not api_key:
        return "Grok is not configured. Add XAI_API_KEY to .env or enter it in this dashboard session."
    try:
        from openai import OpenAI
        client=OpenAI(api_key=api_key,base_url="https://api.x.ai/v1")
        response=client.responses.create(
            model=model,
            instructions=SYSTEM,
            input=f"ANALYSIS CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}",
        )
        return response.output_text
    except Exception as e:
        return f"Grok request failed: {e}"
