from __future__ import annotations

import math
import numpy as np
import pandas as pd


def _entropy_score(values: pd.Series) -> float:
    if values.empty: return 0.0
    p=values.value_counts(normalize=True).values.astype(float)
    if len(p)<=1: return 35.0
    h=float(-(p*np.log(p+1e-12)).sum())
    return 100.0*h/math.log(len(p))


def compute_ahs(df: pd.DataFrame) -> tuple[float,dict[str,float]]:
    if df.empty:
        return 0.0, {k:0.0 for k in ["Sentiment","Emotion","Engagement","Topic Diversity","Language Diversity","Evidence Trust"]}
    sent_map={"positive":1.0,"neutral":0.55,"negative":0.10}
    sentiment=100*float(df["final_sentiment"].map(sent_map).fillna(0.5).mean())
    emo_map={"joy":1.0,"love":0.95,"admiration":0.9,"surprise":0.65,"neutral":0.55,"sadness":0.30,"fear":0.25,"anger":0.15,"disgust":0.10,"disapproval":0.20}
    emotion=100*float(df["emotion"].map(emo_map).fillna(0.5).mean())
    eng_raw=np.log1p(df.get("like_count",pd.Series([0]*len(df))).astype(float)+2*df.get("reply_count",pd.Series([0]*len(df))).astype(float))
    engagement=100*float(eng_raw.mean()/(eng_raw.max()+1e-9)) if eng_raw.max()>0 else 50.0
    topic_div=_entropy_score(df["topic_name"])
    lang_div=_entropy_score(df["language"])
    evidence=100*float(df["final_confidence"].clip(0,1).mean())
    components={
        "Sentiment":sentiment,
        "Emotion":emotion,
        "Engagement":engagement,
        "Topic Diversity":topic_div,
        "Language Diversity":lang_div,
        "Evidence Trust":evidence,
    }
    # Equal weighting keeps the index interpretable and avoids hidden tuning.
    ahs=float(np.mean(list(components.values())))
    return ahs,{k:round(v,2) for k,v in components.items()}
