from __future__ import annotations
import pandas as pd

def summary_dict(output):
    df=output.dataframe
    return {
        "execution_mode":output.mode,
        "comments":int(len(df)),
        "audience_health_score":output.ahs,
        "ahs_components":output.ahs_components,
        "sentiment_distribution_pct":df["final_sentiment"].value_counts(normalize=True).mul(100).round(2).to_dict(),
        "emotion_distribution_pct":df["emotion"].value_counts(normalize=True).mul(100).round(2).to_dict(),
        "language_distribution":df["language"].value_counts().to_dict(),
        "topics":df["topic_name"].value_counts().to_dict(),
        "root_causes":output.root_causes,
        "timings":output.timings,
    }
