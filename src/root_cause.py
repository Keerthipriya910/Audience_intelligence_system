from __future__ import annotations

import numpy as np
import pandas as pd


def _cos(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    na=np.linalg.norm(a); nb=np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))


def link_root_causes(df: pd.DataFrame, topic_names: dict[int,str], topic_embeddings: dict[int,np.ndarray], threshold: float=0.58) -> list[dict]:
    topics=[t for t in sorted(set(df["topic_id"].tolist())) if t != -1]
    parent={t:t for t in topics}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    for i,a in enumerate(topics):
        for b in topics[i+1:]:
            if a in topic_embeddings and b in topic_embeddings and _cos(topic_embeddings[a],topic_embeddings[b])>=threshold:
                union(a,b)
    groups={}
    for t in topics: groups.setdefault(find(t),[]).append(t)
    out=[]
    for gid,members in groups.items():
        sub=df[df["topic_id"].isin(members)]
        if sub.empty: continue
        counts=sub["final_sentiment"].value_counts(normalize=True)
        neg=float(counts.get("negative",0.0))
        reps=sub.sort_values(["like_count","final_confidence"],ascending=False).head(3)["text"].tolist()
        out.append({
            "root_cause": " + ".join(topic_names.get(t,f"Topic {t}") for t in members),
            "topic_ids": members,
            "comments": int(len(sub)),
            "negative_share": neg,
            "flagged": bool(neg>=0.5),
            "representative_comments": reps,
        })
    return sorted(out,key=lambda x:(x["flagged"],x["comments"]),reverse=True)
