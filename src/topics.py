from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class TopicResult:
    topic_ids: list[int]
    topic_names: dict[int, str]
    topic_embeddings: dict[int, np.ndarray]

class TopicEngine:
    def __init__(self, embedding_model: str, demo_mode: bool = False):
        self.embedding_model = embedding_model
        self.demo_mode = demo_mode

    def fit_transform(self, texts: list[str], embeddings: np.ndarray | None = None) -> TopicResult:
        if len(texts) < 6:
            return TopicResult([0] * len(texts), {0: "General Discussion"}, {0: np.mean(embeddings, axis=0) if embeddings is not None and len(embeddings) else np.zeros(4)})
        if self.demo_mode:
            # Deterministic lexical buckets for UI demos only.
            keys = {
                0: ("audio", "sound", "volume", "hear", "voice"),
                1: ("video", "editing", "quality", "camera"),
                2: ("game", "gameplay", "play"),
                3: ("explain", "tutorial", "learn", "helpful"),
            }
            ids=[]
            for t in texts:
                low=t.lower(); found=4
                for k, words in keys.items():
                    if any(w in low for w in words): found=k; break
                ids.append(found)
            names={0:"Audio Quality",1:"Video / Editing",2:"Gameplay",3:"Learning Value",4:"General Discussion"}
            temb={}
            if embeddings is not None:
                for k in sorted(set(ids)):
                    idx=[i for i,x in enumerate(ids) if x==k]
                    temb[k]=np.mean(embeddings[idx],axis=0)
            return TopicResult(ids,names,temb)
        from bertopic import BERTopic
        model = BERTopic(
            embedding_model=None if embeddings is not None else self.embedding_model,
            min_topic_size=max(3, min(10, len(texts)//8)),
            calculate_probabilities=False,
            verbose=False,
        )
        topics, _ = model.fit_transform(texts, embeddings=embeddings)
        info=model.get_topic_info()
        names={int(r.Topic): str(r.Name).replace("_"," ") for _,r in info.iterrows()}
        temb={}
        for tid in set(topics):
            idx=[i for i,x in enumerate(topics) if x==tid]
            if embeddings is not None and idx:
                temb[int(tid)]=np.mean(embeddings[idx],axis=0)
        return TopicResult([int(x) for x in topics], names, temb)
