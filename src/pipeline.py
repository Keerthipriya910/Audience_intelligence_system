from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import settings
from .preprocessing import preprocess
from .sentiment import SentimentEngine
from .emotion import EmotionEngine
from .embeddings import EmbeddingEngine
from .evidence import EvidenceIndex
from .ewtc import apply_ewtc
from .explainability import ExplainabilityEngine
from .topics import TopicEngine
from .root_cause import link_root_causes
from .audience_health import compute_ahs

@dataclass
class PipelineOutput:
    dataframe: pd.DataFrame
    ahs: float
    ahs_components: dict[str,float]
    root_causes: list[dict]
    timings: dict[str,float]
    mode: str

class AudiencePipeline:
    def __init__(self, demo_mode: bool | None = None, threshold: float | None = None, top_k: int | None = None):
        self.demo_mode = settings.demo_mode if demo_mode is None else demo_mode
        self.threshold = settings.confidence_threshold if threshold is None else threshold
        self.top_k = settings.top_k if top_k is None else top_k
        self.sentiment = SentimentEngine(settings.sentiment_model, self.demo_mode)
        self.emotion = EmotionEngine(settings.emotion_model, self.demo_mode)
        self.embedding = EmbeddingEngine(settings.embedding_model, self.demo_mode)
        self.explainer = ExplainabilityEngine(self.demo_mode)
        self.topic_engine = TopicEngine(settings.embedding_model, self.demo_mode)

    def run(self, raw_df: pd.DataFrame, enable_topics: bool=True, enable_emotion: bool=True) -> PipelineOutput:
        start_all=time.perf_counter()
        if "text" not in raw_df.columns:
            raise ValueError("Input dataset must contain a 'text' column.")
        df=raw_df.copy().reset_index(drop=True)
        for col,default in {"comment_id":None,"published_at":"","like_count":0,"reply_count":0,"video_id":"dataset"}.items():
            if col not in df.columns:
                df[col]=[f"row-{i}" for i in range(len(df))] if col=="comment_id" else default
        df=df.drop_duplicates(subset=["comment_id"]).copy()
        df=df[df["text"].astype(str).str.strip().ne("")].reset_index(drop=True)

        t=time.perf_counter()
        processed=[preprocess(x) for x in df["text"].astype(str)]
        df["clean_text"]=[x.clean for x in processed]
        df["language"]=[x.language for x in processed]
        prep_s=time.perf_counter()-t

        t=time.perf_counter()
        preds=[self.sentiment.predict(x) for x in df["clean_text"]]
        df["original_sentiment"]=[p.label for p in preds]
        df["original_confidence"]=[p.confidence for p in preds]
        df["sentiment_probs"]=[p.probabilities for p in preds]
        sentiment_s=time.perf_counter()-t
        if not self.demo_mode:
            import gc
            self.sentiment._pipe=None
            gc.collect()

        t=time.perf_counter()
        embeddings=self.embedding.encode(df["clean_text"].tolist())
        embed_s=time.perf_counter()-t
        if not self.demo_mode:
            import gc
            self.embedding._model=None
            gc.collect()
        evidence_index=EvidenceIndex(embeddings,require_faiss=not self.demo_mode)

        final_labels=[]; final_conf=[]; agreements=[]; lambdas=[]; sigmas=[]; routed=[]; evidence_rows=[]; naive_labels=[]; retrieval_latencies=[]
        t=time.perf_counter()
        for i,row in df.iterrows():
            label=row["original_sentiment"]; c=float(row["original_confidence"])
            if c >= self.threshold or len(df)<=1:
                final_labels.append(label); final_conf.append(c); agreements.append(0.0); lambdas.append(1-c); sigmas.append(0.0); routed.append(False); evidence_rows.append([]); naive_labels.append(label)
                continue
            rt=time.perf_counter()
            hits=evidence_index.search(embeddings[i],k=min(self.top_k,max(1,len(df)-1)),exclude_index=i)
            retrieval_latencies.append((time.perf_counter()-rt)*1000.0)
            labs=[str(df.iloc[h.index]["original_sentiment"]) for h in hits]
            dists=[h.distance for h in hits]
            ewtc=apply_ewtc(label,c,labs,dists)
            if labs:
                counts={}
                for lab in labs: counts[lab]=counts.get(lab,0)+1
                naive=max(counts,key=counts.get)
            else:
                naive=label
            naive_labels.append(naive)
            final_labels.append(ewtc.final_label); final_conf.append(ewtc.final_confidence)
            agreements.append(ewtc.weighted_agreement); lambdas.append(ewtc.correction_strength); sigmas.append(ewtc.sigma); routed.append(True)
            evidence_rows.append([{
                "comment_index":h.index,
                "comment":str(df.iloc[h.index]["text"]),
                "label":str(df.iloc[h.index]["original_sentiment"]),
                "similarity":round(h.similarity,4),
                "distance":round(h.distance,4),
            } for h in hits])
        ewtc_s=time.perf_counter()-t
        df["naive_faiss_sentiment"]=naive_labels
        df["final_sentiment"]=final_labels
        df["final_confidence"]=final_conf
        df["ewtc_agreement"]=agreements
        df["ewtc_lambda"]=lambdas
        df["ewtc_sigma"]=sigmas
        df["evidence_routed"]=routed
        df["evidence"]=evidence_rows

        t=time.perf_counter()
        if enable_emotion:
            emos=[self.emotion.predict(x) for x in df["clean_text"]]
            df["emotion"]=[x.label for x in emos]
            df["emotion_confidence"]=[x.confidence for x in emos]
        else:
            df["emotion"]="not-run"; df["emotion_confidence"]=0.0
        emotion_s=time.perf_counter()-t
        if not self.demo_mode:
            import gc
            self.emotion._pipe=None
            gc.collect()

        t=time.perf_counter()
        df["explanation"]=[self.explainer.explain(t,l) for t,l in zip(df["clean_text"],df["final_sentiment"])]
        explain_s=time.perf_counter()-t

        t=time.perf_counter()
        if enable_topics:
            tr=self.topic_engine.fit_transform(df["clean_text"].tolist(),embeddings)
            df["topic_id"]=tr.topic_ids
            df["topic_name"]=[tr.topic_names.get(int(x),f"Topic {x}") for x in tr.topic_ids]
            root_causes=link_root_causes(df,tr.topic_names,tr.topic_embeddings,settings.root_topic_similarity)
        else:
            df["topic_id"]=0; df["topic_name"]="Not Run"; root_causes=[]
        topics_s=time.perf_counter()-t

        ahs,components=compute_ahs(df)
        total=time.perf_counter()-start_all
        timings={
            "preprocessing_s":round(prep_s,4),"sentiment_s":round(sentiment_s,4),"embedding_s":round(embed_s,4),
            "ewtc_retrieval_s":round(ewtc_s,4),"emotion_s":round(emotion_s,4),"explainability_s":round(explain_s,4),
            "topics_rootcause_s":round(topics_s,4),"total_s":round(total,4),
            "comments_per_second":round(len(df)/max(total,1e-9),3),
            "evidence_route_pct":round(100*float(df["evidence_routed"].mean()),2),
            "high_confidence_pct":round(100*(1.0-float(df["evidence_routed"].mean())),2),
            "avg_retrieval_latency_ms":round(float(np.mean(retrieval_latencies)) if retrieval_latencies else 0.0,3),
        }
        return PipelineOutput(df,round(ahs,2),components,root_causes,timings,"DEMO" if self.demo_mode else "RESEARCH")
