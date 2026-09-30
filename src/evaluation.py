from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

LABELS=["negative","neutral","positive"]

def classification_metrics(df: pd.DataFrame, truth_col: str="label", pred_col: str="final_sentiment") -> dict:
    if truth_col not in df.columns:
        return {"available":False,"reason":f"No '{truth_col}' column in dataset."}
    y_true=df[truth_col].astype(str).str.lower()
    y_pred=df[pred_col].astype(str).str.lower()
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,labels=LABELS,average="macro",zero_division=0)
    wp,wr,wf,_=precision_recall_fscore_support(y_true,y_pred,labels=LABELS,average="weighted",zero_division=0)
    return {
        "available":True,
        "accuracy":float(accuracy_score(y_true,y_pred)),
        "macro_precision":float(p),"macro_recall":float(r),"macro_f1":float(f),
        "weighted_precision":float(wp),"weighted_recall":float(wr),"weighted_f1":float(wf),
        "confusion_matrix":confusion_matrix(y_true,y_pred,labels=LABELS).tolist(),
        "labels":LABELS,
        "report":classification_report(y_true,y_pred,labels=LABELS,zero_division=0,output_dict=True),
    }

def multilingual_metrics(df: pd.DataFrame) -> list[dict]:
    if "label" not in df.columns: return []
    out=[]
    for lang,g in df.groupby("language"):
        if len(g)<2: continue
        m=classification_metrics(g)
        if m.get("available"):
            out.append({"language":lang,"n":len(g),"accuracy":m["accuracy"],"macro_f1":m["macro_f1"]})
    return out

def trust_evaluation(df: pd.DataFrame) -> dict:
    result={"available":True}
    if "label" in df.columns:
        truth=df["label"].astype(str).str.lower()
        before=df["original_sentiment"].astype(str).str.lower()
        after=df["final_sentiment"].astype(str).str.lower()
        routed=df.get("evidence_routed",pd.Series([True]*len(df),index=df.index)).astype(bool)
        wrong_before=(before!=truth) & routed
        correct_before=(before==truth) & routed
        corrected=wrong_before & (after==truth)
        harmed=correct_before & (after!=truth)
        result.update({
            "routed_n":int(routed.sum()),
            "wrong_to_correct_n":int(corrected.sum()),
            "correct_to_wrong_n":int(harmed.sum()),
            "wrong_to_correct_rate":float(corrected.sum()/max(1,wrong_before.sum())),
            "correct_to_wrong_rate":float(harmed.sum()/max(1,correct_before.sum())),
        })
    else:
        result.update({"ground_truth_available":False})
    if "human_trust" in df.columns:
        a=pd.to_numeric(df["human_trust"],errors="coerce")
        b=pd.to_numeric(df["final_confidence"],errors="coerce")
        valid=a.notna() & b.notna()
        result["human_trust_correlation"]=float(a[valid].corr(b[valid])) if valid.sum()>=3 else None
        result["human_trust_n"]=int(valid.sum())
    else:
        result["human_trust_correlation"]=None
        result["human_trust_note"]="Add a numeric human_trust column to compute the proposal's confidence-vs-human-trust correlation."
    return result

def ablation_summary(df: pd.DataFrame) -> list[dict]:
    rows=[]
    if "label" in df.columns:
        base=df.copy(); base["baseline_pred"]=base["original_sentiment"]
        mb=classification_metrics(base,pred_col="baseline_pred")
        mn=classification_metrics(df,pred_col="naive_faiss_sentiment") if "naive_faiss_sentiment" in df.columns else {}
        mf=classification_metrics(df,pred_col="final_sentiment")
        rows.append({"configuration":"Baseline 1 — XLM-R only","accuracy":mb.get("accuracy"),"macro_f1":mb.get("macro_f1"),"notes":"No retrieval or EWTC"})
        rows.append({"configuration":"Baseline 2 — XLM-R + FAISS majority","accuracy":mn.get("accuracy"),"macro_f1":mn.get("macro_f1"),"notes":"Naive majority vote for evidence-routed comments"})
        rows.append({"configuration":"Baseline 3 — XLM-R + FAISS + EWTC","accuracy":mf.get("accuracy"),"macro_f1":mf.get("macro_f1"),"notes":"Adaptive evidence correction"})
    else:
        for name,note in [("Baseline 1 — XLM-R only","No retrieval or EWTC"),("Baseline 2 — XLM-R + FAISS majority","Naive evidence baseline"),("Baseline 3 — XLM-R + FAISS + EWTC","Adaptive evidence correction")]:
            rows.append({"configuration":name,"accuracy":None,"macro_f1":None,"notes":note+"; requires ground-truth label column"})
    rows.append({"configuration":"Full framework","accuracy":rows[-1].get("accuracy"),"macro_f1":rows[-1].get("macro_f1"),"notes":"Adds explainability, emotion, topics, root causes and AHS; descriptive layers do not alter the sentiment label"})
    return rows
