from __future__ import annotations
import argparse, json, platform, sys, time
from pathlib import Path
import pandas as pd
from src.pipeline import AudiencePipeline
from src.evaluation import classification_metrics, multilingual_metrics, ablation_summary, trust_evaluation
from src.analytics import summary_dict
from src.config import settings

p=argparse.ArgumentParser(description="Reproducible experiment runner")
p.add_argument("--input",default="data/sample_comments.csv")
p.add_argument("--mode",choices=["demo","research"],default="research")
p.add_argument("--thresholds",default="0.60,0.66,0.72,0.78,0.84")
p.add_argument("--top-k",type=int,default=5)
args=p.parse_args()

df=pd.read_csv(args.input)
rows=[]; best=None
for threshold in [float(x) for x in args.thresholds.split(",")]:
    out=AudiencePipeline(demo_mode=args.mode=="demo",threshold=threshold,top_k=args.top_k).run(df)
    m=classification_metrics(out.dataframe)
    row={"threshold":threshold,"mode":out.mode,"comments":len(out.dataframe),"route_pct":out.timings["evidence_route_pct"],"total_s":out.timings["total_s"],"throughput":out.timings["comments_per_second"],"ahs":out.ahs,"accuracy":m.get("accuracy"),"macro_f1":m.get("macro_f1")}
    rows.append(row)
    if best is None or ((row.get("macro_f1") or -1) > (best[0].get("macro_f1") or -1)): best=(row,out)

result_df=pd.DataFrame(rows)
settings.export_dir.mkdir(parents=True,exist_ok=True)
result_df.to_csv(settings.export_dir/"threshold_sweep.csv",index=False)
selected=best[1] if best else AudiencePipeline(demo_mode=args.mode=="demo").run(df)
report={
    "provenance":{"mode":selected.mode,"python":sys.version,"platform":platform.platform(),"sentiment_model":settings.sentiment_model,"embedding_model":settings.embedding_model,"emotion_model":settings.emotion_model,"seed":settings.random_seed},
    "selected_summary":summary_dict(selected),
    "classification":classification_metrics(selected.dataframe),
    "multilingual":multilingual_metrics(selected.dataframe),
    "ablation":ablation_summary(selected.dataframe),
    "trust_evaluation":trust_evaluation(selected.dataframe),
    "threshold_sweep":rows,
}
(settings.export_dir/"experiment_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
print(result_df.to_string(index=False))
print("\nSaved exports/threshold_sweep.csv and exports/experiment_report.json")
