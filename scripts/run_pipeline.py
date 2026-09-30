from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from src.pipeline import AudiencePipeline
from src.analytics import summary_dict
from src.storage import export_bundle
from src.config import settings

p=argparse.ArgumentParser()
p.add_argument("--input",default="data/sample_comments.csv")
p.add_argument("--mode",choices=["demo","research"],default="demo")
p.add_argument("--threshold",type=float,default=settings.confidence_threshold)
p.add_argument("--top-k",type=int,default=settings.top_k)
args=p.parse_args()

df=pd.read_csv(args.input)
out=AudiencePipeline(demo_mode=args.mode=="demo",threshold=args.threshold,top_k=args.top_k).run(df)
paths=export_bundle(out.dataframe,summary_dict(out),settings.export_dir,f"pipeline_{args.mode}")
print(f"Mode: {out.mode}")
print(f"Comments: {len(out.dataframe)} | AHS: {out.ahs} | Total seconds: {out.timings['total_s']}")
print(paths)
