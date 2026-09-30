from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


def export_bundle(df: pd.DataFrame, summary: dict, export_dir: Path, prefix: str="analysis") -> dict[str,str]:
    export_dir.mkdir(parents=True,exist_ok=True)
    csv_path=export_dir/f"{prefix}_comments.csv"
    json_path=export_dir/f"{prefix}_summary.json"
    df.to_csv(csv_path,index=False,encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding="utf-8")
    return {"csv":str(csv_path),"json":str(json_path)}
