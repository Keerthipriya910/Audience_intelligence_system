from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

root=Path(__file__).resolve().parents[1]
exp=root/"exports"/"experiment_report.json"
if not exp.exists():
    raise SystemExit("Run `python -m scripts.run_experiments --mode research` first.")
r=json.loads(exp.read_text(encoding="utf-8"))
mode=r.get("provenance",{}).get("mode","UNKNOWN")
cls=r.get("classification",{})
ml=r.get("multilingual",[])
abl=r.get("ablation",[])
sweep=r.get("threshold_sweep",[])
warning="" if mode=="RESEARCH" else "> **DEMO OUTPUT — DO NOT REPORT AS EXPERIMENTAL EVIDENCE.**\n\n"
lines=[f"# PrismPulse Experimental Results\n",warning,f"Execution mode: **{mode}**\n",
       "## Reproducibility provenance\n",
       "```json\n"+json.dumps(r.get("provenance",{}),indent=2)+"\n```\n",
       "## Classification\n"]
if cls.get("available"):
    lines += [f"- Accuracy: {cls['accuracy']:.4f}\n",f"- Macro-F1: {cls['macro_f1']:.4f}\n",f"- Weighted-F1: {cls['weighted_f1']:.4f}\n"]
else:
    lines += ["Ground-truth classification metrics were unavailable for this dataset.\n"]
lines += ["\n## Multilingual slices\n",pd.DataFrame(ml).to_markdown(index=False) if ml else "No multilingual slice metrics available.","\n\n## Ablation\n",pd.DataFrame(abl).to_markdown(index=False),"\n\n## Threshold sweep\n",pd.DataFrame(sweep).to_markdown(index=False),"\n\n## Reporting note\n","Only values produced by a **RESEARCH** run should be copied into a paper as measured results. Preserve the exported JSON/CSV alongside the paper for traceability.\n"]
out=root/"exports"/"paper_results.md"
out.write_text("".join(lines),encoding="utf-8")
print(out)
