from __future__ import annotations

from io import StringIO
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from src.pipeline import AudiencePipeline
from src.analytics import summary_dict
from src.youtube_client import fetch_comments
from src.config import settings

app=FastAPI(title="PrismPulse AI API",version="1.0.0")

class YouTubeRequest(BaseModel):
    video: str
    api_key: str | None = None
    max_comments: int = 200
    demo_mode: bool = False
    confidence_threshold: float = 0.72
    top_k: int = 5

@app.get("/health")
def health():
    return {"status":"ok","service":"PrismPulse AI"}

@app.post("/analyze/youtube")
def analyze_youtube(req: YouTubeRequest):
    try:
        df=fetch_comments(req.api_key or settings.youtube_api_key,req.video,req.max_comments)
        out=AudiencePipeline(demo_mode=req.demo_mode,threshold=req.confidence_threshold,top_k=req.top_k).run(df)
        return summary_dict(out)
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))

@app.post("/analyze/csv")
async def analyze_csv(
    file: UploadFile=File(...),
    demo_mode: bool=Form(False),
    confidence_threshold: float=Form(0.72),
    top_k: int=Form(5),
):
    try:
        raw=(await file.read()).decode("utf-8-sig")
        df=pd.read_csv(StringIO(raw))
        out=AudiencePipeline(demo_mode=demo_mode,threshold=confidence_threshold,top_k=top_k).run(df)
        return summary_dict(out)
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))
