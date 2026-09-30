from __future__ import annotations

import re
from typing import Iterable
import pandas as pd

VIDEO_ID_PATTERNS = [
    re.compile(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})"),
    re.compile(r"^([A-Za-z0-9_-]{11})$")
]


def parse_video_id(value: str) -> str:
    value = (value or "").strip()
    for pattern in VIDEO_ID_PATTERNS:
        m = pattern.search(value)
        if m:
            return m.group(1)
    raise ValueError("Enter a valid YouTube video URL or 11-character video ID.")


def fetch_comments(api_key: str, video: str, max_comments: int = 400) -> pd.DataFrame:
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is missing. Add it to .env or paste it in the dashboard session.")
    from googleapiclient.discovery import build

    video_id = parse_video_id(video)
    youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
    rows: list[dict] = []
    page_token = None

    while len(rows) < max_comments:
        req = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=min(100, max_comments - len(rows)),
            pageToken=page_token,
            textFormat="plainText",
            order="time",
        )
        resp = req.execute()
        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]
            s = top["snippet"]
            rows.append({
                "comment_id": top["id"],
                "text": s.get("textDisplay", ""),
                "published_at": s.get("publishedAt", ""),
                "like_count": int(s.get("likeCount", 0) or 0),
                "reply_count": int(item["snippet"].get("totalReplyCount", 0) or 0),
                "video_id": video_id,
            })
            if len(rows) >= max_comments:
                break
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["comment_id"]).reset_index(drop=True)
    return df
