from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import settings
from src.youtube_client import fetch_comments
from src.pipeline import AudiencePipeline
from src.analytics import summary_dict
from src.evaluation import (
    ablation_summary,
    classification_metrics,
    multilingual_metrics,
    trust_evaluation,
)
from src.storage import export_bundle
from src.grok_client import ask_grok, build_context
from src.integrated_gradients import explain_with_integrated_gradients

ROOT = Path(__file__).resolve().parent
SAMPLE = ROOT / "data" / "sample_comments.csv"
PALETTE = [
    "#7CF8D3",  # mint
    "#B38BFF",  # violet
    "#FF6FA7",  # coral pink
    "#FFD166",  # amber
    "#68E1FD",  # sky
    "#B7F171",  # lime
    "#FF9B71",  # peach
    "#DDB7FF",  # soft lilac
]
SENTIMENT_COLORS = {"positive": PALETTE[0], "neutral": PALETTE[3], "negative": PALETTE[2]}

st.set_page_config(
    page_title="PrismPulse AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
  --bg: #0B1220;
  --bg-soft: #111827;
  --panel: #111B2D;
  --panel-2: #162235;
  --line: rgba(255,255,255,.08);
  --line-strong: rgba(136,229,210,.20);
  --text: #F5F7FA;
  --muted: #9AA7B8;
  --teal: #78D6C6;
  --teal-strong: #8BE0D2;
  --blue: #7AA2F7;
  --silver: #C7D0DB;
  --gold: #E6C87A;
  --rose: #E89AAA;
}

html, body, [class*=\"css\"], [data-testid=\"stAppViewContainer\"] {
  font-family: 'Inter', sans-serif;
  color: var(--text);
}

.stApp {
  background:
    radial-gradient(circle at 18% 0%, rgba(122,162,247,.10), transparent 28%),
    radial-gradient(circle at 85% 5%, rgba(120,214,198,.08), transparent 24%),
    linear-gradient(180deg, #0A1120 0%, #0C1423 48%, #0A111C 100%);
}

.main .block-container {
  max-width: 1520px;
  padding-top: 1.1rem;
  padding-bottom: 2rem;
}

[data-testid=\"stSidebar\"] {
  background: linear-gradient(180deg, #0A1220 0%, #0D1726 100%);
  border-right: 1px solid var(--line);
}

[data-testid=\"stSidebar\"] h3 {
  font-family: 'Manrope', sans-serif;
  letter-spacing: -0.02em;
}

.hero-wrap {
  position: relative;
  overflow: hidden;
  padding: 30px 32px;
  border-radius: 24px;
  border: 1px solid rgba(255,255,255,.09);
  background:
    linear-gradient(135deg, rgba(255,255,255,.045), rgba(255,255,255,.018)),
    #101A2B;
  box-shadow: 0 24px 70px rgba(0,0,0,.24);
  margin-bottom: 22px;
}

.hero-wrap::after {
  content: \"\";
  position: absolute;
  inset: 0 0 auto auto;
  width: 340px;
  height: 100%;
  background: linear-gradient(115deg, transparent 10%, rgba(120,214,198,.055));
  pointer-events: none;
}

.hero-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.7fr) minmax(320px, .95fr);
  gap: 26px;
  align-items: stretch;
}

.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 14px;
  padding: 7px 11px;
  border-radius: 8px;
  border: 1px solid rgba(120,214,198,.20);
  background: rgba(120,214,198,.07);
  color: #A9E6DB;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.hero-title {
  font-family: 'Manrope', sans-serif;
  font-size: clamp(2.5rem, 3.8vw, 4rem);
  line-height: 1.02;
  margin: 0;
  color: #F7F9FC;
  font-weight: 800;
  letter-spacing: -0.05em;
}

.hero-title::after {
  content: \" / Executive Intelligence\";
  display: block;
  margin-top: 10px;
  color: var(--teal);
  font-family: 'Inter', sans-serif;
  font-size: .85rem;
  font-weight: 600;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.hero-sub {
  margin-top: 16px;
  max-width: 920px;
  color: #B8C2D0;
  font-size: 1rem;
  line-height: 1.72;
}

.hero-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 9px;
  margin-top: 18px;
}

.hero-chip {
  padding: 7px 10px;
  border-radius: 8px;
  background: rgba(255,255,255,.035);
  border: 1px solid var(--line);
  color: #D8DEE8;
  font-size: .75rem;
  font-weight: 600;
}

.hero-side {
  display: grid;
  grid-template-columns: repeat(2, minmax(0,1fr));
  gap: 11px;
}

.hero-side-card {
  padding: 15px;
  border-radius: 16px;
  background: #131F32;
  border: 1px solid var(--line);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
}

.hero-side-label {
  color: #8FA0B5;
  font-size: .7rem;
  text-transform: uppercase;
  letter-spacing: .08em;
  font-weight: 700;
}

.hero-side-value {
  color: #F5F7FA;
  font-size: 1.04rem;
  font-weight: 700;
  margin-top: 9px;
}

.hero-side-note {
  color: #9CA9B9;
  font-size: .77rem;
  margin-top: 7px;
  line-height: 1.45;
}

.metric-card {
  position: relative;
  padding: 17px 17px 15px;
  border-radius: 16px;
  background: #111C2E;
  border: 1px solid var(--line);
  min-height: 124px;
  box-shadow: 0 14px 34px rgba(0,0,0,.16);
}

.metric-card::before {
  content: \"\";
  position: absolute;
  top: 0; left: 0; bottom: 0;
  width: 3px;
  border-radius: 16px 0 0 16px;
  background: var(--teal);
}
.metric-card[data-accent=\"violet\"]::before { background: var(--blue); }
.metric-card[data-accent=\"pink\"]::before { background: var(--rose); }
.metric-card[data-accent=\"amber\"]::before { background: var(--gold); }
.metric-card[data-accent=\"sky\"]::before { background: #7CC4E8; }

.metric-title {
  font-size: .7rem;
  color: #92A0B3;
  text-transform: uppercase;
  letter-spacing: .08em;
  font-weight: 700;
}

.metric-value {
  font-family: 'Manrope', sans-serif;
  font-size: 1.95rem;
  font-weight: 800;
  color: #F7F9FC;
  margin-top: 10px;
  letter-spacing: -0.04em;
}

.metric-note {
  color: #9EB0C4;
  font-size: .76rem;
  margin-top: 7px;
  line-height: 1.4;
}

.dashboard-panel {
  padding: 16px 18px 13px;
  border-radius: 16px;
  border: 1px solid var(--line);
  background: #111C2D;
  box-shadow: 0 12px 32px rgba(0,0,0,.12);
  margin-bottom: 14px;
}

.section-title {
  font-family: 'Manrope', sans-serif;
  font-size: 1.16rem;
  font-weight: 800;
  color: #F6F8FB;
  margin-bottom: 4px;
  letter-spacing: -0.02em;
}

.section-sub {
  color: #94A3B8;
  font-size: .86rem;
  line-height: 1.55;
}

.mode-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 10px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,.035);
  font-size: .78rem;
  font-weight: 700;
  margin-bottom: 10px;
}
.mode-research { color: var(--teal); }
.mode-demo { color: var(--gold); }

.insight {
  padding: 15px 16px;
  border-radius: 12px;
  border: 1px solid rgba(120,214,198,.16);
  background: rgba(120,214,198,.055);
  color: #EDF2F7;
  margin: 10px 0 12px;
  line-height: 1.62;
}

.stTabs [data-baseweb=\"tab-list\"] {
  gap: 6px;
  background: #0E1727;
  border: 1px solid var(--line);
  padding: 6px;
  border-radius: 12px;
}

.stTabs [data-baseweb=\"tab\"] {
  border-radius: 8px;
  padding: 10px 14px;
  color: #B3BECC;
  font-weight: 600;
}

.stTabs [aria-selected=\"true\"] {
  background: #17253A;
  color: #F7F9FC;
  box-shadow: inset 0 -2px 0 var(--teal);
}

.stButton > button, .stDownloadButton > button {
  border-radius: 10px;
  border: 1px solid rgba(120,214,198,.22);
  background: linear-gradient(180deg, #173346, #122839);
  color: #F8FAFC;
  font-weight: 700;
  box-shadow: 0 8px 18px rgba(0,0,0,.12);
}

.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: rgba(120,214,198,.42);
  background: linear-gradient(180deg, #1A3A4F, #143044);
}

[data-testid=\"stTextInput\"] input,
[data-testid=\"stTextArea\"] textarea,
[data-testid=\"stNumberInput\"] input,
[data-testid=\"stSelectbox\"] div[data-baseweb=\"select\"],
[data-testid=\"stFileUploader\"] section {
  background: #0F1929 !important;
  border-radius: 10px !important;
  border: 1px solid rgba(255,255,255,.08) !important;
}

[data-testid=\"stSlider\"] [role=\"slider\"] {
  background: var(--teal);
}

div[data-testid=\"stDataFrame\"] {
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
}

[data-testid=\"stExpander\"] {
  border: 1px solid var(--line);
  border-radius: 12px;
  background: rgba(255,255,255,.02);
}

small, .stCaption, label, .stMarkdown p {
  color: #C8D0DB;
}

hr { border-color: var(--line); }

@media (max-width: 980px) {
  .hero-grid { grid-template-columns: 1fr; }
}
@media (max-width: 640px) {
  .hero-side { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_sample():
    return pd.read_csv(SAMPLE)


def metric_card(title: str, value: str, note: str = "", accent: str = "mint") -> None:
    st.markdown(
        f'''
        <div class="metric-card" data-accent="{accent}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'''
        <div class="dashboard-panel">
            <div class="section-title">{title}</div>
            <div class="section-sub">{subtitle}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def fig_layout(fig, height: int = 390):
    fig.update_layout(
        height=height,
        margin=dict(l=15, r=15, t=44, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#DCD8EB", family="Manrope"),
        legend_title_text="",
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        title=dict(font=dict(size=17, color="#F5F3FF", family="Manrope")),
        hoverlabel=dict(bgcolor="#0D1226", bordercolor="rgba(255,255,255,.12)", font=dict(color="#F6F5FE")),
    )
    try:
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,.08)", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,.08)", zeroline=False)
    except Exception:
        pass
    return fig


st.markdown(
    """
    <div class="hero-wrap">
      <div class="hero-grid">
        <div>
          <div class="hero-kicker">Executive-grade audience intelligence</div>
          <h1 class="hero-title">PrismPulse AI</h1>
          <div class="hero-sub">
            Beyond the sentiment score — adaptive XLM-R analysis, real-comment evidence retrieval,
            EWTC trust correction, explainability, emotion detection, topic/root-cause discovery,
            and grounded conversational audience intelligence inside a premium analytic experience.
          </div>
          <div class="hero-chip-row">
            <div class="hero-chip">XLM-RoBERTa</div>
            <div class="hero-chip">FAISS Evidence Retrieval</div>
            <div class="hero-chip">EWTC Trust Correction</div>
            <div class="hero-chip">BERTopic & Root Causes</div>
            <div class="hero-chip">Audience Health Score</div>
          </div>
        </div>
        <div class="hero-side">
          <div class="hero-side-card">
            <div class="hero-side-label">Interface Mood</div>
            <div class="hero-side-value">Executive Dark</div>
            <div class="hero-side-note">Boardroom-ready visual system with restrained contrast, enterprise spacing, and premium depth.</div>
          </div>
          <div class="hero-side-card">
            <div class="hero-side-label">Analysis Modes</div>
            <div class="hero-side-value">Operational + Research</div>
            <div class="hero-side-note">Clear execution modes for controlled demos and reproducible research workflows.</div>
          </div>
          <div class="hero-side-card">
            <div class="hero-side-label">Evidence Lens</div>
            <div class="hero-side-value">Auditable</div>
            <div class="hero-side-note">Uncertain predictions remain traceable through evidence retrieval and EWTC signals.</div>
          </div>
          <div class="hero-side-card">
            <div class="hero-side-label">Exports</div>
            <div class="hero-side-value">Structured</div>
            <div class="hero-side-note">Structured CSV and JSON outputs for reporting, validation, and downstream research workflows.</div>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ◈ Analysis Console")
    st.caption("Configure the input source, evidence controls, and run mode for your dashboard session.")
    execution = st.radio(
        "Execution mode",
        ["Demo Mode", "Research Mode"],
        index=0,
        help="Research Mode runs Hugging Face models and may download model weights on first use.",
    )
    demo_mode = execution == "Demo Mode"
    source = st.selectbox("Input source", ["Sample research dataset", "Upload CSV", "YouTube Data API v3"])
    df_in = None
    if source == "Sample research dataset":
        df_in = load_sample()
        st.caption("Bundled multilingual sample with labels for evaluation.")
    elif source == "Upload CSV":
        upload = st.file_uploader("CSV with a text column", type=["csv"])
        if upload is not None:
            df_in = pd.read_csv(upload)
    else:
        yt_key = st.text_input("YouTube API key", value=os.getenv("YOUTUBE_API_KEY", ""), type="password")
        video = st.text_input("YouTube URL / video ID")
        max_comments = st.slider("Max comments", 50, 1000, min(settings.max_comments, 400), 50)
        if st.button("Fetch comments", use_container_width=True):
            try:
                with st.spinner("Collecting YouTube comments..."):
                    st.session_state["fetched_df"] = fetch_comments(yt_key, video, max_comments)
                st.success(f"Fetched {len(st.session_state['fetched_df'])} comments")
            except Exception as e:
                st.error(str(e))
        df_in = st.session_state.get("fetched_df")
    st.markdown("---")
    threshold = st.slider("Confidence threshold", 0.50, 0.95, float(settings.confidence_threshold), 0.01)
    top_k = st.slider("Evidence Top-K", 1, 12, int(settings.top_k), 1)
    enable_emotion = st.toggle("Emotion analysis", True)
    enable_topics = st.toggle("BERTopic / root causes", True)
    st.markdown("---")
    st.caption("Credentials stay local in your session / .env and are not written into exported analysis files.")
    run = st.button(
        "✦ Run Audience Intelligence",
        type="primary",
        use_container_width=True,
        disabled=df_in is None,
    )

if run:
    try:
        with st.spinner("Running the complete analysis pipeline..."):
            pipe = AudiencePipeline(demo_mode=demo_mode, threshold=threshold, top_k=top_k)
            st.session_state["output"] = pipe.run(df_in, enable_topics=enable_topics, enable_emotion=enable_emotion)
        st.success("Analysis complete.")
    except Exception as e:
        st.exception(e)

output = st.session_state.get("output")
if output is None:
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Pipeline", "Ready", "Choose a source and launch the analysis", "mint")
    with c2:
        metric_card("Research Core", "EWTC", "Median-kernel evidence correction", "violet")
    with c3:
        metric_card("LLM Layer", "Optional", "Grok is used only for grounded conversation", "pink")
    section_header(
        "What this build executes",
        "The dashboard is connected to the actual pipeline. Research Mode uses real transformer inference and semantic embeddings; Demo Mode remains explicitly separated for fast presentation and offline-friendly checks.",
    )
    st.stop()


df = output.dataframe
summary = summary_dict(output)
mode_class = "mode-demo" if output.mode == "DEMO" else "mode-research"
st.markdown(
    f'Execution provenance: <span class="mode-pill">Current run · <span class="{mode_class}">{output.mode} MODE</span></span>',
    unsafe_allow_html=True,
)

sent = df["final_sentiment"].value_counts(normalize=True).mul(100)
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    metric_card("Comments", f"{len(df):,}", f"{df['language'].nunique()} language groups", "mint")
with c2:
    metric_card("Audience Health", f"{output.ahs:.1f}", "0–100 proposed research index", "violet")
with c3:
    metric_card("Positive", f"{sent.get('positive', 0):.1f}%", "trust-adjusted sentiment", "pink")
with c4:
    metric_card("Evidence Routed", f"{output.timings['evidence_route_pct']:.1f}%", f"C < {threshold:.2f}", "amber")
with c5:
    metric_card("Throughput", f"{output.timings['comments_per_second']:.2f}/s", f"total {output.timings['total_s']:.2f}s", "sky")

tabs = st.tabs([
    "Overview",
    "Sentiment",
    "Evidence & Trust",
    "Explainability",
    "Emotion",
    "Topics & Root Causes",
    "Audience Intelligence",
    "Experiments",
    "Grok Analyst",
    "Data & Export",
])

with tabs[0]:
    section_header(
        "Audience signal overview",
        "A high-level view of trust-adjusted sentiment, detected languages, confidence spread, and engagement shape.",
    )
    a, b = st.columns([1, 1])
    with a:
        vc = df["final_sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Comments")
        fig = px.pie(
            vc,
            names="Sentiment",
            values="Comments",
            hole=.68,
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
        )
        fig.update_traces(textposition="inside", textfont_size=13)
        st.plotly_chart(fig_layout(fig), use_container_width=True)
    with b:
        lc = df["language"].value_counts().head(10).rename_axis("Language").reset_index(name="Comments")
        fig = px.bar(lc, x="Comments", y="Language", orientation="h", color="Language", color_discrete_sequence=PALETTE)
        st.plotly_chart(fig_layout(fig), use_container_width=True)
    conf = px.histogram(
        df,
        x="final_confidence",
        color="final_sentiment",
        nbins=20,
        color_discrete_map=SENTIMENT_COLORS,
        title="Trust-adjusted confidence distribution",
    )
    st.plotly_chart(fig_layout(conf, 350), use_container_width=True)

with tabs[1]:
    section_header(
        "Sentiment dynamics",
        "Compare sentiment across languages, engagement intensity, and time-based movement when timestamps are available.",
    )
    a, b = st.columns(2)
    with a:
        s = df.groupby(["language", "final_sentiment"]).size().reset_index(name="Comments")
        fig = px.bar(
            s,
            x="language",
            y="Comments",
            color="final_sentiment",
            barmode="group",
            color_discrete_map=SENTIMENT_COLORS,
            title="Sentiment by language",
        )
        st.plotly_chart(fig_layout(fig), use_container_width=True)
    with b:
        fig = px.scatter(
            df,
            x="like_count",
            y="final_confidence",
            size=(df["reply_count"] + 1),
            color="final_sentiment",
            hover_data=["text"],
            color_discrete_map=SENTIMENT_COLORS,
            title="Engagement × confidence",
        )
        st.plotly_chart(fig_layout(fig), use_container_width=True)
    if pd.to_datetime(df["published_at"], errors="coerce").notna().sum() > 2:
        temp = df.copy()
        temp["published_at"] = pd.to_datetime(temp["published_at"], errors="coerce")
        temp = temp.dropna(subset=["published_at"])
        temp["day"] = temp["published_at"].dt.date
        trend = temp.groupby(["day", "final_sentiment"]).size().reset_index(name="Comments")
        fig = px.line(
            trend,
            x="day",
            y="Comments",
            color="final_sentiment",
            markers=True,
            color_discrete_map=SENTIMENT_COLORS,
            title="Sentiment over time",
        )
        st.plotly_chart(fig_layout(fig, 350), use_container_width=True)

with tabs[2]:
    section_header(
        "EWTC evidence inspector",
        "Inspect the real comments used to support or contradict an uncertain prediction routed through evidence retrieval.",
    )
    routed = df[df["evidence_routed"]]
    if routed.empty:
        st.info("No comments fell below the selected confidence threshold. Raise the threshold to route more comments through evidence retrieval.")
    else:
        choices = {f"#{i} · {str(r.text)[:85]}": i for i, r in routed.iterrows()}
        chosen = st.selectbox("Select an evidence-routed comment", list(choices.keys()))
        r = df.loc[choices[chosen]]
        a, b, c, d = st.columns(4)
        with a:
            metric_card("Original", r["original_sentiment"].title(), f"C = {r['original_confidence']:.3f}", "violet")
        with b:
            metric_card("EWTC", r["final_sentiment"].title(), f"C_final = {r['final_confidence']:.3f}", "mint")
        with c:
            metric_card("Agreement", f"{r['ewtc_agreement']:+.3f}", "weighted [-1, +1]", "pink")
        with d:
            metric_card("λ", f"{r['ewtc_lambda']:.3f}", "derived as 1 − C", "amber")
        st.markdown(f'<div class="insight"><b>Comment</b><br>{r["text"]}</div>', unsafe_allow_html=True)
        ev = pd.DataFrame(r["evidence"])
        if not ev.empty:
            st.dataframe(ev[["comment", "label", "similarity", "distance"]], use_container_width=True, hide_index=True)
            fig = px.bar(
                ev,
                x="similarity",
                y="comment",
                orientation="h",
                color="label",
                color_discrete_map=SENTIMENT_COLORS,
                title="Retrieved evidence similarity",
            )
            st.plotly_chart(fig_layout(fig, 360), use_container_width=True)

with tabs[3]:
    section_header(
        "Prediction explainability",
        "Explore token-level cues and, in Research Mode, compute transformer-level Captum Integrated Gradients for a selected comment.",
    )
    idx = st.selectbox("Comment to explain", df.index, format_func=lambda i: f"#{i} · {df.loc[i, 'text'][:95]}")
    r = df.loc[idx]
    st.markdown(
        f'<div class="insight"><b>{r["final_sentiment"].title()}</b> · confidence {r["final_confidence"]:.3f}<br>{r["text"]}</div>',
        unsafe_allow_html=True,
    )
    exp = pd.DataFrame(r["explanation"])
    if exp.empty:
        st.info("No strong token cues were extracted for this comment.")
    else:
        fig = px.bar(
            exp,
            x="contribution",
            y="token",
            orientation="h",
            color="contribution",
            color_continuous_scale=[[0, PALETTE[2]], [.5, "#6E6A86"], [1, PALETTE[0]]],
            title="Token contribution cues",
        )
        st.plotly_chart(fig_layout(fig, 350), use_container_width=True)
    if output.mode == "RESEARCH":
        if st.button("Compute transformer Integrated Gradients", key="ig_button"):
            with st.spinner("Computing Captum Integrated Gradients on the selected XLM-R prediction..."):
                try:
                    ig = pd.DataFrame(
                        explain_with_integrated_gradients(
                            AudiencePipeline(demo_mode=False).sentiment,
                            r["clean_text"],
                            r["final_sentiment"],
                        )
                    )
                    st.session_state["ig_result"] = ig
                except Exception as e:
                    st.error(str(e))
        if isinstance(st.session_state.get("ig_result"), pd.DataFrame) and not st.session_state["ig_result"].empty:
            ig = st.session_state["ig_result"]
            fig = px.bar(
                ig,
                x="contribution",
                y="token",
                orientation="h",
                color="contribution",
                color_continuous_scale=[[0, PALETTE[2]], [.5, "#6E6A86"], [1, PALETTE[0]]],
                title="Captum Integrated Gradients",
            )
            st.plotly_chart(fig_layout(fig, 390), use_container_width=True)
    else:
        st.caption("Demo Mode shows lexical explanation cues. Switch to Research Mode to compute transformer Integrated Gradients with Captum on demand.")

with tabs[4]:
    section_header(
        "Emotion map",
        "A compact view of the audience emotion spectrum and how emotional patterns distribute across detected languages.",
    )
    ev = df["emotion"].value_counts().rename_axis("Emotion").reset_index(name="Comments")
    fig = px.bar(ev, x="Emotion", y="Comments", color="Emotion", color_discrete_sequence=PALETTE, title="Audience emotion spectrum")
    st.plotly_chart(fig_layout(fig), use_container_width=True)
    e2 = df.groupby(["language", "emotion"]).size().reset_index(name="Comments")
    fig = px.sunburst(e2, path=["language", "emotion"], values="Comments", color="emotion", color_discrete_sequence=PALETTE, title="Emotion by language")
    st.plotly_chart(fig_layout(fig, 480), use_container_width=True)

with tabs[5]:
    section_header(
        "Topics & root-cause candidates",
        "See the discovered discussion landscape, its sentiment profile, and linked root-cause candidate clusters.",
    )
    tdf = df["topic_name"].value_counts().rename_axis("Topic").reset_index(name="Comments")
    fig = px.treemap(tdf, path=["Topic"], values="Comments", color="Comments", color_continuous_scale=[PALETTE[1], PALETTE[0]], title="Discovered discussion landscape")
    st.plotly_chart(fig_layout(fig, 450), use_container_width=True)
    ts = df.groupby(["topic_name", "final_sentiment"]).size().reset_index(name="Comments")
    fig = px.bar(ts, x="topic_name", y="Comments", color="final_sentiment", barmode="stack", color_discrete_map=SENTIMENT_COLORS, title="Topic sentiment profile")
    st.plotly_chart(fig_layout(fig, 370), use_container_width=True)
    st.markdown("#### Linked root-cause candidates")
    if not output.root_causes:
        st.info("No root-cause clusters were produced for this run.")
    for rc in output.root_causes[:8]:
        flag = "⚠ negative-dominant" if rc["flagged"] else "mixed / non-negative"
        with st.expander(f"{rc['root_cause']} · {rc['comments']} comments · {flag}"):
            st.write(f"Negative share: **{rc['negative_share']*100:.1f}%**")
            for x in rc["representative_comments"]:
                st.write("•", x)

with tabs[6]:
    section_header(
        "Audience Health Score",
        "Inspect the proposed audience-intelligence index and the transparent component-level breakdown behind the final score.",
    )
    vals = list(output.ahs_components.values())
    cats = list(output.ahs_components.keys())
    fig = go.Figure(
        go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            line=dict(color=PALETTE[0], width=3),
            fillcolor="rgba(124,248,211,.16)",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,.12)", linecolor="rgba(255,255,255,.12)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,.08)", linecolor="rgba(255,255,255,.10)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
    )
    st.plotly_chart(fig_layout(fig, 470), use_container_width=True)
    st.dataframe(pd.DataFrame({"Component": cats, "Score": vals}), use_container_width=True, hide_index=True)
    st.info("AHS is a proposed audience-intelligence index, not a clinical or psychological measure. The implementation uses transparent equal weighting to avoid hidden tuning.")

with tabs[7]:
    section_header(
        "Evaluation & experiments",
        "Ground-truth metrics, multilingual slices, ablation views, EWTC trust checks, and timing/throughput measurements.",
    )
    metrics = classification_metrics(df)
    st.markdown("#### Classification evaluation")
    if not metrics.get("available"):
        st.warning("Ground-truth evaluation requires an input column named `label` with negative/neutral/positive values. The bundled sample dataset includes this column.")
    else:
        a, b, c = st.columns(3)
        with a:
            metric_card("Accuracy", f"{metrics['accuracy']:.3f}", "ground truth vs final", "mint")
        with b:
            metric_card("Macro-F1", f"{metrics['macro_f1']:.3f}", "equal class weighting", "violet")
        with c:
            metric_card("Weighted-F1", f"{metrics['weighted_f1']:.3f}", "support-weighted", "pink")
        cm = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale=[[0, "#151526"], [.5, PALETTE[1]], [1, PALETTE[0]]], title="Confusion matrix")
        st.plotly_chart(fig_layout(fig, 390), use_container_width=True)
    mm = pd.DataFrame(multilingual_metrics(df))
    if not mm.empty:
        st.markdown("#### Multilingual slices")
        st.dataframe(mm, use_container_width=True, hide_index=True)
    st.markdown("#### Ablation table")
    st.dataframe(pd.DataFrame(ablation_summary(df)), use_container_width=True, hide_index=True)
    st.markdown("#### Trust correction evaluation")
    st.json(trust_evaluation(df))
    st.markdown("#### Efficiency")
    st.json(output.timings)

with tabs[8]:
    section_header(
        "Grounded Grok analyst",
        "Ask questions about this run. Grok only receives the computed analysis context and is instructed not to invent evidence.",
    )
    xai_key = st.text_input("xAI API key", value=os.getenv("XAI_API_KEY", ""), type="password", key="xai_tab")
    model = st.text_input("Grok model", value=os.getenv("GROK_MODEL", settings.grok_model))
    q = st.text_area("Ask about this audience", placeholder="What are the main negative root causes, and which evidence supports them?")
    if st.button("Ask Grok", key="ask_grok") and q.strip():
        context = build_context(df, output.ahs, output.ahs_components, output.root_causes)
        with st.spinner("Grounding the response in this run's analysis..."):
            st.session_state["grok_answer"] = ask_grok(q, context, xai_key, model)
    if st.session_state.get("grok_answer"):
        st.markdown(st.session_state["grok_answer"])

with tabs[9]:
    section_header(
        "Data & export center",
        "Review the comment-level table and export both detailed analysis rows and research-summary artifacts.",
    )
    st.dataframe(df.drop(columns=["sentiment_probs", "evidence", "explanation"], errors="ignore"), use_container_width=True, hide_index=True)
    bundle = export_bundle(df, summary, settings.export_dir, "latest")
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    json_bytes = json.dumps(summary, indent=2, ensure_ascii=False, default=str).encode("utf-8")
    a, b = st.columns(2)
    with a:
        st.download_button("Download comment-level CSV", csv_bytes, "prismpulse_analysis.csv", "text/csv", use_container_width=True)
    with b:
        st.download_button("Download research summary JSON", json_bytes, "prismpulse_summary.json", "application/json", use_container_width=True)
    st.caption(f"Local exports also written to {bundle['csv']} and {bundle['json']}")
