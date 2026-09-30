# PrismPulse AI
### Beyond the Sentiment Score: An Adaptive, Explainable Framework for Evidence-Backed Multilingual YouTube Audience Intelligence

PrismPulse AI is an end-to-end implementation of the supplied research proposal. It analyses fixed research datasets or YouTube comments with multilingual sentiment, confidence-based routing, semantic evidence retrieval, EWTC trust correction, explainability, emotion, topics, root-cause linking, an Audience Health Score, and an optional Grok conversational layer.

## 1. What is included

- YouTube Data API v3 collection and CSV upload
- text preprocessing + English/Hindi/Telugu/code-mixed language handling
- XLM-RoBERTa sentiment inference with probabilities
- adaptive confidence routing
- multilingual SentenceTransformer embeddings
- FAISS cosine-similarity retrieval
- EWTC with median-derived kernel width and `lambda = 1 - confidence`
- real evidence inspector
- on-demand Captum Integrated Gradients in Research Mode
- emotion analysis
- BERTopic + root-cause topic linking
- engagement and language-diversity signals
- transparent Audience Health Score
- Grok grounded conversational analysis
- stunning Streamlit + Plotly dashboard
- baseline/ablation evaluation, multilingual metrics and confusion matrix
- threshold sweeps, latency, throughput and evidence-routing measurements
- CSV/JSON exports and paper-results Markdown generator
- FastAPI endpoints
- one-click Windows scripts

## 2. Important execution modes

**Research Mode** runs the actual Hugging Face sentiment model, multilingual embeddings, FAISS and BERTopic. The first run downloads model weights and therefore needs internet access.

**Demo Mode** uses deterministic lightweight substitutes where necessary. It exists for UI/offline demonstration only. Demo outputs are clearly tagged and must not be used as research claims.

## 3. Fastest Windows setup

Install **Python 3.11 (64-bit)** and then, from the project folder:

```bat
setup_windows.bat
run_dashboard.bat
```

The dashboard opens in your browser. No API key is required for the bundled sample dataset.

In PowerShell, use `.\setup_windows.bat` and `.\run_dashboard.bat`.
The dashboard launcher uses the project's virtual environment directly and runs
setup if Python or Streamlit is missing. Run setup again to repair an incomplete
environment; it preserves an existing `.env`.

## 4. API keys

Copy `.env.example` to `.env` (the setup script does this automatically if `.env` does not exist), then add keys if needed:

```env
YOUTUBE_API_KEY=your_youtube_data_api_v3_key
XAI_API_KEY=your_xai_key
```

Never commit `.env`. The application never writes these keys into result exports.

## 5. Run the dashboard manually

```bat
.venv\Scripts\python.exe -m streamlit run app.py
```

## 6. Run the pipeline from CLI

Fast demo:

```bat
python -m scripts.run_pipeline --mode demo --input data/sample_comments.csv
```

Research run:

```bat
python -m scripts.run_pipeline --mode research --input data/sample_comments.csv
```

## 7. Run reproducible experiments

```bat
python -m scripts.run_experiments --mode research --input data/sample_comments.csv
python -m scripts.generate_paper_results
```

Outputs appear in `exports/`:

- `threshold_sweep.csv`
- `experiment_report.json`
- `paper_results.md`

The report includes execution provenance and will warn if the source run was Demo Mode.

## 8. Dataset format

Required column:

```text
text
```

Recommended columns:

```text
comment_id,text,label,published_at,like_count,reply_count,video_id
```

For accuracy/F1 evaluation, `label` must contain `negative`, `neutral`, or `positive` ground truth.

## 9. Run the REST API

```bat
uvicorn api:app --reload --port 8000
```

Endpoints:

- `GET /health`
- `POST /analyze/csv`
- `POST /analyze/youtube`

FastAPI Swagger UI is available at `/docs` while the server is running.

## 10. EWTC implementation

For an uncertain model prediction with confidence `C`:

1. Retrieve Top-K semantic neighbors.
2. Convert cosine distance into a Gaussian kernel weight.
3. Derive `sigma` from the median of the retrieved distances.
4. Compute weighted agreement `A` on `[-1, 1]`.
5. Derive correction strength `lambda = 1 - C`.
6. Compute:

```text
C_final = clip(C + lambda * A * (1 - C), 0, 1)
```

This matches the proposal's confidence-scaled correction structure without introducing a manually tuned fusion coefficient.

## 11. CPU / 8-GB-RAM guidance

Use Python 3.11, close memory-heavy applications, keep YouTube runs around 100–400 comments while developing, and run BERTopic after sentiment/FAISS are verified. Model inference is CPU-compatible but Research Mode is substantially slower than Demo Mode.

## 12. Research integrity

The bundled sample labels are a functional test fixture, not a published benchmark. Do not copy Demo Mode scores into a paper. For publication experiments, run Research Mode on the frozen research dataset, keep `experiment_report.json`, and report the model IDs/threshold/Top-K/hardware used.
