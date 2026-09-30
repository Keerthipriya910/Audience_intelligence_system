@echo off
if not exist .venv\Scripts\python.exe call setup_windows.bat
call .venv\Scripts\activate
python -m scripts.run_pipeline --mode demo --input data/sample_comments.csv
streamlit run app.py
