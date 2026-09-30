@echo off
if not exist .venv\Scripts\python.exe call setup_windows.bat
call .venv\Scripts\activate
python -m scripts.run_experiments --mode research --input data/sample_comments.csv
python -m scripts.generate_paper_results
pause
