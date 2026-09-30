@echo off
if not exist .venv\Scripts\python.exe call setup_windows.bat
call .venv\Scripts\activate
uvicorn api:app --reload --port 8000
