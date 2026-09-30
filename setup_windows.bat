@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  python -m venv .venv
) else (
  py -3.11 -m venv .venv
)
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
if not exist .env copy .env.example .env >nul
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 goto failed
 echo.
echo Setup complete. Edit .env only if you want YouTube API or Grok.
exit /b 0

:failed
echo Setup failed. Resolve the error above, then run setup_windows.bat again.
exit /b 1
