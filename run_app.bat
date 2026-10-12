@echo off
REM FORGETRACE - double-click to start. Close this window (or press Ctrl+C) to stop the app.
cd /d "%~dp0"
set PYTHONUTF8=1
echo Starting FORGETRACE... first start can take a minute (OCR models load).
echo Open http://localhost:8501 if the browser does not open by itself.
python -m streamlit run app/app.py
pause
