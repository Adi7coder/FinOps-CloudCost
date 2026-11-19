@echo off
REM Start the Streamlit dashboard (http://localhost:8501)
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m streamlit run dashboard.py
pause
