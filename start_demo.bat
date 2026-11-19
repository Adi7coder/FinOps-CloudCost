@echo off
REM Open backend and dashboard in two separate windows, then open Swagger
cd /d "%~dp0"
start "FinOps Backend (FastAPI)" cmd /k call run_backend.bat
start "FinOps Dashboard (Streamlit)" cmd /k call run_dashboard.bat
timeout /t 6 /nobreak >nul
start "" http://127.0.0.1:8000/docs
