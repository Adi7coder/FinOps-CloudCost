@echo off
REM Start the FastAPI backend (Swagger: http://127.0.0.1:8000/docs)
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
pause
