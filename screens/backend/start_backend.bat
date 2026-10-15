@echo off
echo.
echo  NeuroPath AI — Starting Backend
echo  ====================================
echo  Backend: http://localhost:8000
echo  API docs: http://localhost:8000/docs
echo.
cd /d "%~dp0"
set PYTHON=C:\Users\asaad\anaconda3\python.exe
%PYTHON% -m pip install -r requirements.txt --quiet
%PYTHON% -m uvicorn main:app --reload --port 8000
