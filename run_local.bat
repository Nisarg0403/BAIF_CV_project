@echo off
title BAIF CattleWeightAI - Local Development Launcher
echo ========================================================
echo   Starting BAIF CattleWeightAI Locally (FastAPI + React)
echo ========================================================

REM 1. Start FastAPI Backend Server on port 8000 using whitelisted Python runtime
start "CattleWeightAI Backend (FastAPI)" cmd /k "C:\Users\NISARG\PYTHON_DATA\python.exe -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000"

REM 2. Start React Vite Frontend on port 5173
start "CattleWeightAI Frontend (React Vite)" cmd /k "cd frontend && npm run dev"

echo.
echo [OK] Both servers launched in separate windows!
echo   - Backend API:  http://127.0.0.1:8000
echo   - React UI:     http://localhost:5173
echo.
echo Open http://localhost:5173 in your browser to test.
echo ========================================================
pause
