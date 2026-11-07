@echo off
title NL DocChat — SA-CMS Neural Memory Studio (100%% Offline Edition)
cd /d "%~dp0"

echo ==============================================================================
echo [NL DocChat / SA-CMS] KHOI DONG HE THONG HOI DAP TAI LIEU 100%% STRICT OFFLINE
echo ==============================================================================
echo Backbone:   SmolLM2-135M (Frozen 100%%) + 5.31M SA-CMS Continuum Memory
echo Checkpoint: cms_3lvl_extended.pt (Extended Bilingual GPU Verified)
echo Mode:       100%% Local Offline (Zero External CDN / Zero Internet Egress)
echo ==============================================================================

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set SA_CMS_OFFLINE_MODE=1

REM Xac thuc moi truong Python co day du uvicorn + fastapi + torch
set PYTHON_EXE=
if exist "D:\Applaptrinh\Python\python.exe" (
    "D:\Applaptrinh\Python\python.exe" -c "import uvicorn, fastapi, torch" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=D:\Applaptrinh\Python\python.exe"
)
if "%PYTHON_EXE%"=="" if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import uvicorn, fastapi, torch" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=.venv\Scripts\python.exe"
)
if "%PYTHON_EXE%"=="" if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" -c "import uvicorn, fastapi, torch" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=..\.venv\Scripts\python.exe"
)
if "%PYTHON_EXE%"=="" set PYTHON_EXE=python

echo Dang mo trinh duyet tai http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"

echo Dang khoi chay backend API tai cong 8000 su dung "%PYTHON_EXE%"...
"%PYTHON_EXE%" run_offline.py --host 127.0.0.1 --port 8000
pause
