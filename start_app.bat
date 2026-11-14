@echo off
title NL DocChat — SA-CMS Desktop App Studio
cd /d "%~dp0"

echo ==============================================================================
echo [NL DocChat / SA-CMS] KHOI DONG UNG DUNG CHATBOT HOI DAP TAI LIEU (APP MODE)
echo ==============================================================================
echo Backbone:   SmolLM2-135M (Frozen 100%%) + 5.31M SA-CMS Continuum Memory
echo Checkpoint: cms_3lvl_extended.pt (Extended Bilingual GPU Verified)
echo Mode:       Desktop App Window (100%% Local Offline)
echo ==============================================================================

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set SA_CMS_OFFLINE_MODE=1

REM Xac thuc moi truong Python tren may thay hoac may local
set PYTHON_EXE=
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import uvicorn, fastapi" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=.venv\Scripts\python.exe"
)
if "%PYTHON_EXE%"=="" if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" -c "import uvicorn, fastapi" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=..\.venv\Scripts\python.exe"
)
if "%PYTHON_EXE%"=="" if exist "D:\Applaptrinh\Python\python.exe" (
    "D:\Applaptrinh\Python\python.exe" -c "import uvicorn, fastapi" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=D:\Applaptrinh\Python\python.exe"
)
if "%PYTHON_EXE%"=="" (
    python -c "import uvicorn, fastapi" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=python"
)
if "%PYTHON_EXE%"=="" (
    py -3 -c "import uvicorn, fastapi" >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=py -3"
)
if "%PYTHON_EXE%"=="" (
    set PYTHON_EXE=python
    echo [Setup] Dang kiem tra va cai dat thu vien can thiet tu requirements.txt...
    python -m pip install -r requirements.txt
)

REM Mo cua so Desktop App doc lap (khong thanh dia chi URL, khong tab trinh duyet)
echo Dang mo cua so ung dung tai http://127.0.0.1:8000 ...
start /b "" cmd /c "timeout /t 2 >nul & (if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --app=http://127.0.0.1:8000) else if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --app=http://127.0.0.1:8000) else (start "" "http://127.0.0.1:8000"))"

echo Dang khoi chay backend API tai cong 8000...
"%PYTHON_EXE%" run_offline.py --host 127.0.0.1 --port 8000
pause

