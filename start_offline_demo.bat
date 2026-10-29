@echo off
title SA-CMS Neural Memory Studio (100%% Offline Edition)
cd /d "%~dp0"

echo ==============================================================================
echo [SA-CMS] KHOI DONG STUDIO HOI DAP BO NHO LIEN TUC DA THANG (100%% STRICT OFFLINE)
echo ==============================================================================
echo Backbone: SmolLM2-135M (Frozen) + 5.31M SA-CMS Continuum Memory
echo Mode: 100%% Local Offline (Zero External CDN / Zero Internet Socket Egress)
echo VRAM Peak: 743.86 MB (RTX 3050 Laptop Verified)
echo ==============================================================================

set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set SA_CMS_OFFLINE_MODE=1

REM Tim duong dan Python phu hop
set PYTHON_EXE=python
if exist "..\.venv\Scripts\python.exe" (
    set PYTHON_EXE=..\.venv\Scripts\python.exe
) else if exist ".venv\Scripts\python.exe" (
    set PYTHON_EXE=.venv\Scripts\python.exe
) else if exist "D:\Applaptrinh\Python\python.exe" (
    set PYTHON_EXE=D:\Applaptrinh\Python\python.exe
)

echo Dang mo trinh duyet tai http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"

echo Dang khoi chay backend API tai cong 8000 su dung %PYTHON_EXE%...
"%PYTHON_EXE%" -m uvicorn src.web.app:app --host 127.0.0.1 --port 8000 --log-level info
pause
