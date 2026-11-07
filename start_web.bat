@echo off
title NL DocChat — SA-CMS Neural Memory Studio (Web Chatbot)
cd /d "%~dp0"

if exist "%~dp0DACN-NCKH-main\start_offline_demo.bat" (
    cd /d "%~dp0DACN-NCKH-main"
    call start_offline_demo.bat
    exit /b %ERRORLEVEL%
)

call "%~dp0start_offline_demo.bat"
