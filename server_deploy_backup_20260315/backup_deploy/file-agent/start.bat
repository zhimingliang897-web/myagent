@echo off
chcp 65001 >nul
echo ========================================
echo   Private File System v1.0
echo ========================================
echo.

cd /d "%~dp0"

call conda activate file-agent
if errorlevel 1 (
    echo [ERROR] Please create conda env first: conda create -n file-agent python=3.11
    pause
    exit /b 1
)

if not exist "config.json" (
    echo [INFO] Copying config file...
    copy config.example.json config.json >nul
)

echo [INFO] Checking dependencies...
pip install -r requirements.txt -q 2>nul

echo [INFO] Starting server...
echo.
python app.py

pause