@echo off
chcp 65001 >nul
title AI辩论赛
cls

echo ============================================
echo   AI辩论赛 — 一键启动
echo ============================================
echo.

:: 切换到脚本所在目录（确保相对路径正确）
cd /d "%~dp0"

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.10+
    echo        下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

:: 安装/更新依赖（静默模式）
echo [1/2] 检查依赖包...
pip install -r requirements.txt -q --no-warn-script-location
if errorlevel 1 (
    echo [警告] 部分依赖可能安装失败，尝试继续...
)
echo 依赖就绪。
echo.

:: 3秒后自动打开浏览器（在独立进程中）
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:5000"

echo [2/2] 启动服务，浏览器将在3秒后自动打开...
echo.
echo ============================================
echo   访问地址: http://localhost:5000
echo   按 Ctrl+C 可停止服务
echo ============================================
echo.

:: 前台运行 Flask（可以看到日志，Ctrl+C 停止）
python app.py

echo.
echo 服务已停止。
pause
