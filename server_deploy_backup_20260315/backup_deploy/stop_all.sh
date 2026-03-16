#!/bin/bash
# 服务器停止脚本 - 停止所有服务

echo "===== 停止所有服务 ====="

# 停止所有Python服务
echo "停止所有Python服务..."
pkill -f "python app.py"
pkill -f "python webui.py"
pkill -f "gunicorn"

sleep 2

# 检查是否还有残留进程
remaining=$(ps aux | grep -E "python.*app.py|python.*webui.py|gunicorn" | grep -v grep | wc -l)
if [ $remaining -gt 0 ]; then
    echo "强制杀掉残留进程..."
    pkill -9 -f "python app.py"
    pkill -9 -f "python webui.py"
    pkill -9 -f "gunicorn"
fi

echo ""
echo "===== 所有服务已停止 ====="
echo "查看进程: ps aux | grep python"
