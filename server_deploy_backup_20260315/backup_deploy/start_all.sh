#!/bin/bash
# 服务器启动脚本 - 启动所有服务

echo "===== 启动所有服务 ====="

# 启动MyAgent (端口7860)
echo "启动 MyAgent (端口7860)..."
cd /root/myagent
source /root/miniconda3/bin/activate myagent
nohup python webui.py > /tmp/myagent.log 2>&1 &
echo "MyAgent 启动完成"

# 启动AI辩论赛 (端口5001)
echo "启动 AI辩论赛 (端口5001)..."
cd /opt/integrity/5AI辩论赛
nohup /root/miniconda3/bin/python app.py > /tmp/debate.log 2>&1 &
echo "AI辩论赛 启动完成"

# 启动PDF工具 (端口5002)
echo "启动 PDF工具 (端口5002)..."
cd /opt/integrity/12pdffun
nohup /root/miniconda3/bin/python app.py > /tmp/pdf.log 2>&1 &
echo "PDF工具 启动完成"

# 启动台词学习 (端口5003)
echo "启动 台词学习 (端口5003)..."
cd /opt/integrity/2台词
nohup /root/miniconda3/bin/python app.py > /tmp/lines.log 2>&1 &
echo "台词学习 启动完成"

# 启动File Agent (端口5004)
echo "启动 File Agent (端口5004)..."
cd /opt/integrity/14.1file-agent-v1
nohup /root/miniconda3/bin/python app.py > /tmp/file-agent.log 2>&1 &
echo "File Agent 启动完成"

# 启动视频下载 (端口5005)
echo "启动 视频下载 (端口5005)..."
cd /opt/integrity/21视频下载/web
nohup /root/miniconda3/bin/python app.py > /tmp/video.log 2>&1 &
echo "视频下载 启动完成"

# 启动Integrity Tools (端口5006)
echo "启动 Integrity Tools (端口5006)..."
cd /opt/integrity/integrity-tools
nohup /root/miniconda3/bin/python3.13 /root/miniconda3/bin/gunicorn -c gunicorn.conf.py run:app > /tmp/gunicorn.log 2>&1 &
echo "Integrity Tools 启动完成"

# 重启Nginx
echo "重启 Nginx..."
systemctl reload nginx

echo ""
echo "===== 所有服务启动完成 ====="
echo "查看服务状态: ps aux | grep python"
echo "查看端口监听: netstat -tlnp"
