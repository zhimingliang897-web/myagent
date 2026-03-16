# 服务器部署指南

## 服务器信息
- IP: 8.138.164.133
- 系统: CentOS Linux release 7.9.2009
- Python: miniconda3 (Python 3.13)

## 目录结构

```
/opt/integrity/
├── 5AI辩论赛/          # 端口5001 - AI辩论赛服务
├── 12pdffun/           # 端口5002 - PDF工具
├── 2台词/              # 端口5003 - 台词学习服务
├── 14.1file-agent-v1/  # 端口5004 - 文件智能体
├── 21视频下载/web/      # 端口5005 - 视频下载服务
└── integrity-tools/    # 端口5006 - 工具集主站

/root/myagent/          # 端口7860 - MyAgent智能体
```

## 一、环境准备

### 1. 安装miniconda3
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 2. 创建Python环境
```bash
# 主环境
conda create -n myagent python=3.11 -y
conda activate myagent

# 或者使用系统Python 3.13
/root/miniconda3/bin/python3.13
```

### 3. 安装系统依赖
```bash
yum install -y nginx
yum install -y python3-pip
yum install -y git
```

## 二、服务部署

### 1. MyAgent智能体 (端口7860)

**目录**: `/root/myagent`

**启动命令**:
```bash
cd /root/myagent
conda activate myagent
nohup python webui.py > /tmp/myagent.log 2>&1 &
```

**依赖安装**:
```bash
pip install -r requirements.txt
```

**环境变量**: 创建 `.env` 文件配置API密钥

---

### 2. AI辩论赛 (端口5001)

**目录**: `/opt/integrity/5AI辩论赛`

**启动命令**:
```bash
cd /opt/integrity/5AI辩论赛
nohup /root/miniconda3/bin/python app.py > /tmp/debate.log 2>&1 &
```

**配置**: 创建 `.env` 文件配置API密钥

---

### 3. PDF工具 (端口5002)

**目录**: `/opt/integrity/12pdffun`

**启动命令**:
```bash
cd /opt/integrity/12pdffun
nohup /root/miniconda3/bin/python app.py > /tmp/pdf.log 2>&1 &
```

---

### 4. 台词学习 (端口5003)

**目录**: `/opt/integrity/2台词`

**启动命令**:
```bash
cd /opt/integrity/2台词
nohup /root/miniconda3/bin/python app.py > /tmp/lines.log 2>&1 &
```

---

### 5. File Agent (端口5004)

**目录**: `/opt/integrity/14.1file-agent-v1`

**启动命令**:
```bash
cd /opt/integrity/14.1file-agent-v1
nohup /root/miniconda3/bin/python app.py > /tmp/file-agent.log 2>&1 &
```

---

### 6. 视频下载 (端口5005)

**目录**: `/opt/integrity/21视频下载/web`

**启动命令**:
```bash
cd /opt/integrity/21视频下载/web
nohup /root/miniconda3/bin/python app.py > /tmp/video.log 2>&1 &
```

---

### 7. Integrity Tools主站 (端口5006)

**目录**: `/opt/integrity/integrity-tools`

**启动命令**:
```bash
cd /opt/integrity/integrity-tools
nohup /root/miniconda3/bin/python3.13 /root/miniconda3/bin/gunicorn -c gunicorn.conf.py run:app > /tmp/gunicorn.log 2>&1 &
```

**初始化数据库**:
```bash
python init_db.py
```

---

## 三、Nginx配置

### 配置文件位置
- 主配置: `/etc/nginx/nginx.conf`
- 站点配置: `/etc/nginx/conf.d/`

### 部署步骤

1. 复制配置文件:
```bash
cp nginx/api.liangyiren.top.conf /etc/nginx/conf.d/
cp nginx/integrity-services.conf /etc/nginx/conf.d/
```

2. 创建SSL证书目录:
```bash
mkdir -p /etc/nginx/ssl
# 复制证书文件到该目录
```

3. 测试并重载:
```bash
nginx -t
systemctl reload nginx
```

### 路由说明

| 路径 | 后端服务 | 端口 |
|------|----------|------|
| / | Integrity Tools | 5006 |
| /agent/ | MyAgent | 7860 |
| /debate | AI辩论赛 | 5001 |
| /pdf | PDF工具 | 5002 |
| /files/ | File Agent | 5004 |
| /video/ | Video Downloader | 5005 |
| /api/ | Integrity Tools API | 5006 |

---

## 四、端口说明

| 端口 | 服务 | 备注 |
|------|------|------|
| 22 | SSH | |
| 80 | Nginx HTTP | |
| 443 | Nginx HTTPS | (当前未启用) |
| 5001 | AI辩论赛 | |
| 5002 | PDF工具 | |
| 5003 | 台词学习 | |
| 5004 | File Agent | |
| 5005 | 视频下载 | |
| 5006 | Integrity Tools | gunicorn |
| 7860 | MyAgent | Gradio |
| 8000 | Nginx备用 | HTTP直连 |

---

## 五、防火墙配置

```bash
# 开放端口
firewall-cmd --permanent --add-port=80/tcp
firewall-cmd --permanent --add-port=443/tcp
firewall-cmd --permanent --add-port=5001-5006/tcp
firewall-cmd --permanent --add-port=7860/tcp
firewall-cmd --permanent --add-port=8000/tcp
firewall-cmd --reload

# 或者关闭防火墙
systemctl stop firewalld
systemctl disable firewalld
```

---

## 六、常用命令

### 查看服务状态
```bash
ps aux | grep python
netstat -tlnp
```

### 查看日志
```bash
tail -f /tmp/myagent.log
tail -f /tmp/gunicorn.log
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log
```

### 重启服务
```bash
# 杀掉所有Python服务
pkill -f python

# 重新启动各服务
# ... 按上面的启动命令执行
```

---

## 七、API密钥配置

各服务需要配置以下环境变量:

### AI辩论赛 (.env)
```
DASHSCOPE_API_KEY=xxx
DOUBAO_API_KEY=xxx
KIMI_API_KEY=xxx
DEEPSEEK_API_KEY=xxx
```

### MyAgent (.env)
```
OPENAI_API_KEY=xxx
# 或其他LLM配置
```

---

## 八、备份与恢复

### 备份
```bash
tar -czvf server_backup_$(date +%Y%m%d).tar.gz /root/backup_deploy/
```

### 恢复
1. 解压备份文件
2. 按照本指南重新部署
3. 恢复.env配置文件
4. 重启服务

---

生成时间: 2026-03-15
