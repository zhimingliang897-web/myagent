# Integrity Tools 部署文档

## 一、架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Pages (HTTPS)                      │
│                 zhimingliang897-web.github.io/integrity          │
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   首页展示   │    │   Demo页面  │    │  源码浏览   │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                           │                                       │
│                           │ 点击"登录"按钮                         │
│                           ▼                                       │
│                  跳转到云服务器登录页                              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      云服务器 (HTTP)                              │
│                     8.138.164.133:5000                           │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Gunicorn + Flask                       │   │
│  │                                                           │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐     │   │
│  │  │  认证   │  │ AI辩论赛│  │ Token   │  │ AI对比  │     │   │
│  │  │  模块   │  │         │  │ 计算    │  │         │     │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────┘     │   │
│  │                                                           │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────────────┐   │   │
│  │  │ 台词学习│  │PDF工具集│  │   LLM API Providers     │   │   │
│  │  │         │  │         │  │ Qwen|Doubao|Kimi|DeepSeek│   │   │
│  │  └─────────┘  └─────────┘  └─────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   SQLite    │  │  WireGuard  │  │   systemd   │              │
│  │   数据库    │  │   (VPN)     │  │   服务管理   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

## 二、目录结构

```
/root/integrity-tools/
├── .env                     # 环境变量（API Keys、邀请码）
├── config.py                # 全局配置
├── requirements.txt         # Python 依赖
├── run.py                   # 应用入口
├── init_db.py               # 数据库初始化脚本
├── gunicorn.conf.py         # Gunicorn 配置
├── deploy.sh                # 部署脚本
├── integrity-tools.service  # systemd 服务文件
├── DEPLOY.md                # 本文档
│
├── data/
│   └── app.db               # SQLite 数据库
│
└── app/
    ├── __init__.py          # Flask 应用工厂
    ├── models.py            # 数据模型 (User, UserSession)
    ├── routes.py            # 前端页面路由
    │
    ├── auth/                # 认证模块
    │   ├── __init__.py
    │   └── routes.py        # 登录/注册/验证 API
    │
    ├── tools/               # 工具模块
    │   ├── debate/          # AI 辩论赛
    │   │   ├── __init__.py
    │   │   ├── engine.py    # 辩论引擎核心逻辑
    │   │   ├── llm.py       # LLM 客户端封装
    │   │   ├── tts.py       # Edge TTS 语音合成
    │   │   ├── routes.py    # API 路由
    │   │   └── output/      # 输出目录（音频等）
    │   │
    │   ├── tokens/          # Token 计算
    │   │   ├── __init__.py
    │   │   └── routes.py
    │   │
    │   ├── lines/           # 台词学习
    │   │   ├── __init__.py
    │   │   ├── routes.py
    │   │   └── data/        # 学习数据存储
    │   │
    │   ├── compare/         # AI 模型对比
    │   │   ├── __init__.py
    │   │   └── routes.py
    │   │
    │   └── pdf/             # PDF 工具集
    │       ├── __init__.py
    │       └── routes.py    # 图片转PDF/合并/删页/重排
    │
    └── templates/           # 前端页面
        ├── index.html       # 首页/工具导航
        ├── login.html       # 登录页
        ├── register.html    # 注册页
        ├── debate.html      # AI 辩论赛页面
        ├── tokens.html      # Token 计算页面
        ├── lines.html       # 台词学习页面
        ├── compare.html     # AI 对比页面
        └── pdf.html         # PDF 工具集页面
```

## 三、工具列表

| 工具 | 路径 | 功能 | 认证 |
|------|------|------|------|
| AI 辩论赛 | `/debate` | 多模型驱动的AI辩论，支持SSE流式输出和Edge TTS语音合成 | 需登录 |
| Token 计算 | `/tokens` | 计算不同模型的Token消耗和成本 | 无需 |
| AI 模型对比 | `/compare` | 同一问题横向对比多个大模型，并发调用 | 需登录 |
| 台词学习 | `/lines` | 上传PDF生成学习笔记 | 需登录 |
| PDF 工具集 | `/pdf` | 图片转PDF、PDF合并、删页、页面重排 | 无需 |

## 四、API 路由

### 认证 API

| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/auth/register` | POST | 注册（需邀请码） |
| `/api/auth/login` | POST | 登录 |
| `/api/auth/verify` | GET | 验证 Token |

### AI 辩论赛 API

| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/debate/config` | GET | 获取辩论配置 |
| `/api/debate/start` | POST | 开始辩论 |
| `/api/debate/stream` | GET | SSE 流式输出 |
| `/api/debate/stop` | POST | 停止辩论 |
| `/api/debate/status` | GET | 获取状态 |

### Token 计算 API

| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/tokens/calc` | POST | Token 计算 |
| `/api/tokens/providers` | GET | 获取可用模型列表 |
| `/api/tokens/compare` | POST | 多模型对比 |

### AI 对比 API

| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/compare/providers` | GET | 获取可用提供商 |
| `/api/compare` | POST | 并发对比多个模型 |

### PDF 工具 API

| 路由 | 方法 | 说明 |
|------|------|------|
| `/api/pdf/info` | POST | 获取PDF信息 |
| `/api/pdf/images_to_pdf` | POST | 图片转PDF |
| `/api/pdf/merge` | POST | PDF合并 |
| `/api/pdf/remove_pages` | POST | 删除页面 |
| `/api/pdf/download/<filename>` | GET | 下载文件 |

## 五、添加新工具步骤

### 步骤 1：创建工具目录

```bash
mkdir -p app/tools/new_tool
touch app/tools/new_tool/__init__.py
touch app/tools/new_tool/routes.py
```

### 步骤 2：编写蓝图

**`app/tools/new_tool/__init__.py`**

```python
from flask import Blueprint
new_tool_bp = Blueprint('new_tool', __name__)
from app.tools.new_tool.routes import *
```

**`app/tools/new_tool/routes.py`**

```python
from flask import request, jsonify
from app.tools.new_tool import new_tool_bp
from app.auth.routes import token_required  # 如需认证

@new_tool_bp.route('/action', methods=['POST'])
@token_required  # 需要登录
def action():
    data = request.json
    # 业务逻辑
    return jsonify({'result': 'ok'})
```

### 步骤 3：注册蓝图

**`app/__init__.py`** 添加：

```python
from app.tools.new_tool import new_tool_bp
app.register_blueprint(new_tool_bp, url_prefix='/api/new_tool')
```

### 步骤 4：添加前端页面

**`app/templates/new_tool.html`**

### 步骤 5：添加页面路由

**`app/routes.py`** 添加：

```python
@app.route('/new_tool')
def new_tool_page():
    return render_template('new_tool.html')
```

### 步骤 6：添加到首页

**`app/templates/index.html`** 在 `.tools-grid` 中添加工具卡片

### 步骤 7：部署

```bash
scp -r integrity-tools root@8.138.164.133:/root/
ssh root@8.138.164.133 "systemctl restart integrity-tools"
```

## 六、配置文件

### .env

```bash
SECRET_KEY=integrity-lab-secret-key-2026
INVITE_CODES=demo2026,friend2026,test2026

# LLM API Keys
DASHSCOPE_API_KEY=sk-xxx
DOUBAO_API_KEY=xxx
KIMI_API_KEY=sk-xxx
DEEPSEEK_API_KEY=sk-xxx
```

### config.py 关键配置

```python
SECRET_KEY          # JWT 签名密钥
INVITE_CODES        # 有效邀请码集合
LLM_PROVIDERS       # LLM 提供商配置
DEBATERS            # 辩手配置
MAX_WORDS           # 每次发言最大字数 (200)
FREE_DEBATE_ROUNDS  # 自由辩论轮数 (6)
```

## 七、部署命令速查

### 日常维护

```bash
# 查看服务状态
systemctl status integrity-tools

# 重启服务
systemctl restart integrity-tools

# 查看日志
journalctl -u integrity-tools -f

# 查看最近 100 行日志
journalctl -u integrity-tools -n 100
```

### 更新部署

```bash
# 完整重新部署
cd /root/integrity-tools && ./deploy.sh

# 只重启服务（代码已更新）
systemctl restart integrity-tools
```

### 数据库操作

```bash
cd /root/integrity-tools
python3

>>> from app import create_app, db
>>> from app.models import User
>>> app = create_app()
>>> with app.app_context():
...     users = User.query.all()
...     for u in users:
...         print(u.username, u.invite_code)
```

### 添加邀请码

```bash
vim /root/integrity-tools/.env
# 修改 INVITE_CODES=demo2026,friend2026,newcode
systemctl restart integrity-tools
```

## 八、服务器信息

| 项目 | 值 |
|------|-----|
| IP | 8.138.164.133 |
| SSH 端口 | 22 |
| 服务端口 | 5000 |
| Python | /root/miniconda3/bin/python3 |
| 项目目录 | /root/integrity-tools |
| 数据库 | /root/integrity-tools/data/app.db |
| 日志 | journalctl -u integrity-tools |

### 保留的服务

| 服务 | 端口 | 说明 |
|------|------|------|
| WireGuard | 51820 | VPN 服务，不要修改 |

## 九、常见问题

### Q1: 登录后 Token 多久过期？

默认 7 天。修改 `app/auth/routes.py`：

```python
'exp': datetime.datetime.utcnow() + datetime.timedelta(days=7)
```

### Q2: AI辩论赛卡住怎么办？

检查日志：
```bash
journalctl -u integrity-tools -n 50
```

常见原因：
1. API Key 失效
2. 网络超时
3. Gunicorn worker 超时（已设置300秒）

### Q3: PDF工具下载文件404？

PDF工具使用临时目录，服务重启后文件会清空。需要重新处理。

### Q4: GitHub Pages 登录按钮无法使用？

确保 `docs/assets/js/tools-auth.js` 中的 `SERVER_URL` 正确：

```javascript
const SERVER_URL = 'http://8.138.164.133:5000';
```

然后提交并推送到 GitHub。

## 十、依赖列表

```
flask>=3.0.0
flask-sqlalchemy>=3.0.0
flask-cors>=4.0.0
pyjwt>=2.8.0
werkzeug>=3.0.0
python-dotenv>=1.0.0
openai>=1.0.0
edge-tts>=6.1.0
requests>=2.31.0
pdfplumber>=0.10.0
Pillow>=10.0.0
pypdf>=3.0.0
gunicorn>=21.0.0
aiohttp>=3.9.0
```

## 十一、更新日志

### 2026-03-14

- 新增 PDF 工具集（图片转PDF、合并、删页、重排）
- 修复 AI 辩论赛 TTS 容错处理
- 增加 Gunicorn timeout 到 300 秒
- 修复前端 SSE 连接时序问题
- 删除废弃的 docs/server 目录
- 完善部署文档