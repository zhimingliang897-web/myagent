# 私人文件系统 v1

本项目是一个集「本地文件管理 + 远程访问 + AI 助手」于一体的 Web 文件系统，支持本地文件浏览、上传下载、搜索预览，并可选配阿里百炼 LLM 与 Natapp 内网穿透。

## 功能特色

| 功能 | 说明 |
|------|------|
| 文件浏览 | 文件夹导航、排序、分页 |
| 文件上传 | 多文件上传、拖拽上传、大文件支持 |
| 文件下载 | 单文件下载、多文件打包ZIP下载 |
| 在线预览 | 图片/视频/音频/PDF/Word/文本文件 |
| 文件操作 | 新建文件夹、重命名、移动、复制、删除 |
| 回收站 | 移入回收站、恢复、永久删除、清空 |
| 全文搜索 | 关键词搜索、按类型分类搜索 |
| 挂载目录 | 挂载多个本地目录、只读权限控制 |
| AI助手 | 对接阿里百炼LLM，自然语言管理文件 |
| 内网穿透 | 通过Natapp实现外网访问 |

## 目录结构

```text
14.1file-agent-v1/
  app.py                 # FastAPI 入口
  config.json            # 实际运行配置
  config.example.json    # 配置示例
  requirements.txt       # 依赖列表
  start.bat              # Windows 启动脚本
  natapp.exe             # Natapp 客户端（可选）
  app/                   # 应用逻辑 / 配置 / 依赖
  models/                # 数据模型
  services/              # 业务逻辑
  routers/               # API 路由
  templates/             # 前端模板
  static/                # 静态资源
  data/                  # SQLite 等数据文件
```

## 快速开始

### 1. 创建并激活虚拟环境

```bash
conda create -n file-agent python=3.11 -y
conda activate file-agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备配置文件

```bash
copy config.example.json config.json
```

编辑 `config.json`，根据自己的环境修改路径和密钥：

```json
{
  "storage": {
    "root": "F:\\MyFiles",
    "uploads": "F:\\MyFiles\\uploads",
    "trash": "F:\\MyFiles\\.trash"
  },
  "auth": {
    "password": "your_password",
    "session_secret": "random-string-for-encryption",
    "session_expire_hours": 24
  },
  "llm": {
    "api_key": "your-llm-api-key",
    "model": "qwen-plus",
    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
  },
  "natapp": {
    "token": "your-natapp-token",
    "enabled": true
  },
  "server": {
    "host": "0.0.0.0",
    "port": 5000
  }
}
```

### 4. 启动服务

方式一（推荐，在 Windows 上双击即可）：

```bash
start.bat
```

方式二（手动启动）：

```bash
conda activate file-agent
python app.py
```

启动后：

- Web UI：`http://localhost:5000`
- API 文档：`http://localhost:5000/docs`

## 配置说明

### storage 存储配置

| 字段 | 说明 |
|------|------|
| root | 主文件存储目录 |
| uploads | 上传文件临时目录 |
| trash | 回收站目录 |
| mounts | 挂载目录列表 |

### auth 认证配置

| 字段 | 说明 |
|------|------|
| password | 登录密码 |
| session_secret | Session加密密钥（建议修改） |
| session_expire_hours | Session过期时间（小时） |

### llm AI配置

| 字段 | 说明 |
|------|------|
| api_key | 阿里百炼API密钥 |
| model | 模型名称，如 qwen-plus |
| base_url | API地址 |

### natapp 内网穿透配置

| 字段 | 说明 |
|------|------|
| token | Natapp隧道Token |
| enabled | 是否启用内网穿透 |

### server 服务器配置

| 字段 | 说明 |
|------|------|
| host | 监听地址，0.0.0.0 表示所有网卡 |
| port | 监听端口 |

## 挂载目录（Mounts）

挂载目录用于把其他磁盘/路径统一纳入该系统管理，便于集中浏览和使用。

- 访问：`http://localhost:5000/mounts`
- 或从 `/settings` 页面点击 `Mounts Manager`

使用流程：

1. 打开 Mounts Manager  
2. 使用右下角的 Folder Browser 选择一个文件夹  
3. 设置名称 / 读写权限后，点击 Add 完成挂载  

挂载信息会自动写入 `config.json` 的 `storage.mounts` 配置中。

### 挂载目录权限

- **只读模式**：只能浏览、下载、预览，不能上传、删除、重命名
- **读写模式**：拥有完整操作权限

## 支持的预览格式

| 类型 | 格式 |
|------|------|
| 图片 | jpg, jpeg, png, gif, bmp, webp, svg, ico |
| 视频 | mp4, webm, ogg, mov |
| 音频 | mp3, wav, flac, aac, ogg, m4a |
| 文档 | pdf, doc, docx |
| 文本 | txt, md, py, js, ts, html, css, json, xml, log, csv |

## API 接口

### 文件管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/files | 获取文件列表 |
| POST | /api/files/upload | 上传文件 |
| POST | /api/files/folder | 创建文件夹 |
| GET | /api/files/download | 下载文件 |
| DELETE | /api/files | 删除文件（移入回收站） |
| POST | /api/files/move | 移动文件 |
| POST | /api/files/copy | 复制文件 |
| PUT | /api/files/rename | 重命名文件 |

### 预览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/preview | 获取文件预览信息 |
| GET | /api/preview/file | 获取文件流 |
| GET | /api/preview/thumb | 获取图片缩略图 |

### 搜索

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/search | 关键词搜索 |
| GET | /api/search/type/{type} | 按类型搜索 |

### 回收站

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/trash | 获取回收站列表 |
| POST | /api/trash/restore | 恢复文件 |
| DELETE | /api/trash | 永久删除 |
| DELETE | /api/trash/empty | 清空回收站 |

### AI助手

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/agent/chat | AI对话 |

## 常见问题

### Q: 内网穿透访问时预览/下载失败？

A: 这通常是浏览器缓存导致的，请按 `Ctrl + Shift + R` 强制刷新页面。

### Q: 如何修改默认密码？

A: 编辑 `config.json` 中的 `auth.password` 字段，或登录后在 `/settings` 页面修改。

### Q: Word文档预览显示乱码？

A: Word文档预览仅支持文本内容提取，复杂格式建议下载后查看。

### Q: 上传文件大小限制？

A: 默认限制 2GB，可在 `config.json` 的 `upload.max_size_mb` 中修改。

## 安全建议

1. **修改默认密码**：将 `auth.password` 改为强密码
2. **修改 Session 密钥**：将 `auth.session_secret` 改为随机字符串
3. **限制挂载权限**：对外部目录使用只读模式
4. **使用 HTTPS**：在生产环境中配置 SSL 证书

## License

MIT