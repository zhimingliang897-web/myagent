# Docker 部署

## 1. 配置 API Key

```bash
cd docker
cp .env.example .env
```

编辑 `.env`，填入你的 API 密钥：

```ini
DASHSCOPE_API_KEY=sk-xxxxxxxx
```

## 2. 启动

```bash
docker compose up -d
```

浏览器打开 `http://localhost:5000`

## 3. 停止

```bash
docker compose down
```

## 说明

- `word/` 目录通过 volume 挂载，数据保存在本地，不在容器内
- `.env` 文件已加入 `.gitignore`（`docker/.env`），不会上传到 GitHub
- 重建镜像：`docker compose up -d --build`
