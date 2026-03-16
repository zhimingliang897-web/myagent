# Integrity Tools

Web 工具集：台词对比、辩论生成、Token 统计等，使用 Flask + 多模型 LLM。

## 运行

```bash
pip install -r requirements.txt
python init_db.py   # 首次创建数据库
python run.py       # 开发环境
# 或 gunicorn -c gunicorn.conf.py run:app
```

## 需补充的隐私/本地配置（未随仓库提交）

| 文件名 | 说明（对他人） | 样式/格式 |
|--------|----------------|-----------|
| **.env** | 服务密钥与各 LLM 的 API Key，用于登录、邀请码与辩论/台词等能力 | 纯文本，每行 `KEY=value`。需包含：`SECRET_KEY`（Flask 会话密钥）、`INVITE_CODES`（逗号分隔邀请码）、`DASHSCOPE_API_KEY`、`DOUBAO_API_KEY`、`KIMI_API_KEY`、`DEEPSEEK_API_KEY`、`OPENAI_API_KEY`（可选）。 |

**自己使用**：从本仓库的 **`_secrets/integrity-tools/.env`** 拷贝到本项目 `integrity-tools/` 目录下（文件名保持不变）即可。
