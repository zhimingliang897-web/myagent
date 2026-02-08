# Changelog

## Day 2 — 2026-02-08 — Phase 2: RAG + 多格式支持

### 新增

- **RAG 子包** (`agent/rag/`)
  - `loader.py` — 文档加载器，支持 txt/md/pdf/docx/xlsx/pptx + 图片/音频/视频
  - `vectorstore.py` — FAISS 向量存储，支持分 batch 向量化和增量追加
  - `retriever.py` — 将检索器包装为 Agent 工具 `knowledge_search`
- **索引脚本** (`scripts/index_docs.py`)
  - 支持 `--append` 增量模式
  - 支持 `--batch-size`、`--chunk-size`、`--chunk-overlap` 参数
- **多格式文档加载**
  - Office: Word (.docx)、Excel (.xlsx)、PowerPoint (.pptx)
  - 多媒体: 图片→视觉模型描述、音频→语音转文字、视频→视觉模型理解
- **模型配置集中化** — 所有模型名称收拢到 `config.py`
- `main.py` 集成 RAG 工具，启动时自动检测向量索引
- `requirements.txt` 新增 `faiss-cpu`、`python-docx`、`openpyxl`、`python-pptx`

### 变更

- `config.py` 新增 `EMBEDDING_MODEL`、`VISION_MODEL`、`ASR_MODEL` 配置项
- `vectorstore.py` 的 `build_vectorstore()` 新增 `batch_size` 和 `append` 参数
- 系统提示词增加 `knowledge_search` 工具说明和来源引用规则

---

## Day 1 — 2026-02-07 — Phase 1: 基础 Agent

### 新增

- **项目基础设施**: `.gitignore`、`.env`、`requirements.txt`
- **Agent 核心包** (`agent/`)
  - `config.py` — 环境变量加载 (python-dotenv)
  - `llm.py` — ChatTongyi LLM 封装 (DeepSeek V3.1 via DashScope)
  - `tools.py` — 3 个工具: 日期时间、安全计算器 (AST)、DuckDuckGo 搜索
  - `callbacks.py` — Token 用量统计回调 (用户自行添加)
- **CLI 入口** (`main.py`) — `create_react_agent` + 交互式聊天循环
- 基础 LCEL chain 测试 (`test.py`)

### 变更

- `test.py` 移除硬编码 API key，改用 `agent.config`
