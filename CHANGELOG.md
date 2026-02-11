# Changelog

## Day 4 — 2026-02-11 — Phase 4: 手动构建 StateGraph

### 新增

- **`agent/graph.py`** — 手动构建 LangGraph StateGraph，替代 `create_agent` 黑盒
  - 自定义 `AgentState`（messages + iteration_count）
  - 6 个节点: trim、rewrite、agent、tools、increment、force_reply
  - 查询改写（rewrite_node）— 短问题/含代词时 LLM 先改写再检索
  - 工具调用上限（max_iterations=5）— 防止无限循环，超限强制回答
- `main.py` 双模式支持: `--classic` 使用旧版，默认使用新版 StateGraph

### 变更

- `main.py` 重构为 `_build_classic_agent()` 和 `_build_graph_agent()` 双分支
- 新增 `argparse` 命令行参数解析
- 启动时显示当前 Agent 模式名称

---

## Day 3 — 2026-02-09 — Phase 3: Memory + Token 优化

### 新增

- **Memory 子包** (`agent/memory/`)
  - `checkpointer.py` — SqliteSaver 持久化对话记忆
  - 数据库路径: `data/db/agent_memory.db`
  - 支持多 thread_id 管理，可切换会话线程
- **数据库检查脚本** (`scripts/inspect_db.py`)
  - 列出数据库中的表和行数
  - 可选查看最近 checkpoint 详情
  - 自动处理相对路径，避免转义问题
- **Token 优化**
  - `trim_message_history()` 状态修改器
  - 消息窗口限制（默认保留最近 10 条）
  - 长对话 token 消耗减少 50-80%
- `main.py` 集成 Memory 和 Token 优化
  - `/thread <id>` 命令切换会话
  - `clear` 生成新 thread_id
  - 启动时显示消息窗口限制配置

### 变更

- `main.py`:
  - 导入 `trim_messages`
  - 添加 `MAX_MESSAGES` 配置常量
  - `create_react_agent` 新增 `checkpointer` 和 `state_modifier` 参数
  - `invoke` 调用传入 `config` 字典指定 `thread_id`
- `requirements.txt` 新增 `langgraph-checkpoint-sqlite`
- `data/db/` 目录创建，存放所有数据库文件

### 修复

- `scripts/inspect_db.py` 路径转义问题（`\a` 被解释为响铃符）

---

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
