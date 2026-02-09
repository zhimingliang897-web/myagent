# MyAgent

基于 LangChain + LangGraph 的个人知识库问答智能体。

**核心能力**:
- 📚 **RAG 知识库**: 多格式文档加载、向量检索、基于文档回答
- 🧠 **持久化记忆**: 跨会话对话记忆，支持多线程管理
- 🛠️ **工具调用**: 日期时间、计算器、网页搜索
- 💰 **Token 优化**: 消息窗口限制，长对话节省 50-80% token

## 快速开始

```bash
# 1. 安装依赖（conda 环境）
conda activate myagent
pip install -r requirements.txt

# 2. 配置 API key
#    在项目根目录创建 .env 文件：
#    DASHSCOPE_API_KEY=sk-your-key-here

# 3. 索引文档
python scripts/index_docs.py data/

# 4. 启动对话
python main.py
```

## 项目结构

```
MyAgent/
│
├── main.py                    # CLI 入口，交互式聊天循环
├── test.py                    # 基础 LCEL chain 测试
├── requirements.txt           # Python 依赖
├── .env                       # API key（不入 git）
├── .gitignore
│
├── agent/                     # 核心 Agent 包
│   ├── __init__.py
│   ├── config.py              # 集中配置：API key + 所有模型名称
│   ├── llm.py                 # ChatTongyi LLM 封装
│   ├── callbacks.py           # Token 用量统计回调
│   ├── tools.py               # Agent 工具（日期、计算、搜索）
│   ├── rag/                   # RAG 子包
│   │   ├── __init__.py
│   │   ├── loader.py          # 多格式文档加载器
│   │   ├── vectorstore.py     # 向量存储（分块 + FAISS）
│   │   └── retriever.py       # 检索工具（包装为 Agent tool）
│   └── memory/                # Memory 子包
│       ├── __init__.py
│       └── checkpointer.py    # SqliteSaver 持久化记忆
│
├── scripts/
│   ├── index_docs.py          # 一键文档索引脚本
│   └── inspect_db.py          # 数据库检查脚本
│
├── data/                      # 用户文档 + 数据库目录
│   └── db/                    # 数据库文件（运行后生成）
│       └── agent_memory.db
├── vectorstore/               # FAISS 向量索引（自动生成，已 gitignore）
└── 工作日志/                   # 开发日志
```

## 模块说明

### config.py — 集中配置

所有模型名称统一管理，换模型只改这一个文件：


| 变量              | 用途             | 当前值                    |
| ----------------- | ---------------- | ------------------------- |
| `MODEL_NAME`      | Agent 对话主模型 | `deepseek-v3.1`           |
| `EMBEDDING_MODEL` | 向量嵌入         | `text-embedding-v3`       |
| `VISION_MODEL`    | 图片/视频理解    | `qwen-vl-plus-2025-08-15` |
| `ASR_MODEL`       | 语音识别         | `fun-asr-2025-11-07`      |

### tools.py — Agent 工具


| 工具                   | 功能                     | 依赖   |
| ---------------------- | ------------------------ | ------ |
| `get_current_datetime` | 获取当前日期时间         | 标准库 |
| `calculate`            | 安全数学计算（基于 AST） | 标准库 |
| `web_search`           | DuckDuckGo 网页搜索      | httpx  |

### loader.py — 文档加载器

**文本类**（直接提取文字）：


| 格式         | 处理方式                                 |
| ------------ | ---------------------------------------- |
| `.txt` `.md` | 读取原文                                 |
| `.pdf`       | PyPDFLoader 按页提取                     |
| `.docx`      | python-docx 提取段落                     |
| `.xlsx`      | openpyxl 按 sheet 提取，单元格用`|` 分隔 |
| `.pptx`      | python-pptx 按 slide 提取文本框          |

**多媒体类**（调 DashScope API 转文字后索引）：


| 格式                                        | 处理方式                  |
| ------------------------------------------- | ------------------------- |
| `.png` `.jpg` `.jpeg` `.bmp` `.webp` `.gif` | 视觉模型生成描述 + OCR    |
| `.mp3` `.wav` `.flac` `.m4a` `.ogg` `.aac`  | 语音识别转文字            |
| `.mp4` `.avi` `.mov` `.mkv` `.webm`         | 视觉模型视频理解（<50MB） |

### vectorstore.py — 向量存储

- 分块：`RecursiveCharacterTextSplitter`（默认 500 字符/块，100 重叠）
- 嵌入：DashScope `text-embedding-v3`（1024 维）
- 存储：FAISS 本地向量索引
- 支持分 batch 向量化（默认 25/批，DashScope API 限制）
- 支持增量追加（`--append`）

### retriever.py — 检索工具

将 FAISS retriever 包装为 `knowledge_search` Agent 工具。Agent 启动时自动检测向量索引是否存在，存在则加载，否则跳过（优雅降级）。

### callbacks.py — Token 统计

每次对话后输出累计的 prompt/completion/total tokens。

### checkpointer.py — 持久化记忆

- 使用 `SqliteSaver` 将对话状态存储到 SQLite 数据库
- 支持多 thread_id 线程管理，可切换不同对话上下文
- 命令：`/thread <id>` 切换线程，`clear` 生成新线程

### Token 优化

- **消息窗口限制**: 默认只保留最近 10 条消息发送给 LLM
- **配置**: 修改 `main.py` 中的 `MAX_MESSAGES` 变量
- **效果**: 长对话 token 消耗减少 50-80%
- **数据不丢失**: 所有历史仍完整存储在数据库中

## 索引文档

```bash
# 全量索引（覆盖旧索引）
python scripts/index_docs.py data/

# 增量追加新文档
python scripts/index_docs.py data/新文档.pdf --append

# 自定义参数
python scripts/index_docs.py data/ --batch-size 10 --chunk-size 800 --chunk-overlap 200


更新版(更稳健)
先试这组（中文/笔记类很常见）：

python scripts/index_docs.py data/ --chunk-size 800 --chunk-overlap 120 --batch-size 10

遇到限流/不稳定，再加 sleep
python scripts/index_docs.py data/ --chunk-size 800 --chunk-overlap 120 --batch-size 10 --sleep 0.5

增量追加
python scripts/index_docs.py data/ --append --batch-size 10

```

## 技术栈

- **LLM**: DeepSeek V3.1（通过阿里云 DashScope 调用）
- **框架**: LangChain + LangGraph
- **向量数据库**: FAISS
- **Embedding**: DashScope text-embedding-v3
- **语言**: Python 3.11（conda 环境）
