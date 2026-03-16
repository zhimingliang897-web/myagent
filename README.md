# MyAgent (V2.0 多智能体版)

基于 LangChain + LangGraph 打造的个人专属知识库问答智能体。从基础的命令行工具一路进化为支持**多智能体协作**的现代化 AI 助理系统。

## 🌟 核心特性

### V2.0 新增：多智能体协作系统
- 🤖 **Supervisor + Workers 架构**: 智能任务分解与调度
- 💻 **Code Agent**: 代码生成、调试、语法检查
- 📊 **Data Agent**: 数据分析、统计计算、知识库检索
- ✍️ **Writer Agent**: 文章撰写、内容润色、格式化
- 🔄 **自动协作**: 复杂任务自动拆分，多 Agent 接力完成

### V1.0 基础能力
- 🎨 **Web UI 界面**: 基于 Gradio 构建的现代聊天交互窗口
- 🧠 **长短期双重记忆**:
  - 短期：基于 SQLite 持久化的会话上下文（Thread ID 管理）
  - 长期：自适应提取并保存跨会话的用户画像/偏好
- 📚 **混合检索 RAG**:
  - 支持多格式（TXT/MD/PDF/DOCX）加载
  - **Hybrid Search**: FAISS 向量检索 + BM25 词频检索，RRF 融合打分
  - **原生 Markdown 解析**: 带层级关系感知的语义切块
- 🛠️ **工具链网络**: 网页搜索、数学计算、时间查询等

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活 Python 虚拟环境 (推荐 Python 3.10+)
conda activate myagent

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
# 必填：对话与 RAG 等基础能力
DASHSCOPE_API_KEY=sk-your-key-here
```

**可选能力（可能产生费用）**：以下功能需在 `.env` 中**显式开启**后才会启用，默认关闭，避免误扣费。

| 能力 | 说明 | 配置方式 |
|------|------|----------|
| **文生图** | 根据文字描述生成图片（通义万相） | 待实现：`ENABLE_TEXT_TO_IMAGE=1` 并确保 DashScope 已开通万相 |
| **图生文** | 根据图片生成文字描述（视觉模型） | 待实现：`ENABLE_IMAGE_TO_TEXT=1` 并确保视觉模型可用 |

未配置或未开启时，系统不会调用上述 API，不会产生额外费用。

### 3. 构建本地知识库 (可选)

```bash
python scripts/index_docs.py data/
```

### 4. 启动应用

**方式一：一键启动脚本 (Windows 推荐)**

直接双击项目根目录下的 **`启动GUI.bat`**：
- 自动清理 `7860` 端口占用残留
- 自动识别并使用本机的 conda 虚拟环境 (`myagent`)
- 自动打开浏览器进入 Web 图形界面

**方式二：手动运行 Web 图形界面**
```bash
python webui.py
# 浏览器打开 http://127.0.0.1:7860
# 左侧控制面板可切换「单智能体」或「多智能体协作」模式
```

**方式三：命令行界面**
```bash
python main.py              # 单智能体模式（默认）
python main.py --multi      # 多智能体协作模式 ✨
python main.py --classic    # 经典 create_agent 模式
```

## 🤖 多智能体系统

### 架构图

```
用户输入: "分析数据趋势，写一份报告，并生成可视化代码"
                              │
                              ▼
                    ┌─────────────────┐
                    │   Supervisor    │  意图识别 + 任务分解
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Data Agent   │ ─► │ Writer Agent  │ ─► │  Code Agent   │
│  数据分析专家  │    │   写作专家     │    │   代码专家    │
└───────────────┘    └───────────────┘    └───────────────┘
        │                    │                    │
        └────────────────────┴────────────────────┘
                             │
                    ┌────────▼────────┐
                    │    Aggregator   │  结果汇总
                    └─────────────────┘
                             │
                             ▼
                        最终输出
```

### 工作流程

1. **Supervisor** 分析用户请求，创建有序任务计划（Task List）
2. 按计划依次调度 **Worker Agents** 执行子任务
3. 每个 Worker 完成后，结果作为上下文（`handoff_context`）传递给下一个 Worker
4. 若某个 Worker **执行失败**，自动触发 **Re-plan** 重新规划剩余任务
5. **Aggregator** 汇总所有结果，生成最终回答

### Re-plan 机制

当某个 Worker 执行失败时，系统不会直接放弃，而是启动智能重规划：

```
Worker 失败
    │
    ▼
increment → supervisor (检测到 needs_replan=True)
    │
    ▼
  LLM 重新规划        ← 了解失败原因 + 已完成任务上下文
    │
    ▼
  新任务计划          ← 保留已完成的任务，追加新子任务
    │
    ▼
继续执行... (最多重试 MAX_REPLAN_COUNT 次)
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `MAX_REPLAN_COUNT` | 2 | 最大重规划次数，防止无限循环 |
| `MAX_ITERATIONS` | 10 | 最大总迭代轮次保护 |

**Re-plan 策略**：Supervisor 收到失败信息后，知晓已完成的任务和失败原因，可以：调整任务顺序、拆分任务为更小步骤、更换执行 Agent、添加前置信息获取任务。

### 示例场景

| 用户请求 | 调度流程 |
|---------|---------|
| "写一段快速排序代码" | Supervisor → Code Agent |
| "帮我查一下今天日期" | Supervisor → Data Agent |
| "根据文档写一篇博客" | Supervisor → Data Agent → Writer Agent |
| "分析数据并生成图表代码" | Supervisor → Data Agent → Code Agent |

## 📁 目录结构

```text
MyAgent/
├── 启动GUI.bat                # Windows 一键启动 Web 界面（推荐）
├── webui.py                   # ✨ Gradio Web 界面入口
├── main.py                    # 💻 CLI 命令行入口
├── test_multi_agent.py        # 多智能体测试脚本
├── requirements.txt           # Python 依赖清单
├── .env                       # API 环境变量
│
├── agent/                     # 核心智能体模块
│   ├── graph.py               # 单智能体 StateGraph
│   ├── tools.py               # 基础工具库
│   ├── llm.py                 # LLM 封装
│   │
│   ├── multi/                 # 🆕 多智能体模块
│   │   ├── __init__.py
│   │   ├── state.py           # MultiAgentState 状态定义
│   │   ├── supervisor.py      # Supervisor 路由逻辑
│   │   ├── graph.py           # 多智能体 StateGraph
│   │   └── workers/           # Worker Agents
│   │       ├── base.py        # Worker 基类
│   │       ├── code_agent.py  # 代码专家
│   │       ├── data_agent.py  # 数据分析专家
│   │       └── writer_agent.py# 写作专家
│   │
│   ├── memory/
│   │   ├── checkpointer.py    # 短期会话记忆
│   │   └── profile.py         # 长期用户画像
│   │
│   └── rag/
│       ├── loader.py          # 文档加载器
│       ├── vectorstore.py     # 向量存储
│       └── retriever.py       # 混合检索器
│
├── scripts/
│   └── index_docs.py          # 建库脚本
├── data/                      # 知识库源文件
├── vectorstore/               # 向量索引（自动生成）
└── 工作日志/                   # 开发日志
```

## 🛠️ 技术架构

### 单智能体 ReAct 循环
```
START → trim → rewrite → agent → tools → increment → agent → ... → END
```
- **trim**: 裁剪历史消息，防止超出上下文窗口
- **rewrite**: 查询改写，提高 RAG 命中率
- **force_reply**: 工具调用上限保护（max=5）

### 多智能体 StateGraph
```
START → supervisor → route → worker → increment → supervisor → ... → aggregator → END
```
- **supervisor**: 任务分解与调度
- **route**: 动态路由到对应 Worker
- **aggregator**: 结果汇总

### 混合检索 (Hybrid RAG)
- **FAISS**: 语义向量检索
- **BM25**: 词频精确匹配
- **RRF**: 倒数排名融合

## 📜 版本历史

| 版本 | 日期 | 主要更新 |
|-----|------|---------|
| V2.0 | 2026-03 | 多智能体协作系统 (Supervisor + Workers) |
| V1.0 | 2026-02 | 完整单智能体系统 (RAG + Memory + WebUI) |

## 📄 许可证

本项目为个人 AI 学习渐进式练手工程，自由参考与取用！
