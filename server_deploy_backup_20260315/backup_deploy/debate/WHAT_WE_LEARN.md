# 📚 What We Learn - 技术实现与心得

本文档记录了 **AI 辩论赛** 项目的核心技术实现、架构设计以及开发过程中的关键经验。

---

## 💡 核心技术特性

### 1. 标准华语辩论赛赛制 (v3.0)

项目采用正规大学辩论赛的5阶段赛制：

```
阶段1: 开篇立论    正一(立论) → 反一(立论)
阶段2: 攻辩质询    正二质询反一 → 反二质询正一 → 正一质询反二 → 反一质询正二
阶段3: 攻辩小结    正一(小结) → 反一(小结)
阶段4: 自由辩论    正反交替发言 × 6轮
阶段5: 总结陈词    反二(总结) → 正二(总结)
裁判点评           AI裁判评分 + 宣判
```

- **交叉质询**：质询方和回答方交叉配对，避免同一组合重复，模拟真实赛制的战术博弈。
- **攻辩小结**：一辩在质询结束后进行阶段性总结，将零散交锋归纳为系统性结论。

### 2. 提示词工程 (Prompt Engineering)

每位辩手拥有独立的辩论风格定位，对标华语辩论圈的经典风格：

| 辩手 | 风格定位 | 参考风格 | 核心能力 |
|------|---------|---------|---------|
| 正一（千问） | 立论型 | 黄执中 | 定义权争夺、三段论框架、数据论证 |
| 正二（豆包） | 攻辩型 | 陈铭 | 类比案例、连环设问、情感共鸣 |
| 反一（Kimi） | 反驳型 | 马薇薇 | 归谬法、漏洞追击、节奏压迫 |
| 反二（深思） | 思辨型 | 庞颖 | 概念拆解、视角切换、价值升华 |

每个环节提供3-4条具体策略指导，例如攻辩质询环节要求"设计问题陷阱——无论对方如何回答，都可以为己方所用"。

### 3. 💰 按需生成 (Lazy Generation)
- **机制**: 后端采用 `threading.Event` 锁机制实现"步进式"生成。
- **流程**: LLM 生成一轮 -> 后端挂起 -> 前端播放音频 -> 播放结束 -> 前端发送 `/api/continue` -> 后端解锁下一轮。
- **价值**: 彻底杜绝了 Token 浪费，只有在用户真正还在"听"的时候，才会消耗 Token 生成下一句。

### 4. 📺 方案 B (同步播放)
- **前端逻辑**: 摒弃了传统的 SSE 流式打字效果，改为 **全文本直接显示 + 语音同步播放**。
- **用户体验**: 类似于观看带字幕的视频，解决了"文字太快、语音太慢"导致的音画不同步问题。

### 5. 🛑 关闭即停止 (Graceful Shutdown)
- **双重保险**:
  - 前端监听 `beforeunload` 事件，发送 `navigator.sendBeacon("/api/stop")`。
  - 后端检测 `finished` 标志和长久的等待，确保进程不会在后台空转。

---

## 🛠️ 技术栈

- **后端**: Flask (Python Web 框架)
- **AI 模型**: OpenAI 兼容接口 (支持 Qwen, Doubao, Kimi, DeepSeek)
- **语音合成**: Microsoft Edge TTS (通过 `edge-tts` 库)
- **实时通信**: Server-Sent Events (SSE)
- **前端**: 原生 HTML/CSS/JavaScript (无繁重框架)
- **多媒体**: FFmpeg (视频导出)

---

## ⚙️ 详细配置说明

### 环境变量

运行前需设置以下环境变量：

```bash
set QWEN_API_KEY=sk-你的通义千问密钥
set DOUBAO_API_KEY=你的豆包密钥
set KIMI_API_KEY=sk-你的Kimi密钥
set DEEPSEEK_API_KEY=sk-你的DeepSeek密钥
```

### `config.py` 关键参数

```python
# 辩手配置示例
DEBATERS = [
    {
        "id": "pro_1",
        "name": "千问·论道",
        "side": "pro",              # pro=正方, con=反方
        "role": "一辩",
        "provider": "qwen",         # LLM 服务商
        "personality": "立论型辩手，风格类似黄执中...",
        "voice": "zh-CN-YunxiNeural",  # TTS 语音
    },
]

# 辩论控制参数
MAX_HISTORY = 20          # 上下文记忆深度
MAX_WORDS = 200           # 单次发言字数限制
FREE_DEBATE_ROUNDS = 6    # 自由辩论轮数
LLM_TIMEOUT = 60          # API 调用超时（秒）
DEBATE_TIME_LIMIT = 300   # 辩论总时间上限（秒）
```

---

## 🗂️ 项目结构解析

```
AI辩论赛/
├── app.py                 # Flask 主入口：路由、SSE 推送、API 接口
├── debate_engine.py       # 核心引擎：标准赛制流程、LLM 调度、Lazy Gen
├── tts_engine.py          # 语音引擎：Edge TTS 生成、音频拼接
├── llm_client.py          # LLM 客户端：封装 OpenAI 调用（含超时配置）
├── video_export.py        # 视频生成：PIL 卡片 + FFmpeg 合成
├── config.py              # 集中配置：环境变量读取、辩手定义、参数
├── topics.py              # 辩题库：10个高话题度预设辩题
├── templates/
│   └── index.html         # 前端：SSE 监听、音频播放、UI 交互
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 🐛 踩坑与经验

### 1. Windows 上的 Asyncio
在 Flask 线程中直接调用 `asyncio.run()` 会导致 `RuntimeError: Event loop is closed`。
**解决方案**: 使用 `asyncio.new_event_loop()` 和 `asyncio.set_event_loop()` 手动管理事件循环。

### 2. 音画同步难点
流式文字（Typing Effect）和 TTS 语音的速度完全不同，单纯的并行播放会导致严重的割裂感。
**解决方案**: 放弃纯流式打字效果，采用"字幕式"同步，即 `Text Ready -> Text Show -> Audio Play -> Next Signal`。

### 3. 浏览器资源管控
Web 页面关闭时，后端如果不感知，会继续跑完昂贵的 LLM 流程。
**解决方案**: 引入 Beacon API 发送离线信号，配合后端的 Wait 锁机制，实现秒级止损。

### 4. Python 字符串中的中文引号
`config.py` 中 personality 描述包含中文引号 `"..."` 时，会和 Python 的字符串定界符冲突导致 SyntaxError。
**解决方案**: 使用单书名号 `「...」` 替代中文双引号。

---

## 🔧 开发指南

### 添加新的 LLM 服务商
1. 在 `config.py` 的 `LLM_PROVIDERS` 中添加配置。
2. 设置对应的环境变量。
3. 确保 API URL 和 Key 符合 OpenAI 规范。

### 自定义辩题
编辑 `topics.py`，添加包含 `topic`、`pro_position`、`con_position` 的字典即可。

### 调整辩论节奏
- `FREE_DEBATE_ROUNDS`: 自由辩论轮数（默认6轮，正反各发言一次算一轮）
- `MAX_WORDS`: 每次发言字数上限（影响辩论节奏快慢）
- `DEBATE_TIME_LIMIT`: 辩论总时间上限（超时后跳至裁判环节）
