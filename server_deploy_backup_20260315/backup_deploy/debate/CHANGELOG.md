# 更新日志

## v4.0.0 — 网页配置中心 + 一键启动

### 新功能
- **网页内配置面板**（`⚙️ 高级配置`，无需修改任何代码）：
  - **API 密钥管理**：为每个 LLM 服务商（通义千问、豆包、Kimi、DeepSeek）单独配置密钥，自动保存到浏览器 localStorage，不会上传
  - **模型选择**：每位辩手可独立切换 LLM 服务商，并可填写自定义模型名
  - **音色选择**：每位辩手从 8 种中文 TTS 音色中自由选择
  - **人设编辑**：每位辩手的性格描述完全可在网页内修改
  - **辩论参数**：单次发言字数、自由辩论轮数、时间上限均可调节
- **一键启动脚本** `启动.bat`：自动安装依赖、启动服务、3 秒后打开浏览器，零命令行操作
- **音量滑块**：辩论界面新增音量控制
- **错误提示 Toast**：统一的错误提示替代 alert()

### API 新增
- `GET /api/config`：返回完整配置信息（API Key 仅返回是否已配置，不含明文）
- `POST /api/start` 现接受 `config` 字段，支持覆盖 providers（api_key/model）、debaters（provider/model_override/voice/personality）、params（max_words/free_debate_rounds/debate_time_limit）

### Bug 修复
- 🐛 **修复超时空气泡 bug**：辩论超时时，`_emit_turn()` 现在会显示"（此环节已超时跳过）"而非卡在加载中
- 🐛 **修复 export_status 不重置**：开始新辩论时正确重置导出状态，避免导出按钮状态残留
- 🐛 **修复 debate_clients 并发访问**：推送事件时改用 `list(debate_clients)` 快照，避免迭代时修改
- 🐛 **修复裁判发言的 skipped TTS**：跳过的发言不再尝试生成语音，避免 edge-tts 空文本报错

### 架构优化
- `DebateEngine` 现在通过构造函数接受 `debaters`、`providers`、`params` 参数，实现运行时配置注入（不修改全局 config.py）
- `_is_timeout()` 现在使用实例变量 `self._debate_time_limit`（支持运行时覆盖）
- `init_clients()` 支持辩手级别的 `model_override` 字段
- SSE heartbeat timeout 从 300s 缩短至 30s，更快检测断连

---

## v3.0.0 — 标准赛制 + 提示词重构

### 赛制重构
- **标准华语辩论赛流程**：从4阶段升级为5阶段，对齐正规大学辩论赛赛制
  - 开篇立论（仅一辩）→ 攻辩质询（交叉配对）→ **攻辩小结（新增）** → 自由辩论 → 总结陈词（仅二辩）
- **交叉质询配对**：正二质询反一 → 反二质询正一 → 正一质询反二 → 反一质询正二
- **攻辩小结环节**：双方一辩总结质询阶段的交锋成果，承上启下

### 提示词优化
- **辩手性格差异化**：每位辩手对标真实辩论风格（黄执中/陈铭/马薇薇/庞颖），立论、攻辩、反驳、思辨各有侧重
- **环节专属策略**：每个阶段提供3-4条具体辩论策略指导，而非笼统的一句话
- **辩论赛用语规范**：要求称"对方辩友"，论证分层（观点→论据→结论），口语化表达

### 辩题库更新
- 全部替换为10个高话题度辩题：AI内容标注、大城市卷vs小城市躺、孔乙己的长衫、精神离职等

### 安全与修复
- 🔒 **API Key 改用环境变量**：`os.environ.get()` 读取，不再硬编码在代码中
- 🐛 修复裁判评分函数中 transcript 重复拼接的 bug
- 🐛 修复 `_is_timeout()` 未生效（`_start_time` 未赋值）的问题
- 🐛 修复前端 `resetDebate()` 中引用未定义变量的问题
- 🐛 修复视频导出时正反方立场丢失的问题
- ⏱️ `LLMClient` 现在使用 `LLM_TIMEOUT` 配置（60秒超时）

### 代码优化
- 提取 `_emit_turn()` 辅助方法，消除 `run_debate()` 中十余处重复的事件发射代码

---

## v2.0.0 — 优化与节能

- 💰 **按需生成 (Lazy Generation)**：后端生成完一句后自动挂起，等待前端信号，彻底杜绝 Token 浪费
- 📺 **同步播放 (Plan B)**：取消流式打字效果，改为"显示完整文本 → 播放语音"的同步模式，体验更佳
- 🛑 **关闭即停止**：网页关闭时发送信号强制停止后端进程，双重保险防跑票
- 📝 **字数限制**：单次发言限制调整为 200 字，辩论更精简

---

## v1.0.1 — Bug 修复

- 修复 Windows 上 `RuntimeError: Event loop is closed` 错误（`tts_engine.py` 改用 `get_event_loop()` + `run_until_complete()`）
- 配置所有 LLM API 密钥（Qwen、Doubao、Kimi、DeepSeek）
- 更新模型名称：Qwen → `qwen3-vl-flash-2026-01-22`，Kimi → `kimi-k2-turbo-preview`
- 新增 `README.md` 和 `CHANGELOG.md` 文档

---

## v1.0.0 — 初始版本

- 完整四阶段辩论流程（开篇陈词 → 攻辩质询 → 自由辩论 → 总结陈词）
- AI 裁判评分和点评系统
- 支持 4 位辩手，每位可配置不同 LLM 模型和性格
- 多 LLM 服务商支持（Qwen、Doubao、Kimi、DeepSeek）
- Edge TTS 实时语音合成，支持多种中文语音
- 现代化 Web 界面，深色主题，SSE 实时推送
- 预设辩题库，快速开始
- 辩论视频导出功能（FFmpeg）
