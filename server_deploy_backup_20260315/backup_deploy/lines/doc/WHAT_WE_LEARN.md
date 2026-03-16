# What We Learn

这个项目过程中涉及的原理和技术点。

## PDF 文本提取

用 `pdfplumber` 读取 PDF，每个字符对象带有 `size`（字号）、`text`（内容）等属性。
通过筛选最大字号的字符，提取出需要学习的单词。

提取出来的字符是连续的（如 `renewalbarter`），程序不知道哪里断词，
所以调 LLM（deepseek-v3.2）来判断边界，拆分成独立单词。

## SPA 网站的数据获取

popmystic.com 是一个 Vue 单页应用（SPA），页面内容由 JavaScript 动态渲染。
直接用 `requests.get` 请求网页 URL 只能拿到空壳 HTML，看不到搜索结果。

```
浏览器访问:  URL → 下载 HTML → 执行 JS → JS 调 API → 渲染结果到页面  ✓
requests:   URL → 下载 HTML → 完了（不执行 JS）→ 空页面              ✗
直接调 API:  API URL → POST JSON → 直接拿到数据                      ✓ ← 我们用这个
```

**怎么找到真正的 API：**

1. 打开 Chrome，按 F12 打开开发者工具
2. 切到 Network 标签，勾选 "Fetch/XHR"
3. 在 popmystic.com 搜索一个词
4. 观察 Network 面板，会看到一个 POST 请求发往：
   `https://pop-opensearch-api-myxi6.ondigitalocean.app/search-scroll`
5. 点击请求查看 Request Payload 和 Response

这是一个 OpenSearch（Elasticsearch 的开源分支）查询接口。
用同样的 JSON 格式直接 POST，就能拿到搜索结果，不需要浏览器渲染。

## LLM 整理学习材料

原始台词数据量大且杂乱（每个词数百条引用），不适合直接阅读。
调 LLM 做筛选和加工：

- 从大量结果中挑出 2-3 段最能体现自然用法的台词
- 补充上下文，还原为对话形式
- 添加中英释义和中文翻译

Prompt 的关键是给出明确的输出格式，让 LLM 返回结构化的 Markdown，方便后续解析。

## Flask 本地 Web 应用

用 Flask 搭建本地服务器，前端是单个 HTML 文件（SPA 风格，用 JS 控制路由和渲染）。

关键设计：

- **Markdown 解析不依赖库**：`_study.md` 格式固定，用正则按 `## word` 分块提取结构化数据，返回 JSON 给前端渲染
- **后台异步处理**：上传 PDF 后用 `threading.Thread` 在后台依次跑三个脚本，前端轮询 `/api/status` 获取进度
- **SPA 路由**：用 `history.pushState` + `popstate` 事件实现前端路由，Flask 端对 `/day/<n>` 统一返回同一个 HTML

## 运行时配置注入

API Key 不能写死在代码里，也不能上传到 GitHub。两种方案并存：

- **环境变量**：`set DASHSCOPE_API_KEY=xxx` 后启动，代码里用 `os.getenv()` 读取
- **config.json**：网页端保存配置到本地文件，启动时读取并写入 `os.environ`（不覆盖已有环境变量）

优先级：环境变量 > config.json。`.gitignore` 排除 config.json 防止泄露。

## Free Dictionary API 获取音标

`https://api.dictionaryapi.dev/api/v2/entries/en/{word}` 是免费的词典 API，返回：
- `phonetic`：IPA 音标文本（如 `/rɪˈnjuːəl/`）
- `phonetics[].audio`：真人发音 mp3 URL

通过后端代理请求（避免前端 CORS 差异），内存缓存避免重复请求。

## edge-tts 离线语音生成

浏览器内置的 `speechSynthesis` API 在 Windows 上很不稳定（随缘出声）。
解决方案是用 `edge-tts`（微软 Edge 的 TTS 引擎）**预生成** mp3 文件：

- `edge-tts` 是 Python 包，调用微软 Edge 的神经网络语音，质量接近真人
- 在处理流水线中提前生成，网页端只做播放，零延迟
- 文件命名约定：`w_{词序号}.mp3`（单词）、`d_{词序号}_{对话序号}_{行序号}.mp3`（台词）
- 纯符号文本（如 `---`）需要过滤，edge-tts 对这类输入会生成空文件
