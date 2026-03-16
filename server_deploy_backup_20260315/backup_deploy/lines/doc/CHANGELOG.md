# 更新日志

## v8 — 启动脚本与 Docker 优化

- 启动脚本增强：本地创建/使用 `./.venv` 虚拟环境，依赖缺失自动安装（flask、requests、pdfplumber、edge-tts）
- 服务就绪检测：等待后端返回 200 后再打开浏览器，避免偶发 “Not Found”
- 无 Python 回退：未检测到 Python 且存在 Docker Desktop 时，自动使用 `docker compose up -d` 启动
- 改用 `python -m pip` 安装，兼容不同系统配置；修正 `activate.bat` 调用
- Docker 镜像加入 `edge-tts`，复制 `tts_generator.py`，容器内可完整到 TTS 阶段
- 启动脚本与 Docker 路径：`2台词/启动.bat`、`2台词/docker/Dockerfile`、`2台词/docker/docker-compose.yml`

## v7 — TTS 语音生成

- 新增 `tts_generator.py`：用 edge-tts 为单词和台词批量生成 mp3 音频
- 处理流水线新增第 4 步：study.md → audio/（自动在上传 PDF 时执行）
- 前端台词行旁加播放按钮，点击播放本地预生成的 mp3
- 单词发音优先本地音频，无本地音频时 fallback 到 Free Dictionary API 音频
- 过滤无效文本（如 `---` 分隔线），自动清理空音频文件
- 支持命令行独立运行：`python tts_generator.py word/day1/day1_study.md`

## v6 — 音标 + 发音 + 双击启动 + 网页设置

- 新增 `启动.bat`：双击启动，自动打开浏览器
- 新增网页端 API Key 设置：右上角齿轮按钮，设置保存到 `config.json`
- config.json 已加入 `.gitignore`，API Key 不会上传 GitHub
- 两种 API Key 方式并存：命令行 `set`（优先）或网页设置
- 新增 `/api/phonetics/<word>`：代理 Free Dictionary API，返回 IPA 音标 + 音频 URL
- 单词标题旁显示音标 + 发音按钮
- 未设置 API Key 时首页显示橙色提示条
- 处理进度新增 TTS 步骤标签

## v5 — 删除 + 上传修复

- Day 卡片右上角新增删除按钮，hover 显示，点击二次确认后删除整个 day 文件夹
- 修复拖拽上传 `No file` 报错（`closeModal` 提前清空了文件引用）
- 上传弹窗自动建议下一个 Day 编号，支持手动修改
- 已存在的 Day 拒绝覆盖，防止误操作

## v4 — 本地 Web 学习界面

- 新增 `app.py`：Flask 本地 Web 应用，替代直接看 Markdown
- 单词卡片式美化展示：大标题、释义高亮、对话分块、影视来源标签
- 顶部快速导航栏，点击单词跳转
- 拖拽上传 PDF：弹窗指定 Day 编号，自动创建文件夹并后台执行完整处理流水线
- 处理进度实时轮询（extracting → scraping → formatting → done）

## v3 — LLM 生成学习笔记

- 新增 `quote_formatter.py`：读取 JSON 台词数据，调 LLM 生成学习笔记
- 每个单词生成中英释义 + 2-3 段精选影视对话
- 对话附中文翻译，标注影视来源和年份
- 输出 `dayN_study.md`，按 `## word` 分节、`---` 分隔

## v2 — 影视台词搜索

- 新增 `pop_mystic_scraper.py`：通过 popmystic.com OpenSearch API 批量查询台词
- 每个单词返回数百条影视引用（标题、年份、剧集、台词原文）
- 输出 `dayN.json` + `dayN.csv`，已有文件自动跳过

## v1 — PDF 单词提取

- 新增 `pdf_extractor.py`：从 PDF 提取最大字号字符
- LLM 自动拆分连续字符为独立单词（deepseek-v3.2）
- 支持 auto / interactive 两种模式
- 按 `word/day*/` 目录结构批量扫描处理
