# 台词学习工具

从 PDF 提取单词 -> 查询影视台词 -> 整理成学习材料 -> 生成语音 -> 网页美化展示。

## 版本历史

> By Liang — v1 (1.30) PDF提取 · v2 (1.31) 网页界面 · v3 (2.05) 双击启动/音标/TTS
>
> 完整更新日志见 [CHANGELOG.md](CHANGELOG.md)

## 文件结构

```text
2台词\
├── app.py                   # Web 界面（Flask）
├── pdf_extractor.py         # 步骤1: PDF 提取单词
├── pop_mystic_scraper.py    # 步骤2: 查询影视台词
├── quote_formatter.py       # 步骤3: LLM 整理学习笔记
├── tts_generator.py         # 步骤4: edge-tts 生成语音
├── 启动.bat                 # 双击启动（自动开浏览器）
├── templates/
│   └── index.html           # 前端页面
├── config.json              # [需自行补充] API Key 等配置（已 gitignore）
└── word/
    ├── day1/
    │   ├── day1.pdf         # [你放的] PDF 原件
    │   ├── day1.txt         # [自动生成] 提取的单词
    │   ├── day1.json        # [自动生成] 台词原始数据
    │   ├── day1.csv         # [自动生成] 台词表格
    │   ├── day1_study.md    # [自动生成] 学习笔记
    │   └── audio/           # [自动生成] TTS 语音文件
    │       ├── w_0.mp3      # 单词发音
    │       ├── d_0_0_0.mp3  # 台词朗读
    │       └── ...
    ├── day2/
    └── ...
```

### 需补充的隐私/本地配置（未随仓库提交）

| 文件名 | 说明（对他人） | 样式/格式 |
|--------|----------------|-----------|
| **config.json** | API Key 等配置，用于台词查询与 LLM 整理 | JSON，含 `DASHSCOPE_API_KEY` 等。首次运行可在网页右上角齿轮中设置并自动保存。 |

**自己使用**：从 **`_secrets/2台词/config.json`** 拷贝到本项目 `2台词/` 根目录（文件名保持不变）即可。

## 依赖安装

```bat
pip install -r requirements.txt
```

## 运行方式

### 方式一：双击启动（推荐）

1. 双击 `启动.bat`，浏览器自动打开
2. 首次使用点右上角齿轮设置 API Key（自动保存，以后不用再设）
3. 拖拽 PDF 上传，等待处理完成即可学习

说明：

- 启动脚本会在项目目录创建并使用虚拟环境：`./.venv`
- 如依赖缺失会自动安装（读取 `requirements.txt`）
- 会等待后端服务就绪后再打开浏览器，避免偶发 "Not Found"
- 如果电脑没有 Python，但安装了 Docker Desktop，会自动用 Docker 启动
- 虚拟环境与依赖均在项目目录中，互不影响系统环境

### 方式二：命令行

```bat
set DASHSCOPE_API_KEY=你的API密钥
python app.py
```

浏览器打开 `http://localhost:8765`。

### 方式三：命令行逐步执行

```bat
set DASHSCOPE_API_KEY=你的API密钥

python pdf_extractor.py        & :: 步骤1: PDF -> 单词
python pop_mystic_scraper.py   & :: 步骤2: 单词 -> 台词
python quote_formatter.py      & :: 步骤3: 台词 -> 学习笔记
python tts_generator.py word/day1/day1_study.md  & :: 步骤4: 生成语音
```

所有脚本都有跳过机制：已生成的文件不会重复处理。

### 方式四：Docker 快速启动（无需本机 Python）

1. 安装 Docker Desktop
2. 在 `2台词/docker` 目录执行：
   ```bat
   docker compose up -d
   ```

3. 浏览器打开 `http://localhost:8765`

镜像已包含运行所需依赖（含 edge-tts），并挂载 `word/` 为数据卷。

## 功能特性

- 音标显示（Free Dictionary API）
- 单词真人发音 + 台词 TTS 朗读（edge-tts 预生成 mp3）
- 网页端设置 API Key（保存到本地 config.json，不上传 GitHub）
- 拖拽上传 PDF，自动完成全部处理流水线
- 处理进度实时显示（extracting → scraping → formatting → tts → story → done）
- 情景故事生成：根据当天生词数量自适应生成短文 + 对话（迷你/标准/加长/双场景）
