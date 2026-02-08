import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY not found. Please create a .env file with your API key.")

# ──────────────────────── 模型配置 ────────────────────────
# 所有模型名称集中管理，换模型只改这里

# 对话模型（Agent 主模型）
MODEL_NAME = "deepseek-v3.1"

# 向量嵌入模型
EMBEDDING_MODEL = "text-embedding-v3"

# 视觉理解模型（图片描述 / 视频理解）
VISION_MODEL = "qwen-vl-plus-2025-08-15"

# 语音识别模型（音频转文字）
ASR_MODEL = "fun-asr-2025-11-07"
