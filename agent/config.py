import os
from dotenv import load_dotenv

load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY not found. Please create a .env file with your API key.")

# ──────────────────────── 模型配置 ────────────────────────
# 所有模型名称集中管理，换模型只改这里

# 对话模型（Agent 主模型）
MODEL_NAME = "qwen-plus"

# 向量嵌入模型
EMBEDDING_MODEL = "text-embedding-v3"

# 视觉理解模型（图片描述 / 视频理解）
VISION_MODEL = "qwen-vl-plus-2025-08-15"

# 语音识别模型（音频转文字）
ASR_MODEL = "fun-asr-2025-11-07"

# ──────────────────────── 可选付费能力（默认关闭，避免误扣费）────────────────────────
# 文生图（通义万相）：在 .env 中设置 ENABLE_TEXT_TO_IMAGE=1 并确保已开通万相后才会启用
# 图生文（视觉模型）：在 .env 中设置 ENABLE_IMAGE_TO_TEXT=1 后才会启用
ENABLE_TEXT_TO_IMAGE = os.getenv("ENABLE_TEXT_TO_IMAGE", "0").strip().lower() in ("1", "true", "yes")
ENABLE_IMAGE_TO_TEXT = os.getenv("ENABLE_IMAGE_TO_TEXT", "0").strip().lower() in ("1", "true", "yes")
