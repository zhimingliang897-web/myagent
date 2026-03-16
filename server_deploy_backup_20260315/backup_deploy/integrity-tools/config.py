import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.environ.get('SECRET_KEY', 'default-secret-key')
INVITE_CODES = set(os.environ.get('INVITE_CODES', 'demo2026,test2026').split(','))
DATABASE_URI = 'sqlite:///data/app.db'

LLM_PROVIDERS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "model": "qwen-plus",
    },
    "qwen_max": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "model": "qwen-max",
    },
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "api_key": os.environ.get("DOUBAO_API_KEY", ""),
        "model": "doubao-1.5-pro-32k-250115",
    },
    "kimi": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key": os.environ.get("KIMI_API_KEY", ""),
        "model": "kimi-k2-turbo-preview",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "model": "deepseek-chat",
    },
}

DEBATERS = [
    {
        "id": "pro_1",
        "name": "千问·论道",
        "side": "pro",
        "role": "一辩",
        "provider": "qwen",
        "personality": "立论型辩手，风格类似黄执中。擅长在开篇就抢占定义权和价值高地，用严密的三段论构建论证框架。",
        "voice": "zh-CN-YunxiNeural",
    },
    {
        "id": "pro_2",
        "name": "豆包·善辩",
        "side": "pro",
        "role": "二辩",
        "provider": "doubao",
        "personality": "攻辩型辩手，风格类似陈铭。擅长用生动的类比和真实案例把抽象问题具象化。",
        "voice": "zh-CN-XiaoyiNeural",
    },
    {
        "id": "con_1",
        "name": "Kimi·锐评",
        "side": "con",
        "role": "一辩",
        "provider": "kimi",
        "personality": "反驳型辩手，风格类似马薇薇。语速快、攻击性强，擅长抓住对方论述中的逻辑漏洞穷追猛打。",
        "voice": "zh-CN-YunjianNeural",
    },
    {
        "id": "con_2",
        "name": "深思·明辨",
        "side": "con",
        "role": "二辩",
        "provider": "deepseek",
        "personality": "思辨型辩手，风格类似庞颖。说话沉稳有条理，擅长从哲学和社会学的底层逻辑出发。",
        "voice": "zh-CN-XiaoxiaoNeural",
    },
]

JUDGE = {
    "provider": "qwen_max",
    "voice": "zh-CN-YunyangNeural",
}

ZH_VOICES = {
    "zh-CN-YunxiNeural": "云希（年轻男声）",
    "zh-CN-YunjianNeural": "云健（成熟男声）",
    "zh-CN-YunyangNeural": "云扬（播报男声）",
    "zh-CN-YunzeNeural": "云泽（沉稳男声）",
    "zh-CN-XiaoxiaoNeural": "晓晓（活泼女声）",
    "zh-CN-XiaoyiNeural": "晓伊（温柔女声）",
    "zh-CN-XiaohanNeural": "晓涵（知性女声）",
    "zh-CN-XiaomoNeural": "晓墨（沉稳女声）",
}

MAX_HISTORY = 20
MAX_WORDS = 200
FREE_DEBATE_ROUNDS = 6
LLM_TIMEOUT = 60
DEBATE_TIME_LIMIT = 5 * 60

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")