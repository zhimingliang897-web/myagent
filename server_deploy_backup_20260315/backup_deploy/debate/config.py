"""
配置文件 — LLM API、辩手定义、TTS声音

使用前请填写各服务商的 API Key（或设置环境变量）
"""

import os

# ========== LLM 服务商配置 ==========
# 支持任意 OpenAI 兼容接口，修改 base_url / model 即可切换

LLM_PROVIDERS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.environ.get("QWEN_API_KEY", ""),
        "model": "qwen3-vl-flash-2026-01-22",
    },
    "qwen_max": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.environ.get("QWEN_API_KEY", ""),
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

# ========== 辩手阵容 ==========
# 每个辩手绑定: LLM服务商、性格描述、TTS声音

DEBATERS = [
    {
        "id": "pro_1",
        "name": "千问·论道",
        "side": "pro",
        "role": "一辩",
        "provider": "qwen",
        "personality": "你是立论型辩手，风格类似黄执中。擅长在开篇就抢占定义权和价值高地，用严密的三段论构建论证框架。"
                       "说话铿锵有力，喜欢用「我方的第一个论点是……」「让我们回到问题的本质」这样的结构化表达。"
                       "善于引用社会学、经济学理论和权威数据来支撑观点。",
        "voice": "zh-CN-YunxiNeural",
    },
    {
        "id": "pro_2",
        "name": "豆包·善辩",
        "side": "pro",
        "role": "二辩",
        "provider": "doubao",
        "personality": "你是攻辩型辩手，风格类似陈铭。擅长用生动的类比和真实案例把抽象问题具象化，让听众产生共鸣。"
                       "质询时善于设置连环问题，引导对方进入预设的逻辑陷阱。"
                       "语言富有感染力，善用排比和反问制造节奏感，经常说「我想请对方辩友回答一个简单的问题」。",
        "voice": "zh-CN-XiaoyiNeural",
    },
    {
        "id": "con_1",
        "name": "Kimi·锐评",
        "side": "con",
        "role": "一辩",
        "provider": "kimi",
        "personality": "你是反驳型辩手，风格类似马薇薇。语速快、攻击性强，擅长抓住对方论述中的逻辑漏洞穷追猛打。"
                       "喜欢用归谬法——「按照对方辩友的逻辑，那岂不是……」来暴露对方的荒谬之处。"
                       "自由辩论时存在感极强，善于打断对方节奏，用短促有力的反驳制造压迫感。",
        "voice": "zh-CN-YunjianNeural",
    },
    {
        "id": "con_2",
        "name": "深思·明辨",
        "side": "con",
        "role": "二辩",
        "provider": "deepseek",
        "personality": "你是思辨型辩手，风格类似庞颖。说话沉稳有条理，擅长从哲学和社会学的底层逻辑出发，"
                       "对辩题中的核心概念进行深度拆解。善于用「我们不妨换一个角度来看这个问题」引入全新视角。"
                       "总结陈词时能把全场交锋升华到价值层面，用温和而坚定的语气说服裁判。",
        "voice": "zh-CN-XiaoxiaoNeural",
    },
]

# ========== 裁判配置 ==========

JUDGE = {
    "provider": "qwen_max",
    "voice": "zh-CN-YunyangNeural",
}

# ========== TTS 声音池（中文） ==========

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

# ========== 辩论参数 ==========

MAX_HISTORY = 20          # 上下文最多保留的发言条数
MAX_WORDS = 200           # 每次发言最大字数
FREE_DEBATE_ROUNDS = 6    # 自由辩论轮数（正反各发言一次算一轮）
LLM_TIMEOUT = 60          # LLM 调用超时（秒）
SPEECH_GAP = 0.5          # 语音片段间的静音间隔（秒）
DEBATE_TIME_LIMIT = 5 * 60  # 辩论总时间上限（秒），默认5分钟，超时后跳过剩余环节直接进入裁判点评

# ========== 路径 ==========

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
