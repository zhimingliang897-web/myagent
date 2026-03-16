from langchain_community.chat_models import ChatTongyi
from agent.config import DASHSCOPE_API_KEY, MODEL_NAME
from agent.callbacks import get_usage_callback


def get_llm(callbacks=None, streaming=True):
    """Create a ChatTongyi LLM instance."""
    # 默认挂载全局 Token 统计回调
    if callbacks is None:
        callbacks = [get_usage_callback()]
    elif isinstance(callbacks, list) and get_usage_callback() not in callbacks:
        callbacks.append(get_usage_callback())

    return ChatTongyi(
        model=MODEL_NAME,
        api_key=DASHSCOPE_API_KEY,
        temperature=0.7,
        streaming=streaming,
        callbacks=callbacks
    )
