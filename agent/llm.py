from langchain_community.chat_models import ChatTongyi
from agent.config import DASHSCOPE_API_KEY, MODEL_NAME


def get_llm(callbacks=None):
    """Create a ChatTongyi LLM instance."""
    return ChatTongyi(
        model=MODEL_NAME,
        api_key=DASHSCOPE_API_KEY,
        temperature=0.7,
        callbacks=callbacks
    )  
