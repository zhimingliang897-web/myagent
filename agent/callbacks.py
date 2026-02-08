# callbacks.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class TokenCounter:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0  # ✅ 新增：记录统计到 usage 的次数

    def add(self, usage: Dict[str, Any]) -> None:
        pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        tt = int(usage.get("total_tokens") or (pt + ct) or 0)

        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.total_tokens += tt
        self.calls += 1  # ✅ 新增


class UsageCallback(BaseCallbackHandler):
    """把每次 LLM 调用的 token usage 累加到 counter 上。"""

    def __init__(self, counter: TokenCounter):
        self.counter = counter

    def on_llm_end(self, response, **kwargs: Any) -> None:
        # response 通常是 LLMResult，不同模型把 usage 放的位置不一样
        usage: Optional[Dict[str, Any]] = None

        # 常见 1：response.llm_output["token_usage"]
        if getattr(response, "llm_output", None):
            lo = response.llm_output or {}
            usage = lo.get("token_usage") or lo.get("usage")

        # 常见 2：generation_info 里有 usage
        if not usage and getattr(response, "generations", None):
            try:
                gi = response.generations[0][0].generation_info or {}
                usage = gi.get("token_usage") or gi.get("usage")
            except Exception:
                pass

        if usage:
            self.counter.add(usage)
