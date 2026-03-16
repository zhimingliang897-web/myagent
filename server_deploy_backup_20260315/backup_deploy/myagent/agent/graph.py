"""手动构建 LangGraph StateGraph — 替代 create_agent 的自定义 ReAct 循环

图结构:
    START → trim → rewrite → agent → should_continue
                                        ├─ "tools" → tools → check_iterations
                                        │                       ├─ "continue" → agent (循环)
                                        │                       └─ "limit" → force_reply (强制结束)
                                        └─ END (直接回复)

增强功能:
  1. 查询改写 (rewrite_node) — 首轮对话时改写用户问题，提高 RAG 命中率
  2. 工具调用上限 (max_iterations) — 防止 LLM 陷入无限调工具循环
"""

from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ──────────────── 状态定义 ────────────────

class AgentState(TypedDict):
    """Agent 状态，在 MessagesState 基础上增加迭代计数。"""
    messages: Annotated[list, add_messages]
    iteration_count: int  # 工具调用轮数计数


# ──────────────── 默认配置 ────────────────

DEFAULT_MAX_ITERATIONS = 5  # 最多调 5 轮工具


def build_agent(
    llm,
    tools,
    checkpointer,
    system_prompt: str,
    max_messages: int = 10,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
):
    """构建手动 StateGraph Agent。

    Args:
        llm: ChatTongyi 实例（已带 callbacks）
        tools: 工具列表（@tool 装饰的函数）
        checkpointer: SqliteSaver 检查点器
        system_prompt: 系统提示词
        max_messages: 消息窗口大小（保留最近 N 条非系统消息）
        max_iterations: 工具调用最大轮数（防止无限循环）

    Returns:
        编译后的 CompiledGraph，用法与 create_agent 返回的对象一致
    """

    # 绑定工具到 LLM
    llm_with_tools = llm.bind_tools(tools)

    # 查询改写用的 prompt (增强版，为 Hybrid Search 准备)
    REWRITE_PROMPT = (
        "你是一个专门为搜索引擎（包含关键词匹配和向量匹配）准备查询语句的助手。\n"
        "请将用户的原始问题改写成更精确、更详细、且包含核心关键词的形式。\n"
        "规则：\n"
        "- 只输出改写后的句子，不要解释\n"
        "- 补全所有的代词（他/这/那个），结合上下文还原所指对象\n"
        "- 适当扩充同义词，或把口语化的词语转换为书面的专业术语关键词（这对命中搜索引擎极度重要）\n"
        "- 保持用户的语言（如中文）\n"
    )

    # ──────────────── 节点定义 ────────────────

    def trim_node(state: AgentState) -> dict:
        """裁剪消息历史 + 重置迭代计数。"""
        messages = state["messages"]

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(other_msgs) > max_messages:
            other_msgs = other_msgs[-max_messages:]

        return {"messages": system_msgs + other_msgs, "iteration_count": 0}

    def rewrite_node(state: AgentState) -> dict:
        """查询改写：对用户最新问题进行改写，提高检索命中率。

        只在用户消息较短或模糊时改写，明确的问题直接透传。
        改写结果替换原始用户消息。
        """
        messages = state["messages"]
        if not messages:
            return {}

        # 找到最后一条用户消息
        last_human = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_human = m
                break

        if not last_human:
            return {}

        original = last_human.content.strip()

        # 短问题（<6字）或包含代词（那个/这个/它）才改写，避免不必要的 API 调用
        needs_rewrite = len(original) < 6 or any(
            w in original for w in ["那个", "这个", "它", "他", "她", "这", "那", "上面", "之前"]
        )

        if not needs_rewrite:
            return {}

        # 用 LLM 改写（不带工具绑定，纯文本生成）
        rewrite_messages = [
            SystemMessage(content=REWRITE_PROMPT),
            HumanMessage(content=f"原始问题：{original}"),
        ]
        rewritten = llm.invoke(rewrite_messages)
        new_query = rewritten.content.strip()

        if new_query and new_query != original:
            print(f"  [查询改写] {original} → {new_query}", flush=True)
            # 替换最后一条用户消息
            new_messages = []
            replaced = False
            for m in reversed(messages):
                if isinstance(m, HumanMessage) and not replaced:
                    new_messages.append(HumanMessage(content=new_query))
                    replaced = True
                else:
                    new_messages.append(m)
            new_messages.reverse()
            return {"messages": new_messages}

        return {}

    def agent_node(state: AgentState) -> dict:
        """调用 LLM：注入系统提示词和动态长期记忆。"""
        messages = state["messages"]

        from agent.memory.profile import get_profile_summary
        dynamic_system_prompt = system_prompt + "\n\n" + get_profile_summary()

        # 如果第一条不是系统消息，注入 dynamic_system_prompt
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=dynamic_system_prompt)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # 使用 langgraph 内置的 ToolNode 来执行工具调用
    tool_node = ToolNode(tools)

    def force_reply(state: AgentState) -> dict:
        """工具调用超过上限时，强制 LLM 给出最终回答。"""
        messages = state["messages"]

        force_msg = SystemMessage(
            content=f"你已经调用了 {max_iterations} 轮工具，请立即基于已有信息给出最终回答，不要再调用工具。"
        )

        from agent.memory.profile import get_profile_summary
        dynamic_system_prompt = system_prompt + "\n\n" + get_profile_summary()

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=dynamic_system_prompt)] + messages

        # 用不绑定工具的 LLM，确保不会再调工具
        response = llm.invoke(messages + [force_msg])
        return {"messages": [response]}

    # ──────────────── 条件路由 ────────────────

    def should_continue(state: AgentState) -> str:
        """agent 回复后：有 tool_calls → 去执行工具，否则结束。"""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    def check_iterations(state: AgentState) -> str:
        """工具执行完后：检查是否超过迭代上限。"""
        count = state.get("iteration_count", 0) + 1
        if count >= max_iterations:
            print(f"  [警告] 工具调用已达上限 ({max_iterations} 轮)，强制结束", flush=True)
            return "limit"
        return "continue"

    def increment_counter(state: AgentState) -> dict:
        """工具执行后递增计数器（通过 tools 节点后的包装）。"""
        return {"iteration_count": state.get("iteration_count", 0) + 1}

    # ──────────────── 构建图 ────────────────

    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("trim", trim_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("increment", increment_counter)
    graph.add_node("force_reply", force_reply)

    # 添加边
    graph.set_entry_point("trim")                  # START → trim
    graph.add_edge("trim", "rewrite")              # trim → rewrite
    graph.add_edge("rewrite", "agent")             # rewrite → agent
    graph.add_conditional_edges(                   # agent → tools 或 END
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "increment")           # tools → increment（计数）
    graph.add_conditional_edges(                   # increment → agent 或 force_reply
        "increment",
        check_iterations,
        {"continue": "agent", "limit": "force_reply"},
    )
    graph.add_edge("force_reply", END)             # force_reply → END

    # 编译
    return graph.compile(checkpointer=checkpointer)
