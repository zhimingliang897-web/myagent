"""MyAgent - 基于 LangChain + LangGraph 的智能体 CLI

用法:
    python main.py              # 默认: 手动 StateGraph 模式
    python main.py --classic    # 使用原来的 create_agent 封装
    python main.py --multi      # 多智能体模式 (Supervisor + Workers)
"""

import argparse
import uuid
from langchain_core.messages import HumanMessage

from agent.callbacks import TokenCounter, UsageCallback
from agent.llm import get_llm
from agent.tools import ALL_TOOLS
from agent.rag.retriever import create_rag_tool
from agent.memory.checkpointer import get_checkpointer_context

SYSTEM_PROMPT = """你是一个有用的 AI 助手，可以使用工具来帮助回答问题。

可用工具:
- get_current_datetime: 获取当前日期和时间
- calculate: 计算数学表达式
- web_search: 搜索网络信息
- knowledge_search: 在用户的个人知识库中搜索（如果可用）

规则:
- 当用户询问其文档或知识库中的内容时，优先使用 knowledge_search 工具。
- 需要计算时，务必使用 calculate 工具，不要心算。
- 需要时事信息或不确定的事实时，使用 web_search 工具。
- 需要日期时间时，使用 get_current_datetime 工具。
- 不需要工具时，直接回答。
- 用用户的语言回答。
- 使用 knowledge_search 时，在回答中注明信息来源。
"""

# 消息窗口限制配置
MAX_MESSAGES = 10  # 保留最近 10 条消息


def _build_classic_agent(llm, tools, memory):
    """经典模式：使用 create_agent 高层封装。"""
    from langchain.agents import create_agent
    from langchain.agents.middleware import before_model

    @before_model
    def trim_messages_mw(state, config=None):
        messages = state.get("messages", [])
        system_msgs = [m for m in messages if getattr(m, "type", None) == "system"]
        other_msgs  = [m for m in messages if getattr(m, "type", None) != "system"]
        if len(other_msgs) > MAX_MESSAGES:
            other_msgs = other_msgs[-MAX_MESSAGES:]
        return {**state, "messages": system_msgs + other_msgs}

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
        middleware=[trim_messages_mw],
    )


def _build_graph_agent(llm, tools, memory):
    """StateGraph 模式：手动构建 LangGraph 状态图。"""
    from agent.graph import build_agent
    return build_agent(llm, tools, memory, SYSTEM_PROMPT, MAX_MESSAGES)


def main():
    parser = argparse.ArgumentParser(description="MyAgent 智能体 CLI")
    parser.add_argument(
        "--classic", action="store_true",
        help="使用原来的 create_agent 封装（默认使用手动 StateGraph）",
    )
    parser.add_argument(
        "--multi", action="store_true",
        help="多智能体模式：Supervisor + Workers (Code/Data/Writer)",
    )
    parser.add_argument(
        "--rag", type=str, choices=["classic", "advanced"], default="advanced",
        help="RAG检索模式：classic (仅FAISS) 或 advanced (语义分块+混合检索)，默认 advanced",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="关闭流式输出，等待回答完全生成后再一次性打印",
    )
    args = parser.parse_args()

    # --multi 和 --classic 互斥
    if args.multi and args.classic:
        print("[错误] --multi 和 --classic 不能同时使用")
        return

    # 将是否使用流式输出作为标志，默认流式 (即不仅用 --no-stream)
    use_stream = not args.no_stream

    # 确定运行模式
    if args.multi:
        mode_name = "Multi-Agent (Supervisor + Workers)"
    elif args.classic:
        mode_name = "Classic (create_agent)"
    else:
        mode_name = "StateGraph (手动构建)"

    print("=" * 50)
    print("  MyAgent - 智能体")
    print(f"  模式: {mode_name}")
    if args.multi:
        print("  Workers: Code / Data / Writer")
    print(f"  输出: {'流式 (Streaming)' if use_stream else '整块 (Blocking)'}")
    print("  输入 'quit' 退出 | 'clear' 清空对话")
    if not args.multi:
        print("  输入 '/thread <id>' 切换对话线程")
    print("=" * 50)

    # 组装工具列表：基础工具 + RAG 工具（如果向量索引存在）
    tools = list(ALL_TOOLS)
    rag_tool = create_rag_tool(mode=args.rag)
    if rag_tool:
        tools.append(rag_tool)
        print(f"  [知识库已加载] (RAG模式: {args.rag})")
    else:
        print("  [知识库未建立，跳过 RAG 工具]")

    print("  [记忆模块已启用 (SQLite)]")

    # Token 追踪
    counter = TokenCounter()
    cb = UsageCallback(counter)
    llm = get_llm(callbacks=[cb])

    if not args.multi:
        print(f"  [消息窗口: 最近 {MAX_MESSAGES} 条]")

    # 默认线程 ID
    thread_id = "default"
    if not args.multi:
        print(f"  [当前会话 ID: {thread_id}]")

    # ================== 多智能体模式 ==================
    if args.multi:
        from agent.multi import build_multi_agent_graph

        print("\n[Multi-Agent 系统就绪]")
        print("  - Supervisor: 任务分解与调度")
        print("  - Code Agent: 代码生成与调试")
        print("  - Data Agent: 数据分析与检索")
        print("  - Writer Agent: 文章撰写与润色")
        print()

        graph = build_multi_agent_graph()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("再见!")
                break
            if user_input.lower() == "clear":
                print("[对话已清空]")
                continue

            # 执行多智能体流程（使用包含 replan 字段的初始状态）
            from agent.multi.state import create_initial_state
            initial_state = create_initial_state(user_input)

            try:
                result = graph.invoke(initial_state)
                messages = result.get("messages", [])
                if messages:
                    final_message = messages[-1]
                    content = final_message.content if hasattr(final_message, "content") else str(final_message)
                    print(f"\nAgent:\n{content}")
                else:
                    print("\n[处理完成，但没有生成回答]")
            except Exception as e:
                print(f"\n[错误]: {e}")
                import traceback
                traceback.print_exc()

            print()

        return  # 多智能体模式不进入下面的单智能体流程

    # ================== 单智能体模式 ==================
    async def process_chat():
        nonlocal thread_id

        # 在异步上下文中初始化 checkpointer 和 agent
        async with get_checkpointer_context() as memory:
            if args.classic:
                agent = _build_classic_agent(llm, tools, memory)
            else:
                agent = _build_graph_agent(llm, tools, memory)

            while True:
                try:
                    user_input = input(f"\nYou ({thread_id}): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n再见!")
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    print("再见!")
                    break

                # 切换线程命令
                if user_input.startswith("/thread "):
                    new_id = user_input.split(" ", 1)[1].strip()
                    if new_id:
                        thread_id = new_id
                        print(f"[已切换到会话: {thread_id}]")
                    continue

                if user_input.lower() == "clear":
                    thread_id = str(uuid.uuid4())[:8]
                    print(f"[对话已清空 - 新会话 ID: {thread_id}]")
                    continue

                config = {"configurable": {"thread_id": thread_id}}
                print(f"\nAgent: ", end="", flush=True)

                try:
                    if use_stream:
                        # 流式模式：使用 astream_events 实现真正的流式输出
                        async for event in agent.astream_events(
                            {"messages": [HumanMessage(content=user_input)]},
                            config=config,
                            version="v2"
                        ):
                            kind = event.get("event", "")

                            if kind == "on_chat_model_stream":
                                # 流式 token
                                chunk = event["data"].get("chunk")
                                if chunk and hasattr(chunk, "content") and chunk.content:
                                    content = chunk.content
                                    if isinstance(content, str):
                                        print(content, end="", flush=True)
                                    elif isinstance(content, list):
                                        for item in content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                print(item.get("text", ""), end="", flush=True)
                                            elif isinstance(item, str):
                                                print(item, end="", flush=True)

                            elif kind == "on_tool_start":
                                # 工具开始执行
                                tool_name = event.get("name", "unknown")
                                tool_input = event.get("data", {}).get("input", {})
                                print(f"\n  [调用工具: {tool_name}, 参数: {tool_input}]", flush=True)
                                print("Agent: ", end="", flush=True)

                        print()  # 换行
                    else:
                        # 非流式(整块)模式
                        result = await agent.ainvoke({"messages": [HumanMessage(content=user_input)]}, config=config)
                        ai_message = result["messages"][-1]
                        print(ai_message.content)

                        print(
                            f"\n[usage] calls={counter.calls} "
                            f"prompt={counter.prompt_tokens} "
                            f"completion={counter.completion_tokens} "
                            f"total={counter.total_tokens}"
                        )
                except Exception as e:
                    print(f"\n[错误]: {e}")

    # 需要用 asyncio 运行
    import asyncio
    asyncio.run(process_chat())

if __name__ == "__main__":
    main()
