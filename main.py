"""MyAgent - 基于 LangChain + LangGraph 的智能体 CLI"""

import uuid
from langchain_core.messages import HumanMessage, trim_messages
# from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent

# agent = create_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)
from agent.callbacks import TokenCounter, UsageCallback
from langchain.agents.middleware import before_model

counter = TokenCounter()
cb = UsageCallback(counter)
from agent.llm import get_llm
from agent.tools import ALL_TOOLS
from agent.rag.retriever import create_rag_tool
from agent.memory.checkpointer import get_checkpointer

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
@before_model
def trim_messages_mw(state, config=None):
    messages = state.get("messages", [])

    system_msgs = [m for m in messages if getattr(m, "type", None) == "system"]
    other_msgs  = [m for m in messages if getattr(m, "type", None) != "system"]

    if len(other_msgs) > MAX_MESSAGES:
        other_msgs = other_msgs[-MAX_MESSAGES:]

    return {**state, "messages": system_msgs + other_msgs}


def main():
    print("=" * 50)
    print("  MyAgent - 智能体")
    print("  输入 'quit' 退出 | 'clear' 清空对话")
    print("  输入 '/thread <id>' 切换对话线程")
    print("=" * 50)

    # 组装工具列表：基础工具 + RAG 工具（如果向量索引存在）
    tools = list(ALL_TOOLS)
    rag_tool = create_rag_tool()
    if rag_tool:
        tools.append(rag_tool)
        print("  [知识库已加载]")
    else:
        print("  [知识库未建立，跳过 RAG 工具]")

    # 初始化记忆
    memory = get_checkpointer()
    print("  [记忆模块已启用 (SQLite)]")

    llm = get_llm(callbacks=[cb])
    # 传入 checkpointer 和 state_modifier 来限制消息窗口
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=memory,
        middleware=[trim_messages_mw],
    )


    # 限制消息历史，节省 token)
    # agent = create_react_agent(
    #     llm, 
    #     tools, 
    #     prompt=SYSTEM_PROMPT, 
    #     checkpointer=memory,
    #     state_modifier=trim_message_history  # 限制消息历史，节省 token
    # )
    print(f"  [消息窗口: 最近 {MAX_MESSAGES} 条]")
    
    # 默认线程 ID
    thread_id = "default"
    print(f"  [当前会话 ID: {thread_id}]")

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
            # 清空当前线程的记忆？
            # LangGraph checkpointer doesn't have a direct "clear" but we can generate a new thread_id
            # or just rely on the user to switch threads.
            # For now, let's keep the old behavior of clearing local logic if needed, 
            # but with persistent memory, the agent state is in the DB.
            # So "clear" effectively means "start new thread" or "ignore previous".
            # Let's just generate a new random thread ID for "clearing" effectively.
            thread_id = str(uuid.uuid4())[:8]
            print(f"[对话已清空 - 新会话 ID: {thread_id}]")
            continue

        # 使用 persistent memory，不需要手动维护 messages 列表传给 invoke
        # 只需要传入新消息和 thread_id
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # 传入 messages=[HumanMessage(...)]，LangGraph 会自动追加到历史
            result = agent.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)
            
            # result["messages"] 包含所有历史，取最后一条 AI 回复
            ai_message = result["messages"][-1]
            print(f"\nAgent: {ai_message.content}")
            
            # ✅ 打印累计 token（每次对话后）
            print(
                f"[usage] calls={counter.calls} "
                f"prompt={counter.prompt_tokens} "
                f"completion={counter.completion_tokens} "
                f"total={counter.total_tokens}"
            )
        except Exception as e:
            print(f"\n[错误]: {e}")


if __name__ == "__main__":
    main()
