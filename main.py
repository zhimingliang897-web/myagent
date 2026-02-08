"""MyAgent - 基于 LangChain + LangGraph 的智能体 CLI"""

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
# from langchain.agents import create_agent

# agent = create_agent(llm, ALL_TOOLS, prompt=SYSTEM_PROMPT)
from agent.callbacks import TokenCounter, UsageCallback

counter = TokenCounter()
cb = UsageCallback(counter)
from agent.llm import get_llm
from agent.tools import ALL_TOOLS
from agent.rag.retriever import create_rag_tool

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


def main():
    print("=" * 50)
    print("  MyAgent - 智能体")
    print("  输入 'quit' 退出 | 'clear' 清空对话")
    print("=" * 50)

    # 组装工具列表：基础工具 + RAG 工具（如果向量索引存在）
    tools = list(ALL_TOOLS)
    rag_tool = create_rag_tool()
    if rag_tool:
        tools.append(rag_tool)
        print("  [知识库已加载]")
    else:
        print("  [知识库未建立，跳过 RAG 工具]")

    llm = get_llm(callbacks=[cb])
    agent = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    messages = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("再见!")
            break
        if user_input.lower() == "clear":
            messages = []
            print("[对话已清空]")
            continue

        messages.append(HumanMessage(content=user_input))

        try:
            result = agent.invoke({"messages": messages})
            ai_message = result["messages"][-1]
            print(f"\nAgent: {ai_message.content}")
            messages = result["messages"]
            
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
