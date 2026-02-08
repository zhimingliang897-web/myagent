"""RAG 检索工具 — 包装为 LangChain Tool 供 Agent 调用"""

from langchain_core.tools import tool

from agent.rag.vectorstore import load_vectorstore


def create_rag_tool(store_path=None):
    """创建一个 RAG 检索工具。返回 None 如果向量存储不存在。"""
    try:
        vs = load_vectorstore(store_path)
    except FileNotFoundError:
        return None

    retriever = vs.as_retriever(search_kwargs={"k": 4})

    @tool
    def knowledge_search(query: str) -> str:
        """在用户的个人知识库中搜索相关信息。
        当用户询问关于其文档、笔记或知识库中的内容时使用此工具。
        返回最相关的文档片段。"""
        docs = retriever.invoke(query)
        if not docs:
            return "知识库中未找到相关内容。"

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            # 只取文件名
            source = source.split("\\")[-1].split("/")[-1]
            results.append(f"[{i}] 来源: {source}\n{doc.page_content}")

        return "\n\n---\n\n".join(results)

    return knowledge_search
