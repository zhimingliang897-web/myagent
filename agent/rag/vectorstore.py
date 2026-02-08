"""向量存储 — 文档分块 + FAISS 向量化 + 持久化"""

import time
from pathlib import Path
from typing import List, Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.config import DASHSCOPE_API_KEY, EMBEDDING_MODEL

# 默认持久化路径
DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "vectorstore"

# DashScope text-embedding-v3 限制: 每次最多 25 个文本
# 参考: https://help.aliyun.com/zh/model-studio/text-embedding-api
DEFAULT_BATCH_SIZE = 25


def get_embeddings() -> DashScopeEmbeddings:
    return DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dashscope_api_key=DASHSCOPE_API_KEY,
    )


def split_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """将文档分块。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"  分块完成: {len(docs)} 个文档 → {len(chunks)} 个片段")
    return chunks


def build_vectorstore(
    chunks: List[Document],
    store_path: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    append: bool = False,
) -> FAISS:
    """从文档片段构建 FAISS 向量存储并持久化。

    Args:
        chunks: 文档片段列表
        store_path: 持久化路径
        batch_size: 每批发送给 embedding API 的文档数量
                    DashScope text-embedding-v3 限制每次最多 25 个
        append: 是否追加到已有的向量存储（增量索引）
    """
    save_dir = Path(store_path) if store_path else DEFAULT_STORE_PATH
    embeddings = get_embeddings()

    # 如果是增量模式，先加载已有索引
    vectorstore = None
    if append and (save_dir / "index.faiss").exists():
        vectorstore = FAISS.load_local(
            str(save_dir), embeddings, allow_dangerous_deserialization=True
        )
        print(f"  已加载现有索引，将追加 {len(chunks)} 个片段")

    # 分 batch 向量化
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        end = min(i + batch_size, total)
        print(f"  向量化 [{i + 1}-{end}] / {total} ...")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

        # 批次间等待，避免触发 API 限流
        if end < total:
            time.sleep(0.5)

    save_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(save_dir))
    print(f"  向量存储已保存到 {save_dir} (共 {total} 个片段)")

    return vectorstore


def load_vectorstore(store_path: Optional[str] = None) -> FAISS:
    """从磁盘加载已有的 FAISS 向量存储。"""
    load_dir = Path(store_path) if store_path else DEFAULT_STORE_PATH

    if not (load_dir / "index.faiss").exists():
        raise FileNotFoundError(
            f"向量存储不存在: {load_dir}\n"
            "请先运行 python scripts/index_docs.py <文档路径> 建立索引"
        )

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(load_dir), embeddings, allow_dangerous_deserialization=True
    )
