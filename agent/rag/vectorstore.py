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
    vectorstore: Optional[FAISS] = None,       # ✅ 允许复用已加载的 vs，避免反复 load_local
    save_every_batch: bool = True,             # ✅ 每批落盘，防止中断白跑
    sleep_s: float = 0.0,                      # ✅ 批间等待（限流时可调大）
) -> FAISS:
    """从文档片段构建/追加 FAISS 向量存储并持久化。"""
    save_dir = Path(store_path) if store_path else DEFAULT_STORE_PATH
    embeddings = get_embeddings()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 需要追加且还没传入 vectorstore 时，尝试从磁盘加载
    if vectorstore is None and append and (save_dir / "index.faiss").exists():
        vectorstore = FAISS.load_local(
            str(save_dir), embeddings, allow_dangerous_deserialization=True
        )
        print(f"  已加载现有索引，将追加 {len(chunks)} 个片段", flush=True)

    total = len(chunks)

    try:
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            end = min(i + batch_size, total)
            print(f"  向量化 [{i + 1}-{end}] / {total} ...", flush=True)

            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)

            if save_every_batch:
                vectorstore.save_local(str(save_dir))
                # 可选：打印更少一点就把下面这行注释掉
                print(f"  已保存进度到 {save_dir}", flush=True)

            if sleep_s and end < total:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\n  收到 Ctrl+C，中断构建；正在保存当前索引...", flush=True)
        if vectorstore is not None:
            vectorstore.save_local(str(save_dir))
            print("  已保存。退出。", flush=True)
        sys.exit(130)

    # 最后再保存一次
    vectorstore.save_local(str(save_dir))
    print(f"  向量存储已保存到 {save_dir} (本次处理 {total} 个片段)", flush=True)
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
