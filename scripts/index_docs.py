"""一键索引文档脚本

用法:
    python scripts/index_docs.py <文档路径>              # 全量索引（覆盖）
    python scripts/index_docs.py <文档路径> --append      # 增量追加到已有索引
    python scripts/index_docs.py <文档路径> --batch-size 10  # 自定义 batch 大小

支持: .txt, .md, .pdf
"""

import argparse
import sys
from pathlib import Path

# 让 import agent 正常工作
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.rag.loader import load_documents
from agent.rag.vectorstore import split_documents, build_vectorstore, DEFAULT_BATCH_SIZE


def main():
    parser = argparse.ArgumentParser(description="索引文档到向量存储")
    parser.add_argument("path", help="文档文件或目录路径")
    parser.add_argument(
        "--append", action="store_true",
        help="追加到已有索引（默认覆盖重建）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"每批发送给 embedding API 的文档数 (默认 {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="分块大小，单位字符 (默认 500)",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=100,
        help="分块重叠字符数 (默认 100)",
    )
    args = parser.parse_args()

    print(f"[1/3] 加载文档: {args.path}")
    docs = load_documents(args.path)

    if not docs:
        print("未找到任何文档，退出。")
        sys.exit(1)

    print(f"\n[2/3] 分块 (chunk_size={args.chunk_size}, overlap={args.chunk_overlap})...")
    chunks = split_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    mode = "追加" if args.append else "全量"
    print(f"\n[3/3] 构建向量索引 (模式={mode}, batch_size={args.batch_size})...")
    build_vectorstore(chunks, batch_size=args.batch_size, append=args.append)

    print(f"\n完成! 共索引 {len(chunks)} 个片段。")
    print("现在可以运行 python main.py 并提问知识库相关问题。")


if __name__ == "__main__":
    main()
