"""RAG 查询演示：混合检索 + 重排；可选对答案做 Self-RAG 自评估。"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.stdio import fix_console

fix_console()

from src.rag.factory import build_retriever, build_self_rag


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 查询")
    parser.add_argument("--query", required=True)
    parser.add_argument("--config", default="configs/rag/retriever.yaml")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--answer", default=None, help="可选：对给定回答做 Self-RAG 自评估")
    args = parser.parse_args()

    retriever = build_retriever(args.config)
    results = retriever.retrieve(args.query, top_k=args.top_k)
    print(f"\n查询：{args.query}\n")
    for r in results:
        c = r.chunk
        print(
            f"#{r.rank} [{c.chunk_id}] 分数={r.score:.4f} 来源={','.join(r.sources)} "
            f"页码={c.page} 章节={c.section_path or '无'}"
        )
        print(f"   {c.text[:180].replace(chr(10), ' ')}...\n")

    if args.answer:
        self_rag = build_self_rag(args.config, embedder=retriever.embedder)
        chunks = [r.chunk for r in results]
        report = self_rag.evaluate(args.query, args.answer, chunks)
        print("=== Self-RAG 自评估 ===")
        print(f"相关性达标: {report.relevant}")
        print(f"支持率: {report.grounded_ratio:.0%} ({len([v for v in report.verdicts if v.supported])}/{len(report.verdicts)})")
        print(f"引用: {report.citations}")
        print(f"结论: {report.summary}")


if __name__ == "__main__":
    sys.exit(main())
