"""CI 冒烟测试：不依赖 GPU 与在线模型下载，验证核心链路可运行。

覆盖：版面结构 -> 层级化切块 -> BM25 稀疏检索。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.layout import Block, LayoutDocument  # noqa: E402
from src.rag.bm25 import BM25Index  # noqa: E402
from src.rag.chunker import HierarchicalChunker  # noqa: E402


def main() -> None:
    doc = LayoutDocument(
        source=Path("ci_smoke.txt"),
        title="Attention Is All You Need",
        abstract="We propose a new architecture based solely on attention mechanisms.",
        blocks=[
            Block(type="heading", text="1 Introduction", level=1),
            Block(
                type="paragraph",
                text=(
                    "The dominant sequence transduction models are based on "
                    "recurrent neural networks and convolutional networks."
                ),
            ),
            Block(
                type="paragraph",
                text=(
                    "Attention mechanisms allow modeling of dependencies "
                    "without regard to their distance in input or output sequences."
                ),
            ),
            Block(type="heading", text="2 Method", level=1),
            Block(
                type="paragraph",
                text=(
                    "The Transformer uses self-attention with multi-head attention "
                    "and position-wise feed forward layers."
                ),
            ),
        ],
    )

    chunker = HierarchicalChunker(chunk_size=200, overlap=30)
    chunks = chunker.chunk_document(doc, "arxiv_ci_001")
    assert len(chunks) > 0, "切块结果为空"

    bm25 = BM25Index().fit(chunks)
    results = bm25.search("transformer attention architecture", top_k=3)
    assert len(results) > 0, "BM25 检索结果为空"

    top_text = results[0][0].text.lower()
    assert "attention" in top_text, f"检索结果不相关: {top_text[:80]}"

    print(f"[ok] chunks={len(chunks)} top_score={results[0][1]:.3f}")
    print("CI smoke test passed")


if __name__ == "__main__":
    main()
