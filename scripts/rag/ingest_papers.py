"""构建 RAG 索引：版面解析 -> 层级切块 -> 稠密（Milvus/内存）+ 稀疏（BM25）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.stdio import fix_console

fix_console()

from src.config import PROJECT_ROOT, load_config
from src.data.layout import parse_pdf
from src.rag.bm25 import BM25Index
from src.rag.chunker import HierarchicalChunker
from src.rag.embedder import Embedder
from src.rag.store import InMemoryStore, MilvusStore


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 RAG 索引")
    parser.add_argument("--pdf-dir", default="work/data/arxiv_pdfs")
    parser.add_argument("--config", default="configs/rag/retriever.yaml")
    parser.add_argument("--force-in-memory", action="store_true", help="跳过 Milvus 使用内存向量库")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    chunk_cfg = cfg["chunk"]
    emb_cfg = cfg["embedder"]
    bm25_cfg = cfg.get("bm25", {})
    milvus_cfg = cfg.get("milvus", {})

    embedder = Embedder(
        name=emb_cfg["name"],
        device=emb_cfg.get("device", "cpu"),
        batch_size=emb_cfg.get("batch_size", 16),
        normalize=emb_cfg.get("normalize", True),
        dim=emb_cfg.get("dim"),
    )

    if args.force_in_memory:
        store = InMemoryStore(embedder.dim)
    else:
        try:
            store = MilvusStore(
                uri=milvus_cfg.get("uri", "work/data/milvus_demo.db"),
                collection=milvus_cfg.get("collection", "paper_chunks"),
                dim=embedder.dim,
                index_type=milvus_cfg.get("index_type", "FLAT"),
            )
        except Exception as e:  # noqa: BLE001
            print(f"[warn] Milvus 不可用，降级为内存向量库: {e}")
            store = InMemoryStore(embedder.dim)

    chunker = HierarchicalChunker(
        chunk_size=chunk_cfg.get("chunk_size", 600),
        overlap=chunk_cfg.get("overlap", 80),
    )

    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.is_absolute():
        pdf_dir = PROJECT_ROOT / pdf_dir
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    all_chunks = []
    for i, pdf in enumerate(pdfs):
        try:
            doc = parse_pdf(pdf)
            chunks = chunker.chunk_document(doc, doc_id=pdf.stem)
            all_chunks.extend(chunks)
            print(f"[{i + 1}/{len(pdfs)}] {pdf.stem}: {len(chunks)} 块")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {pdf.name}: {e}")

    print(f"编码 {len(all_chunks)} 个块 ...")
    embeddings = embedder.encode([c.text for c in all_chunks])
    store.add(all_chunks, embeddings)
    print(f"稠密索引完成，共 {store.count()} 条")

    bm25 = BM25Index(k1=bm25_cfg.get("k1", 1.5), b=bm25_cfg.get("b", 0.75)).fit(all_chunks)
    cache = bm25.save(bm25_cfg.get("cache_file", "work/data/bm25_index.pkl"))
    print(f"BM25 索引完成，缓存 -> {cache}")


if __name__ == "__main__":
    sys.exit(main())
