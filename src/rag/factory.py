"""从 YAML 配置构建检索器（便于脚本与 API 复用同一套装配）。"""

from __future__ import annotations

from src.config import PROJECT_ROOT, load_config
from src.rag.bm25 import BM25Index
from src.rag.embedder import Embedder
from src.rag.reranker import Reranker
from src.rag.retriever import HybridRetriever
from src.rag.self_rag import SelfRAG
from src.rag.store import InMemoryStore, MilvusStore


def build_retriever(
    config_path: str = "configs/rag/retriever.yaml",
    force_in_memory: bool = False,
) -> HybridRetriever:
    cfg = load_config(config_path)
    emb_cfg = cfg["embedder"]
    embedder = Embedder(
        name=emb_cfg["name"],
        device=emb_cfg.get("device", "cpu"),
        batch_size=emb_cfg.get("batch_size", 16),
        normalize=emb_cfg.get("normalize", True),
        dim=emb_cfg.get("dim"),
    )

    store: InMemoryStore | MilvusStore
    if force_in_memory:
        store = InMemoryStore(embedder.dim)
    else:
        try:
            milvus = cfg["milvus"]
            store = MilvusStore(
                uri=cfg.get("milvus", {}).get("uri", "work/data/milvus_demo.db"),
                collection=milvus.get("collection", "paper_chunks"),
                dim=embedder.dim,
                index_type=milvus.get("index_type", "FLAT"),
            )
        except Exception as e:  # noqa: BLE001
            print(f"[warn] Milvus 不可用，降级为内存向量库: {e}")
            store = InMemoryStore(embedder.dim)

    bm25_cfg = cfg.get("bm25", {})
    bm25 = BM25Index.load(bm25_cfg.get("cache_file", "work/data/bm25_index.pkl"))
    if bm25 is None:
        bm25 = BM25Index(k1=bm25_cfg.get("k1", 1.5), b=bm25_cfg.get("b", 0.75))

    reranker = None
    if cfg.get("reranker", {}).get("name"):
        r_cfg = cfg["reranker"]
        reranker = Reranker(
            name=r_cfg["name"], top_k=r_cfg.get("top_k", 5), device=r_cfg.get("device", "cpu")
        )

    fusion = cfg.get("fusion", {})
    return HybridRetriever(
        embedder=embedder,
        dense_store=store,
        bm25=bm25,
        reranker=reranker,
        fusion=fusion.get("method", "rrf"),
        rrf_k=fusion.get("rrf_k", 60),
        weights=fusion.get("weights", {"dense": 1.0, "sparse": 1.0}),
        dense_k=cfg.get("dense_k", 20),
        sparse_k=cfg.get("sparse_k", 20),
    )


def build_self_rag(
    config_path: str = "configs/rag/retriever.yaml",
    embedder: Embedder | None = None,
) -> SelfRAG:
    cfg = load_config(config_path)
    sr = cfg.get("self_rag", {})
    return SelfRAG(
        judge_model=sr.get("judge_model", "cross-encoder/nli-deberta-v3-xsmall"),
        relevance_threshold=sr.get("relevance_threshold", 0.30),
        faithfulness_threshold=sr.get("faithfulness_threshold", 0.50),
        citation_threshold=sr.get("citation_threshold", 0.35),
        device=sr.get("device", "cpu"),
        embedder=embedder,
    )
