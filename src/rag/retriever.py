"""混合检索：稠密 + 稀疏 + RRF 融合 + 可选重排。"""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.chunker import Chunk
from src.rag.embedder import Embedder
from src.rag.store import VectorStore


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    rank: int
    sources: list[str]


class HybridRetriever:
    def __init__(
        self,
        embedder: Embedder,
        dense_store: VectorStore,
        bm25=None,
        reranker=None,
        fusion: str = "rrf",
        rrf_k: int = 60,
        weights: dict | None = None,
        dense_k: int = 20,
        sparse_k: int = 20,
    ):
        self.embedder = embedder
        self.dense_store = dense_store
        self.bm25 = bm25
        self.reranker = reranker
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.weights = weights or {"dense": 1.0, "sparse": 1.0}
        self.dense_k = dense_k
        self.sparse_k = sparse_k

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        qv = self.embedder.encode([query])[0]
        dense = self.dense_store.search(qv, self.dense_k)
        sparse = self.bm25.search(query, self.sparse_k) if self.bm25 else []
        fused = self._rrf(dense, sparse)
        candidates = [c for c, _ in fused[: max(top_k * 3, 30)]]

        if self.reranker is not None and candidates:
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            return [
                RetrievalResult(chunk=c, score=s, rank=i + 1, sources=["dense", "sparse", "rerank"])
                for i, (c, s) in enumerate(reranked)
            ]

        return [
            RetrievalResult(chunk=c, score=s, rank=i + 1, sources=["dense", "sparse"])
            for i, (c, s) in enumerate(fused[:top_k])
        ]

    def _rrf(self, dense, sparse) -> list[tuple[Chunk, float]]:
        k = self.rrf_k
        wd = self.weights.get("dense", 1.0)
        ws = self.weights.get("sparse", 1.0)
        acc: dict[str, dict] = {}
        for rank, (c, _) in enumerate(dense, start=1):
            acc.setdefault(c.chunk_id, {"chunk": c, "score": 0.0})["score"] += wd / (k + rank)
        for rank, (c, _) in enumerate(sparse, start=1):
            if c.chunk_id in acc:
                acc[c.chunk_id]["score"] += ws / (k + rank)
            else:
                acc[c.chunk_id] = {"chunk": c, "score": ws / (k + rank)}
        ranked = sorted(acc.values(), key=lambda x: x["score"], reverse=True)
        return [(d["chunk"], d["score"]) for d in ranked]
