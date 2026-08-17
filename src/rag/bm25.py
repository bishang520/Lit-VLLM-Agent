"""BM25 稀疏检索（rank_bm25），索引可持久化。"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from src.config import PROJECT_ROOT
from src.rag.chunker import Chunk

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "with", "from", "as", "by", "at", "that",
    "this", "these", "those", "we", "our", "it", "its", "they", "their",
}


class BM25Index:
    """对 chunk 集合构建 BM25Okapi 并检索。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._bm25 = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9+#.-]{2,}", text.lower())
        return [t for t in tokens if t not in STOPWORDS]

    def fit(self, chunks: list[Chunk]) -> "BM25Index":
        """用 chunk 集合构建索引。"""
        from rank_bm25 import BM25Okapi

        self.chunks = list(chunks)
        corpus = [self.tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)
        return self

    def search(self, query: str, top_k: int = 20) -> list[tuple[Chunk, float]]:
        if self._bm25 is None or not self.chunks:
            return []
        scores = self._bm25.get_scores(self.tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump({"chunks": self.chunks, "k1": self.k1, "b": self.b}, f)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index | None":
        p = Path(path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            return None
        with p.open("rb") as f:
            data = pickle.load(f)
        return cls(k1=data["k1"], b=data["b"]).fit(data["chunks"])
