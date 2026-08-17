"""交叉编码器重排序（BGE Reranker / MiniLM）。"""

from __future__ import annotations

from src.rag.chunker import Chunk


class Reranker:
    def __init__(self, name: str, top_k: int = 5, device: str = "cpu"):
        self.name = name
        self.top_k = top_k
        self.device = device
        self._model = None

    def _load(self):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.name, device=self.device)

    def rerank(
        self, query: str, candidates: list[Chunk], top_k: int | None = None
    ) -> list[tuple[Chunk, float]]:
        """返回 (chunk, score) 按分数降序，最多 top_k 个。"""
        import numpy as np

        if not candidates:
            return []
        top_k = top_k or self.top_k
        try:
            if self._model is None:
                self._load()
            pairs = [(query, c.text[:512]) for c in candidates]
            scores = self._model.predict(pairs, show_progress_bar=False)
            scores = np.asarray(scores, dtype=float).tolist()
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            return [(c, float(s)) for c, s in ranked[:top_k]]
        except Exception as e:  # noqa: BLE001
            print(f"[warn] 重排序模型不可用，跳过重排: {e}")
            return [(c, 0.0) for c in candidates[:top_k]]
