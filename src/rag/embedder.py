"""稠密向量编码器（sentence-transformers，BGE 系列）。"""

from __future__ import annotations


class Embedder:
    """懒加载的稠密编码器，encode 返回归一化 float32 矩阵。"""

    def __init__(
        self,
        name: str,
        device: str = "cpu",
        batch_size: int = 16,
        normalize: bool = True,
        dim: int | None = None,
    ):
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._dim = dim
        self._model = None

    def _load(self):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.name, device=self.device)

    @property
    def dim(self) -> int:
        if self._dim:
            return self._dim
        if self._model is None:
            self._load()
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        import numpy as np

        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._model is None:
            self._load()
        emb = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return np.asarray(emb, dtype=np.float32)
