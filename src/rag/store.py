"""向量存储抽象：Milvus（生产）/ 内存 NumPy（本机演示）。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.rag.chunker import Chunk


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[Chunk, float]]: ...

    @abstractmethod
    def count(self) -> int: ...


class MilvusStore(VectorStore):
    """Milvus / Milvus Lite 存储。字段与 Chunk 对齐，检索时还原 Chunk。"""

    _OUTPUT_FIELDS = [
        "chunk_id", "doc_id", "text", "section", "headings",
        "page", "parent_id", "metadata",
    ]

    def __init__(self, uri: str, collection: str, dim: int, index_type: str = "FLAT"):
        from pymilvus import DataType, MilvusClient

        self.client = MilvusClient(uri=str(uri))
        self.name = collection
        self.dim = dim
        self.index_type = index_type
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        from pymilvus import DataType

        if self.client.has_collection(self.name):
            self.client.load_collection(self.name)
            return
        schema = self.client.create_schema(auto_id=False)
        schema.add_field("id", DataType.VARCHAR, max_length=128, is_primary=True)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=128)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
        schema.add_field("text", DataType.VARCHAR, max_length=16384)
        schema.add_field("section", DataType.VARCHAR, max_length=512)
        schema.add_field("headings", DataType.VARCHAR, max_length=512)
        schema.add_field("page", DataType.INT64)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=128)
        schema.add_field("metadata", DataType.JSON)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dim)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type=self.index_type, metric_type="COSINE"
        )
        self.client.create_collection(
            self.name, schema=schema, index_params=index_params
        )
        self.client.load_collection(self.name)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        import numpy as np

        rows = []
        for c, emb in zip(chunks, embeddings):
            rows.append(
                {
                    "id": c.chunk_id,
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text[:16000],
                    "section": c.section_path,
                    "headings": c.headings,
                    "page": int(c.page),
                    "parent_id": c.parent_id or "",
                    "metadata": c.metadata,
                    "vector": np.asarray(emb, dtype=np.float32).tolist(),
                }
            )
        if rows:
            self.client.insert(self.name, data=rows)

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[Chunk, float]]:
        import numpy as np

        if self.count() == 0:
            return []
        res = self.client.search(
            self.name,
            data=[np.asarray(query_vec, dtype=np.float32).tolist()],
            limit=top_k,
            output_fields=self._OUTPUT_FIELDS,
            search_params={"metric_type": "COSINE", "params": {}},
        )
        hits = res[0] if res else []
        return [(self._to_chunk(h), float(h["distance"])) for h in hits]

    @staticmethod
    def _to_chunk(hit: dict) -> Chunk:
        e = hit.get("entity", {})
        return Chunk(
            chunk_id=e.get("chunk_id", ""),
            doc_id=e.get("doc_id", ""),
            text=e.get("text", ""),
            section_path=e.get("section", ""),
            headings=e.get("headings", ""),
            page=int(e.get("page", 0) or 0),
            parent_id=e.get("parent_id") or None,
            metadata=dict(e.get("metadata", {}) or {}),
        )

    def count(self) -> int:
        try:
            stats = self.client.get_collection_stats(self.name)
            return int(stats.get("row_count", 0))
        except Exception:  # noqa: BLE001
            return 0


class InMemoryStore(VectorStore):
    """内存向量库：用于无 Milvus 的本机演示与单测。"""

    def __init__(self, dim: int):
        import numpy as np

        self.dim = dim
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, dim), dtype=np.float32)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        import numpy as np

        if not chunks:
            return
        self.chunks.extend(chunks)
        self.matrix = np.vstack([self.matrix, np.asarray(embeddings, dtype=np.float32)])

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[tuple[Chunk, float]]:
        import numpy as np

        if not self.chunks:
            return []
        sims = self.matrix @ np.asarray(query_vec, dtype=np.float32)
        order = np.argsort(-sims)[:top_k]
        return [(self.chunks[int(i)], float(sims[int(i)])) for i in order]

    def count(self) -> int:
        return len(self.chunks)
