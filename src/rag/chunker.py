"""层级化切块：论文 -> 章节 -> 段落块（重叠），保留父子引用与页码。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.data.layout import LayoutDocument, Section


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    section_path: str = ""
    headings: str = ""
    page: int = 0
    parent_id: str | None = None
    metadata: dict = field(default_factory=dict)


class HierarchicalChunker:
    """按章节层级切块，子块携带父章节 id 以便回跳整节上下文。"""

    def __init__(self, chunk_size: int = 600, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._counter = 0

    def _next_id(self, doc_id: str) -> str:
        self._counter += 1
        return f"{doc_id}__{self._counter:05d}"

    def chunk_document(self, doc: LayoutDocument, doc_id: str) -> list[Chunk]:
        self._counter = 0
        chunks: list[Chunk] = []
        chunks.extend(self.chunk_sections(doc, doc_id, _sections(doc)))
        return chunks

    def chunk_sections(
        self,
        doc: LayoutDocument,
        doc_id: str,
        sections: list[Section],
        path: tuple[str, ...] = (),
        parent_id: str | None = None,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for sec in sections:
            sec_path = " > ".join((*path, sec.title)) if sec.title else " > ".join(path)
            text = sec.text.strip()
            if not text:
                chunks.extend(
                    self.chunk_sections(doc, doc_id, sec.subsections, (*path, sec.title), parent_id)
                )
                continue
            # 父块：整节文本，用于回跳上下文
            sec_id = self._next_id(doc_id)
            chunks.append(
                Chunk(
                    chunk_id=sec_id,
                    doc_id=doc_id,
                    text=text[: self.chunk_size * 4],
                    section_path=sec_path,
                    headings=sec.title,
                    page=sec.page,
                    parent_id=parent_id,
                    metadata={"level": sec.level, "is_section": True},
                )
            )
            # 子块：段落级重叠切分，parent 指向本节点
            for i, piece in enumerate(self._split_text(text)):
                chunks.append(
                    Chunk(
                        chunk_id=self._next_id(doc_id),
                        doc_id=doc_id,
                        text=piece,
                        section_path=sec_path,
                        headings=sec.title,
                        page=sec.page,
                        parent_id=sec_id,
                        metadata={"level": sec.level, "is_section": False, "piece": i},
                    )
                )
            chunks.extend(
                self.chunk_sections(doc, doc_id, sec.subsections, (*path, sec.title), sec_id)
            )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """按段落合并到目标大小，超长段落硬切并加重叠。"""
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: list[str] = []
        buf = ""
        for p in paragraphs:
            if len(p) > self.chunk_size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                start = 0
                while start < len(p):
                    end = min(start + self.chunk_size, len(p))
                    chunks.append(p[start:end].strip())
                    if end >= len(p):
                        break
                    start = max(start, end - self.overlap)
            elif len(buf) + len(p) + 2 <= self.chunk_size:
                buf = f"{buf}\n\n{p}" if buf else p
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return chunks


def _sections(doc: LayoutDocument) -> list[Section]:
    from src.data.layout import to_sections

    return to_sections(doc)
