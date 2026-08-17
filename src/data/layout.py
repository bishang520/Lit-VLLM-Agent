"""版面解析：PDF -> 层级化文档结构（标题/章节/段落，带页码）。"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Block:
    """文档块：heading / paragraph / table / caption。"""

    type: str
    text: str
    level: int = 0
    page: int = 0
    bbox: tuple | None = None


@dataclass
class Section:
    """章节树节点。"""

    title: str
    level: int
    page: int
    text: str = ""
    subsections: list["Section"] = field(default_factory=list)

    def walk(self):
        yield self
        for sub in self.subsections:
            yield from sub.walk()


@dataclass
class LayoutDocument:
    source: Path
    title: str = ""
    abstract: str = ""
    blocks: list[Block] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def parse_pdf(path: str | Path, backend: str = "pymupdf") -> LayoutDocument:
    """解析 PDF。默认 PyMuPDF；backend 预留 marker/unstructured 扩展。"""
    p = Path(path)
    if backend == "pymupdf":
        return _parse_pymupdf(p)
    raise ValueError(f"未知版面解析后端: {backend}")


def _parse_pymupdf(path: Path) -> LayoutDocument:
    try:
        import pymupdf  # type: ignore
    except ImportError:  # pragma: no cover
        import fitz as pymupdf  # type: ignore

    doc = pymupdf.open(str(path))
    page_spans: list[tuple[int, list[tuple[float, str, tuple]]]] = []
    body_sizes: list[float] = []
    for pno, page in enumerate(doc):
        spans: list[tuple[float, str, tuple]] = []
        try:
            d = page.get_text("dict")
            for b in d.get("blocks", []):
                if b.get("type") != 0:
                    continue
                for line in b.get("lines", []):
                    for s in line.get("spans", []):
                        text = (s.get("text") or "").strip()
                        if text:
                            spans.append(
                                (float(s.get("size", 10)), text, tuple(s.get("bbox", ())))
                            )
        except Exception:  # noqa: BLE001
            continue
        page_spans.append((pno, spans))
        body_sizes.extend(sz for sz, _, _ in spans if sz > 0)

    body = statistics.median(body_sizes) if body_sizes else 12.0
    blocks: list[Block] = []
    title = ""
    for pno, spans in page_spans:
        for sz, text, bbox in spans:
            if not title and sz > body * 1.35:
                title = text
                blocks.append(Block("heading", text, level=0, page=pno, bbox=bbox))
            elif sz >= body * 1.12:
                level = 1 if sz >= body * 1.22 else 2
                blocks.append(Block("heading", text, level=level, page=pno, bbox=bbox))
            else:
                blocks.append(Block("paragraph", text, page=pno, bbox=bbox))

    blocks = _merge_paragraphs(blocks)
    doc_ = LayoutDocument(source=path, title=title, blocks=blocks)
    doc_.abstract = _extract_abstract(blocks)
    return doc_


def _merge_paragraphs(blocks: list[Block]) -> list[Block]:
    """合并同页相邻的正文行，避免被 PDF 行切碎。"""
    merged: list[Block] = []
    for b in blocks:
        if (
            b.type == "paragraph"
            and merged
            and merged[-1].type == "paragraph"
            and merged[-1].page == b.page
        ):
            merged[-1].text += " " + b.text
        else:
            merged.append(b)
    return merged


def _extract_abstract(blocks: list[Block]) -> str:
    for i, b in enumerate(blocks):
        if b.type == "heading" and re.search(r"\babstract\b", b.text, re.I):
            parts = []
            for nb in blocks[i + 1 :]:
                if nb.type == "heading":
                    break
                parts.append(nb.text)
            return "\n\n".join(parts).strip()
    return ""


def to_sections(doc: LayoutDocument) -> list[Section]:
    """把块序列组织为章节树（按 heading level 入栈）。"""
    root = Section("", 0, 0)
    stack = [root]
    cur = root
    for b in doc.blocks:
        if b.type == "heading" and b.level >= 1:
            sec = Section(b.text, b.level, b.page)
            while stack and stack[-1].level >= b.level:
                stack.pop()
            stack[-1].subsections.append(sec)
            stack.append(sec)
            cur = sec
        elif b.type == "paragraph":
            cur.text = (cur.text + "\n\n" + b.text) if cur.text else b.text
    return root.subsections
