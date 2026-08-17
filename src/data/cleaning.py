"""文本清洗与质量过滤。"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Iterable


def clean_text(text: str) -> str:
    """归一化、折叠空白、清理常见 PDF/LaTeX 残留。"""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    # 常见 LaTeX 残留
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)  # \textbf{x} -> x
    text = re.sub(r"\\[a-zA-Z]+", " ", text)  # \emph 等孤立的控制序列
    text = re.sub(r"[{}]", " ", text)
    # 页眉页脚常见残留
    text = re.sub(
        r"arXiv:\d{4}\.\d{4,5}\s+v\d+\s+\[[^\]]+\]\s+[\d\s:UTC]+", " ", text
    )
    text = re.sub(r"\bSubmitted to\b.*$", " ", text, flags=re.I)
    # 空白折叠
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def quality_filter(
    text: str,
    min_chars: int = 80,
    min_sentences: int = 2,
    max_repeat_ratio: float = 0.5,
) -> bool:
    """启发式质量过滤。"""
    text = text.strip()
    if len(text) < min_chars:
        return False
    sentences = re.split(r"[.!?。！？]\s", text)
    if len([s for s in sentences if s.strip()]) < min_sentences:
        return False
    if not text:
        return False
    counts = Counter(text.lower())
    most_common = counts.most_common(1)[0][1] if counts else 0
    if most_common / max(len(text), 1) > max_repeat_ratio:
        return False
    # 基本语言判定：正文需包含一定比例字母
    alpha = sum(1 for c in text if c.isalpha())
    if alpha / max(len(text), 1) < 0.15:
        return False
    return True


def shingle_hash(text: str, k: int = 8) -> str:
    """k-shingle 哈希，用于去重。"""
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    if len(tokens) < k:
        return hashlib.md5(" ".join(tokens).encode()).hexdigest()
    shingles = [" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)]
    return hashlib.md5(" ".join(sorted(set(shingles))).encode()).hexdigest()


def dedup_records(records: Iterable[dict], key_field: str = "text") -> list[dict]:
    """按 shingle 哈希去重，保留首个出现。"""
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        h = shingle_hash(str(r.get(key_field, "")))
        if h in seen:
            continue
        seen.add(h)
        out.append(r)
    return out
