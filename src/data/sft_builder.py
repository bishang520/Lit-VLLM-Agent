"""SFT 多轮指令数据集构建。

两种模式：
- GPT-4 辅助：调用 OpenAI API 根据论文片段生成高质量 QA（含 CoT 与引用）；
- mock：无 API Key 时基于章节标题与正文模板离线造数（用于开发联调）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT
from src.data.layout import LayoutDocument, Section, to_sections

SYSTEM_PROMPT = (
    "你是一个严谨的学术文献助手。回答必须严格基于提供的论文原文，"
    "给出逐步推理过程，并在每个关键论断后标注引用的片段编号 [chunk_id]。"
    "如果原文没有足够信息，请明确回答'原文未提供相关信息'，禁止编造。"
)

QA_GENERATION_SYSTEM = (
    "你是数据构建助手。给定一篇论文的片段，生成 {num} 个高质量的问答对。"
    "要求：问题覆盖方法原理、关键公式含义、实验结论等；"
    "回答必须包含逐步推理（CoT），并引用片段编号 [1]..[n]；"
    "严格输出 JSON 数组：{{\"qa\": [{{\"question\": ..., \"answer\": ...}}]}}。"
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _section_blocks(sections: list[Section], limit: int = 6) -> list[tuple[str, str]]:
    """取章节标题 + 文本片段，用于造数。"""
    out = []
    for sec in sections:
        text = sec.text.strip()
        if text:
            out.append((sec.title, text[:3000]))
        for sub in sec.subsections:
            if sub.text.strip():
                out.append((sub.title, sub.text[:3000]))
        if len(out) >= limit:
            break
    return out


def mock_qa(doc: LayoutDocument, num: int = 3) -> list[dict]:
    """离线模板造数：从章节正文抽取关键句并套用 QA 模板。"""
    sections = to_sections(doc)
    blocks = _section_blocks(sections, limit=8)
    if not blocks:
        return []
    qas: list[dict] = []
    for i, (title, text) in enumerate(blocks[:num]):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        core = " ".join(sentences[:2]) if sentences else text
        qas.append(
            {
                "section_title": title,
                "question": f"请根据原文总结章节“{title}”的核心内容，并说明关键结论。",
                "answer": (
                    f"以下是逐步推理：\n1. 该章节讨论的是“{title}”。\n"
                    f"2. 原文核心表述：{core}\n"
                    f"3. 关键结论需要结合原文上下文理解。\n"
                    f"引用：[chunk_{i + 1}]"
                ),
            }
        )
    return qas


def gpt4_qa(
    doc: LayoutDocument,
    num: int = 3,
    client: Any | None = None,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """调用 GPT-4 生成 QA 对；失败时降级为 mock。"""
    if client is None:
        return mock_qa(doc, num)
    blocks = _section_blocks(to_sections(doc), limit=6)
    if not blocks:
        return []
    paper_text = "\n\n".join(f"[{i + 1}] ({t})\n{text}" for i, (t, text) in enumerate(blocks))
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QA_GENERATION_SYSTEM.format(num=num)},
                {"role": "user", "content": paper_text[:12000]},
            ],
            temperature=0.6,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        items = data.get("qa", data.get("items", []))
        if isinstance(items, list) and items:
            return [
                {"question": q.get("question"), "answer": q.get("answer")}
                for q in items
            ]
    except Exception as e:  # noqa: BLE001
        print(f"[warn] GPT-4 造数失败，降级 mock: {e}")
    return mock_qa(doc, num)


def build_sft_examples(
    doc: LayoutDocument,
    doc_id: str,
    num: int = 3,
    client: Any | None = None,
    mock: bool = True,
) -> list[dict]:
    """构建 SFT 样本：{"messages": [...]}。"""
    qas = mock_qa(doc, num) if mock else gpt4_qa(doc, num, client)
    examples = []
    for qa in qas:
        context = _build_context(doc, focus=qa.get("section_title"))
        examples.append(
            {
                "doc_id": doc_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"论文原文：\n{context}\n\n问题：{qa['question']}",
                    },
                    {"role": "assistant", "content": qa["answer"]},
                ],
            }
        )
    return examples


def _build_context(
    doc: LayoutDocument,
    max_chars: int = 2000,
    focus: str | None = None,
) -> str:
    """渲染上下文；focus 指定章节时只包含该章节及其子章节。"""
    sections = to_sections(doc)
    parts = []

    def walk(secs: list[Section], level: int) -> None:
        for sec in secs:
            if focus and focus not in sec.title and not any(
                focus in s.title for s in _all_subsections(sec)
            ):
                continue
            if sec.text.strip():
                parts.append(f"{'  ' * level}### {sec.title}\n{sec.text}")
            walk(sec.subsections, level + 1)

    walk(sections, 0)
    if not parts:  # focus 未命中时回退全文
        focus = None
        walk(sections, 0)
    text = "\n\n".join(parts)
    return text[:max_chars]


def _all_subsections(sec: Section) -> list[Section]:
    out = [sec]
    for sub in sec.subsections:
        out.extend(_all_subsections(sub))
    return out


def write_jsonl(records: list[dict], path: str | Path) -> Path:
    out = _resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out
