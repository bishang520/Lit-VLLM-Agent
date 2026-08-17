"""DPO 偏好数据集构建：构造 (prompt, chosen, rejected) 三元组。"""

from __future__ import annotations

import json
import random
import re
from functools import partial
from pathlib import Path

from src.config import PROJECT_ROOT
from src.data.layout import LayoutDocument
from src.data.sft_builder import SYSTEM_PROMPT, _build_context


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _corrupt_numbers(text: str, rng: random.Random) -> str:
    """幻觉型负样本：把关键数字替换为近似值。"""
    numbers = re.findall(r"\b\d+(?:\.\d+)?(?:%|x|×)?\b", text)
    if not numbers:
        return text
    for n in rng.sample(numbers, min(2, len(numbers))):
        try:
            v = float(n)
            corrupted = v * (1 + rng.choice([0.1, 0.2, -0.1]))
            text = text.replace(n, f"{corrupted:.1f}", 1)
        except ValueError:
            continue
    return text


def _shuffle_reasoning(text: str) -> str:
    """推理错序型负样本：打乱编号步骤。"""
    steps = re.findall(r"(\d+[\.、]\s*[^\n]+)", text)
    if len(steps) >= 3:
        shuffled = steps[1:] + steps[:1]
        for old, new in zip(steps, shuffled):
            text = text.replace(old, new, 1)
    return text


def _strip_citations(text: str) -> str:
    """无引用型负样本：删除 [chunk_x] 标注。"""
    return re.sub(r"\[chunk_[^\]]+\]", "", text)


def _grounded_answer(qa: dict) -> str:
    """构造忠实答案（chosen）：CoT + 引用完整。"""
    answer = qa.get("answer", "")
    if "[chunk_" not in answer:
        answer = answer + "\n引用：[chunk_1]"
    return answer


def build_preference_pairs(
    qa_list: list[dict],
    num: int = 2,
    rng: random.Random | None = None,
) -> list[dict]:
    """由 QA 对构造偏好三元组。"""
    rng = rng or random.Random(42)
    pairs: list[dict] = []
    for qa in qa_list:
        prompt = qa.get("question", "")
        if not prompt:
            continue
        chosen = _grounded_answer(qa)
        strategies = [
            partial(_corrupt_numbers, rng=rng),
            _strip_citations,
            _shuffle_reasoning,
        ]
        for strat in rng.sample(strategies, min(num, len(strategies))):
            rejected = strat(chosen)
            if rejected.strip() == chosen.strip():
                rejected = _strip_citations(chosen)
            pairs.append(
                {
                    "prompt": [{"role": "user", "content": prompt}],
                    "chosen": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": chosen},
                    ],
                    "rejected": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": rejected},
                    ],
                }
            )
    return pairs


def build_dpo_examples(
    doc: LayoutDocument,
    doc_id: str,
    qa_list: list[dict],
    num: int = 2,
) -> list[dict]:
    """组装 DPO 样本，prompt 附带论文上下文。"""
    context = _build_context(doc)
    examples = []
    for p in build_preference_pairs(qa_list, num=num):
        prompt_text = f"论文原文：\n{context}\n\n问题：{p['prompt'][0]['content']}"
        examples.append(
            {
                "doc_id": doc_id,
                "prompt": [{"role": "user", "content": prompt_text}],
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            }
        )
    return examples


def write_jsonl(records: list[dict], path: str | Path) -> Path:
    out = _resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out
