"""分词与 chat template 工具：assistant 掩码 + 扩展词表加载。"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import PROJECT_ROOT


def load_tokenizer(base_name_or_dir: str, extended_tokenizer_dir: str | None = None):
    """加载 tokenizer；若提供扩展词表目录则直接加载扩展版（含新增 token）。"""
    from transformers import AutoTokenizer

    if extended_tokenizer_dir:
        p = Path(extended_tokenizer_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists() and (p / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        print(f"[warn] 扩展词表目录不存在，回退基座 tokenizer: {p}")

    tokenizer = AutoTokenizer.from_pretrained(base_name_or_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_chat(example: dict, tokenizer, max_length: int = 2048) -> dict:
    """多轮 SFT 样本 -> input_ids/attention_mask/labels（仅 assistant 计算 loss）。

    transformers 5.x 移除了 apply_chat_template 的 assistant_masks 返回，
    这里用“渲染文本 + 字符偏移映射”自行计算 assistant 区间（对 4.x/5.x 均鲁棒）。
    """
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
    )
    input_ids = enc["input_ids"]
    offsets = enc.get("offset_mapping")
    labels = input_ids[:]
    if offsets:
        spans = _assistant_spans(text)

        def inside(span: tuple[int, int], off: tuple[int, int]) -> bool:
            mid = (off[0] + off[1]) / 2
            return span[0] <= mid <= span[1] and off != (0, 0)

        labels = [
            tok_id if any(inside(s, o) for s in spans) else -100
            for tok_id, o in zip(input_ids, offsets)
        ]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _assistant_spans(text: str) -> list[tuple[int, int]]:
    """在渲染文本中定位所有 assistant 内容区间（含 system/user 保留）。"""
    marker_start = "<|im_start|>assistant"
    marker_end = "<|im_end|>"
    spans: list[tuple[int, int]] = []
    idx = 0
    while True:
        s = text.find(marker_start, idx)
        if s < 0:
            break
        body_start = s + len(marker_start)
        if body_start < len(text) and text[body_start] == "\n":
            body_start += 1
        e = text.find(marker_end, body_start)
        if e < 0:
            break
        spans.append((body_start, e))
        idx = e + len(marker_end)
    return spans


def format_dpo_example(example: dict, tokenizer) -> dict:
    """把 DPO 样本的 messages 渲染为字符串（TRL DPOTrainer 要求）。"""

    def render(msgs, add_generation_prompt: bool) -> str:
        if isinstance(msgs, str):
            return msgs
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt
        )

    return {
        "prompt": render(example["prompt"], add_generation_prompt=True),
        "chosen": render(example["chosen"], add_generation_prompt=False),
        "rejected": render(example["rejected"], add_generation_prompt=False),
    }


class ChatDataCollator:
    """把变长样本 padding 成 batch；labels 用 -100 填充。"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict:
        import torch

        pad_id = self.tokenizer.pad_token_id or 0
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            n = len(f["input_ids"])
            pad_n = max_len - n
            input_ids.append(f["input_ids"] + [pad_id] * pad_n)
            attention_mask.append([1] * n + [0] * pad_n)
            labels.append(f["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def load_added_tokens(extended_dir: str | Path) -> list[str]:
    p = Path(extended_dir)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    added_file = p / "added_tokens.json"
    if not added_file.exists():
        return []
    data = json.loads(added_file.read_text(encoding="utf-8"))
    return list(data.keys())
