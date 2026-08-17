"""BPE 领域词表扩充。

流程：学术语料训练 BPE -> 按“压缩收益 = 频率 x (基座分词片断数 - 1)”
筛选高频学术术语 -> 扩展基座 tokenizer -> 输出新增 token 列表。
训练阶段需 resize_token_embeddings 并随 QLoRA 一并微调新 token。
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.data.cleaning import clean_text


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def iter_corpus_lines(
    corpus_files: Iterable[str | Path], max_chars: int = 500
) -> Iterable[str]:
    """逐行产出清洗后的语料（限制行长，避免超长行拖慢训练）。"""
    for f in corpus_files:
        p = Path(f)
        if p.is_dir():
            files = sorted(p.rglob("*.txt"))
        elif p.is_file():
            files = [p]
        else:
            continue
        for fp in files:
            try:
                for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = clean_text(line)
                    if 10 <= len(line) <= max_chars:
                        yield line
            except Exception:  # noqa: BLE001
                continue


def train_bpe(
    corpus_files: list[str | Path],
    out_dir: str | Path,
    vocab_size: int = 32768,
    min_frequency: int = 50,
) -> Path:
    """在语料上训练 BPE 模型并保存。"""
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("缺少 tokenizers，请安装 requirements-train.txt") from e

    out = _resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>"],
        show_progress=True,
    )
    files = []
    for f in corpus_files:
        p = Path(f)
        if p.is_dir():
            files.extend(str(x) for x in p.rglob("*.txt"))
        elif p.is_file():
            files.append(str(p))
    tokenizer.train(files, trainer)
    model_path = out / "bpe_domain.json"
    tokenizer.save(str(model_path))
    return model_path


def select_domain_tokens(
    base_tokenizer,
    trained_tokenizer,
    corpus_files: list[str | Path],
    top_n: int = 2000,
    min_freq: int = 5,
    max_lines: int = 200_000,
) -> list[str]:
    """按压缩收益筛选领域 token，返回按收益降序的新增 token。"""
    vocab = trained_tokenizer.get_vocab()
    freq = Counter()
    for i, line in tqdm(
        enumerate(iter_corpus_lines(corpus_files)),
        desc="统计领域 token 频率",
        total=max_lines,
    ):
        if i >= max_lines:
            break
        try:
            freq.update(trained_tokenizer.encode(line).tokens)
        except Exception:  # noqa: BLE001
            continue

    existing = set(base_tokenizer.get_vocab().keys())
    scored: list[tuple[float, str]] = []
    for token, count in freq.items():
        if count < min_freq:
            continue
        if token in existing:
            continue
        if len(token) < 2 or not re.search(r"[A-Za-z]", token):
            continue
        alpha_ratio = sum(1 for c in token if c.isalnum()) / max(len(token), 1)
        if alpha_ratio < 0.5:
            continue
        try:
            pieces = base_tokenizer.tokenize(token)
        except Exception:  # noqa: BLE001
            continue
        saved = len(pieces) - 1
        if saved < 1:
            continue
        scored.append((count * saved, token))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [token for _, token in scored[:top_n]]


def extend_tokenizer(
    base_name_or_dir: str,
    new_tokens: list[str],
    out_dir: str | Path,
) -> Path:
    """扩展基座 tokenizer，保存到 out_dir，并写出新增 token 清单。"""
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("缺少 transformers，请安装 requirements-train.txt") from e

    out = _resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_name_or_dir, trust_remote_code=True)
    added = tokenizer.add_tokens(sorted(set(new_tokens), key=len, reverse=True))
    tokenizer.save_pretrained(str(out))
    (out / "domain_tokens.txt").write_text(
        "\n".join(sorted(set(new_tokens))), encoding="utf-8"
    )
    (out / "added_count.txt").write_text(str(added), encoding="utf-8")
    print(f"新增 token 数: {added}；扩展 tokenizer 已保存到 {out}")
    return out
