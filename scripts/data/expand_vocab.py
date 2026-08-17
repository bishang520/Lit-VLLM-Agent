"""BPE 领域词表扩充：训练 BPE -> 筛选领域 token -> 扩展基座 tokenizer。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.stdio import fix_console

fix_console()

from src.data.vocab import extend_tokenizer, select_domain_tokens, train_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description="BPE 领域词表扩充")
    parser.add_argument("--corpus-dir", required=True, help="学术语料目录（*.txt）")
    parser.add_argument("--base-tokenizer", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", default="work/models/extended_tokenizer")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-freq", type=int, default=50, help="BPE 训练最小频率")
    parser.add_argument("--top-n", type=int, default=2000, help="新增 token 数量")
    parser.add_argument("--sel-min-freq", type=int, default=5, help="筛选最小语料频率")
    parser.add_argument("--max-lines", type=int, default=200_000)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    corpus_dirs = [args.corpus_dir]
    print("1/3 训练 BPE ...")
    bpe_path = train_bpe(
        corpus_dirs,
        str(Path(args.out) / "bpe"),
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
    )

    print("2/3 筛选领域 token ...")
    from tokenizers import Tokenizer

    trained = Tokenizer.from_file(str(bpe_path))
    base = AutoTokenizer.from_pretrained(args.base_tokenizer, trust_remote_code=True)
    tokens = select_domain_tokens(
        base,
        trained,
        corpus_dirs,
        top_n=args.top_n,
        min_freq=args.sel_min_freq,
        max_lines=args.max_lines,
    )
    print(f"筛选出 {len(tokens)} 个候选 token，示例: {tokens[:10]}")

    print("3/3 扩展 tokenizer ...")
    extend_tokenizer(args.base_tokenizer, tokens, args.out)


if __name__ == "__main__":
    sys.exit(main())
