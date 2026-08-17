"""从论文 PDF 构建 SFT 多轮指令数据集（GPT-4 辅助或 mock）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.stdio import fix_console

fix_console()

from src.data.layout import parse_pdf
from src.data.sft_builder import build_sft_examples, write_jsonl


def _openai_client(model: str):
    import os

    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[warn] 未设置 OPENAI_API_KEY，将使用 mock 模式")
        return None
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 SFT 数据集")
    parser.add_argument("--pdf-dir", default="work/data/arxiv_pdfs")
    parser.add_argument("--out", default="work/data/sft_train.jsonl")
    parser.add_argument("--num-per-doc", type=int, default=3)
    parser.add_argument("--gpt4", action="store_true", help="使用 GPT-4 造数（需 API Key）")
    parser.add_argument("--gpt4-model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 篇（0=全部）")
    args = parser.parse_args()

    client = _openai_client(args.gpt4_model) if args.gpt4 else None
    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    records = []
    for i, pdf in enumerate(pdfs):
        try:
            doc = parse_pdf(pdf)
            examples = build_sft_examples(
                doc, doc_id=pdf.stem, num=args.num_per_doc, client=client, mock=client is None
            )
            records.extend(examples)
            print(f"[{i + 1}/{len(pdfs)}] {pdf.stem}: {len(examples)} 条")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {pdf.name}: {e}")

    out = write_jsonl(records, args.out)
    print(f"完成：{len(records)} 条 SFT 样本 -> {out}")


if __name__ == "__main__":
    sys.exit(main())
