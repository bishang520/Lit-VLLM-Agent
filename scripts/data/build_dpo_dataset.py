"""从论文 PDF 构建 DPO 偏好数据集。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.stdio import fix_console

fix_console()

from src.data.dpo_builder import build_dpo_examples, write_jsonl
from src.data.layout import parse_pdf
from src.data.sft_builder import mock_qa


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 DPO 数据集")
    parser.add_argument("--pdf-dir", default="work/data/arxiv_pdfs")
    parser.add_argument("--out", default="work/data/dpo_train.jsonl")
    parser.add_argument("--num-per-doc", type=int, default=3, help="每篇生成的 QA 数")
    parser.add_argument("--pairs-per-qa", type=int, default=2, help="每个 QA 构造的偏好对数")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    records = []
    for i, pdf in enumerate(pdfs):
        try:
            doc = parse_pdf(pdf)
            qas = mock_qa(doc, num=args.num_per_doc)
            examples = build_dpo_examples(doc, pdf.stem, qas, num=args.pairs_per_qa)
            records.extend(examples)
            print(f"[{i + 1}/{len(pdfs)}] {pdf.stem}: {len(examples)} 条")
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {pdf.name}: {e}")

    out = write_jsonl(records, args.out)
    print(f"完成：{len(records)} 条 DPO 样本 -> {out}")


if __name__ == "__main__":
    sys.exit(main())
