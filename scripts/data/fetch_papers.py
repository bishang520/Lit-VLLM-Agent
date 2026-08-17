"""采集 ArXiv 论文 PDF（按 ID 或查询词），可选提取为结构化文本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.stdio import fix_console

fix_console()

from src.data.arxiv import download_by_ids, extract_pdfs, search_and_download


def main() -> None:
    parser = argparse.ArgumentParser(description="采集 ArXiv 论文")
    parser.add_argument("--ids", default=None, help="逗号分隔的 arXiv ID，如 1706.03762,2005.11401")
    parser.add_argument("--query", default=None, help="检索词，如 retrieval augmented generation")
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--out", default="work/data/arxiv_pdfs", help="PDF 输出目录")
    parser.add_argument("--extract", action="store_true", help="解析为结构化文本")
    args = parser.parse_args()

    if not args.ids and not args.query:
        parser.error("请提供 --ids 或 --query")

    if args.ids:
        ids = [i.strip() for i in args.ids.split(",") if i.strip()]
        download_by_ids(ids, args.out)
    if args.query:
        search_and_download(args.query, max_results=args.max_results, download_dir=args.out)

    if args.extract:
        files = extract_pdfs(args.out, "work/data/arxiv_txt")
        print(f"已提取 {len(files)} 篇论文文本到 work/data/arxiv_txt")


if __name__ == "__main__":
    sys.exit(main())
