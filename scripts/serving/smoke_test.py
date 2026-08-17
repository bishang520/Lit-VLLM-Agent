"""API 冒烟测试：健康检查 + 检索 + 问答（连上服务后运行）。"""

from __future__ import annotations

import argparse
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.stdio import fix_console

fix_console()

from src.serving.client import AcademicAgentClient


def main() -> None:
    parser = argparse.ArgumentParser(description="API 冒烟测试")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--query", default="论文的主要贡献是什么？")
    args = parser.parse_args()

    client = AcademicAgentClient(args.base_url)
    print("健康检查:", client.health())
    ret = client.retrieve(args.query, top_k=3)
    print(f"检索到 {len(ret['results'])} 个片段")
    for r in ret["results"]:
        print(f"  [{r['chunk_id']}] 分数={r['score']} {r['text'][:80]}...")
    resp = client.chat([{"role": "user", "content": args.query}], stream=False)
    content = resp["choices"][0]["message"]["content"]
    print("问答响应:", content[:500])


if __name__ == "__main__":
    sys.exit(main())
