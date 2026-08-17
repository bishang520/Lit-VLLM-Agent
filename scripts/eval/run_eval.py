"""端到端评测：检索 Recall@K / MRR、忠实度（NLI 支持率）、引用命中、延迟。

检索指标通过 API 的 /v1/retrieve 计算（测真实服务链路，避免 Milvus 锁冲突）。
查询为章节首句（单句检索），gold 为该句所属片段——标准的 passage retrieval 评测设定。

用法（服务器上，先启动 API）：
    python scripts/eval/run_eval.py \
        --api http://localhost:8000 \
        --sft work/data/sft_train.jsonl \
        --num-queries 40 --num-rag 10
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.stdio import fix_console

fix_console()

from src.rag.chunker import Chunk


def build_queries(bm25_chunks, sft_path: str, num: int, seed: int = 42) -> list[dict]:
    """采样子片段：查询=前 1-2 句（≤140 字符），gold=该片段自身。"""
    candidates = []
    for c in bm25_chunks:
        if c.metadata.get("is_section"):
            continue
        text = " ".join(c.text.split())
        if len(text) < 250:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", text)
        query = " ".join(sentences[:2])[:140]
        if len(query) < 40:
            continue
        candidates.append(
            {"query": query, "gold": [c.chunk_id], "doc_id": c.doc_id}
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return candidates[:num]


def recall_at_k(pred_ids: list[str], gold: set[str], k: int) -> float:
    hit = set(pred_ids[:k]) & gold
    return len(hit) / len(gold) if gold else 0.0


def mrr_at_k(pred_ids: list[str], gold: set[str], k: int = 10) -> float:
    for i, cid in enumerate(pred_ids[:k]):
        if cid in gold:
            return 1.0 / (i + 1)
    return 0.0


def mean(x):
    return round(statistics.mean(x), 4) if x else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="端到端评测")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--sft", default="work/data/sft_train.jsonl")
    parser.add_argument("--num-queries", type=int, default=40)
    parser.add_argument("--num-rag", type=int, default=10, help="做生成评测的查询数")
    parser.add_argument("--judge-model", default="cross-encoder/nli-deberta-v3-xsmall")
    parser.add_argument("--out", default="work/eval_report.json")
    args = parser.parse_args()

    import requests

    from src.config import PROJECT_ROOT
    from src.rag.factory import build_retriever

    print("[1/4] 构建检索器（仅用于取片段元数据，指标走 API）...")
    retriever = build_retriever()
    bm25_chunks = retriever.bm25.chunks if retriever.bm25 else []

    print(f"[2/4] 从 SFT 数据构建 {args.num_queries} 个查询 ...")
    queries = build_queries(bm25_chunks, args.sft, args.num_queries)
    print(f"      有效查询: {len(queries)}")
    if not queries:
        raise SystemExit("无有效查询，请检查 --sft 与索引片段")

    methods = {"dense": {}, "sparse": {}, "hybrid": {}}
    for m in methods.values():
        m["recall5"], m["recall10"], m["mrr"] = [], [], []
    retrieval_latency = []

    rag_queries = queries[: args.num_rag]
    gen_latency = []
    faithfulness = []
    cit_recalls, cit_precisions, cit_grounding = [], [], []

    from src.eval.citation import citation_metrics
    from src.eval.faithfulness import faithfulness_report

    print("[3/4] 检索指标（走 API /v1/retrieve）...")
    for i, q in enumerate(queries):
        gold = set(q["gold"])
        for method in methods:
            t0 = time.time()
            resp = requests.post(
                f"{args.api}/v1/retrieve",
                json={"query": q["query"], "top_k": 10, "method": method},
                timeout=60,
            )
            elapsed = (time.time() - t0) * 1000
            if resp.status_code != 200:
                print(f"      [warn] {method} HTTP {resp.status_code}: {resp.text[:100]}")
                continue
            preds = [r["chunk_id"] for r in resp.json()["results"]]
            if method == "hybrid":
                retrieval_latency.append(elapsed)
            methods[method]["recall5"].append(recall_at_k(preds, gold, 5))
            methods[method]["recall10"].append(recall_at_k(preds, gold, 10))
            methods[method]["mrr"].append(mrr_at_k(preds, gold))
        if (i + 1) % 10 == 0:
            print(f"      检索进度 {i + 1}/{len(queries)}")

    print("\n=== 检索指标（均值）===")
    print(f"{'方法':<10}{'Recall@5':<10}{'Recall@10':<10}{'MRR@10':<10}")
    for name, m in methods.items():
        print(f"{name:<10}{mean(m['recall5']):<10}{mean(m['recall10']):<10}{mean(m['mrr']):<10}")
    print(f"平均检索延迟(hybrid): {mean(retrieval_latency)} ms\n")

    print(f"[4/4] RAG 生成评测（{len(rag_queries)} 个查询）...")
    for i, q in enumerate(rag_queries):
        try:
            t0 = time.time()
            resp = requests.post(
                f"{args.api}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": q["query"]}],
                    "temperature": 0.2,
                    "max_tokens": 512,
                },
                timeout=180,
            )
            gen_latency.append(time.time() - t0)
            if resp.status_code != 200:
                print(f"      [skip] q#{i} HTTP {resp.status_code}")
                continue
            payload = json.loads(resp.json()["choices"][0]["message"]["content"])
            ret = requests.post(
                f"{args.api}/v1/retrieve",
                json={"query": q["query"], "top_k": 5, "method": "hybrid"},
                timeout=60,
            ).json()
            chunks = [r["chunk_id"] for r in ret["results"]]
            pred_cites = [
                str(c.get("chunk_id", "")).strip("[]\"' ")
                for c in payload.get("citations", [])
                if c.get("chunk_id")
            ]
            evidence = [
                Chunk(chunk_id=r["chunk_id"], doc_id=r.get("doc_id", ""), text=r["text"])
                for r in ret["results"]
            ]
            report = faithfulness_report(payload.get("answer", ""), evidence, judge_model=args.judge_model)
            faithfulness.append(report["support_ratio"])
            gold = set(q["gold"])
            cit = citation_metrics(pred_cites, list(gold))
            cit_recalls.append(cit["citation_recall"])
            cit_precisions.append(cit["citation_precision"])
            grounding = len(set(pred_cites) & set(chunks)) / len(set(pred_cites)) if pred_cites else 0.0
            cit_grounding.append(grounding)
            if i < 3:
                print(f"      q#{i} [{q['doc_id']}] 预测引用={pred_cites[:3]} 检索={chunks[:3]}")
            print(f"      q#{i}: 忠实度={report['support_ratio']:.2f} "
                  f"引用R={cit['citation_recall']:.2f} P={cit['citation_precision']:.2f} "
                  f"引用对检索准确率={grounding:.2f} 延迟={gen_latency[-1]:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(f"      [skip] q#{i}: {e}")

    report = {
        "retrieval": {
            name: {k: mean(v) for k, v in m.items()} for name, m in methods.items()
        },
        "avg_retrieval_latency_ms": mean(retrieval_latency),
        "rag": {
            "n": len(faithfulness),
            "avg_faithfulness_support_ratio": mean(faithfulness),
            "avg_citation_recall": mean(cit_recalls),
            "avg_citation_precision": mean(cit_precisions),
            "avg_citation_grounding": mean(cit_grounding),
            "avg_generation_latency_s": mean(gen_latency),
        },
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告已保存: {out}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
