"""忠实度评测：主张级 NLI 支持率。"""

from __future__ import annotations

import statistics

from src.rag.chunker import Chunk
from src.rag.self_rag import SelfRAG


def faithfulness_report(
    answer: str,
    chunks: list[Chunk],
    judge_model: str = "cross-encoder/nli-deberta-v3-xsmall",
    device: str = "cpu",
) -> dict:
    """对回答做主张级忠实度评测。"""
    rag = SelfRAG(judge_model=judge_model, device=device)
    claims = rag.extract_claims(answer)
    verdicts = rag.judge_claims(claims, chunks)
    supported = [v for v in verdicts if v.supported]
    scores = [v.score for v in verdicts]
    return {
        "n_claims": len(verdicts),
        "n_supported": len(supported),
        "support_ratio": len(supported) / len(verdicts) if verdicts else 0.0,
        "avg_score": statistics.mean(scores) if scores else 0.0,
        "verdicts": [
            {
                "claim": v.claim,
                "chunk_id": v.chunk_id,
                "score": round(v.score, 4),
                "supported": v.supported,
            }
            for v in verdicts
        ],
    }
