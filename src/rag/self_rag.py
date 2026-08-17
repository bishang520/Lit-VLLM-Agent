"""Self-RAG 自评估：相关性门控 + 主张级忠实度 + 引用可溯源性。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.rag.chunker import Chunk
from src.rag.embedder import Embedder


@dataclass
class ClaimVerdict:
    claim: str
    chunk_id: str | None
    chunk_text: str
    score: float
    supported: bool


@dataclass
class SelfRagReport:
    query: str
    relevant: bool
    verdicts: list[ClaimVerdict]
    grounded_ratio: float
    citations: list[str]
    summary: str


class SelfRAG:
    """三段式自评估。

    1. 检索相关性门控（embedding 相似度）；
    2. 主张级 NLI 忠实度（交叉编码器判定证据是否蕴含主张）；
    3. 引用可溯源性（每条主张映射到 chunk，分数低于阈值判为不可溯源）。
    """

    def __init__(
        self,
        judge_model: str = "cross-encoder/nli-deberta-v3-xsmall",
        relevance_threshold: float = 0.30,
        faithfulness_threshold: float = 0.50,
        citation_threshold: float = 0.35,
        device: str = "cpu",
        embedder: Embedder | None = None,
    ):
        self.judge_model = judge_model
        self.relevance_threshold = relevance_threshold
        self.faithfulness_threshold = faithfulness_threshold
        self.citation_threshold = citation_threshold
        self.device = device
        self.embedder = embedder
        self._judge = None
        self._is_nli = "nli" in judge_model.lower()

    def _load_judge(self):
        from sentence_transformers import CrossEncoder

        self._judge = CrossEncoder(self.judge_model, device=self.device)

    def check_relevance(self, query: str, chunks: list[Chunk]) -> bool:
        """检索相关性门控：query 与检索片段最高相似度是否达标。"""
        import numpy as np

        if self.embedder is None or not chunks:
            return True
        qv = self.embedder.encode([query])[0]
        evs = self.embedder.encode([c.text[:1024] for c in chunks])
        sims = evs @ qv
        return bool(sims.max() >= self.relevance_threshold)

    @staticmethod
    def extract_claims(answer: str, max_claims: int = 8) -> list[str]:
        """启发式主张切分：按句子切分并过滤过短句子。"""
        if not answer or not answer.strip():
            return []
        sentences = re.split(r"(?<=[.!?。！？])\s+", answer.strip())
        claims = [s.strip() for s in sentences if len(s.strip()) >= 20]
        if not claims:
            claims = [answer.strip()] if len(answer.strip()) >= 20 else []
        return claims[:max_claims]

    def judge_claims(
        self, claims: list[str], chunks: list[Chunk], evidence_k: int = 6
    ) -> list[ClaimVerdict]:
        """对每条主张判定最相关证据及其支持分数。"""
        import numpy as np

        if not claims or not chunks:
            return []
        if self._judge is None:
            self._load_judge()
        evidence = chunks[:evidence_k]
        verdicts = []
        for claim in claims:
            pairs = [(claim, c.text[:512]) for c in evidence]
            scores = np.asarray(self._judge.predict(pairs, show_progress_bar=False), dtype=float)
            if self._is_nli and scores.ndim == 2 and scores.shape[1] == 3:
                exp = np.exp(scores - scores.max(axis=1, keepdims=True))
                probs = exp / exp.sum(axis=1, keepdims=True)
                conf = float(probs[:, 2].max())  # entailment 列
            else:
                conf = float(1.0 / (1.0 + np.exp(-scores.max())))
            best = int(np.argmax(scores if not (self._is_nli and scores.ndim == 2) else probs[:, 2]))
            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    chunk_id=evidence[best].chunk_id,
                    chunk_text=evidence[best].text[:200],
                    score=conf,
                    supported=conf >= self.faithfulness_threshold,
                )
            )
        return verdicts

    def evaluate(self, query: str, answer: str, chunks: list[Chunk]) -> SelfRagReport:
        """端到端自评估。"""
        relevant = self.check_relevance(query, chunks)
        claims = self.extract_claims(answer)
        verdicts = self.judge_claims(claims, chunks)
        grounded = [
            v
            for v in verdicts
            if v.supported and v.score >= self.citation_threshold
        ]
        ratio = len(grounded) / len(verdicts) if verdicts else 0.0
        citations = [v.chunk_id for v in grounded if v.chunk_id]
        if not relevant:
            summary = "检索相关性不足：请更换问法或补充领域关键词。"
        elif not verdicts:
            summary = "回答过短或无法切分主张，建议人工复核。"
        elif ratio >= 0.8:
            summary = f"回答基本忠实（支持率 {ratio:.0%}），可放心展示。"
        elif ratio >= 0.5:
            summary = f"回答部分忠实（支持率 {ratio:.0%}），建议人工复核低分主张。"
        else:
            summary = f"回答忠实度偏低（支持率 {ratio:.0%}），建议拒答或改写。"
        return SelfRagReport(
            query=query,
            relevant=relevant,
            verdicts=verdicts,
            grounded_ratio=ratio,
            citations=citations,
            summary=summary,
        )
