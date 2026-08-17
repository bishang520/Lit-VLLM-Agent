"""引用评测：Citation Recall / Precision / F1。"""

from __future__ import annotations


def citation_metrics(pred_ids: list[str], gold_ids: list[str]) -> dict:
    pred = set(pred_ids)
    gold = set(gold_ids)
    inter = pred & gold
    recall = len(inter) / len(gold) if gold else 0.0
    precision = len(inter) / len(pred) if pred else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "citation_recall": round(recall, 4),
        "citation_precision": round(precision, 4),
        "citation_f1": round(f1, 4),
        "hit": sorted(inter),
    }


def answer_citation_metrics(pred_payload: dict, gold_payload: dict) -> dict:
    """从结构化回答 payload 中提取 citations 后计算指标。"""
    pred = [c.get("chunk_id") for c in pred_payload.get("citations", [])]
    gold = [c.get("chunk_id") for c in gold_payload.get("citations", [])]
    return citation_metrics(pred, gold)
