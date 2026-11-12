"""Metrics for the results tables.

Retrieval: recall@k, precision@k, MRR, nDCG@k -- these answer the
"RAG 召回率" question in the idea doc.
Generation: token-level F1 and exact match as a cheap reference-based score;
add ROUGE/BLEU here if the report compares them (Week 4 content).
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[一-鿿]")


def _norm(items) -> list[str]:
    return [str(i).strip().lower() for i in items]


def recall_at_k(retrieved, relevant, k: int) -> float:
    """Fraction of the relevant set that appears in the top ``k``."""
    rel = set(_norm(relevant))
    if not rel:
        return float("nan")
    got = set(_norm(retrieved)[:k])
    return len(got & rel) / len(rel)


def precision_at_k(retrieved, relevant, k: int) -> float:
    rel = set(_norm(relevant))
    top = _norm(retrieved)[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in rel) / len(top)


def hit_rate_at_k(retrieved, relevant, k: int) -> float:
    """1.0 if any relevant item is in the top ``k``."""
    return 1.0 if set(_norm(retrieved)[:k]) & set(_norm(relevant)) else 0.0


def mean_reciprocal_rank(retrieved, relevant) -> float:
    rel = set(_norm(relevant))
    for i, item in enumerate(_norm(retrieved), start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved, relevant, k: int) -> float:
    """Binary-gain nDCG@k."""
    rel = set(_norm(relevant))
    if not rel:
        return float("nan")
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, item in enumerate(_norm(retrieved)[:k], start=1)
        if item in rel
    )
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg else 0.0


def answer_token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token F1 between a generated answer and a reference."""
    pred = Counter(_TOKEN.findall(prediction.lower()))
    ref = Counter(_TOKEN.findall(reference.lower()))
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(pred.values()), 1)
    recall = overlap / max(sum(ref.values()), 1)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(
        " ".join(_TOKEN.findall(prediction.lower()))
        == " ".join(_TOKEN.findall(reference.lower()))
    )
