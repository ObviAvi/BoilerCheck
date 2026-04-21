"""
Retrieval quality metrics.
"""

import math
from typing import List, Set


def hit_rate(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """1.0 if at least one relevant doc appears in top-k, else 0.0."""
    for doc_id in retrieved_ids[:k]:
        if doc_id in relevant_ids:
            return 1.0
    return 0.0


def mrr(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Mean Reciprocal Rank: 1/(rank of first relevant doc) within top-k."""
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k (binary relevance)."""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG: all relevant docs at the top
    ideal_count = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_count))

    return dcg / idcg if idcg > 0 else 0.0


def context_precision(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant (precision@k)."""
    if k == 0:
        return 0.0
    relevant_count = sum(1 for doc_id in retrieved_ids[:k] if doc_id in relevant_ids)
    return relevant_count / k


def compute_all_metrics(
    retrieved_ids: List[str], relevant_ids: Set[str], k: int
) -> dict:
    """Compute all metrics for a single query."""
    rel_set = set(relevant_ids)
    return {
        "hit_rate@k": hit_rate(retrieved_ids, rel_set, k),
        "mrr@k": mrr(retrieved_ids, rel_set, k),
        "ndcg@k": ndcg(retrieved_ids, rel_set, k),
        "context_precision@k": context_precision(retrieved_ids, rel_set, k),
    }
