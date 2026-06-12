"""Retrieval/answer metric contracts: recall@k, MRR, nDCG, citation faithfulness.

Pure functions over ranked chunk-id lists and gold sets — deterministic, no IO. Populated
benchmark datasets are out of M1 scope; the contracts exist now so eval is not "deferred".
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from jera.evaluation_contracts.dataset import GoldChunk


def recall_at_k(ranked_ids: Sequence[str], gold: Sequence[GoldChunk], k: int) -> float:
    if not gold:
        return 0.0
    gold_ids = {g.chunk_id for g in gold}
    hits = sum(1 for cid in ranked_ids[:k] if cid in gold_ids)
    return hits / len(gold_ids)


def mrr(ranked_ids: Sequence[str], gold: Sequence[GoldChunk]) -> float:
    gold_ids = {g.chunk_id for g in gold}
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], gold: Sequence[GoldChunk], k: int) -> float:
    rel = {g.chunk_id: g.relevance for g in gold}
    dcg = sum(rel.get(cid, 0.0) / math.log2(i + 2) for i, cid in enumerate(ranked_ids[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def citation_faithfulness(cited_ids: Sequence[str], retrieved_ids: Sequence[str]) -> float:
    """Fraction of citations that resolve to actually-retrieved chunks (1.0 = fully grounded)."""
    if not cited_ids:
        return 1.0
    retrieved = set(retrieved_ids)
    grounded = sum(1 for cid in cited_ids if cid in retrieved)
    return grounded / len(cited_ids)
