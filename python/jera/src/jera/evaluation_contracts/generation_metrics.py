"""RAGAS-lite generation-quality metric contracts (pure, deterministic, no IO).

Jera's retrieval metrics (`metrics.py`) grade *what was retrieved*; these grade *the generated
answer* — the RAGAS quartet, in deterministic offline form:

- ``faithfulness``       — is every answer claim grounded in the context? (anti-hallucination)
- ``answer_relevance``   — does the answer address the query? (cosine over supplied vectors)
- ``answer_correctness`` — token-F1 of the answer against a reference answer
- ``context_precision``  — are the relevant contexts ranked near the top? (average-precision)

Deterministic by construction: ``faithfulness``/``answer_correctness`` use token-set overlap
(no model); ``answer_relevance`` takes pre-computed embedding vectors so the eval harness can
supply them from the embedding port without this module importing one. The same contracts grade
real LLM answers under the local/prod profiles without code changes.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from jera.evaluation_contracts.dataset import GoldChunk

_TOKEN = re.compile(r"\w+", re.UNICODE)
_SENT_SPLIT = re.compile(r"[.!?。！？\n]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text)}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _containment(sentence: set[str], context: set[str]) -> float:
    """Fraction of a sentence's tokens present in the context (precision-style)."""
    if not sentence:
        return 0.0
    return len(sentence & context) / len(sentence)


def faithfulness(
    answer_text: str,
    context_texts: Sequence[str],
    *,
    support_threshold: float = 0.6,
) -> float:
    """Fraction of answer sentences supported by the retrieved context (1.0 = fully grounded).

    A sentence is "supported" when ≥ ``support_threshold`` of its tokens appear in the union of
    context tokens (containment, not Jaccard — a long context must not dilute support, and a
    short answer sentence sharing only stop-words must not count as grounded). An answer with no
    sentences is vacuously faithful (1.0). Deterministic stand-in for RAGAS faithfulness (an LLM
    judge) with the same contract: grounded-claim ratio.
    """
    answer_sents = _sentences(answer_text)
    if not answer_sents:
        return 1.0
    context_tokens: set[str] = set()
    for ctx in context_texts:
        context_tokens |= _tokens(ctx)
    if not context_tokens:
        return 0.0
    grounded = sum(
        1
        for sent in answer_sents
        if _containment(_tokens(sent), context_tokens) >= support_threshold
    )
    return grounded / len(answer_sents)


def answer_relevance(answer_vec: Sequence[float], query_vec: Sequence[float]) -> float:
    """Cosine similarity between the answer and query embeddings, clamped to [0, 1].

    Vectors are supplied pre-computed (by the embedding port) so this stays pure and model-free.
    Mirrors RAGAS answer-relevance (which embeds a question generated from the answer).
    """
    if len(answer_vec) != len(query_vec):
        raise ValueError(f"vector dim mismatch: {len(answer_vec)} != {len(query_vec)}")
    dot = sum(a * b for a, b in zip(answer_vec, query_vec, strict=True))
    na = math.sqrt(sum(a * a for a in answer_vec))
    nb = math.sqrt(sum(b * b for b in query_vec))
    if na == 0.0 or nb == 0.0:
        return 0.0
    cos = dot / (na * nb)
    return max(0.0, min(1.0, cos))


def answer_correctness(answer_text: str, reference_text: str) -> float:
    """Token-level F1 between the answer and a reference answer (0..1).

    Precision = shared / answer tokens, recall = shared / reference tokens, F1 their harmonic
    mean. Multiset-aware (repeated tokens counted) so padding an answer cannot inflate the score.
    """
    ans = [t.lower() for t in _TOKEN.findall(answer_text)]
    ref = [t.lower() for t in _TOKEN.findall(reference_text)]
    if not ans and not ref:
        return 1.0
    if not ans or not ref:
        return 0.0
    shared = 0
    ref_pool = list(ref)
    for tok in ans:
        if tok in ref_pool:
            ref_pool.remove(tok)
            shared += 1
    if shared == 0:
        return 0.0
    precision = shared / len(ans)
    recall = shared / len(ref)
    return 2 * precision * recall / (precision + recall)


def context_precision(ranked_ids: Sequence[str], gold: Sequence[GoldChunk], k: int) -> float:
    """Average precision over the ranked context — relevant items ranked higher score better.

    AP = mean of precision@i taken at each rank i (≤ k) that holds a relevant chunk, divided by
    the number of relevant gold chunks (capped at k). 0.0 when nothing relevant is retrieved.
    This rewards putting the right context first — what generation quality actually depends on.
    """
    gold_ids = {g.chunk_id for g in gold}
    if not gold_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, cid in enumerate(ranked_ids[:k], start=1):
        if cid in gold_ids:
            hits += 1
            precision_sum += hits / i
    denom = min(len(gold_ids), k)
    return precision_sum / denom if denom else 0.0
