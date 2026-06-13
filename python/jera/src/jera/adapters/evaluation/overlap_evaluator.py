"""OverlapRetrievalEvaluator — deterministic retrieval grader (no LLM).

Grades a ``RetrievalResult`` using two thresholds:

1. **Score threshold** (``score_threshold``): if the collection is empty or the top
   result's score is below this value the retrieval is ``INCORRECT`` — the ranker
   found nothing useful.

2. **Jaccard overlap threshold** (``overlap_threshold``): Jaccard similarity between
   the *query token set* and the *top-chunk token set*.  At or above the threshold →
   ``CORRECT``; below → ``AMBIGUOUS``.

Both thresholds are constructor-level so the evaluator is easily tuned per use-case.

**LLM-judge variant** (future opt-in): an ``LlmRetrievalEvaluator`` following the
same ``RetrievalEvaluator`` protocol could call a generative model to score relevance
as in the original CRAG paper (Yan et al., 2025, arXiv:2401.15884).  That variant is
*not* built here to keep this adapter offline and dependency-free.
"""

from __future__ import annotations

import re

from jera.domain.retrieval import RetrievalResult
from jera.ports.retrieval_evaluator import RetrievalGrade

_WORD = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity: |A ∩ B| / |A ∪ B|.  Returns 0.0 when both sets are empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class OverlapRetrievalEvaluator:
    """Deterministic retrieval grader based on score and Jaccard token overlap.

    Parameters
    ----------
    score_threshold:
        Minimum score the top result must reach to avoid an ``INCORRECT`` grade.
        Default ``0.0`` means any non-empty result passes the score gate.
    overlap_threshold:
        Minimum Jaccard(query tokens, top-chunk tokens) to earn a ``CORRECT`` grade.
        Results that pass the score gate but fall below this are ``AMBIGUOUS``.
        Default ``0.1`` is intentionally low: even a single shared token in a small
        query is enough for partial relevance.
    """

    def __init__(
        self,
        *,
        score_threshold: float = 0.0,
        overlap_threshold: float = 0.1,
    ) -> None:
        self._score_threshold = score_threshold
        self._overlap_threshold = overlap_threshold

    def grade(self, query: str, result: RetrievalResult) -> RetrievalGrade:
        """Grade the retrieval result.

        Decision tree
        -------------
        1. No results **or** top score < ``score_threshold`` → ``INCORRECT``.
        2. Jaccard(query, top chunk) ≥ ``overlap_threshold``  → ``CORRECT``.
        3. Otherwise                                           → ``AMBIGUOUS``.
        """
        if not result.results:
            return RetrievalGrade.INCORRECT

        top = result.results[0]
        if top.score < self._score_threshold:
            return RetrievalGrade.INCORRECT

        # Require the top chunk to have text for a meaningful overlap check.
        if top.chunk is None or not top.chunk.text.strip():
            return RetrievalGrade.AMBIGUOUS

        query_tokens = _tokens(query)
        chunk_tokens = _tokens(top.chunk.text)
        overlap = _jaccard(query_tokens, chunk_tokens)

        if overlap >= self._overlap_threshold:
            return RetrievalGrade.CORRECT
        return RetrievalGrade.AMBIGUOUS
