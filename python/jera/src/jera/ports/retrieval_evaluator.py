"""RetrievalEvaluator port — grades a RetrievalResult against a query.

Corrective RAG (Yan et al., "Corrective Retrieval Augmented Generation", ICLR 2025,
arXiv:2401.15884) requires a retrieval quality signal to decide whether to accept the
initial retrieval, trigger a corrective re-query, or fall back to a web search.  This
port defines that signal as a ``RetrievalGrade`` and the ``RetrievalEvaluator`` protocol
that produces it.

Two concrete implementations are envisioned:

* **OverlapRetrievalEvaluator** (``jera.adapters.evaluation.overlap_evaluator``) — fully
  deterministic, no LLM, suitable for CI.  Grades via score threshold + Jaccard overlap.
* **LlmRetrievalEvaluator** (future opt-in) — calls an LLM to judge relevance, mirrors
  the original CRAG paper's relevance scoring step.  Only built when cloud is enabled.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from jera.domain.retrieval import RetrievalResult


class RetrievalGrade(StrEnum):
    """Three-way quality signal for a retrieval result.

    * ``CORRECT``   — the top result is confidently on-topic; proceed to generation.
    * ``AMBIGUOUS`` — partial relevance; the corrective loop may improve quality.
    * ``INCORRECT`` — no relevant results; the corrective loop should run.
    """

    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@runtime_checkable
class RetrievalEvaluator(Protocol):
    """Grades a ``RetrievalResult`` as CORRECT / AMBIGUOUS / INCORRECT.

    Implementations must be deterministic given the same inputs so that
    ``CorrectiveQueryPipeline`` behaves reproducibly in tests and CI.
    """

    def grade(self, query: str, result: RetrievalResult) -> RetrievalGrade: ...
