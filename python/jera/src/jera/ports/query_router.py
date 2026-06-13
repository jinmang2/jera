"""QueryRouter port.

Adaptive-RAG complexity routing: classify each query into one of three tiers so the pipeline
can skip retrieval entirely (NO_RETRIEVAL), run a standard single-step search (SINGLE_STEP),
or expand into multi-hop sub-queries (MULTI_STEP).  The port is intentionally thin — any
deterministic heuristic or trained classifier (TF-IDF+SVM, etc.) can satisfy it.

Reference: Jeong et al., "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity" (NAACL 2024).  RAGRouter-Bench 2026 reports TF-IDF+SVM
achieving ~93 % routing accuracy; a trainable variant of this interface can be swapped in
without touching the pipeline layer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable


class QueryComplexity(StrEnum):
    """Three-tier query complexity for Adaptive-RAG routing.

    NO_RETRIEVAL  — the query is self-contained (arithmetic, trivially short, etc.); the LLM
                    can answer from parametric knowledge alone → zero vector-store calls.
    SINGLE_STEP   — one targeted retrieval pass is sufficient.
    MULTI_STEP    — multi-hop reasoning required; expand into sub-queries and fuse rankings.
    """

    NO_RETRIEVAL = "no_retrieval"
    SINGLE_STEP = "single_step"
    MULTI_STEP = "multi_step"


@runtime_checkable
class QueryRouter(Protocol):
    """Classify a query string into a ``QueryComplexity`` tier.

    Implementations must be deterministic and require no network call so that the routing
    decision adds negligible latency.  A trained ML variant (TF-IDF+SVM or fine-tuned
    classifier) is a valid future implementation of this same interface.
    """

    def route(self, query: str) -> QueryComplexity: ...
