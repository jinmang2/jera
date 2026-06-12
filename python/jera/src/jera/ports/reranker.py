"""Reranker port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jera.domain.retrieval import ScoredChunk


@runtime_checkable
class Reranker(Protocol):
    """Reorders retrieved candidates for precision after first-stage recall."""

    model_id: str

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]: ...
