"""Retrieval-side domain models: queries, fusion methods, scored results."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from jera.domain.chunk import Chunk


class FusionMethod(StrEnum):
    RRF = "rrf"  # Reciprocal Rank Fusion (rank-based; robust to score scale)
    DBSF = "dbsf"  # Distribution-Based Score Fusion (min-max normalized scores summed)


class RetrievalMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class Query(BaseModel):
    """A normalized retrieval query."""

    model_config = {"frozen": True}

    text: str
    top_k: int = 10
    mode: RetrievalMode = RetrievalMode.HYBRID
    fusion: FusionMethod = FusionMethod.RRF


class ScoredChunk(BaseModel):
    """A chunk id with a score and optional resolved chunk + per-component scores."""

    model_config = {"frozen": True}

    chunk_id: str
    score: float
    components: dict[str, float] = {}  # e.g. {"dense": .., "sparse": .., "rerank": ..}
    chunk: Chunk | None = None

    def with_chunk(self, chunk: Chunk) -> ScoredChunk:
        return self.model_copy(update={"chunk": chunk})


class RetrievalResult(BaseModel):
    """Ordered results for a query at a named pipeline stage."""

    query: Query
    stage: str  # "dense" | "sparse" | "hybrid" | "rerank"
    results: list[ScoredChunk]
