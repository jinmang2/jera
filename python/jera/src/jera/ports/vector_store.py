"""VectorStore port + collection/record value objects.

The search signature mirrors Qdrant's named-vector + prefetch + fusion model so the
in-memory and Qdrant adapters are config-swappable (parity is goal-not-guaranteed in M1;
pinned by a golden-file fusion contract — see the in-memory adapter).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from jera.domain.retrieval import FusionMethod, ScoredChunk
from jera.domain.vectors import DenseVector, SparseVector


class Distance(StrEnum):
    COSINE = "cosine"
    DOT = "dot"


class CollectionSpec(BaseModel):
    """Declares a collection's named dense/sparse vectors and the embedding identity."""

    model_config = {"frozen": True}

    name: str
    dense_dim: int
    distance: Distance = Distance.COSINE
    has_sparse: bool = True
    embedding_model_id: str = ""


class VectorRecord(BaseModel):
    """A vector row: named dense+sparse vectors plus a payload pointing back to Postgres/SQLite."""

    model_config = {"frozen": True}

    chunk_id: str
    document_id: str
    dense: DenseVector | None = None
    sparse: SparseVector | None = None
    payload: dict[str, object] = {}


@runtime_checkable
class VectorStore(Protocol):
    def ensure_collection(self, spec: CollectionSpec) -> None: ...

    def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None: ...

    def delete(self, collection: str, chunk_ids: Sequence[str]) -> None: ...

    def search(
        self,
        collection: str,
        *,
        dense: DenseVector | None = None,
        sparse: SparseVector | None = None,
        top_k: int = 10,
        fusion: FusionMethod = FusionMethod.RRF,
        prefetch_limit: int = 100,
    ) -> list[ScoredChunk]:
        """Dense-only (sparse=None), sparse-only (dense=None), or hybrid (both) search.

        Raises if a provided dense vector's dimension differs from the collection's.
        """
        ...
