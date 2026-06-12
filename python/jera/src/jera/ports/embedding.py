"""EmbeddingProvider port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jera.domain.vectors import DenseVector


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Produces dense vectors. ``model_id``/``dimensions`` are recorded in collection
    metadata because changing them requires re-indexing."""

    model_id: str
    dimensions: int
    context_limit: int | None

    def embed(self, texts: Sequence[str]) -> list[DenseVector]: ...

    def embed_query(self, text: str) -> DenseVector: ...
