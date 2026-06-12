"""SparseVectorProvider port (+ optional Fittable for corpus-statistics providers)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jera.domain.vectors import SparseVector


@runtime_checkable
class SparseVectorProvider(Protocol):
    """Produces sparse lexical vectors (BM25/SPLADE-style)."""

    model_id: str

    def encode(self, texts: Sequence[str]) -> list[SparseVector]: ...

    def encode_query(self, text: str) -> SparseVector: ...


@runtime_checkable
class Fittable(Protocol):
    """A provider that needs corpus statistics (e.g. BM25 idf/avgdl) before encoding.

    The ingest pipeline calls :meth:`fit` on the batch being ingested. Providers that do
    not need fitting simply do not implement this protocol.
    """

    def fit(self, corpus: Sequence[str]) -> None: ...

    @property
    def is_fitted(self) -> bool: ...
