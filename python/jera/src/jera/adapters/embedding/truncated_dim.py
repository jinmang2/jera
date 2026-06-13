"""Matryoshka dimension truncation wrapper for any EmbeddingProvider.

Truncates the first ``dims`` coordinates of each embedding and L2-renormalizes,
implementing the Matryoshka Representation Learning (MRL) inference trick
(Kusupati et al., NeurIPS 2022, arXiv:2205.13147).

This lets a single model trained with MRL serve multiple accuracy/latency
trade-off points by simply slicing the output dimension.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from jera.domain.vectors import DenseVector
from jera.ports.embedding import EmbeddingProvider


def _truncate_and_renorm(vec: DenseVector, dims: int) -> DenseVector:
    """Return the first ``dims`` elements of ``vec``, L2-renormalized."""
    trunc = vec[:dims]
    norm = math.sqrt(sum(v * v for v in trunc))
    if norm == 0.0:
        return trunc
    return [v / norm for v in trunc]


class TruncatedDimEmbedding:
    """Wraps any EmbeddingProvider; truncates each vector to ``dims`` dims and re-normalizes.

    Parameters
    ----------
    base:
        The underlying EmbeddingProvider to delegate ``embed`` / ``embed_query`` to.
    dims:
        Target dimensionality.  Must be <= ``base.dimensions``.

    Attributes
    ----------
    model_id:
        ``base.model_id + "-trunc{dims}"``
    dimensions:
        ``dims``
    context_limit:
        Forwarded from ``base``.
    """

    def __init__(self, base: EmbeddingProvider, dims: int) -> None:
        if dims <= 0:
            raise ValueError(f"dims must be > 0, got {dims}")
        if dims > base.dimensions:
            raise ValueError(f"dims {dims} exceeds base model dimensions {base.dimensions}")
        self._base = base
        self._dims = dims
        self.model_id: str = f"{base.model_id}-trunc{dims}"
        self.dimensions: int = dims
        self.context_limit: int | None = base.context_limit

    def embed(self, texts: Sequence[str]) -> list[DenseVector]:
        """Embed texts and truncate+renormalize to ``self.dimensions`` dims."""
        full_vecs = self._base.embed(texts)
        return [_truncate_and_renorm(v, self._dims) for v in full_vecs]

    def embed_query(self, text: str) -> DenseVector:
        """Embed a single query and truncate+renormalize to ``self.dimensions`` dims."""
        full_vec = self._base.embed_query(text)
        return _truncate_and_renorm(full_vec, self._dims)
