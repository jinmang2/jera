"""Late Chunking embedding adapter (Günther et al., Jina AI, arXiv:2409.04701).

Real-model path (opt-in, not implemented here):
    A long-context encoder (e.g. jina-embeddings-v3) embeds the *entire* document in
    one forward pass so every token attends to its full context.  Each chunk's embedding
    is then produced by mean-pooling over that chunk's token-span positions in the final
    hidden state.  The result: a chunk vector that implicitly carries cross-chunk context
    (coreference, discourse) without altering the chunk text.

Deterministic CI analogue (this module):
    Given a document's ordered chunk texts we approximate the same effect in pure Python:

        raw_i  = base.embed(chunk_i)
        ctx_i  = mean( raw_j  for j in [i-window, …, i+window] )
        out_i  = L2_normalize( (1 - alpha) * raw_i  +  alpha * ctx_i )

    When alpha=0 the output is identical to the base embedding (no mixing).
    When alpha>0 neighboring-chunk vocabulary "bleeds" into each chunk vector so that,
    for example, a pronoun chunk (no entity tokens) gains similarity to a query that
    names the antecedent entity — exactly the coreference-resolution property the paper
    demonstrates.

Usage (ingest pipeline)::

    embedder = LateChunkingEmbedding(HashEmbedding())
    chunk_vecs = embedder.embed_document_chunks(["Tesla announced …", "It will ship …"])

    # Query side — unchanged, delegates to base:
    q_vec = embedder.embed_query("Tesla EV release")
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from jera.domain.vectors import DenseVector
from jera.ports.embedding import EmbeddingProvider


def _l2_norm(vec: list[float]) -> list[float]:
    """Return the L2-normalised copy of *vec*; returns the zero vector unchanged."""
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return list(vec)
    return [v / norm for v in vec]


def _mean_vecs(vecs: list[list[float]]) -> list[float]:
    """Element-wise mean of a non-empty list of equal-length vectors."""
    dim = len(vecs[0])
    result = [0.0] * dim
    n = len(vecs)
    for v in vecs:
        for k in range(dim):
            result[k] += v[k]
    return [x / n for x in result]


def _add_scaled(
    a: list[float],
    scale_a: float,
    b: list[float],
    scale_b: float,
) -> list[float]:
    """Return scale_a*a + scale_b*b element-wise."""
    return [scale_a * x + scale_b * y for x, y in zip(a, b, strict=True)]


class LateChunkingEmbedding:
    """Context-preserving chunk embeddings via neighbor-window mean-pooling.

    Parameters
    ----------
    base:
        Any :class:`~jera.ports.embedding.EmbeddingProvider`.  ``embed`` and
        ``embed_query`` are delegated to it unchanged so this wrapper satisfies the
        same protocol.
    alpha:
        Mixing weight for the context (neighbor) contribution.  ``0.0`` → pure base
        embedding; ``1.0`` → pure context mean.  Default ``0.3``.
    window:
        Number of neighboring chunks on each side to include in the context mean.
        Default ``1`` (immediate left + right neighbours).
    """

    def __init__(
        self,
        base: EmbeddingProvider,
        *,
        alpha: float = 0.3,
        window: int = 1,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if window < 0:
            raise ValueError(f"window must be >= 0, got {window}")
        self._base = base
        self._alpha = alpha
        self._window = window

        # EmbeddingProvider protocol attributes
        self.model_id: str = base.model_id + "-latechunk"
        self.dimensions: int = base.dimensions
        self.context_limit: int | None = base.context_limit

    # ------------------------------------------------------------------
    # EmbeddingProvider protocol — pass-through to base
    # ------------------------------------------------------------------

    def embed(self, texts: Sequence[str]) -> list[DenseVector]:
        """Batch-embed arbitrary texts (no context mixing); satisfies the protocol."""
        return self._base.embed(texts)

    def embed_query(self, text: str) -> DenseVector:
        """Embed a single query string; satisfies the protocol."""
        return self._base.embed_query(text)

    # ------------------------------------------------------------------
    # Late-chunking entry point
    # ------------------------------------------------------------------

    def embed_document_chunks(self, chunk_texts: Sequence[str]) -> list[DenseVector]:
        """Produce context-mixed, L2-normalised embeddings for an ordered chunk list.

        Each output vector is::

            out_i = L2_norm( (1-alpha)*raw_i + alpha*mean(raw_j for j in window(i)) )

        where ``window(i)`` spans ``[max(0, i-window), min(n-1, i+window)]``.

        When *alpha* is ``0.0`` the output equals the base embeddings exactly (up to
        floating-point rounding — confirmed by the ``alpha=0`` test).

        Parameters
        ----------
        chunk_texts:
            Ordered chunk texts from a single document.  Ordering matters because the
            window mixes *positional* neighbours.

        Returns
        -------
        list[DenseVector]
            One unit-length vector per input chunk.
        """
        texts = list(chunk_texts)
        n = len(texts)
        if n == 0:
            return []

        # Embed all chunks in one batch for efficiency.
        raw: list[list[float]] = [list(v) for v in self._base.embed(texts)]

        if self._alpha == 0.0:
            # Fast path: no mixing, but still normalise for consistency.
            return [_l2_norm(v) for v in raw]

        result: list[DenseVector] = []
        for i in range(n):
            lo = max(0, i - self._window)
            hi = min(n - 1, i + self._window)
            neighbors = raw[lo : hi + 1]  # includes i itself
            ctx = _mean_vecs(neighbors)
            mixed = _add_scaled(raw[i], 1.0 - self._alpha, ctx, self._alpha)
            result.append(_l2_norm(mixed))

        return result
