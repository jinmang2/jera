"""MultiVectorStore port — late-interaction MaxSim retrieval.

Stores per-chunk token-vector matrices and scores candidates by the ColBERT MaxSim
formula:  score(q, d) = Σ_{qi ∈ q} max_{dj ∈ d} cosine(qi, dj).

The opt-in production adapter would wrap a PLAID/ColBERTv2 index; this port defines
the interface any such adapter must satisfy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jera.domain.retrieval import ScoredChunk


@runtime_checkable
class MultiVectorStore(Protocol):
    """Stores and retrieves chunks via token-level MaxSim scoring.

    Each chunk is stored as a matrix of per-token dense vectors.  At query time every
    query token is matched against every document token for its chunk; the per-query-token
    contribution is the *maximum* cosine similarity over all document tokens, and the
    chunk score is the *sum* of those per-query-token maxima (ColBERT MaxSim).
    """

    def add(self, items: Sequence[tuple[str, list[list[float]]]]) -> None:
        """Index chunks by their token-vector matrices.

        Args:
            items: Sequence of ``(chunk_id, token_vectors)`` pairs where
                   ``token_vectors`` has shape ``[n_tokens, dimensions]``.
        """
        ...

    def search_maxsim(
        self,
        query_vectors: list[list[float]],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Return the top-k chunks ranked by MaxSim score, descending.

        Ties are broken by ``chunk_id`` ascending.

        Args:
            query_vectors: Per-token query matrix, shape ``[n_query_tokens, dimensions]``.
            top_k:         Number of results to return.

        Returns:
            Up to ``top_k`` :class:`~jera.domain.retrieval.ScoredChunk` objects with
            ``components={"maxsim": score}``.
        """
        ...

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove chunks from the index.  No-op for unknown ids."""
        ...
