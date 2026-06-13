"""MaxSim in-memory vector store — late-interaction (ColBERT-style) retrieval.

Implements the ColBERT MaxSim scoring formula:

    score(q, d) = Σ_{qi ∈ q} max_{dj ∈ d} cosine(qi, dj)

Each query-token contributes its *maximum* cosine similarity over all document tokens;
the chunk score is the *sum* of those per-query-token maxima.  This lets a single highly
relevant token in a long document dominate its contribution without being diluted by
unrelated tokens — which is why MaxSim outperforms bag-of-words on partial-overlap cases.

The opt-in production adapter would wrap a PLAID/ColBERTv2 compressed index; this
adapter provides an exact, pure-Python reference implementation for CI and benchmarking.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from jera.domain.retrieval import ScoredChunk


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors.  Returns 0.0 for zero-norm inputs."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _maxsim_score(
    query_vecs: list[list[float]],
    doc_vecs: list[list[float]],
) -> float:
    """ColBERT MaxSim: Σ_qi  max_dj  cosine(qi, dj)."""
    total = 0.0
    for q_tok in query_vecs:
        best = max(_cosine(q_tok, d_tok) for d_tok in doc_vecs)
        total += best
    return total


class MaxSimVectorStore:
    """In-memory store that scores chunks with the ColBERT MaxSim formula.

    Thread-safety: not thread-safe (single-process CI use-case).

    The opt-in production adapter would wrap BGE-M3's ColBERT head with PLAID
    compression; this class provides the exact reference scoring for unit tests and
    offline benchmarking.
    """

    def __init__(self) -> None:
        # chunk_id -> token-vector matrix [n_tokens, dim]
        self._store: dict[str, list[list[float]]] = {}

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    def add(self, items: Sequence[tuple[str, list[list[float]]]]) -> None:
        """Index chunks by their token-vector matrices.

        Later ``add`` calls with the same ``chunk_id`` overwrite the previous entry
        (upsert semantics).

        Args:
            items: Sequence of ``(chunk_id, token_vectors)`` pairs.
        """
        for chunk_id, token_vecs in items:
            self._store[chunk_id] = list(token_vecs)

    def search_maxsim(
        self,
        query_vectors: list[list[float]],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Return the top-k chunks ranked by MaxSim score (descending).

        Ties are broken by ``chunk_id`` ascending (lexicographic) to ensure a
        deterministic, reproducible ranking.

        Args:
            query_vectors: Per-token query matrix ``[n_query_tokens, dimensions]``.
            top_k:         Maximum number of results to return.

        Returns:
            Up to ``top_k`` :class:`~jera.domain.retrieval.ScoredChunk` objects with
            ``components={"maxsim": score}``.
        """
        if not self._store or not query_vectors:
            return []

        scored: list[tuple[str, float]] = []
        for chunk_id, doc_vecs in self._store.items():
            if not doc_vecs:
                continue
            score = _maxsim_score(query_vectors, doc_vecs)
            scored.append((chunk_id, score))

        # Sort: score descending, chunk_id ascending on tie.
        scored.sort(key=lambda kv: (-kv[1], kv[0]))

        return [
            ScoredChunk(chunk_id=cid, score=s, components={"maxsim": s})
            for cid, s in scored[:top_k]
        ]

    def delete(self, chunk_ids: Sequence[str]) -> None:
        """Remove chunks from the index.  No-op for unknown ids."""
        for cid in chunk_ids:
            self._store.pop(cid, None)
