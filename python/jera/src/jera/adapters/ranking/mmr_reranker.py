"""MMR (Maximal Marginal Relevance) reranker.

Balances relevance and diversity using the greedy MMR algorithm (Carbonell & Goldstein 1998).
``lambda_`` interpolates between pure relevance (1.0) and pure diversity (0.0).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from jera.domain.retrieval import ScoredChunk
from jera.domain.vectors import DenseVector
from jera.ports.embedding import EmbeddingProvider


def _cosine(a: DenseVector, b: DenseVector) -> float:
    """Cosine similarity; returns 0.0 when either vector has zero norm."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class MMRReranker:
    """Greedy MMR reranker backed by an :class:`~jera.ports.embedding.EmbeddingProvider`.

    Candidates that carry no :attr:`~jera.domain.retrieval.ScoredChunk.chunk` cannot be
    embedded for diversity scoring and are appended after the MMR-selected set, ordered by
    ``(-score, chunk_id)``.
    """

    def __init__(
        self,
        embedding: EmbeddingProvider,
        *,
        lambda_: float = 0.7,
        model_id: str = "mmr-rerank-v1",
    ) -> None:
        if not (0.0 <= lambda_ <= 1.0):
            raise ValueError(f"lambda_ must be in [0, 1], got {lambda_!r}")
        self.embedding = embedding
        self.lambda_ = lambda_
        self.model_id = model_id

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        """Return up to *top_k* candidates reranked by MMR.

        Chunks without an attached :class:`~jera.domain.chunk.Chunk` are appended last,
        ordered by ``(-score, chunk_id)``.
        """
        # Split into embeddable and un-embeddable candidates.
        with_chunk = [c for c in candidates if c.chunk is not None]
        without_chunk = sorted(candidates, key=lambda c: (-c.score, c.chunk_id))
        without_chunk = [c for c in without_chunk if c.chunk is None]

        if not with_chunk:
            return list(without_chunk[:top_k])

        # Embed query and all embeddable candidate texts.
        query_vec: DenseVector = self.embedding.embed_query(query)
        chunk_texts = [c.chunk.embedding_text for c in with_chunk]  # type: ignore[union-attr]
        doc_vecs: list[DenseVector] = self.embedding.embed(chunk_texts)

        # Pre-compute relevance scores (cosine to query).
        relevances = [_cosine(query_vec, dv) for dv in doc_vecs]

        # Greedy MMR selection.
        remaining: list[int] = list(range(len(with_chunk)))
        selected_indices: list[int] = []
        selected_vecs: list[DenseVector] = []
        selected_chunks: list[ScoredChunk] = []

        while remaining and len(selected_chunks) < top_k:
            best_idx: int | None = None
            best_mmr: float = float("-inf")
            best_chunk_id: str = ""

            for i in remaining:
                rel = relevances[i]
                if selected_vecs:
                    max_sim = max(_cosine(doc_vecs[i], sv) for sv in selected_vecs)
                else:
                    max_sim = 0.0
                mmr_score = self.lambda_ * rel - (1.0 - self.lambda_) * max_sim

                # Tie-break: higher mmr score wins; then chunk_id ascending.
                if (
                    best_idx is None
                    or mmr_score > best_mmr
                    or (mmr_score == best_mmr and with_chunk[i].chunk_id < best_chunk_id)
                ):
                    best_idx = i
                    best_mmr = mmr_score
                    best_chunk_id = with_chunk[i].chunk_id

            assert best_idx is not None
            remaining.remove(best_idx)
            selected_indices.append(best_idx)
            selected_vecs.append(doc_vecs[best_idx])

            original = with_chunk[best_idx]
            updated = original.model_copy(
                update={
                    "components": {
                        **original.components,
                        "rerank": best_mmr,
                        "relevance": relevances[best_idx],
                    }
                }
            )
            selected_chunks.append(updated)

        # Append no-chunk candidates if budget remains.
        budget = top_k - len(selected_chunks)
        if budget > 0:
            selected_chunks.extend(without_chunk[:budget])

        return selected_chunks
