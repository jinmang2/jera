"""Identity reranker — TEST default.

Deterministic, score-stable: preserves the first-stage ordering (score desc, chunk_id asc)
and truncates to top_k. It does not invent semantic reordering — honest about being a
passthrough so retrieval-gate results reflect the retriever/fusion, not a fake rerank.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.retrieval import ScoredChunk


class IdentityReranker:
    model_id = "identity-rerank-v1"

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        ordered = sorted(candidates, key=lambda c: (-c.score, c.chunk_id))[:top_k]
        return [
            c.model_copy(update={"components": {**c.components, "rerank": c.score}})
            for c in ordered
        ]
