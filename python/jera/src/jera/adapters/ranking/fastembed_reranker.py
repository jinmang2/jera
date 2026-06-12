"""fastembed cross-encoder reranker — LOCAL dev (extra: local)."""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.retrieval import ScoredChunk

_DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class FastEmbedReranker:
    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "FastEmbedReranker requires the 'local' extra: `uv sync --extra local`."
            ) from exc
        self._model = TextCrossEncoder(model_name=model_name)
        self.model_id = model_name

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        docs = [c.chunk.text if c.chunk else "" for c in candidates]
        scores = list(self._model.rerank(query, docs))
        rescored = [
            c.model_copy(
                update={"score": float(s), "components": {**c.components, "rerank": float(s)}}
            )
            for c, s in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda c: (-c.score, c.chunk_id))[:top_k]
