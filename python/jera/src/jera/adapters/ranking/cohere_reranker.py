"""Cohere Rerank adapter — DISABLED by default (extra: cloud, paid)."""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.retrieval import ScoredChunk


class CohereReranker:
    def __init__(
        self,
        model: str = "rerank-v3.5",
        api_key: str | None = None,
        enabled: bool = False,
    ) -> None:
        if not enabled:
            raise RuntimeError(
                "CohereReranker is disabled by default. Pass enabled=True and an api_key "
                "(paid live calls; never enabled in automated tests)."
            )
        if not api_key:
            raise RuntimeError("CohereReranker requires an api_key when enabled.")
        try:
            import cohere
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "CohereReranker requires the 'cloud' extra: `uv sync --extra cloud`."
            ) from exc
        self._client = cohere.Client(api_key)
        self._model = model
        self.model_id = model

    def rerank(  # pragma: no cover
        self, query: str, candidates: Sequence[ScoredChunk], top_k: int
    ) -> list[ScoredChunk]:
        docs = [c.chunk.text if c.chunk else "" for c in candidates]
        resp = self._client.rerank(query=query, documents=docs, model=self._model, top_n=top_k)
        out: list[ScoredChunk] = []
        for r in resp.results:
            c = candidates[r.index]
            out.append(
                c.model_copy(
                    update={
                        "score": float(r.relevance_score),
                        "components": {**c.components, "rerank": float(r.relevance_score)},
                    }
                )
            )
        return out
