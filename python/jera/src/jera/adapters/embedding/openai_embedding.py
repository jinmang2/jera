"""OpenAI embeddings adapter — DISABLED by default (extra: cloud, paid live calls).

Constructed only when explicitly enabled with an API key. Records model identity so a
dimension/model change forces re-indexing (storage gate). Never invoked in automated tests.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.vectors import DenseVector

_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class OpenAIEmbedding:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        enabled: bool = False,
        dimensions: int | None = None,
    ) -> None:
        if not enabled:
            raise RuntimeError(
                "OpenAIEmbedding is disabled by default. Pass enabled=True and an api_key "
                "to use it (paid live calls; never enabled in automated tests)."
            )
        if not api_key:
            raise RuntimeError("OpenAIEmbedding requires an api_key when enabled.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "OpenAIEmbedding requires the 'cloud' extra: `uv sync --extra cloud`."
            ) from exc
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self.model_id = model
        self.dimensions = dimensions or _DIMS.get(model, 1536)
        self.context_limit: int | None = 8192
        self._dim_override = dimensions

    def embed(self, texts: Sequence[str]) -> list[DenseVector]:  # pragma: no cover
        kwargs: dict[str, object] = {"model": self._model, "input": list(texts)}
        if self._dim_override:
            kwargs["dimensions"] = self._dim_override
        resp = self._client.embeddings.create(**kwargs)
        return [list(d.embedding) for d in resp.data]

    def embed_query(self, text: str) -> DenseVector:  # pragma: no cover
        return next(iter(self.embed([text])))
