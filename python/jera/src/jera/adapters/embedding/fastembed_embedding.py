"""fastembed (ONNX) dense embedding — LOCAL dev default (extra: local).

Lazy import keeps the base install torch-free. Model downloads on first use, so this is the
`local` profile default and is skipped in default CI.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.vectors import DenseVector

# bge-small-en-v1.5 → 384 dims; a strong, light ONNX retrieval encoder.
# bge-m3             → 1024 dims; multilingual (Korean/EN/…); local profile default.
# multilingual-e5-large → 1024 dims; S0 fallback if bge-m3 is absent.
_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-m3": 1024,
    "intfloat/multilingual-e5-large": 1024,
}


class FastEmbedEmbedding:
    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "FastEmbedEmbedding requires the 'local' extra: `uv sync --extra local`."
            ) from exc
        self._model = TextEmbedding(model_name=model_name)
        self.model_id = model_name
        self.dimensions = _DIMS.get(model_name, 384)
        self.context_limit: int | None = 512

    def embed(self, texts: Sequence[str]) -> list[DenseVector]:  # pragma: no cover
        return [list(map(float, v)) for v in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> DenseVector:  # pragma: no cover
        return next(iter(self.embed([text])))
