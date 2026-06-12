"""fastembed SPLADE/BM25 sparse provider (extra: local).

Note the sparse↔fusion coupling (see ADR): SPLADE produces learned logit weights whose
distribution differs from BM25 counts, so swapping this in changes DBSF output even with
identical rankings. RRF (the default fusion) is robust to this; DBSF must be re-checked.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.vectors import SparseVector

_DEFAULT_MODEL = "prithivida/Splade_PP_en_v1"


class FastEmbedSparse:
    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        try:
            from fastembed import SparseTextEmbedding
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "FastEmbedSparse requires the 'local' extra: `uv sync --extra local`."
            ) from exc
        self._model = SparseTextEmbedding(model_name=model_name)
        self.model_id = model_name

    def encode(self, texts: Sequence[str]) -> list[SparseVector]:
        out: list[SparseVector] = []
        for emb in self._model.embed(list(texts)):
            out.append(
                SparseVector(
                    indices=[int(i) for i in emb.indices],
                    values=[float(v) for v in emb.values],
                )
            )
        return out

    def encode_query(self, text: str) -> SparseVector:
        return next(iter(self.encode([text])))
