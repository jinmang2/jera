"""Quantized in-memory vector store — int8 candidate retrieval + float32 rescore.

Two-stage dense retrieval (MRL-compatible):
  1. Score all docs with int8 dot-product arithmetic → oversized candidate set
     of size ``top_k * rescore_multiplier``.
  2. Rescore only those candidates with exact float32 cosine similarity.
  3. Return top_k by float32 score.

Sparse / hybrid / delete / ensure_collection / upsert delegate to an internal
``InMemoryVectorStore`` instance for correctness and DRY-ness.  The dimension
guard and frozen-record semantics are identical to ``InMemoryVectorStore``.

Quantization scheme: scale each coordinate by 127 / max_abs(vec), round to
nearest int, clamp to [-127, 127].  The int8 dot-product is therefore an
approximation of the true cosine score (exact when both query and document
norms are equal after quantization).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from jera.adapters.vector_store.in_memory import InMemoryVectorStore, _cosine
from jera.domain.retrieval import FusionMethod, ScoredChunk
from jera.domain.vectors import DenseVector, SparseVector
from jera.ports.vector_store import CollectionSpec, VectorRecord

# A quantized vector: list of ints in [-127, 127].
_Int8Vec = list[int]


def _quantize(vec: DenseVector) -> _Int8Vec:
    """Scale vec by 127 / max_abs, round, clamp to [-127, 127]."""
    max_abs = max(abs(v) for v in vec) if vec else 0.0
    if max_abs == 0.0:
        return [0] * len(vec)
    scale = 127.0 / max_abs
    return [max(-127, min(127, round(v * scale))) for v in vec]


def _int8_dot(a: _Int8Vec, b: _Int8Vec) -> int:
    """Integer dot product of two int8 vectors."""
    return sum(x * y for x, y in zip(a, b, strict=True))


class QuantizedInMemoryVectorStore:
    """VectorStore with int8 first-pass + float32 rescore for dense search.

    Parameters
    ----------
    rescore_multiplier:
        Candidate set size = ``top_k * rescore_multiplier``.  Must be >= 1.
        Higher values trade speed for recall fidelity.  Default: 4.
    """

    def __init__(self, rescore_multiplier: int = 4) -> None:
        if rescore_multiplier < 1:
            raise ValueError("rescore_multiplier must be >= 1")
        self._rescore_multiplier = rescore_multiplier
        # Delegate sparse / hybrid / collection management to InMemoryVectorStore.
        self._inner = InMemoryVectorStore()
        # Parallel int8 storage: collection → chunk_id → quantized dense vector.
        self._int8: dict[str, dict[str, _Int8Vec]] = {}
        # float32 originals for rescore: collection → chunk_id → DenseVector.
        self._float32: dict[str, dict[str, DenseVector]] = {}

    # ------------------------------------------------------------------
    # VectorStore protocol
    # ------------------------------------------------------------------

    def ensure_collection(self, spec: CollectionSpec) -> None:
        """Register a collection; idempotent."""
        self._inner.ensure_collection(spec)
        self._int8.setdefault(spec.name, {})
        self._float32.setdefault(spec.name, {})

    def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        """Upsert records; also stores int8 + float32 copies of dense vectors."""
        self._inner.upsert(collection, records)
        int8_col = self._int8[collection]
        f32_col = self._float32[collection]
        for rec in records:
            if rec.dense is not None:
                int8_col[rec.chunk_id] = _quantize(rec.dense)
                f32_col[rec.chunk_id] = rec.dense
            else:
                int8_col.pop(rec.chunk_id, None)
                f32_col.pop(rec.chunk_id, None)

    def delete(self, collection: str, chunk_ids: Sequence[str]) -> None:
        """Remove chunk_ids from collection.  No-op for unknown collection or missing ids."""
        self._inner.delete(collection, chunk_ids)
        int8_col = self._int8.get(collection)
        f32_col = self._float32.get(collection)
        for cid in chunk_ids:
            if int8_col is not None:
                int8_col.pop(cid, None)
            if f32_col is not None:
                f32_col.pop(cid, None)

    def search(
        self,
        collection: str,
        *,
        dense: DenseVector | None = None,
        sparse: SparseVector | None = None,
        top_k: int = 10,
        fusion: FusionMethod = FusionMethod.RRF,
        prefetch_limit: int = 100,
    ) -> list[ScoredChunk]:
        """Dense-only: two-stage int8 → float32 rescore.

        Sparse-only and hybrid paths delegate to the inner InMemoryVectorStore
        (no quantization benefit there).

        Raises ValueError if the dense query dimension mismatches the collection.
        """
        if dense is None and sparse is None:
            raise ValueError("search requires at least one of dense/sparse")

        # Delegate sparse-only and hybrid paths to inner store (handles dim guard too).
        if dense is None or sparse is not None:
            return self._inner.search(
                collection,
                dense=dense,
                sparse=sparse,
                top_k=top_k,
                fusion=fusion,
                prefetch_limit=prefetch_limit,
            )

        # Dense-only: apply dimension guard manually (mirrors InMemoryVectorStore).
        specs = self._inner._specs
        if collection not in specs:
            return []
        spec = specs[collection]
        if len(dense) != spec.dense_dim:
            raise ValueError(
                f"query dense dim {len(dense)} != collection dim {spec.dense_dim} "
                f"for {collection!r}; a model/dimension change requires re-indexing"
            )

        int8_col = self._int8.get(collection, {})
        f32_col = self._float32.get(collection, {})
        if not int8_col:
            return []

        # --- Stage 1: int8 dot-product over all docs → oversized candidate set ---
        q_int8 = _quantize(dense)
        candidate_size = top_k * self._rescore_multiplier
        int8_scores: list[tuple[str, int]] = [
            (cid, _int8_dot(q_int8, doc_int8)) for cid, doc_int8 in int8_col.items()
        ]
        # Sort descending by int8 score (stable: break ties by chunk_id for determinism).
        int8_scores.sort(key=lambda kv: (-kv[1], kv[0]))
        candidates = int8_scores[:candidate_size]

        # --- Stage 2: exact float32 cosine rescore over candidates only ---
        qnorm = math.sqrt(sum(v * v for v in dense))
        rescored: list[tuple[str, float]] = []
        for cid, _ in candidates:
            doc_f32 = f32_col.get(cid)
            if doc_f32 is None:
                continue
            rescored.append((cid, _cosine(dense, doc_f32, qnorm)))

        rescored.sort(key=lambda kv: (-kv[1], kv[0]))
        top = rescored[:top_k]

        return [
            ScoredChunk(chunk_id=cid, score=score, components={"dense": score})
            for cid, score in top
        ]
