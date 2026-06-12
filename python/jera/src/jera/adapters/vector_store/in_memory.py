"""In-memory vector store — dev/test default.

Implements dense (cosine), sparse (dot), and hybrid (RRF/DBSF) search with the SAME request
shape as the Qdrant adapter. Parity with Qdrant is a goal, not an M1 guarantee; the fusion
math is pinned by fusion.py + a golden-file contract test.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from jera.adapters.vector_store.fusion import (
    distribution_based_score_fusion,
    reciprocal_rank_fusion,
)
from jera.domain.retrieval import FusionMethod, ScoredChunk
from jera.domain.vectors import DenseVector, SparseVector
from jera.ports.vector_store import CollectionSpec, VectorRecord


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._specs: dict[str, CollectionSpec] = {}
        self._records: dict[str, dict[str, VectorRecord]] = {}

    def ensure_collection(self, spec: CollectionSpec) -> None:
        self._specs[spec.name] = spec
        self._records.setdefault(spec.name, {})

    def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        if collection not in self._records:
            raise KeyError(f"unknown collection {collection!r}; call ensure_collection first")
        store = self._records[collection]
        for rec in records:
            store[rec.chunk_id] = rec

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
        if dense is None and sparse is None:
            raise ValueError("search requires at least one of dense/sparse")
        if collection not in self._specs:
            return []  # nothing ingested yet → empty result (defined empty-index behavior)
        spec = self._specs[collection]
        records = list(self._records[collection].values())
        if not records:
            return []

        if dense is not None and len(dense) != spec.dense_dim:
            raise ValueError(
                f"query dense dim {len(dense)} != collection dim {spec.dense_dim} "
                f"for {collection!r}; a model/dimension change requires re-indexing"
            )

        dense_scores = self._dense_scores(records, dense) if dense is not None else None
        sparse_scores = self._sparse_scores(records, sparse) if sparse is not None else None

        # Single-modality paths.
        if dense_scores is not None and sparse_scores is None:
            return self._top(dense_scores, top_k, component="dense")
        if sparse_scores is not None and dense_scores is None:
            return self._top(sparse_scores, top_k, component="sparse")

        # Hybrid: prefetch top-N per modality, then fuse.
        assert dense_scores is not None and sparse_scores is not None
        dense_pref = _prefetch(dense_scores, prefetch_limit)
        sparse_pref = _prefetch(sparse_scores, prefetch_limit)

        if fusion is FusionMethod.RRF:
            fused = reciprocal_rank_fusion(
                {
                    "dense": [cid for cid, _ in dense_pref],
                    "sparse": [cid for cid, _ in sparse_pref],
                }
            )
        else:
            fused = distribution_based_score_fusion(
                {"dense": dict(dense_pref), "sparse": dict(sparse_pref)}
            )

        dense_map = dict(dense_pref)
        sparse_map = dict(sparse_pref)
        out: list[ScoredChunk] = []
        for chunk_id, score in fused[:top_k]:
            out.append(
                ScoredChunk(
                    chunk_id=chunk_id,
                    score=score,
                    components={
                        "dense": dense_map.get(chunk_id, 0.0),
                        "sparse": sparse_map.get(chunk_id, 0.0),
                    },
                )
            )
        return out

    def _dense_scores(self, records: list[VectorRecord], query: DenseVector) -> dict[str, float]:
        qn = math.sqrt(sum(v * v for v in query))
        scores: dict[str, float] = {}
        for rec in records:
            if rec.dense is None:
                continue
            scores[rec.chunk_id] = _cosine(query, rec.dense, qn)
        return scores

    def _sparse_scores(self, records: list[VectorRecord], query: SparseVector) -> dict[str, float]:
        scores: dict[str, float] = {}
        for rec in records:
            if rec.sparse is None:
                continue
            scores[rec.chunk_id] = rec.sparse.dot(query)
        return scores

    def _top(self, scores: dict[str, float], top_k: int, *, component: str) -> list[ScoredChunk]:
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [ScoredChunk(chunk_id=cid, score=s, components={component: s}) for cid, s in ranked]


def _cosine(query: DenseVector, doc: DenseVector, qnorm: float) -> float:
    dot = sum(x * y for x, y in zip(query, doc, strict=True))
    dn = math.sqrt(sum(y * y for y in doc))
    if qnorm == 0 or dn == 0:
        return 0.0
    return dot / (qnorm * dn)


def _prefetch(scores: dict[str, float], limit: int) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
