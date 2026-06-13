"""Qdrant vector store adapter (extra: qdrant).

Production target: named dense + sparse vectors, RRF/DBSF prefetch fusion, payload pointing
back to the metadata store. Marked parity-unverified until the Qdrant-integration milestone,
which adds a live cross-adapter equivalence test against the in-memory store's fusion output.
Construction is lazy so the base install stays Qdrant-free.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.retrieval import FusionMethod, ScoredChunk
from jera.domain.vectors import DenseVector, SparseVector
from jera.ports.vector_store import CollectionSpec, Distance, VectorRecord

PARITY_VERIFIED = False  # flipped to True only after the live equivalence test exists


class QdrantVectorStore:
    def __init__(self, url: str = "http://localhost:6333", api_key: str | None = None) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "QdrantVectorStore requires the 'qdrant' extra: `uv sync --extra qdrant`."
            ) from exc
        self._client = QdrantClient(url=url, api_key=api_key)

    def ensure_collection(self, spec: CollectionSpec) -> None:
        from qdrant_client import models

        distance = (
            models.Distance.COSINE if spec.distance is Distance.COSINE else models.Distance.DOT
        )
        self._client.recreate_collection(
            collection_name=spec.name,
            vectors_config={"dense": models.VectorParams(size=spec.dense_dim, distance=distance)},
            sparse_vectors_config=(
                {"sparse": models.SparseVectorParams()} if spec.has_sparse else None
            ),
        )

    def upsert(self, collection: str, records: Sequence[VectorRecord]) -> None:
        from qdrant_client import models

        points = []
        for rec in records:
            vectors: dict[str, object] = {}
            if rec.dense is not None:
                vectors["dense"] = rec.dense
            if rec.sparse is not None:
                vectors["sparse"] = models.SparseVector(
                    indices=rec.sparse.indices, values=rec.sparse.values
                )
            points.append(
                models.PointStruct(
                    id=rec.chunk_id,
                    vector=vectors,
                    payload={"document_id": rec.document_id, **rec.payload},
                )
            )
        self._client.upsert(collection_name=collection, points=points)

    def delete(self, collection: str, chunk_ids: Sequence[str]) -> None:  # pragma: no cover
        """Delete points by chunk_id from a Qdrant collection.

        Uses ``QdrantClient.delete`` with a ``PointIdsList`` selector.
        Idempotent: Qdrant ignores ids that do not exist.
        """
        from qdrant_client import models

        self._client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=list(chunk_ids)),
        )

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
        from qdrant_client import models

        prefetch = []
        if dense is not None:
            prefetch.append(models.Prefetch(query=dense, using="dense", limit=prefetch_limit))
        if sparse is not None:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                    using="sparse",
                    limit=prefetch_limit,
                )
            )
        fusion_enum = models.Fusion.RRF if fusion is FusionMethod.RRF else models.Fusion.DBSF
        result = self._client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=fusion_enum),
            limit=top_k,
            with_payload=True,
        )
        return [ScoredChunk(chunk_id=str(p.id), score=float(p.score)) for p in result.points]
