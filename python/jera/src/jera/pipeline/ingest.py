"""IngestPipeline: parse → chunk → (fit sparse) → embed → encode → store.

Vectors land in the VectorStore (dense+sparse named vectors, payload→chunk_id); documents,
chunks, and the job record land in the MetadataStore.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.adapters.parsing.registry import ParserRegistry
from jera.domain.chunk import Chunk
from jera.domain.document import SourceRef
from jera.domain.ids import stable_id
from jera.domain.jobs import IngestionJob, JobStatus
from jera.ports.chunker import Chunker
from jera.ports.contextualizer import Contextualizer
from jera.ports.embedding import EmbeddingProvider
from jera.ports.metadata_store import MetadataStore
from jera.ports.sparse import Fittable, SparseVectorProvider
from jera.ports.vector_store import CollectionSpec, VectorRecord, VectorStore


class IngestPipeline:
    def __init__(
        self,
        *,
        parsers: ParserRegistry,
        chunker: Chunker,
        embedding: EmbeddingProvider,
        sparse: SparseVectorProvider,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        collection: str,
        contextualizer: Contextualizer | None = None,
    ) -> None:
        self._parsers = parsers
        self._chunker = chunker
        self._embedding = embedding
        self._sparse = sparse
        self._vectors = vector_store
        self._meta = metadata_store
        self._collection = collection
        # Contextual Retrieval (Anthropic, 2024): when set, each chunk is situated with a
        # short context that is prepended for embedding/sparse indexing (Chunk.embedding_text).
        self._contextualizer = contextualizer

    def ingest(self, source: SourceRef) -> IngestionJob:
        return self.ingest_many([source])[0]

    def ingest_many(self, sources: Sequence[SourceRef]) -> list[IngestionJob]:
        """Ingest a batch. Sparse stats (BM25 idf) are fit once over the whole batch so the
        corpus statistics are consistent across the documents indexed together."""
        jobs: list[IngestionJob] = []
        all_chunks: list[Chunk] = []
        per_source: list[tuple[IngestionJob, list[Chunk]]] = []

        self._vectors.ensure_collection(
            CollectionSpec(
                name=self._collection,
                dense_dim=self._embedding.dimensions,
                has_sparse=True,
                embedding_model_id=self._embedding.model_id,
            )
        )

        for source in sources:
            job = IngestionJob(
                job_id=stable_id(source.source_id, "job"),
                source_id=source.source_id,
                status=JobStatus.RUNNING,
            )
            try:
                document = self._parsers.parse(source)
                # Idempotent re-ingest: a source's document_id is deterministic, so re-ingesting
                # drops any chunks/vectors from the prior ingest before re-indexing (otherwise
                # content edits would leave orphaned chunks with stale ids).
                stale_ids = self._meta.delete_document(document.document_id)
                if stale_ids:
                    self._vectors.delete(self._collection, stale_ids)
                self._meta.save_document(document)
                chunks = self._chunker.chunk(document)
                if self._contextualizer is not None and chunks:
                    contexts = self._contextualizer.contextualize(document, chunks)
                    chunks = [
                        c.model_copy(update={"context": ctx}) if ctx else c
                        for c, ctx in zip(chunks, contexts, strict=True)
                    ]
                job = job.model_copy(
                    update={"document_id": document.document_id, "chunk_count": len(chunks)}
                )
                per_source.append((job, chunks))
                all_chunks.extend(chunks)
            except Exception as exc:  # noqa: BLE001 - record failure on the job
                jobs.append(job.model_copy(update={"status": JobStatus.FAILED, "error": str(exc)}))
                self._meta.save_job(jobs[-1])
                per_source.append((jobs[-1], []))

        # Fit sparse provider once over the whole batch corpus (if it needs fitting).
        if isinstance(self._sparse, Fittable):
            self._sparse.fit([c.embedding_text for c in all_chunks])

        for job, chunks in per_source:
            if job.status is JobStatus.FAILED:
                continue
            if chunks:
                self._index_chunks(chunks)
                self._meta.save_chunks(chunks)
            done = job.model_copy(update={"status": JobStatus.SUCCEEDED})
            self._meta.save_job(done)
            jobs.append(done)

        # Preserve input order in the returned jobs.
        order = {s.source_id: i for i, s in enumerate(sources)}
        jobs.sort(key=lambda j: order.get(j.source_id, 0))
        return jobs

    def _index_chunks(self, chunks: list[Chunk]) -> None:
        # Index the contextualized text (context + chunk) for both dense and sparse — this is
        # the Contextual Embeddings + Contextual BM25 of Anthropic's recipe. Without a
        # contextualizer, embedding_text == text, so indexing is unchanged.
        texts = [c.embedding_text for c in chunks]
        dense = self._embedding.embed(texts)
        sparse = self._sparse.encode(texts)
        records = [
            VectorRecord(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                dense=d,
                sparse=s,
                payload={"document_id": c.document_id, "section_path": list(c.section_path)},
            )
            for c, d, s in zip(chunks, dense, sparse, strict=True)
        ]
        self._vectors.upsert(self._collection, records)
