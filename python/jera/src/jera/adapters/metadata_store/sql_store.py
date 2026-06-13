"""SqlMetadataStore — engine-agnostic SQLAlchemy implementation of the MetadataStore port.

Used by both the SQLite (dev/test) and Postgres (prod) adapters; only the engine differs.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import Engine, delete, func, select
from sqlalchemy.orm import Session

from jera.adapters.metadata_store.models import (
    Base,
    ChunkRow,
    ConfigSnapshotRow,
    DocumentRow,
    JobRow,
)
from jera.domain.chunk import Chunk
from jera.domain.document import DocumentInfo, PageSpan, ParsedDocument
from jera.domain.jobs import IngestionJob, JobStatus, ProviderConfigSnapshot


class SqlMetadataStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def init_schema(self) -> None:
        Base.metadata.create_all(self._engine)

    # --- documents ---
    def save_document(self, document: ParsedDocument) -> None:
        """Upsert a document row."""
        with Session(self._engine) as session:
            session.merge(
                DocumentRow(
                    document_id=document.document_id,
                    source_id=document.source_id,
                    title=document.title,
                    provenance=document.provenance.model_dump(mode="json"),
                )
            )
            session.commit()

    def list_documents(self) -> list[DocumentInfo]:
        """Return all documents with their chunk counts, ordered by document_id."""
        with Session(self._engine) as session:
            rows = session.execute(
                select(
                    DocumentRow.document_id,
                    DocumentRow.source_id,
                    DocumentRow.title,
                    func.count(ChunkRow.chunk_id).label("chunk_count"),
                )
                .outerjoin(ChunkRow, ChunkRow.document_id == DocumentRow.document_id)
                .group_by(DocumentRow.document_id)
                .order_by(DocumentRow.document_id)
            ).all()
            return [
                DocumentInfo(
                    document_id=row.document_id,
                    source_id=row.source_id,
                    title=row.title,
                    chunk_count=row.chunk_count,
                )
                for row in rows
            ]

    def get_document_info(self, document_id: str) -> DocumentInfo | None:
        """Return a single document's info with chunk count, or None if not found."""
        with Session(self._engine) as session:
            row = session.execute(
                select(
                    DocumentRow.document_id,
                    DocumentRow.source_id,
                    DocumentRow.title,
                    func.count(ChunkRow.chunk_id).label("chunk_count"),
                )
                .outerjoin(ChunkRow, ChunkRow.document_id == DocumentRow.document_id)
                .where(DocumentRow.document_id == document_id)
                .group_by(DocumentRow.document_id)
            ).first()
            if row is None:
                return None
            return DocumentInfo(
                document_id=row.document_id,
                source_id=row.source_id,
                title=row.title,
                chunk_count=row.chunk_count,
            )

    def delete_document(self, document_id: str) -> list[str]:
        """Delete the document and all its chunks; return the deleted chunk_ids.

        Idempotent: returns [] if the document_id is unknown.
        """
        with Session(self._engine) as session:
            chunk_ids: list[str] = list(
                session.scalars(
                    select(ChunkRow.chunk_id)
                    .where(ChunkRow.document_id == document_id)
                    .order_by(ChunkRow.chunk_id)
                ).all()
            )
            session.execute(delete(ChunkRow).where(ChunkRow.document_id == document_id))
            session.execute(delete(DocumentRow).where(DocumentRow.document_id == document_id))
            session.commit()
        return chunk_ids

    def chunk_ids_for_document(self, document_id: str) -> list[str]:
        """Return ordered chunk_ids belonging to a document."""
        with Session(self._engine) as session:
            return list(
                session.scalars(
                    select(ChunkRow.chunk_id)
                    .where(ChunkRow.document_id == document_id)
                    .order_by(ChunkRow.chunk_id)
                ).all()
            )

    def document_id_for_source(self, source_id: str) -> str | None:
        """Return the document_id currently stored for a source_id, or None."""
        with Session(self._engine) as session:
            return session.scalars(
                select(DocumentRow.document_id).where(DocumentRow.source_id == source_id)
            ).first()

    # --- chunks ---
    def save_chunks(self, chunks: Sequence[Chunk]) -> None:
        with Session(self._engine) as session:
            for c in chunks:
                session.merge(
                    ChunkRow(
                        chunk_id=c.chunk_id,
                        document_id=c.document_id,
                        source_id=c.source_id,
                        text=c.text,
                        page_start=c.page_span.start_page,
                        page_end=c.page_span.end_page,
                        section_path=list(c.section_path),
                        element_ids=list(c.element_ids),
                        char_start=c.char_span[0],
                        char_end=c.char_span[1],
                        token_count=c.token_count,
                        chunk_strategy=c.chunk_strategy,
                        chunk_version=c.chunk_version,
                        parent_chunk_id=c.parent_chunk_id,
                        context=c.context,
                    )
                )
            session.commit()

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with Session(self._engine) as session:
            row = session.get(ChunkRow, chunk_id)
            return _to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        with Session(self._engine) as session:
            rows = session.scalars(
                select(ChunkRow).where(ChunkRow.chunk_id.in_(list(chunk_ids)))
            ).all()
            by_id = {r.chunk_id: _to_chunk(r) for r in rows}
        # preserve caller order
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    # --- jobs ---
    def save_job(self, job: IngestionJob) -> None:
        with Session(self._engine) as session:
            session.merge(
                JobRow(
                    job_id=job.job_id,
                    source_id=job.source_id,
                    status=job.status.value,
                    document_id=job.document_id,
                    chunk_count=job.chunk_count,
                    error=job.error,
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                )
            )
            session.commit()

    def get_job(self, job_id: str) -> IngestionJob | None:
        with Session(self._engine) as session:
            row = session.get(JobRow, job_id)
            if not row:
                return None
            return IngestionJob(
                job_id=row.job_id,
                source_id=row.source_id,
                status=JobStatus(row.status),
                document_id=row.document_id,
                chunk_count=row.chunk_count,
                error=row.error,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )

    # --- config snapshots ---
    def save_config_snapshot(self, snapshot: ProviderConfigSnapshot) -> None:
        with Session(self._engine) as session:
            session.merge(
                ConfigSnapshotRow(
                    snapshot_id=snapshot.snapshot_id,
                    profile=snapshot.profile,
                    embedding_model_id=snapshot.embedding_model_id,
                    embedding_dimensions=snapshot.embedding_dimensions,
                    embedding_context_limit=snapshot.embedding_context_limit,
                    sparse_model_id=snapshot.sparse_model_id,
                    reranker_model_id=snapshot.reranker_model_id,
                    generator_model_id=snapshot.generator_model_id,
                    cost_metadata=snapshot.cost_metadata,
                    version=snapshot.version,
                    created_at=snapshot.created_at,
                )
            )
            session.commit()

    def latest_config_snapshot(self) -> ProviderConfigSnapshot | None:
        with Session(self._engine) as session:
            row = session.scalars(
                select(ConfigSnapshotRow).order_by(ConfigSnapshotRow.created_at.desc())
            ).first()
            if not row:
                return None
            return ProviderConfigSnapshot(
                snapshot_id=row.snapshot_id,
                profile=row.profile,
                embedding_model_id=row.embedding_model_id,
                embedding_dimensions=row.embedding_dimensions,
                embedding_context_limit=row.embedding_context_limit,
                sparse_model_id=row.sparse_model_id,
                reranker_model_id=row.reranker_model_id,
                generator_model_id=row.generator_model_id,
                cost_metadata=dict(row.cost_metadata or {}),
                version=row.version,
                created_at=row.created_at,
            )


def _to_chunk(row: ChunkRow) -> Chunk:
    return Chunk(
        chunk_id=row.chunk_id,
        document_id=row.document_id,
        source_id=row.source_id,
        text=row.text,
        page_span=PageSpan(start_page=row.page_start, end_page=row.page_end),
        section_path=tuple(row.section_path),
        element_ids=tuple(row.element_ids),
        char_span=(row.char_start, row.char_end),
        token_count=row.token_count,
        chunk_strategy=row.chunk_strategy,
        chunk_version=row.chunk_version,
        parent_chunk_id=row.parent_chunk_id,
        context=row.context,
    )
