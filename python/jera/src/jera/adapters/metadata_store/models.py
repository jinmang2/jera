"""SQLAlchemy 2.0 ORM models — shared schema for SQLite (dev/test) and Postgres (prod)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(256), index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    provenance: Mapped[dict[str, object]] = mapped_column(JSON)


class ChunkRow(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("documents.document_id"), index=True
    )
    source_id: Mapped[str] = mapped_column(String(256), index=True)
    text: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int] = mapped_column(Integer)
    page_end: Mapped[int] = mapped_column(Integer)
    section_path: Mapped[list[str]] = mapped_column(JSON)
    element_ids: Mapped[list[str]] = mapped_column(JSON)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    token_count: Mapped[int] = mapped_column(Integer)
    chunk_strategy: Mapped[str] = mapped_column(String(64))
    chunk_version: Mapped[str] = mapped_column(String(32))
    parent_chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    context: Mapped[str | None] = mapped_column(Text, nullable=True)  # contextual-retrieval prefix


class JobRow(Base):
    __tablename__ = "ingestion_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(256), index=True)
    status: Mapped[str] = mapped_column(String(16))
    document_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ConfigSnapshotRow(Base):
    __tablename__ = "config_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile: Mapped[str] = mapped_column(String(32))
    embedding_model_id: Mapped[str] = mapped_column(String(128))
    embedding_dimensions: Mapped[int] = mapped_column(Integer)
    embedding_context_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sparse_model_id: Mapped[str] = mapped_column(String(128))
    reranker_model_id: Mapped[str] = mapped_column(String(128))
    generator_model_id: Mapped[str] = mapped_column(String(128))
    cost_metadata: Mapped[dict[str, object]] = mapped_column(JSON, default=dict)
    version: Mapped[str] = mapped_column(String(16), default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
