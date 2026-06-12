"""Ingestion job and provider-config-snapshot records (owned by the metadata store)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _now() -> datetime:
    return datetime.now(UTC)


class IngestionJob(BaseModel):
    job_id: str
    source_id: str
    status: JobStatus = JobStatus.PENDING
    document_id: str | None = None
    chunk_count: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class ProviderConfigSnapshot(BaseModel):
    """A point-in-time record of the provider config used, for reproducibility.

    Persisted so that an embedding model/dimension change is detectable and forces a
    re-index (storage/vector gate).
    """

    snapshot_id: str
    profile: str
    embedding_model_id: str
    embedding_dimensions: int
    embedding_context_limit: int | None = None
    sparse_model_id: str
    reranker_model_id: str
    generator_model_id: str
    cost_metadata: dict[str, object] = {}  # per-provider list pricing (see config/pricing.py)
    version: str = "1"
    created_at: datetime = Field(default_factory=_now)
