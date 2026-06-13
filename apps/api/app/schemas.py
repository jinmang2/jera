"""API request/response schemas (API-only; no domain logic)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from jera.rag import FusionMethod, MediaType, RetrievalMode


class IngestRequest(BaseModel):
    source_id: str
    media_type: MediaType = MediaType.MARKDOWN
    text: str | None = Field(default=None, description="Inline text for text/markdown sources")
    content_b64: str | None = Field(default=None, description="Base64 bytes for binary sources")
    filename: str | None = None


class IngestResponse(BaseModel):
    job_id: str
    status: str
    document_id: str | None
    chunk_count: int


class CitationOut(BaseModel):
    chunk_id: str
    document_id: str
    snippet: str
    score: float
    page_span: tuple[int, int]
    section_path: tuple[str, ...]


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: RetrievalMode = RetrievalMode.HYBRID
    fusion: FusionMethod = FusionMethod.RRF


class QueryStatsOut(BaseModel):
    timings_ms: dict[str, float]
    estimated_cost_usd: float
    model_ids: dict[str, str]


class QueryResponse(BaseModel):
    query: str
    text: str
    citations: list[CitationOut]
    stats: QueryStatsOut | None = None


class JobResponse(BaseModel):
    job_id: str
    source_id: str
    status: str
    document_id: str | None
    chunk_count: int
    error: str | None
    created_at: datetime
    updated_at: datetime


class DocumentInfoOut(BaseModel):
    document_id: str
    source_id: str
    title: str | None
    chunk_count: int


class DeleteResponse(BaseModel):
    document_id: str
    deleted_chunk_count: int
