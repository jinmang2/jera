"""API request/response schemas (API-only; no domain logic)."""

from __future__ import annotations

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


class QueryResponse(BaseModel):
    query: str
    text: str
    citations: list[CitationOut]
