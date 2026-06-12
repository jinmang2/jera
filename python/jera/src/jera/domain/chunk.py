"""Chunk model with the full provenance metadata contract.

Principle (provenance everywhere): every chunk carries enough to cite back to the exact
source location and to reproduce/version the chunking that produced it.
"""

from __future__ import annotations

from pydantic import BaseModel

from jera.domain.document import PageSpan


class Chunk(BaseModel):
    """A retrievable unit of text plus its full provenance metadata."""

    model_config = {"frozen": True}

    chunk_id: str
    document_id: str
    source_id: str
    text: str
    page_span: PageSpan
    section_path: tuple[str, ...]
    element_ids: tuple[str, ...]
    char_span: tuple[int, int]  # [start, end) offsets within the section's concatenated text
    token_count: int
    chunk_strategy: str  # e.g. "heading_aware"
    chunk_version: str  # adapter version that produced this chunk
    parent_chunk_id: str | None = None  # for hierarchical strategies
