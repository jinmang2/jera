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
    context: str | None = None  # contextual-retrieval situating prefix (Anthropic, 2024)

    @property
    def embedding_text(self) -> str:
        """Text used for dense embedding and sparse (BM25) indexing.

        With Contextual Retrieval enabled, the situating ``context`` is prepended so the
        chunk is findable by queries naming entities it never repeats. ``text`` itself stays
        the original chunk content — citations and snippets always quote ``text``, never the
        synthesized context — so provenance (char_span/element_ids) is unaffected.
        """
        if self.context:
            return f"{self.context}\n\n{self.text}"
        return self.text
