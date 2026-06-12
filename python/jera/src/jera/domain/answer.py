"""Answer-side domain models: citations and generated answers."""

from __future__ import annotations

from pydantic import BaseModel


class Citation(BaseModel):
    """A pointer from an answer back to the chunk that supports it."""

    model_config = {"frozen": True}

    chunk_id: str
    document_id: str
    snippet: str
    score: float
    page_span: tuple[int, int]
    section_path: tuple[str, ...]


class Answer(BaseModel):
    """A generated answer with citations resolving to retrieved chunks."""

    model_config = {"frozen": True}

    query: str
    text: str
    citations: list[Citation]

    @property
    def is_empty(self) -> bool:
        return not self.citations and not self.text.strip()
