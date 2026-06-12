"""Document-side domain models: sources, typed elements, parsed documents.

Principle (structure-first): a document is a typed element tree with provenance, never a
flat string. Chunking and citation depend on these types.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, model_validator


class MediaType(StrEnum):
    MARKDOWN = "text/markdown"
    PLAIN = "text/plain"
    PDF = "application/pdf"
    HTML = "text/html"


class ElementType(StrEnum):
    """Typed element kinds produced by parsers (a subset of the Unstructured/Docling space)."""

    TITLE = "Title"
    NARRATIVE_TEXT = "NarrativeText"
    LIST_ITEM = "ListItem"
    TABLE = "Table"
    FIGURE = "Figure"
    CODE = "Code"
    FORMULA = "Formula"
    PAGE_HEADER = "PageHeader"
    PAGE_FOOTER = "PageFooter"


class PageSpan(BaseModel):
    """1-based inclusive page range an element/chunk spans."""

    model_config = {"frozen": True}

    start_page: int
    end_page: int

    @model_validator(mode="after")
    def _check(self) -> PageSpan:
        if self.start_page < 1 or self.end_page < self.start_page:
            raise ValueError(f"invalid page span: {self.start_page}..{self.end_page}")
        return self

    @classmethod
    def single(cls, page: int) -> PageSpan:
        return cls(start_page=page, end_page=page)

    def merge(self, other: PageSpan) -> PageSpan:
        return PageSpan(
            start_page=min(self.start_page, other.start_page),
            end_page=max(self.end_page, other.end_page),
        )


class SourceRef(BaseModel):
    """A reference to an ingestible source — either an on-disk path or in-memory bytes."""

    source_id: str
    media_type: MediaType
    path: Path | None = None
    content: bytes | None = None
    filename: str | None = None

    @model_validator(mode="after")
    def _has_payload(self) -> SourceRef:
        if self.path is None and self.content is None:
            raise ValueError("SourceRef requires either `path` or `content`")
        return self

    def read_bytes(self) -> bytes:
        if self.content is not None:
            return self.content
        assert self.path is not None
        return self.path.read_bytes()

    def read_text(self, encoding: str = "utf-8") -> str:
        return self.read_bytes().decode(encoding)


class Provenance(BaseModel):
    """Where a parsed document came from and how it was produced."""

    model_config = {"frozen": True}

    source_id: str
    parser_name: str
    parser_version: str
    media_type: MediaType


class DocumentElement(BaseModel):
    """A single typed, positioned element with provenance back to the source."""

    model_config = {"frozen": True}

    element_id: str
    type: ElementType
    text: str
    page_span: PageSpan
    order: int  # reading order within the document, 0-based
    section_path: tuple[str, ...] = ()  # heading breadcrumb, e.g. ("Intro", "Goals")
    metadata: dict[str, object] = {}


class ParsedDocument(BaseModel):
    """The structured output of a parser: ordered typed elements + provenance."""

    document_id: str
    source_id: str
    title: str | None
    elements: list[DocumentElement]
    provenance: Provenance

    def element_by_id(self, element_id: str) -> DocumentElement | None:
        return next((e for e in self.elements if e.element_id == element_id), None)
