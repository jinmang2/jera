"""DocumentParser port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jera.domain.document import ParsedDocument, SourceRef


@runtime_checkable
class DocumentParser(Protocol):
    """Converts a raw source into a structured :class:`ParsedDocument`.

    Implementations MUST return typed elements with provenance, never flat text.
    """

    name: str
    version: str

    def supports(self, source: SourceRef) -> bool:
        """Whether this parser can handle the given source (by media type)."""
        ...

    def parse(self, source: SourceRef) -> ParsedDocument: ...
