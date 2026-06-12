"""ParserRegistry — dispatches a SourceRef to the first parser that supports it."""

from __future__ import annotations

from jera.domain.document import ParsedDocument, SourceRef
from jera.ports.parser import DocumentParser


class ParserRegistry:
    def __init__(self, parsers: list[DocumentParser]) -> None:
        self._parsers = parsers

    def parse(self, source: SourceRef) -> ParsedDocument:
        for parser in self._parsers:
            if parser.supports(source):
                return parser.parse(source)
        raise ValueError(f"no parser supports media type {source.media_type!r}")
