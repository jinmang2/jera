"""Chunker port."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jera.domain.chunk import Chunk
from jera.domain.document import ParsedDocument


@runtime_checkable
class Chunker(Protocol):
    """Splits a parsed document into retrievable chunks over its structured elements."""

    strategy: str
    version: str

    def chunk(self, document: ParsedDocument) -> list[Chunk]: ...
