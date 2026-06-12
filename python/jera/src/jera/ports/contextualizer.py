"""Contextualizer port.

Contextual Retrieval (Anthropic, 2024): before a chunk is embedded/indexed, prepend a short
*situating context* that names the document and section it came from, so a chunk that never
repeats an entity is still findable by a query for that entity. The port returns one context
string per input chunk, order-parallel; the ingest pipeline stores it on ``Chunk.context`` and
indexes ``Chunk.embedding_text`` (= context + text). Citations always quote ``Chunk.text``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jera.domain.chunk import Chunk
from jera.domain.document import ParsedDocument


@runtime_checkable
class Contextualizer(Protocol):
    """Produces a situating context string for each chunk of a parsed document."""

    strategy: str
    version: str

    def contextualize(self, document: ParsedDocument, chunks: Sequence[Chunk]) -> list[str]:
        """Return one context string per chunk (order-parallel to ``chunks``).

        An empty string means "no useful context" and leaves the chunk un-contextualized.
        """
        ...
