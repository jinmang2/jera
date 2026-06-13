"""ContextProcessor port — transforms an ordered list of chunks before generation.

Each adapter in ``adapters/context/`` implements this Protocol to apply one stage
of the context-engineering pipeline:

  retrieve → rerank → [reorder] → [curate] → [compress] → generate

The ``process`` method receives chunks in relevance order (best first, as produced
by the reranker) and returns a transformed list.  Order, count, or text content may
change depending on the adapter; the caller assembles the final generation context
from the returned list.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jera.domain.chunk import Chunk


@runtime_checkable
class ContextProcessor(Protocol):
    """Transforms an ordered sequence of chunks before they are assembled for generation."""

    name: str

    def process(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]:
        """Apply this processing stage.

        Parameters
        ----------
        query:
            The original retrieval query string.  Used by compression and curation
            stages; ignored by pure reordering stages.
        chunks:
            Chunks in relevance order (best first) as produced by the reranker.

        Returns
        -------
        list[Chunk]
            Transformed chunk list.  May differ in order, count, or ``text`` content
            from the input; ``chunk_id`` and all provenance fields are always preserved.
        """
        ...
