"""GraphRetriever port.

Defines the contract for graph-based multi-hop retrieval.  Implementations
build an entity graph from indexed chunks and answer queries by propagating
signals across that graph (e.g. Personalized PageRank).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jera.domain.chunk import Chunk
from jera.domain.retrieval import ScoredChunk


@runtime_checkable
class GraphRetriever(Protocol):
    """Index chunks into an entity graph and retrieve via graph signal propagation.

    Lifecycle
    ---------
    1. Call :meth:`index` once (or incrementally) with all :class:`~jera.domain.chunk.Chunk`
       objects to populate the underlying entity graph.
    2. Call :meth:`retrieve` for each query; the implementation seeds a graph
       signal (e.g. Personalized PageRank) on the query entities and scores
       chunks by the aggregated signal over their contained entities.
    """

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Build (or update) the entity graph from *chunks*."""
        ...

    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]:
        """Return up to *top_k* :class:`~jera.domain.retrieval.ScoredChunk` objects.

        Chunks are ranked by their aggregated graph signal score.  When no
        query entities are found or the graph is empty, an empty list is
        returned rather than raising.
        """
        ...
