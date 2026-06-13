"""Context-engineering adapters: reorder, curate, and compress retrieved chunks.

Pipeline position:  retrieve → rerank → reorder → curate → compress → generate

Each adapter implements the :class:`~jera.ports.context_processor.ContextProcessor`
Protocol and can be composed in any order via a plain list comprehension over the
retrieved :class:`~jera.domain.chunk.Chunk` objects.
"""

from __future__ import annotations

from jera.adapters.context.compressor import ExtractiveCompressor
from jera.adapters.context.curator import RedundancyCurator
from jera.adapters.context.reorderer import LostInTheMiddleReorderer

__all__ = [
    "ExtractiveCompressor",
    "LostInTheMiddleReorderer",
    "RedundancyCurator",
]
