"""Public RAG facade — the stable import surface for consumers (API, workers, CLI, eval).

Domain logic lives in ``jera.*`` subpackages; this module re-exports the pieces that
external adapters should depend on. ``apps/api/app`` imports from here, never from internals.
"""

from jera.config import Profile, RagSystem, Settings, build_system
from jera.domain import (
    Answer,
    Chunk,
    Citation,
    FusionMethod,
    IngestionJob,
    ParsedDocument,
    Query,
    RetrievalMode,
    ScoredChunk,
    SourceRef,
)
from jera.domain.document import MediaType
from jera.pipeline import IngestPipeline, QueryPipeline

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "FusionMethod",
    "IngestPipeline",
    "IngestionJob",
    "MediaType",
    "ParsedDocument",
    "Profile",
    "Query",
    "QueryPipeline",
    "RagSystem",
    "RetrievalMode",
    "ScoredChunk",
    "Settings",
    "SourceRef",
    "build_system",
]
