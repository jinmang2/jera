"""Pure domain models — no IO, no vendor SDKs."""

from jera.domain.answer import Answer, Citation
from jera.domain.chunk import Chunk
from jera.domain.document import (
    DocumentElement,
    ElementType,
    MediaType,
    PageSpan,
    ParsedDocument,
    Provenance,
    SourceRef,
)
from jera.domain.ids import stable_id
from jera.domain.jobs import IngestionJob, JobStatus, ProviderConfigSnapshot
from jera.domain.retrieval import (
    FusionMethod,
    Query,
    RetrievalMode,
    RetrievalResult,
    ScoredChunk,
)
from jera.domain.vectors import DenseVector, SparseVector

__all__ = [
    "Answer",
    "Citation",
    "Chunk",
    "DenseVector",
    "DocumentElement",
    "ElementType",
    "FusionMethod",
    "IngestionJob",
    "JobStatus",
    "MediaType",
    "PageSpan",
    "ParsedDocument",
    "Provenance",
    "ProviderConfigSnapshot",
    "Query",
    "RetrievalMode",
    "RetrievalResult",
    "ScoredChunk",
    "SourceRef",
    "SparseVector",
    "stable_id",
]
