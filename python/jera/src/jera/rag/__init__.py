"""Public RAG facade — the stable import surface for consumers (API, workers, CLI, eval).

Domain logic lives in ``jera.*`` subpackages; this module re-exports the pieces that
external adapters should depend on. ``apps/api/app`` imports from here, never from internals.
"""

# M10 — 2025-26 SOTA: late-interaction retrieval + agentic orchestration (CRAG / Adaptive-RAG /
# decomposition). Each is a port + offline adapter + a thin wrapper over QueryPipeline.
from jera.adapters.embedding.hash_multivector import HashMultiVectorEmbedding
from jera.adapters.evaluation.overlap_evaluator import OverlapRetrievalEvaluator
from jera.adapters.query.connective_decomposer import ConnectiveDecomposer
from jera.adapters.query.heuristic_router import HeuristicQueryRouter
from jera.adapters.vector_store.maxsim_store import MaxSimVectorStore
from jera.config import Profile, RagSystem, Settings, build_system
from jera.domain import (
    Answer,
    Chunk,
    Citation,
    DocumentInfo,
    FusionMethod,
    IngestionJob,
    ParsedDocument,
    Query,
    RetrievalMode,
    ScoredChunk,
    SourceRef,
)
from jera.domain.document import MediaType
from jera.evaluation import (
    CaseSpec,
    EvalReport,
    EvalRunner,
    GenerationEvalRunner,
    GenerationReport,
    build_gold_dataset,
)
from jera.pipeline import IngestPipeline, QueryPipeline
from jera.pipeline.adaptive import AdaptiveAnsweredQuery, AdaptiveQueryPipeline
from jera.pipeline.corrective import CorrectiveQueryPipeline, CorrectiveResult
from jera.pipeline.decompositional import DecompositionalQueryPipeline, DecompositionalResult
from jera.ports.multi_vector_embedding import MultiVectorEmbedding
from jera.ports.multi_vector_store import MultiVectorStore
from jera.ports.query_decomposer import QueryDecomposer
from jera.ports.query_router import QueryComplexity, QueryRouter
from jera.ports.retrieval_evaluator import RetrievalEvaluator, RetrievalGrade

__all__ = [
    "AdaptiveAnsweredQuery",
    "AdaptiveQueryPipeline",
    "Answer",
    "CaseSpec",
    "Chunk",
    "Citation",
    "ConnectiveDecomposer",
    "CorrectiveQueryPipeline",
    "CorrectiveResult",
    "DecompositionalQueryPipeline",
    "DecompositionalResult",
    "DocumentInfo",
    "EvalReport",
    "EvalRunner",
    "FusionMethod",
    "GenerationEvalRunner",
    "GenerationReport",
    "HashMultiVectorEmbedding",
    "HeuristicQueryRouter",
    "IngestPipeline",
    "IngestionJob",
    "MaxSimVectorStore",
    "MediaType",
    "MultiVectorEmbedding",
    "MultiVectorStore",
    "OverlapRetrievalEvaluator",
    "ParsedDocument",
    "Profile",
    "Query",
    "QueryComplexity",
    "QueryDecomposer",
    "QueryPipeline",
    "QueryRouter",
    "RagSystem",
    "RetrievalEvaluator",
    "RetrievalGrade",
    "RetrievalMode",
    "ScoredChunk",
    "Settings",
    "SourceRef",
    "build_gold_dataset",
    "build_system",
]
