"""Public RAG facade — the stable import surface for consumers (API, workers, CLI, eval).

Domain logic lives in ``jera.*`` subpackages; this module re-exports the pieces that
external adapters should depend on. ``apps/api/app`` imports from here, never from internals.
"""

# M10 — 2025-26 SOTA: late-interaction retrieval + agentic orchestration (CRAG / Adaptive-RAG /
# decomposition). Each is a port + offline adapter + a thin wrapper over QueryPipeline.
# M12 — context quality & evaluation (claim-level eval metrics live in jera.evaluation_contracts).
from jera.adapters.chunking.proposition import PropositionChunker
from jera.adapters.context.compressor import ExtractiveCompressor
from jera.adapters.context.curator import RedundancyCurator
from jera.adapters.context.reorderer import LostInTheMiddleReorderer
from jera.adapters.embedding.hash_multivector import HashMultiVectorEmbedding

# M11 — research-viable deferred techniques: HippoRAG PPR graph retrieval, int8 quantization,
# listwise reranking, late chunking (all offline-deterministic; real variants opt-in).
from jera.adapters.embedding.late_chunking import LateChunkingEmbedding
from jera.adapters.embedding.truncated_dim import TruncatedDimEmbedding
from jera.adapters.evaluation.overlap_evaluator import OverlapRetrievalEvaluator
from jera.adapters.graph.hippo_retriever import HippoGraphRetriever
from jera.adapters.graph.regex_entity_extractor import RegexEntityExtractor
from jera.adapters.query.bridge_followup_controller import BridgeFollowupController
from jera.adapters.query.connective_decomposer import ConnectiveDecomposer
from jera.adapters.query.heuristic_router import HeuristicQueryRouter
from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker, ListwiseReranker
from jera.adapters.vector_store.maxsim_store import MaxSimVectorStore
from jera.adapters.vector_store.quantized_in_memory import QuantizedInMemoryVectorStore
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
from jera.pipeline.iterative import IterativeResult, IterativeRetrievalPipeline
from jera.ports.context_processor import ContextProcessor
from jera.ports.entity_extractor import EntityExtractor
from jera.ports.followup_controller import FollowupController
from jera.ports.graph_retriever import GraphRetriever
from jera.ports.multi_vector_embedding import MultiVectorEmbedding
from jera.ports.multi_vector_store import MultiVectorStore
from jera.ports.query_decomposer import QueryDecomposer
from jera.ports.query_router import QueryComplexity, QueryRouter
from jera.ports.retrieval_evaluator import RetrievalEvaluator, RetrievalGrade

__all__ = [
    "AdaptiveAnsweredQuery",
    "AdaptiveQueryPipeline",
    "Answer",
    "BridgeFollowupController",
    "CaseSpec",
    "Chunk",
    "Citation",
    "ClaudeListwiseReranker",
    "ConnectiveDecomposer",
    "ContextProcessor",
    "CorrectiveQueryPipeline",
    "CorrectiveResult",
    "DecompositionalQueryPipeline",
    "DecompositionalResult",
    "DocumentInfo",
    "EntityExtractor",
    "EvalReport",
    "EvalRunner",
    "ExtractiveCompressor",
    "FollowupController",
    "FusionMethod",
    "GenerationEvalRunner",
    "GenerationReport",
    "GraphRetriever",
    "HashMultiVectorEmbedding",
    "HeuristicQueryRouter",
    "HippoGraphRetriever",
    "IngestPipeline",
    "IngestionJob",
    "IterativeResult",
    "IterativeRetrievalPipeline",
    "LateChunkingEmbedding",
    "ListwiseReranker",
    "LostInTheMiddleReorderer",
    "MaxSimVectorStore",
    "MediaType",
    "MultiVectorEmbedding",
    "MultiVectorStore",
    "OverlapRetrievalEvaluator",
    "ParsedDocument",
    "Profile",
    "PropositionChunker",
    "QuantizedInMemoryVectorStore",
    "Query",
    "QueryComplexity",
    "QueryDecomposer",
    "QueryPipeline",
    "QueryRouter",
    "RagSystem",
    "RedundancyCurator",
    "RegexEntityExtractor",
    "RetrievalEvaluator",
    "RetrievalGrade",
    "RetrievalMode",
    "ScoredChunk",
    "Settings",
    "SourceRef",
    "TruncatedDimEmbedding",
    "build_gold_dataset",
    "build_system",
]
