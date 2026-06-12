"""ProviderRegistry — builds wired pipelines from Settings, per profile.

Records a ProviderConfigSnapshot in the metadata store so an embedding model/dimension
change is detectable (storage gate). Paid providers are never constructed unless
``enable_cloud`` is set with a key.
"""

from __future__ import annotations

from dataclasses import dataclass

from jera.adapters.chunking.heading_aware import HeadingAwareChunker
from jera.adapters.chunking.hierarchical import HierarchicalChunker
from jera.adapters.chunking.semantic import SemanticChunker
from jera.adapters.embedding.hash_embedding import HashEmbedding
from jera.adapters.generator.extractive_generator import ExtractiveGenerator
from jera.adapters.metadata_store.sqlite_store import make_sqlite_store
from jera.adapters.parsing import MarkdownParser, ParserRegistry, PyMuPDFParser
from jera.adapters.ranking.identity_reranker import IdentityReranker
from jera.adapters.sparse.bm25_local import BM25Local
from jera.adapters.vector_store.in_memory import InMemoryVectorStore
from jera.config.settings import Profile, Settings
from jera.domain.ids import stable_id
from jera.domain.jobs import ProviderConfigSnapshot
from jera.pipeline.ingest import IngestPipeline
from jera.pipeline.query import QueryPipeline
from jera.ports.chunker import Chunker
from jera.ports.contextualizer import Contextualizer
from jera.ports.embedding import EmbeddingProvider
from jera.ports.generator import GeneratorLLM
from jera.ports.metadata_store import MetadataStore
from jera.ports.parser import DocumentParser
from jera.ports.reranker import Reranker
from jera.ports.sparse import SparseVectorProvider
from jera.ports.vector_store import VectorStore


@dataclass
class RagSystem:
    ingest: IngestPipeline
    query: QueryPipeline
    metadata_store: MetadataStore
    embedding: EmbeddingProvider
    sparse: SparseVectorProvider
    reranker: Reranker
    generator: GeneratorLLM
    vector_store: VectorStore


def build_system(settings: Settings | None = None) -> RagSystem:
    settings = settings or Settings()

    parsers = _build_parsers(settings)

    embedding = _build_embedding(settings)
    chunker = _build_chunker(settings, embedding)
    sparse = _build_sparse(settings)
    vector_store = _build_vector_store(settings)
    metadata_store = _build_metadata_store(settings)
    reranker = _build_reranker(settings)
    generator = _build_generator(settings)
    contextualizer = _build_contextualizer(settings)

    ingest = IngestPipeline(
        parsers=parsers,
        chunker=chunker,
        embedding=embedding,
        sparse=sparse,
        vector_store=vector_store,
        metadata_store=metadata_store,
        collection=settings.collection,
        contextualizer=contextualizer,
    )
    query = QueryPipeline(
        embedding=embedding,
        sparse=sparse,
        vector_store=vector_store,
        metadata_store=metadata_store,
        reranker=reranker,
        generator=generator,
        collection=settings.collection,
    )

    metadata_store.save_config_snapshot(
        ProviderConfigSnapshot(
            snapshot_id=stable_id(
                settings.profile.value, embedding.model_id, str(embedding.dimensions)
            ),
            profile=settings.profile.value,
            embedding_model_id=embedding.model_id,
            embedding_dimensions=embedding.dimensions,
            embedding_context_limit=embedding.context_limit,
            sparse_model_id=sparse.model_id,
            reranker_model_id=reranker.model_id,
            generator_model_id=generator.model_id,
            cost_metadata={"note": "placeholder; populate per-provider pricing when wired"},
        )
    )
    return RagSystem(
        ingest=ingest,
        query=query,
        metadata_store=metadata_store,
        embedding=embedding,
        sparse=sparse,
        reranker=reranker,
        generator=generator,
        vector_store=vector_store,
    )


def _build_parsers(settings: Settings) -> ParserRegistry:
    # Markdown/plain → light parser; HWPX → stdlib HwpxParser (always, zero deps).
    # For PDF, first-match precedence: RoutingPdfParser (if use_routing_pdf) → Docling (if
    # use_docling) → PyMuPDF fallback. Both routing flags default off, so default PDF ingestion
    # is unchanged.
    from jera.adapters.parsing.hwpx_parser import HwpxParser

    parsers: list[DocumentParser] = [MarkdownParser(), HwpxParser()]
    if settings.use_routing_pdf:
        from jera.adapters.parsing.routing_pdf_parser import RoutingPdfParser

        parsers.append(RoutingPdfParser())
    if settings.use_docling:
        from jera.adapters.parsing.docling_parser import DoclingParser

        parsers.append(DoclingParser())
    parsers.append(PyMuPDFParser())
    return ParserRegistry(parsers)


def _build_chunker(settings: Settings, embedding: EmbeddingProvider) -> Chunker:
    if settings.chunk_strategy == "semantic":
        return SemanticChunker(embedding, max_tokens=settings.max_tokens)
    if settings.chunk_strategy == "hierarchical":
        return HierarchicalChunker(embedding)
    if settings.chunk_strategy == "heading_aware":
        return HeadingAwareChunker(
            max_tokens=settings.max_tokens, overlap_tokens=settings.overlap_tokens
        )
    raise ValueError(f"unknown chunk_strategy {settings.chunk_strategy!r}")


def _build_contextualizer(settings: Settings) -> Contextualizer | None:
    """Contextual Retrieval is off by default. When enabled, the heuristic adapter is
    offline/CI-real; the ``llm`` adapter needs a real SituateLLM (cloud, opt-in) and is only
    constructed when cloud is enabled with an Anthropic key — CI never builds it."""
    if not settings.use_contextual_retrieval:
        return None
    if settings.contextualizer_kind == "heuristic":
        from jera.adapters.contextual.heuristic_contextualizer import HeuristicContextualizer

        return HeuristicContextualizer()
    if settings.contextualizer_kind == "llm":
        if not (settings.enable_cloud and settings.anthropic_api_key):
            raise RuntimeError(
                "contextualizer_kind='llm' needs enable_cloud=True + an anthropic_api_key "
                "(paid). Use contextualizer_kind='heuristic' for the offline path."
            )
        from jera.adapters.contextual.claude_situate_llm import ClaudeSituateLLM
        from jera.adapters.contextual.llm_contextualizer import LlmContextualizer

        return LlmContextualizer(ClaudeSituateLLM(api_key=settings.anthropic_api_key, enabled=True))
    raise ValueError(f"unknown contextualizer_kind {settings.contextualizer_kind!r}")


def _build_embedding(settings: Settings) -> EmbeddingProvider:
    if settings.profile is Profile.LOCAL:
        from jera.adapters.embedding.fastembed_embedding import FastEmbedEmbedding

        # Default to bge-m3 (multilingual, 1024-dim) when no override is set.
        model = settings.embedding_model or "BAAI/bge-m3"
        return FastEmbedEmbedding(model_name=model)
    if settings.profile is Profile.PROD and settings.enable_cloud and settings.openai_api_key:
        from jera.adapters.embedding.openai_embedding import OpenAIEmbedding

        return OpenAIEmbedding(api_key=settings.openai_api_key, enabled=True)
    return HashEmbedding(dimensions=settings.hash_dimensions)


def _build_sparse(settings: Settings) -> SparseVectorProvider:
    if settings.profile is Profile.LOCAL:
        from jera.adapters.sparse.fastembed_sparse import FastEmbedSparse

        return FastEmbedSparse()
    return BM25Local()


def _build_vector_store(settings: Settings) -> VectorStore:
    if settings.profile is Profile.PROD:
        from jera.adapters.vector_store.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return InMemoryVectorStore()


def _build_metadata_store(settings: Settings) -> MetadataStore:
    if settings.profile is Profile.PROD and settings.postgres_dsn:
        from jera.adapters.metadata_store.postgres_store import make_postgres_store

        return make_postgres_store(settings.postgres_dsn)
    return make_sqlite_store(settings.sqlite_path)


def _build_reranker(settings: Settings) -> Reranker:
    if settings.profile is Profile.LOCAL:
        from jera.adapters.ranking.fastembed_reranker import FastEmbedReranker

        # Default to bge-reranker-v2-m3 (multilingual); S0 fallback: bge-reranker-base.
        model = settings.reranker_model or "BAAI/bge-reranker-v2-m3"
        return FastEmbedReranker(model_name=model)
    if settings.profile is Profile.PROD and settings.enable_cloud and settings.cohere_api_key:
        from jera.adapters.ranking.cohere_reranker import CohereReranker

        return CohereReranker(api_key=settings.cohere_api_key, enabled=True)
    return IdentityReranker()


def _build_generator(settings: Settings) -> GeneratorLLM:
    if settings.generator_kind == "tooluse":
        # Offline-safe: FakeToolUseLLM + CalculatorTool — zero paid calls.
        # ClaudeToolUseGenerator is wired via generator_kind="tooluse" + cloud extra + key
        # in script contexts; CI always lands here.
        from jera.adapters.generator.tool_augmented_generator import ToolAugmentedGenerator
        from jera.tooluse.llm import FakeToolUseLLM
        from jera.tooluse.tools import CalculatorTool

        return ToolAugmentedGenerator(
            llm=FakeToolUseLLM(),
            tools=[CalculatorTool()],
        )
    if settings.profile is Profile.PROD and settings.enable_cloud and settings.anthropic_api_key:
        from jera.adapters.generator.claude_generator import ClaudeGenerator

        return ClaudeGenerator(api_key=settings.anthropic_api_key, enabled=True)
    return ExtractiveGenerator()
