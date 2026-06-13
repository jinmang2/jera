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
from jera.adapters.parsing.routing import OCREngine
from jera.adapters.ranking.identity_reranker import IdentityReranker
from jera.adapters.sparse.bm25_local import BM25Local
from jera.adapters.vector_store.in_memory import InMemoryVectorStore
from jera.config.pricing import snapshot_cost_metadata
from jera.config.settings import Profile, Settings
from jera.domain.ids import stable_id
from jera.domain.jobs import ProviderConfigSnapshot
from jera.pipeline.ingest import IngestPipeline
from jera.pipeline.query import QueryPipeline
from jera.ports.chunker import Chunker
from jera.ports.context_processor import ContextProcessor
from jera.ports.contextualizer import Contextualizer
from jera.ports.embedding import EmbeddingProvider
from jera.ports.generator import GeneratorLLM
from jera.ports.metadata_store import MetadataStore
from jera.ports.parser import DocumentParser
from jera.ports.query_transformer import QueryTransformer
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
    collection: str


def build_system(settings: Settings | None = None) -> RagSystem:
    settings = settings or Settings()

    parsers = _build_parsers(settings)

    embedding = _build_embedding(settings)
    chunker = _build_chunker(settings, embedding)
    sparse = _build_sparse(settings)
    vector_store = _build_vector_store(settings)
    metadata_store = _build_metadata_store(settings)
    reranker = _build_reranker(settings, embedding)
    generator = _build_generator(settings)
    contextualizer = _build_contextualizer(settings)
    query_transformer = _build_query_transformer(settings)

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
        query_transformer=query_transformer,
        context_processors=_build_context_processors(settings),
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
            cost_metadata=snapshot_cost_metadata(
                embedding=embedding.model_id,
                sparse=sparse.model_id,
                reranker=reranker.model_id,
                generator=generator.model_id,
            ),
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
        collection=settings.collection,
    )


def _build_parsers(settings: Settings) -> ParserRegistry:
    # Markdown/plain → light parser; HWPX → stdlib HwpxParser (always, zero deps); legacy .hwp →
    # PyHwpParser (always registered; activates only for application/x-hwp, needs the `hwp` extra).
    # For PDF, first-match precedence: RoutingPdfParser (if use_routing_pdf) → OpenDataLoader (if
    # use_opendataloader) → Docling (if use_docling) → Camelot (if use_camelot; table-only) →
    # PyMuPDF fallback. All opt-in flags default off, so default PDF ingestion is unchanged.
    from jera.adapters.parsing.hwpx_parser import HwpxParser
    from jera.adapters.parsing.pyhwp_parser import PyHwpParser

    parsers: list[DocumentParser] = [MarkdownParser(), HwpxParser(), PyHwpParser()]
    if settings.use_routing_pdf:
        from jera.adapters.parsing.routing_pdf_parser import RoutingPdfParser

        parsers.append(RoutingPdfParser(ocr=_build_ocr_engine(settings)))
    if settings.use_opendataloader:
        from jera.adapters.parsing.opendataloader_parser import OpenDataLoaderParser

        parsers.append(OpenDataLoaderParser())
    if settings.use_docling:
        from jera.adapters.parsing.docling_parser import DoclingParser

        parsers.append(DoclingParser())
    if settings.use_camelot:
        # Table-focused: returns only TABLE elements. Opt-in (the user wants table extraction).
        from jera.adapters.parsing.camelot_parser import CamelotTableParser

        parsers.append(CamelotTableParser())
    parsers.append(PyMuPDFParser())
    return ParserRegistry(parsers)


def _build_ocr_engine(settings: Settings) -> OCREngine | None:
    """OCR engine for RoutingPdfParser's OCR route. ``fake`` (default) → None so the parser uses
    its deterministic FakeOCR; real engines are opt-in (need the `ocr` extra / a CLOVA key)."""
    kind = settings.ocr_engine
    if kind == "fake":
        return None
    if kind == "tesseract":
        from jera.adapters.parsing.ocr import TesseractOCREngine

        return TesseractOCREngine(lang=settings.ocr_lang)
    if kind == "rapidocr":
        from jera.adapters.parsing.ocr import RapidOcrOCREngine

        return RapidOcrOCREngine()
    if kind == "clova":
        if not (settings.clova_invoke_url and settings.clova_secret):
            raise RuntimeError(
                "ocr_engine='clova' needs clova_invoke_url + clova_secret (paid). "
                "Use ocr_engine='tesseract'/'rapidocr' for local OCR."
            )
        from jera.adapters.parsing.ocr import ClovaOCREngine

        return ClovaOCREngine(
            invoke_url=settings.clova_invoke_url, secret=settings.clova_secret, enabled=True
        )
    raise ValueError(f"unknown ocr_engine {kind!r}")


def _build_chunker(settings: Settings, embedding: EmbeddingProvider) -> Chunker:
    if settings.chunk_strategy == "semantic":
        return SemanticChunker(embedding, max_tokens=settings.max_tokens)
    if settings.chunk_strategy == "hierarchical":
        return HierarchicalChunker(embedding)
    if settings.chunk_strategy == "heading_aware":
        return HeadingAwareChunker(
            max_tokens=settings.max_tokens, overlap_tokens=settings.overlap_tokens
        )
    if settings.chunk_strategy == "proposition":
        from jera.adapters.chunking.proposition import PropositionChunker

        return PropositionChunker()
    raise ValueError(f"unknown chunk_strategy {settings.chunk_strategy!r}")


def _build_context_processors(settings: Settings) -> list[ContextProcessor]:
    """Context-engineering chain (M12), applied to retrieved chunks before generation:
    curate near-duplicates → extractively compress → reorder so the best land at the edges."""
    if not settings.use_context_processing:
        return []
    from jera.adapters.context.compressor import ExtractiveCompressor
    from jera.adapters.context.curator import RedundancyCurator
    from jera.adapters.context.reorderer import LostInTheMiddleReorderer

    return [RedundancyCurator(), ExtractiveCompressor(), LostInTheMiddleReorderer()]


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


def _build_query_transformer(settings: Settings) -> QueryTransformer | None:
    """Multi-query retrieval is off by default. ``rule_based`` is offline/CI-real; ``hyde``
    needs a real HypothesisLLM (cloud, opt-in) and is built only with cloud + an Anthropic key."""
    if not settings.use_query_transform:
        return None
    if settings.query_transform_kind == "rule_based":
        from jera.adapters.query.rule_based_expander import RuleBasedExpander

        return RuleBasedExpander()
    if settings.query_transform_kind == "hyde":
        if not (settings.enable_cloud and settings.anthropic_api_key):
            raise RuntimeError(
                "query_transform_kind='hyde' needs enable_cloud=True + an anthropic_api_key "
                "(paid). Use query_transform_kind='rule_based' for the offline path."
            )
        from jera.adapters.query.claude_hypothesis_llm import ClaudeHypothesisLLM
        from jera.adapters.query.hyde import HydeTransformer

        llm = ClaudeHypothesisLLM(api_key=settings.anthropic_api_key, enabled=True)
        return HydeTransformer(llm)
    raise ValueError(f"unknown query_transform_kind {settings.query_transform_kind!r}")


def _build_embedding(settings: Settings) -> EmbeddingProvider:
    base = _build_base_embedding(settings)
    # Instruction-tuned wrapper (query-side task prefix) goes innermost so later wrappers see it.
    if settings.embedding_instruction is not None:
        from jera.adapters.embedding.instruction import InstructionEmbedding

        base = InstructionEmbedding(base, task=settings.embedding_instruction)
    # M11 opt-in wrappers (order: truncate dims first, then late-chunking pools the truncated vecs).
    if settings.embedding_truncate_dims is not None:
        from jera.adapters.embedding.truncated_dim import TruncatedDimEmbedding

        base = TruncatedDimEmbedding(base, dims=settings.embedding_truncate_dims)
    if settings.use_late_chunking:
        from jera.adapters.embedding.late_chunking import LateChunkingEmbedding

        base = LateChunkingEmbedding(base, alpha=settings.late_chunking_alpha)
    return base


def _build_base_embedding(settings: Settings) -> EmbeddingProvider:
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
    if settings.use_quantized_store:
        from jera.adapters.vector_store.quantized_in_memory import QuantizedInMemoryVectorStore

        return QuantizedInMemoryVectorStore()
    return InMemoryVectorStore()


def _build_metadata_store(settings: Settings) -> MetadataStore:
    if settings.profile is Profile.PROD and settings.postgres_dsn:
        from jera.adapters.metadata_store.postgres_store import make_postgres_store

        return make_postgres_store(settings.postgres_dsn)
    return make_sqlite_store(settings.sqlite_path)


def _build_reranker(settings: Settings, embedding: EmbeddingProvider) -> Reranker:
    # Explicit MMR opt-in (any profile): diversity reorder over the embedding — no model/key.
    if settings.reranker_kind == "mmr":
        from jera.adapters.ranking.mmr_reranker import MMRReranker

        return MMRReranker(embedding, lambda_=settings.mmr_lambda)
    if settings.reranker_kind == "listwise":
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        return ListwiseReranker()
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
        # Real tool-use LLM when cloud+key is provided; deterministic FakeToolUseLLM offline.
        from jera.adapters.generator.tool_augmented_generator import ToolAugmentedGenerator
        from jera.tooluse.llm import ToolUseLLM
        from jera.tooluse.tools import CalculatorTool

        llm: ToolUseLLM
        if settings.enable_cloud and settings.anthropic_api_key:
            from jera.tooluse.llm import ClaudeToolUseGenerator

            llm = ClaudeToolUseGenerator(api_key=settings.anthropic_api_key, enabled=True)
        else:
            from jera.tooluse.llm import FakeToolUseLLM

            llm = FakeToolUseLLM()
        return ToolAugmentedGenerator(llm=llm, tools=[CalculatorTool()])
    if settings.profile is Profile.PROD and settings.enable_cloud and settings.anthropic_api_key:
        from jera.adapters.generator.claude_generator import ClaudeGenerator

        return ClaudeGenerator(api_key=settings.anthropic_api_key, enabled=True)
    return ExtractiveGenerator()
