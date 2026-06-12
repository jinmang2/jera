"""ProviderRegistry — builds wired pipelines from Settings, per profile.

Records a ProviderConfigSnapshot in the metadata store so an embedding model/dimension
change is detectable (storage gate). Paid providers are never constructed unless
``enable_cloud`` is set with a key.
"""

from __future__ import annotations

from dataclasses import dataclass

from jera.adapters.chunking.heading_aware import HeadingAwareChunker
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
from jera.ports.embedding import EmbeddingProvider
from jera.ports.generator import GeneratorLLM
from jera.ports.metadata_store import MetadataStore
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

    parsers = ParserRegistry([MarkdownParser(), PyMuPDFParser()])
    chunker = HeadingAwareChunker(
        max_tokens=settings.max_tokens, overlap_tokens=settings.overlap_tokens
    )

    embedding = _build_embedding(settings)
    sparse = _build_sparse(settings)
    vector_store = _build_vector_store(settings)
    metadata_store = _build_metadata_store(settings)
    reranker = _build_reranker(settings)
    generator = _build_generator(settings)

    ingest = IngestPipeline(
        parsers=parsers,
        chunker=chunker,
        embedding=embedding,
        sparse=sparse,
        vector_store=vector_store,
        metadata_store=metadata_store,
        collection=settings.collection,
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


def _build_embedding(settings: Settings) -> EmbeddingProvider:
    if settings.profile is Profile.LOCAL:
        from jera.adapters.embedding.fastembed_embedding import FastEmbedEmbedding

        return FastEmbedEmbedding()
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

        return FastEmbedReranker()
    if settings.profile is Profile.PROD and settings.enable_cloud and settings.cohere_api_key:
        from jera.adapters.ranking.cohere_reranker import CohereReranker

        return CohereReranker(api_key=settings.cohere_api_key, enabled=True)
    return IdentityReranker()


def _build_generator(settings: Settings) -> GeneratorLLM:
    if settings.profile is Profile.PROD and settings.enable_cloud and settings.anthropic_api_key:
        from jera.adapters.generator.claude_generator import ClaudeGenerator

        return ClaudeGenerator(api_key=settings.anthropic_api_key, enabled=True)
    return ExtractiveGenerator()
