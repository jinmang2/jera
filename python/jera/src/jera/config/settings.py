"""Settings — profile-driven configuration (pydantic-settings).

Profiles:
  test  → fully deterministic offline (hash embed, BM25, in-memory store, SQLite :memory:)
  local → real local models (fastembed) + SQLite file; no paid calls
  prod  → Qdrant + Postgres + cloud adapters (cloud disabled unless keys provided)
"""

from __future__ import annotations

from enum import StrEnum

from pydantic_settings import BaseSettings, SettingsConfigDict


class Profile(StrEnum):
    TEST = "test"
    LOCAL = "local"
    PROD = "prod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JERA_", env_file=".env", extra="ignore")

    profile: Profile = Profile.TEST
    collection: str = "jera_chunks"

    # storage
    sqlite_path: str = ":memory:"
    postgres_dsn: str | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # embedding (hash profile)
    hash_dimensions: int = 256

    # parsing
    use_docling: bool = False  # prefer Docling for PDF/HTML (extra: docling; heavier, layout/OCR)
    use_routing_pdf: bool = False  # route PDF pages text|OCR via RoutingPdfParser (before PyMuPDF)
    use_opendataloader: bool = False  # rich PDF parser (extra: opendataloader; Java 11+)
    use_camelot: bool = False  # table-focused PDF parser (extra: tables; TABLE elements only)
    # OCR engine for RoutingPdfParser's OCR route: fake (offline) | tesseract | rapidocr | clova
    ocr_engine: str = "fake"
    ocr_lang: str = "kor+eng"
    clova_invoke_url: str | None = None  # CLOVA OCR endpoint (ocr_engine="clova", paid)
    clova_secret: str | None = None

    # chunking
    chunk_strategy: str = "heading_aware"  # heading_aware | semantic | hierarchical | proposition
    max_tokens: int = 180
    overlap_tokens: int = 40

    # context engineering (M12) — process retrieved chunks before generation:
    # redundancy curation → extractive compression → lost-in-the-middle reorder
    use_context_processing: bool = False

    # contextual retrieval (Anthropic, 2024) — situate chunks before embedding/indexing
    use_contextual_retrieval: bool = False
    contextualizer_kind: str = "heuristic"  # heuristic (offline) | llm (opt-in, needs cloud)

    # retrieval
    top_k: int = 5

    # multi-query retrieval — expand the query and RRF-fuse per-variant rankings
    use_query_transform: bool = False
    query_transform_kind: str = "rule_based"  # rule_based (offline) | hyde (opt-in, needs cloud)

    # reranker back-end: "identity" (default) | "mmr" (diversity) | "listwise" (RankLLM-style)
    reranker_kind: str = "identity"
    mmr_lambda: float = 0.7  # MMRReranker tradeoff: 1.0 = pure relevance, lower = more diverse

    # M11 advanced retrieval (all opt-in, offline-deterministic)
    use_quantized_store: bool = False  # int8 two-stage quantized vector store (MRL rescore)
    embedding_truncate_dims: int | None = None  # Matryoshka: truncate dense vectors to N dims
    use_late_chunking: bool = False  # context-mixed chunk embeddings (Jina late chunking)
    late_chunking_alpha: float = 0.3  # late-chunking neighbor-context mix weight

    # local profile model overrides (None → adapter picks its own default)
    # local dense default: BAAI/bge-m3 (1024-dim multilingual)
    # S0 fallback: intfloat/multilingual-e5-large
    embedding_model: str | None = None
    # local reranker default: BAAI/bge-reranker-v2-m3; S0 fallback: BAAI/bge-reranker-base
    reranker_model: str | None = None

    # generator back-end: "extractive" (default, offline) | "tooluse" (ToolAugmentedGenerator)
    generator_kind: str = "extractive"

    # cloud (disabled unless explicitly enabled with a key)
    enable_cloud: bool = False
    openai_api_key: str | None = None
    cohere_api_key: str | None = None
    anthropic_api_key: str | None = None
