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

    # chunking
    chunk_strategy: str = "heading_aware"  # heading_aware | semantic | hierarchical
    max_tokens: int = 180
    overlap_tokens: int = 40

    # contextual retrieval (Anthropic, 2024) — situate chunks before embedding/indexing
    use_contextual_retrieval: bool = False
    contextualizer_kind: str = "heuristic"  # heuristic (offline) | llm (opt-in, needs cloud)

    # retrieval
    top_k: int = 5

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
