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

    # chunking
    max_tokens: int = 180
    overlap_tokens: int = 40

    # retrieval
    top_k: int = 5

    # cloud (disabled unless explicitly enabled with a key)
    enable_cloud: bool = False
    openai_api_key: str | None = None
    cohere_api_key: str | None = None
    anthropic_api_key: str | None = None
