"""Postgres metadata store — prod (extra: postgres).

Same SQLAlchemy schema as SQLite; only the engine/DSN differs. pgvector is optional and only
needed if the deployment chooses the pgvector-only storage mode instead of Qdrant (see ADR).
alembic owns migrations in prod (SCOPE OUT for M1, where SQLite uses create_all).
"""

from __future__ import annotations

from jera.adapters.metadata_store.sql_store import SqlMetadataStore


def make_postgres_store(dsn: str, *, create_schema: bool = False) -> SqlMetadataStore:
    try:
        from sqlalchemy import create_engine
    except ImportError as exc:  # pragma: no cover
        raise ImportError("SQLAlchemy is required for the Postgres store.") from exc
    # psycopg (v3) driver; requires the 'postgres' extra.
    engine = create_engine(dsn)
    store = SqlMetadataStore(engine)
    if create_schema:  # prod normally runs alembic instead
        store.init_schema()
    return store
