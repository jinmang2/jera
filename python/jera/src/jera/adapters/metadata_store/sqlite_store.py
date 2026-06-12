"""SQLite metadata store — dev/test default (no external service)."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from jera.adapters.metadata_store.sql_store import SqlMetadataStore


def make_sqlite_store(path: str = ":memory:") -> SqlMetadataStore:
    """Create a SqlMetadataStore backed by SQLite.

    ``:memory:`` uses a single shared connection so the schema persists for the process
    (required since each Session would otherwise get a fresh empty in-memory db).
    """
    if path == ":memory:":
        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        engine = create_engine(f"sqlite:///{path}")
    store = SqlMetadataStore(engine)
    store.init_schema()
    return store
