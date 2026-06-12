"""Storage gate (Gate 5): SQLite owns documents/chunks/jobs/config snapshots; round-trips."""

from __future__ import annotations

from sqlalchemy import inspect

from jera.adapters.metadata_store.sqlite_store import make_sqlite_store
from jera.domain.chunk import Chunk
from jera.domain.document import PageSpan
from jera.domain.jobs import IngestionJob, JobStatus, ProviderConfigSnapshot


def _chunk(cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id="doc1",
        source_id="s1",
        text="hello world",
        page_span=PageSpan.single(1),
        section_path=("Intro",),
        element_ids=("e1",),
        char_span=(0, 11),
        token_count=2,
        chunk_strategy="heading_aware",
        chunk_version="1.0.0",
    )


def test_schema_owns_all_four_tables() -> None:
    store = make_sqlite_store(":memory:")
    names = set(inspect(store._engine).get_table_names())
    assert {"documents", "chunks", "ingestion_jobs", "config_snapshots"} <= names


def test_chunk_round_trip_preserves_provenance() -> None:
    store = make_sqlite_store(":memory:")
    store.save_chunks([_chunk("c1"), _chunk("c2")])
    got = store.get_chunks(["c2", "c1"])  # order preserved
    assert [c.chunk_id for c in got] == ["c2", "c1"]
    assert got[0].section_path == ("Intro",)
    assert got[0].char_span == (0, 11)


def test_job_and_config_snapshot_persist() -> None:
    store = make_sqlite_store(":memory:")
    store.save_job(
        IngestionJob(job_id="j1", source_id="s1", status=JobStatus.SUCCEEDED, chunk_count=3)
    )
    assert store.get_job("j1").chunk_count == 3

    snap = ProviderConfigSnapshot(
        snapshot_id="snap1",
        profile="test",
        embedding_model_id="hash-emb-v1-256",
        embedding_dimensions=256,
        sparse_model_id="bm25-local-v1",
        reranker_model_id="identity-rerank-v1",
        generator_model_id="extractive-v1",
    )
    store.save_config_snapshot(snap)
    latest = store.latest_config_snapshot()
    assert latest is not None
    assert latest.embedding_dimensions == 256
    assert latest.embedding_model_id == "hash-emb-v1-256"
