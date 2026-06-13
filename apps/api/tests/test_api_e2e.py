"""E2E gate: ingest → query through the FastAPI app returns a grounded, cited answer.

This is the Milestone-1 definition-of-done: the entire real pipeline (parse→chunk→embed→
sparse→store→fuse→rerank→cite→generate) exercised through the HTTP adapter with deterministic
local providers — not a fake call stack.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.deps import get_system
from app.main import create_app

SAMPLE = """# Jera

Jera is a hybrid retrieval system. The ranking module identifier is ZX9000.

## Fusion

Hybrid retrieval uses reciprocal rank fusion to merge dense and sparse rankings.
"""


@pytest.fixture
def client() -> TestClient:
    get_system.cache_clear()  # fresh deterministic system per test
    return TestClient(create_app())


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_ingest_then_query_returns_cited_answer(client: TestClient) -> None:
    r = client.post(
        "/ingest", json={"source_id": "d1", "media_type": "text/markdown", "text": SAMPLE}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "succeeded"
    assert body["chunk_count"] >= 1

    q = client.post(
        "/query", json={"query": "What does hybrid retrieval use to merge rankings?", "top_k": 3}
    )
    assert q.status_code == 200
    answer = q.json()
    assert answer["citations"], "expected at least one citation"
    # citations carry resolvable provenance
    for c in answer["citations"]:
        assert c["chunk_id"]
        assert c["document_id"] == body["document_id"]
        assert len(c["page_span"]) == 2


def test_ingest_requires_text_or_content(client: TestClient) -> None:
    r = client.post("/ingest", json={"source_id": "d1", "media_type": "text/markdown"})
    assert r.status_code == 422


def _ingest(client: TestClient, source_id: str = "d1", text: str = SAMPLE) -> dict[str, object]:
    r = client.post(
        "/ingest", json={"source_id": source_id, "media_type": "text/markdown", "text": text}
    )
    assert r.status_code == 200
    return r.json()


def test_job_status_is_pollable(client: TestClient) -> None:
    job = _ingest(client)
    r = client.get(f"/jobs/{job['job_id']}")
    assert r.status_code == 200
    assert r.json()["status"] == "succeeded"
    assert client.get("/jobs/nope").status_code == 404


def test_query_response_carries_stats(client: TestClient) -> None:
    _ingest(client)
    q = client.post("/query", json={"query": "fusion", "top_k": 3})
    stats = q.json()["stats"]
    assert stats is not None
    assert set(stats["timings_ms"]) >= {"retrieve", "rerank", "generate", "total"}
    assert stats["estimated_cost_usd"] == 0.0  # deterministic free local models


def test_documents_list_get_delete_lifecycle(client: TestClient) -> None:
    body = _ingest(client)
    document_id = body["document_id"]

    listed = client.get("/documents").json()
    assert any(d["document_id"] == document_id for d in listed)
    assert all(d["chunk_count"] >= 1 for d in listed)

    got = client.get(f"/documents/{document_id}")
    assert got.status_code == 200 and got.json()["document_id"] == document_id
    assert client.get("/documents/missing").status_code == 404

    deleted = client.delete(f"/documents/{document_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted_chunk_count"] >= 1
    # gone from the store AND its vectors are removed → query no longer cites it
    assert client.get(f"/documents/{document_id}").status_code == 404
    assert client.delete(f"/documents/{document_id}").status_code == 404
    q = client.post("/query", json={"query": "fusion", "top_k": 3})
    assert q.json()["citations"] == []


def test_reingest_is_idempotent_not_duplicated(client: TestClient) -> None:
    _ingest(client)
    _ingest(client)  # same source_id again
    docs = client.get("/documents").json()
    assert len(docs) == 1, "re-ingesting a source must not create a duplicate document"
