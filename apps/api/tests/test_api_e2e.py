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
