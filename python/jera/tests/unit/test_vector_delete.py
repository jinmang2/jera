"""Tests for VectorStore.delete — in-memory correctness + Qdrant SDK-boundary mock."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from jera.adapters.vector_store.in_memory import InMemoryVectorStore
from jera.domain.vectors import SparseVector
from jera.ports.vector_store import CollectionSpec, VectorRecord

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPEC = CollectionSpec(name="col", dense_dim=3)


def _rec(chunk_id: str, document_id: str = "doc1") -> VectorRecord:
    return VectorRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        dense=[0.1, 0.2, 0.3],
        sparse=SparseVector(indices=[1], values=[1.0]),
    )


def _populated_store() -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.ensure_collection(_SPEC)
    store.upsert("col", [_rec("a"), _rec("b"), _rec("c")])
    return store


# ---------------------------------------------------------------------------
# 1.  InMemoryVectorStore.delete
# ---------------------------------------------------------------------------


class TestInMemoryDelete:
    def test_deleted_ids_not_returned_by_search(self) -> None:
        store = _populated_store()
        store.delete("col", ["a", "b"])
        results = store.search("col", dense=[0.1, 0.2, 0.3], top_k=10)
        ids = {r.chunk_id for r in results}
        assert "a" not in ids
        assert "b" not in ids

    def test_surviving_records_still_returned_by_search(self) -> None:
        store = _populated_store()
        store.delete("col", ["a", "b"])
        results = store.search("col", dense=[0.1, 0.2, 0.3], top_k=10)
        ids = {r.chunk_id for r in results}
        assert "c" in ids

    def test_delete_unknown_id_is_noop(self) -> None:
        store = _populated_store()
        # Must not raise; all original records remain.
        store.delete("col", ["nonexistent"])
        results = store.search("col", dense=[0.1, 0.2, 0.3], top_k=10)
        assert len(results) == 3

    def test_delete_unknown_collection_is_noop(self) -> None:
        store = _populated_store()
        # Must not raise even when the collection is entirely absent.
        store.delete("no_such_collection", ["a"])

    def test_delete_empty_list_is_noop(self) -> None:
        store = _populated_store()
        store.delete("col", [])
        results = store.search("col", dense=[0.1, 0.2, 0.3], top_k=10)
        assert len(results) == 3

    def test_delete_all_records_yields_empty_search(self) -> None:
        store = _populated_store()
        store.delete("col", ["a", "b", "c"])
        results = store.search("col", dense=[0.1, 0.2, 0.3], top_k=10)
        assert results == []

    def test_delete_is_idempotent_on_repeated_call(self) -> None:
        store = _populated_store()
        store.delete("col", ["a"])
        # Second delete of the same id must not raise.
        store.delete("col", ["a"])
        results = store.search("col", dense=[0.1, 0.2, 0.3], top_k=10)
        ids = {r.chunk_id for r in results}
        assert "a" not in ids
        assert len(ids) == 2


# ---------------------------------------------------------------------------
# 2.  QdrantVectorStore.delete — SDK-boundary mock
# ---------------------------------------------------------------------------


def _install_qdrant_fake(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Install a minimal fake qdrant_client module; return captured delete calls."""
    delete_calls: list[dict[str, Any]] = []

    class _FakePointIdsList:
        def __init__(self, points: list[str]) -> None:
            self.points = points

    class _FakeModels:
        PointIdsList = _FakePointIdsList

        class Distance:
            COSINE = "Cosine"
            DOT = "Dot"

        class VectorParams:
            def __init__(self, size: int, distance: Any) -> None:
                pass

        class SparseVectorParams:
            pass

        class SparseVector:
            def __init__(self, indices: list[int], values: list[float]) -> None:
                pass

        class PointStruct:
            def __init__(self, id: str, vector: Any, payload: Any) -> None:
                pass

        class Prefetch:
            def __init__(self, query: Any, using: str, limit: int) -> None:
                pass

        class Fusion:
            RRF = "rrf"
            DBSF = "dbsf"

        class FusionQuery:
            def __init__(self, fusion: Any) -> None:
                pass

    class _FakeQdrantClient:
        def __init__(self, url: str, api_key: Any = None) -> None:
            pass

        def collection_exists(self, collection_name: str) -> bool:
            return False

        def delete_collection(self, collection_name: str) -> None:
            pass

        def create_collection(self, **kwargs: Any) -> None:
            pass

        def upsert(self, **kwargs: Any) -> None:
            pass

        def delete(self, **kwargs: Any) -> None:
            delete_calls.append(kwargs)

        def query_points(self, **kwargs: Any) -> Any:
            return types.SimpleNamespace(points=[])

    qdrant_mod = types.ModuleType("qdrant_client")
    qdrant_mod.QdrantClient = _FakeQdrantClient  # type: ignore[attr-defined]
    qdrant_mod.models = _FakeModels  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_mod)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", _FakeModels)  # type: ignore[arg-type]

    return delete_calls


class TestQdrantDelete:
    def _make_store(self) -> Any:
        from jera.adapters.vector_store.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(url="http://localhost:6333")

    def test_delete_calls_client_delete_with_collection_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        delete_calls = _install_qdrant_fake(monkeypatch)
        store = self._make_store()
        store.delete("my_col", ["id1", "id2"])
        assert len(delete_calls) == 1
        assert delete_calls[0]["collection_name"] == "my_col"

    def test_delete_passes_chunk_ids_as_points_selector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        delete_calls = _install_qdrant_fake(monkeypatch)
        store = self._make_store()
        store.delete("col", ["id1", "id2", "id3"])
        selector = delete_calls[0]["points_selector"]
        assert selector.points == ["id1", "id2", "id3"]

    def test_delete_empty_list_still_calls_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        delete_calls = _install_qdrant_fake(monkeypatch)
        store = self._make_store()
        store.delete("col", [])
        assert len(delete_calls) == 1
        assert delete_calls[0]["points_selector"].points == []
