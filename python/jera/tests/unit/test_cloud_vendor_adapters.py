"""SDK-boundary tests for opt-in NON-Anthropic vendor adapters.

Each test injects a fake vendor module into sys.modules before constructing
the adapter, capturing request kwargs and feeding canned responses — no
network calls, no API keys, no heavy installs required.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from jera.domain.chunk import Chunk
from jera.domain.document import PageSpan, ParsedDocument, Provenance
from jera.domain.retrieval import FusionMethod, ScoredChunk
from jera.domain.vectors import SparseVector
from jera.ports.vector_store import CollectionSpec, Distance, VectorRecord

# ---------------------------------------------------------------------------
# Helper: install a fake module into sys.modules for the duration of a test
# ---------------------------------------------------------------------------


def _install(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> types.ModuleType:
    """Create and register a fake module with the given attributes."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _make_chunk(chunk_id: str = "c1", text: str = "hello") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        source_id="src1",
        text=text,
        page_span=PageSpan.single(1),
        section_path=("Intro",),
        element_ids=("e1",),
        char_span=(0, len(text)),
        token_count=1,
        chunk_strategy="heading_aware",
        chunk_version="1.0.0",
    )


# ===========================================================================
# 1.  OpenAI Embedding
# ===========================================================================


class TestOpenAIEmbedding:
    """Verify request-building and response-parsing for OpenAIEmbedding."""

    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[dict[str, Any]]]:
        """Install fake openai module; return (adapter, captured_calls list)."""
        captured: list[dict[str, Any]] = []

        class _FakeEmbeddingsCreate:
            def create(self, **kwargs: Any) -> Any:
                captured.append(kwargs)
                # Return a response with .data[i].embedding
                emb0 = types.SimpleNamespace(embedding=[0.1, 0.2, 0.3])
                emb1 = types.SimpleNamespace(embedding=[0.4, 0.5, 0.6])
                return types.SimpleNamespace(data=[emb0, emb1])

        class _FakeOpenAI:
            def __init__(self, api_key: str) -> None:
                self.embeddings = _FakeEmbeddingsCreate()

        _install(monkeypatch, "openai", OpenAI=_FakeOpenAI)

        from jera.adapters.embedding.openai_embedding import OpenAIEmbedding

        adapter = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key="test-key",
            enabled=True,
        )
        return adapter, captured

    def test_embed_sends_correct_model_and_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, captured = self._setup(monkeypatch)
        adapter.embed(["hello", "world"])
        assert len(captured) == 1
        assert captured[0]["model"] == "text-embedding-3-small"
        assert captured[0]["input"] == ["hello", "world"]

    def test_embed_returns_parsed_vectors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, captured = self._setup(monkeypatch)
        result = adapter.embed(["a", "b"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    def test_embed_query_returns_first_vector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, captured = self._setup(monkeypatch)
        result = adapter.embed_query("single")
        assert result == [0.1, 0.2, 0.3]
        assert captured[0]["input"] == ["single"]

    def test_dimensions_default_for_known_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, _ = self._setup(monkeypatch)
        assert adapter.dimensions == 1536

    def test_dim_override_sent_in_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[dict[str, Any]] = []

        class _FakeEmbeddingsCreate:
            def create(self, **kwargs: Any) -> Any:
                captured.append(kwargs)
                emb = types.SimpleNamespace(embedding=[0.0])
                return types.SimpleNamespace(data=[emb])

        class _FakeOpenAI:
            def __init__(self, api_key: str) -> None:
                self.embeddings = _FakeEmbeddingsCreate()

        _install(monkeypatch, "openai", OpenAI=_FakeOpenAI)

        from importlib import reload

        import jera.adapters.embedding.openai_embedding as _mod

        reload(_mod)
        adapter = _mod.OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key="k",
            enabled=True,
            dimensions=256,
        )
        adapter.embed(["x"])
        assert captured[0]["dimensions"] == 256

    def test_disabled_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, "openai", OpenAI=object)
        from jera.adapters.embedding.openai_embedding import OpenAIEmbedding

        with pytest.raises(RuntimeError, match="disabled by default"):
            OpenAIEmbedding(model="text-embedding-3-small")

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, "openai", OpenAI=object)
        from jera.adapters.embedding.openai_embedding import OpenAIEmbedding

        with pytest.raises(RuntimeError, match="api_key"):
            OpenAIEmbedding(model="text-embedding-3-small", enabled=True)


# ===========================================================================
# 2.  Cohere Reranker
# ===========================================================================


class TestCohereReranker:
    """Verify request-building and result-mapping for CohereReranker."""

    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[dict[str, Any]]]:
        captured: list[dict[str, Any]] = []

        class _FakeCohere:
            def __init__(self, api_key: str) -> None:
                self._key = api_key

            def rerank(self, **kwargs: Any) -> Any:
                captured.append(kwargs)
                # Simulate: first doc scores 0.9, second scores 0.4
                r0 = types.SimpleNamespace(index=0, relevance_score=0.9)
                r1 = types.SimpleNamespace(index=1, relevance_score=0.4)
                return types.SimpleNamespace(results=[r0, r1])

        _install(monkeypatch, "cohere", Client=_FakeCohere)
        from jera.adapters.ranking.cohere_reranker import CohereReranker

        adapter = CohereReranker(model="rerank-v3.5", api_key="test-key", enabled=True)
        return adapter, captured

    def test_rerank_calls_sdk_with_correct_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, captured = self._setup(monkeypatch)
        c1 = ScoredChunk(chunk_id="c1", score=0.5, chunk=_make_chunk("c1", "alpha"))
        c2 = ScoredChunk(chunk_id="c2", score=0.3, chunk=_make_chunk("c2", "beta"))
        adapter.rerank("my query", [c1, c2], top_k=2)
        assert len(captured) == 1
        assert captured[0]["query"] == "my query"
        assert captured[0]["documents"] == ["alpha", "beta"]
        assert captured[0]["model"] == "rerank-v3.5"
        assert captured[0]["top_n"] == 2

    def test_rerank_maps_scores_to_candidates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, _ = self._setup(monkeypatch)
        c1 = ScoredChunk(chunk_id="c1", score=0.5, chunk=_make_chunk("c1", "alpha"))
        c2 = ScoredChunk(chunk_id="c2", score=0.3, chunk=_make_chunk("c2", "beta"))
        results = adapter.rerank("q", [c1, c2], top_k=2)
        assert results[0].chunk_id == "c1"
        assert results[0].score == pytest.approx(0.9)
        assert results[0].components["rerank"] == pytest.approx(0.9)
        assert results[1].chunk_id == "c2"
        assert results[1].score == pytest.approx(0.4)

    def test_rerank_preserves_original_chunk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Use a fresh fake that returns exactly 1 result for 1 candidate
        captured: list[dict[str, Any]] = []

        class _FakeCohereOneResult:
            def __init__(self, api_key: str) -> None:
                pass

            def rerank(self, **kwargs: Any) -> Any:
                captured.append(kwargs)
                r0 = types.SimpleNamespace(index=0, relevance_score=0.9)
                return types.SimpleNamespace(results=[r0])

        _install(monkeypatch, "cohere", Client=_FakeCohereOneResult)
        from jera.adapters.ranking.cohere_reranker import CohereReranker

        adapter = CohereReranker(model="rerank-v3.5", api_key="k", enabled=True)
        chunk = _make_chunk("c1", "text")
        sc = ScoredChunk(chunk_id="c1", score=0.5, chunk=chunk)
        results = adapter.rerank("q", [sc], top_k=1)
        assert results[0].chunk is chunk

    def test_disabled_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, "cohere", Client=object)
        from jera.adapters.ranking.cohere_reranker import CohereReranker

        with pytest.raises(RuntimeError, match="disabled by default"):
            CohereReranker()

    def test_missing_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, "cohere", Client=object)
        from jera.adapters.ranking.cohere_reranker import CohereReranker

        with pytest.raises(RuntimeError, match="api_key"):
            CohereReranker(enabled=True)


# ===========================================================================
# 3.  Qdrant Vector Store
# ===========================================================================


class _QdrantFakes:
    """Shared fake qdrant_client module factories."""

    @staticmethod
    def make(
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        recreate_calls: list[dict[str, Any]] = []
        upsert_calls: list[dict[str, Any]] = []
        query_calls: list[dict[str, Any]] = []

        class _FakeModels:
            """Namespace matching qdrant_client.models.*."""

            class Distance:
                COSINE = "Cosine"
                DOT = "Dot"

            class VectorParams:
                def __init__(self, size: int, distance: Any) -> None:
                    self.size = size
                    self.distance = distance

            class SparseVectorParams:
                pass

            class SparseVector:
                def __init__(self, indices: list[int], values: list[float]) -> None:
                    self.indices = indices
                    self.values = values

            class PointStruct:
                def __init__(self, id: str, vector: Any, payload: Any) -> None:
                    self.id = id
                    self.vector = vector
                    self.payload = payload

            class Prefetch:
                def __init__(self, query: Any, using: str, limit: int) -> None:
                    self.query = query
                    self.using = using
                    self.limit = limit

            class Fusion:
                RRF = "rrf"
                DBSF = "dbsf"

            class FusionQuery:
                def __init__(self, fusion: Any) -> None:
                    self.fusion = fusion

        class _FakeQdrantClient:
            def __init__(self, url: str, api_key: Any = None) -> None:
                self.url = url

            def recreate_collection(self, **kwargs: Any) -> None:
                recreate_calls.append(kwargs)

            def upsert(self, **kwargs: Any) -> None:
                upsert_calls.append(kwargs)

            def query_points(self, **kwargs: Any) -> Any:
                query_calls.append(kwargs)
                hit1 = types.SimpleNamespace(id="c1", score=0.95)
                hit2 = types.SimpleNamespace(id="c2", score=0.75)
                return types.SimpleNamespace(points=[hit1, hit2])

        qdrant_mod = types.ModuleType("qdrant_client")
        qdrant_mod.QdrantClient = _FakeQdrantClient  # type: ignore[attr-defined]
        qdrant_mod.models = _FakeModels  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "qdrant_client", qdrant_mod)
        monkeypatch.setitem(sys.modules, "qdrant_client.models", _FakeModels)  # type: ignore[arg-type]

        return recreate_calls, upsert_calls, query_calls


class TestQdrantVectorStore:
    """Verify ensure_collection, upsert, and search for QdrantVectorStore."""

    def _make_store(self) -> Any:
        from jera.adapters.vector_store.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(url="http://localhost:6333")

    # --- ensure_collection ---

    def test_ensure_collection_calls_recreate_with_correct_spec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recreate_calls, _, _ = _QdrantFakes.make(monkeypatch)
        store = self._make_store()
        spec = CollectionSpec(
            name="test_col",
            dense_dim=384,
            distance=Distance.COSINE,
            has_sparse=True,
        )
        store.ensure_collection(spec)
        assert len(recreate_calls) == 1
        call = recreate_calls[0]
        assert call["collection_name"] == "test_col"
        vp = call["vectors_config"]["dense"]
        assert vp.size == 384
        assert call["sparse_vectors_config"] is not None
        assert "sparse" in call["sparse_vectors_config"]

    def test_ensure_collection_no_sparse_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recreate_calls, _, _ = _QdrantFakes.make(monkeypatch)
        store = self._make_store()
        spec = CollectionSpec(name="c", dense_dim=128, has_sparse=False)
        store.ensure_collection(spec)
        assert recreate_calls[0]["sparse_vectors_config"] is None

    def test_ensure_collection_dot_distance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recreate_calls, _, _ = _QdrantFakes.make(monkeypatch)
        store = self._make_store()
        spec = CollectionSpec(name="c", dense_dim=128, distance=Distance.DOT)
        store.ensure_collection(spec)
        vp = recreate_calls[0]["vectors_config"]["dense"]
        assert vp.distance == "Dot"

    # --- upsert ---

    def test_upsert_sends_dense_and_sparse_vectors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, upsert_calls, _ = _QdrantFakes.make(monkeypatch)
        store = self._make_store()
        rec = VectorRecord(
            chunk_id="c1",
            document_id="doc1",
            dense=[0.1, 0.2, 0.3],
            sparse=SparseVector(indices=[10, 20], values=[1.0, 2.0]),
            payload={"source": "s1"},
        )
        store.upsert("my_col", [rec])
        assert len(upsert_calls) == 1
        call = upsert_calls[0]
        assert call["collection_name"] == "my_col"
        points = call["points"]
        assert len(points) == 1
        pt = points[0]
        assert pt.id == "c1"
        assert pt.vector["dense"] == [0.1, 0.2, 0.3]
        sv = pt.vector["sparse"]
        assert sv.indices == [10, 20]
        assert sv.values == [1.0, 2.0]
        assert pt.payload["document_id"] == "doc1"
        assert pt.payload["source"] == "s1"

    def test_upsert_dense_only_record(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, upsert_calls, _ = _QdrantFakes.make(monkeypatch)
        store = self._make_store()
        rec = VectorRecord(chunk_id="c2", document_id="doc1", dense=[0.5, 0.6])
        store.upsert("col", [rec])
        pt = upsert_calls[0]["points"][0]
        assert "dense" in pt.vector
        assert "sparse" not in pt.vector

    def test_upsert_multiple_records(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, upsert_calls, _ = _QdrantFakes.make(monkeypatch)
        store = self._make_store()
        recs = [
            VectorRecord(chunk_id=f"c{i}", document_id="doc1", dense=[float(i)]) for i in range(3)
        ]
        store.upsert("col", recs)
        assert len(upsert_calls[0]["points"]) == 3

    # --- search ---

    def test_search_returns_scored_chunks_from_hits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, query_calls = _QdrantFakes.make(monkeypatch)
        store = self._make_store()
        results = store.search(
            "my_col",
            dense=[0.1, 0.2],
            top_k=5,
        )
        assert len(results) == 2
        assert results[0].chunk_id == "c1"
        assert results[0].score == pytest.approx(0.95)
        assert results[1].chunk_id == "c2"
        assert results[1].score == pytest.approx(0.75)

    def test_search_sends_dense_prefetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, query_calls = _QdrantFakes.make(monkeypatch)
        store = self._make_store()
        store.search("col", dense=[0.1, 0.2], top_k=3)
        call = query_calls[0]
        assert call["collection_name"] == "col"
        assert call["limit"] == 3
        assert call["with_payload"] is True
        # Should have one prefetch for dense
        prefetch = call["prefetch"]
        assert any(p.using == "dense" for p in prefetch)

    def test_search_sends_sparse_prefetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, query_calls = _QdrantFakes.make(monkeypatch)
        store = self._make_store()
        sv = SparseVector(indices=[5], values=[0.8])
        store.search("col", sparse=sv, top_k=5)
        prefetch = query_calls[0]["prefetch"]
        assert any(p.using == "sparse" for p in prefetch)

    def test_search_hybrid_sends_both_prefetches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, query_calls = _QdrantFakes.make(monkeypatch)
        store = self._make_store()
        sv = SparseVector(indices=[1], values=[1.0])
        store.search("col", dense=[0.3, 0.4], sparse=sv, top_k=10)
        prefetch = query_calls[0]["prefetch"]
        usings = {p.using for p in prefetch}
        assert usings == {"dense", "sparse"}

    def test_search_rrf_fusion_used_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, query_calls = _QdrantFakes.make(monkeypatch)
        store = self._make_store()
        store.search("col", dense=[0.1], fusion=FusionMethod.RRF)
        fq = query_calls[0]["query"]
        assert fq.fusion == "rrf"

    def test_search_dbsf_fusion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _, _, query_calls = _QdrantFakes.make(monkeypatch)
        store = self._make_store()
        store.search("col", dense=[0.1], fusion=FusionMethod.DBSF)
        fq = query_calls[0]["query"]
        assert fq.fusion == "dbsf"


# ===========================================================================
# 4.  FastEmbed Sparse
# ===========================================================================


class TestFastEmbedSparse:
    """Verify SparseVector output from FastEmbedSparse."""

    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[Any]]:
        embed_calls: list[Any] = []

        class _FakeSparseResult:
            def __init__(self, indices: list[int], values: list[float]) -> None:
                self.indices = indices
                self.values = values

        class _FakeSparseTextEmbedding:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def embed(self, texts: list[str]) -> Iterator[_FakeSparseResult]:
                embed_calls.append(texts)
                yield _FakeSparseResult([1, 2, 3], [0.5, 0.3, 0.1])
                if len(texts) > 1:
                    yield _FakeSparseResult([4, 5], [0.9, 0.7])

        _install(monkeypatch, "fastembed", SparseTextEmbedding=_FakeSparseTextEmbedding)
        from jera.adapters.sparse.fastembed_sparse import FastEmbedSparse

        adapter = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1")
        return adapter, embed_calls

    def test_encode_returns_sparse_vectors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, calls = self._setup(monkeypatch)
        results = adapter.encode(["hello", "world"])
        assert len(results) == 2
        assert isinstance(results[0], SparseVector)

    def test_encode_yields_correct_indices_and_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter, calls = self._setup(monkeypatch)
        results = adapter.encode(["text"])
        sv = results[0]
        assert sv.indices == [1, 2, 3]
        assert sv.values == pytest.approx([0.5, 0.3, 0.1])

    def test_encode_passes_texts_to_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, calls = self._setup(monkeypatch)
        adapter.encode(["foo", "bar"])
        assert calls[0] == ["foo", "bar"]

    def test_encode_query_returns_single_sparse_vector(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter, _ = self._setup(monkeypatch)
        result = adapter.encode_query("query text")
        assert isinstance(result, SparseVector)
        assert result.indices == [1, 2, 3]

    def test_encode_values_are_floats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, _ = self._setup(monkeypatch)
        sv = adapter.encode_query("q")
        assert all(isinstance(v, float) for v in sv.values)
        assert all(isinstance(i, int) for i in sv.indices)


# ===========================================================================
# 5.  FastEmbed Dense Embedding
# ===========================================================================


class TestFastEmbedEmbedding:
    """Verify DenseVector output from FastEmbedEmbedding."""

    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[Any]]:
        embed_calls: list[Any] = []

        class _FakeTextEmbedding:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def embed(self, texts: list[str]) -> Iterator[list[float]]:
                embed_calls.append(texts)
                for _ in texts:
                    yield [0.1, 0.2, 0.3]

        _install(monkeypatch, "fastembed", TextEmbedding=_FakeTextEmbedding)
        from jera.adapters.embedding.fastembed_embedding import FastEmbedEmbedding

        adapter = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return adapter, embed_calls

    def test_embed_returns_list_of_dense_vectors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, _ = self._setup(monkeypatch)
        results = adapter.embed(["a", "b"])
        assert len(results) == 2
        assert results[0] == pytest.approx([0.1, 0.2, 0.3])
        assert results[1] == pytest.approx([0.1, 0.2, 0.3])

    def test_embed_passes_texts_as_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, calls = self._setup(monkeypatch)
        adapter.embed(["x", "y", "z"])
        assert calls[0] == ["x", "y", "z"]

    def test_embed_query_returns_single_vector(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, _ = self._setup(monkeypatch)
        result = adapter.embed_query("single text")
        assert result == pytest.approx([0.1, 0.2, 0.3])

    def test_values_are_floats(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, _ = self._setup(monkeypatch)
        result = adapter.embed_query("q")
        assert all(isinstance(v, float) for v in result)

    def test_dimensions_reflect_known_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, _ = self._setup(monkeypatch)
        assert adapter.dimensions == 384

    def test_dimensions_for_bge_m3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embed_calls: list[Any] = []

        class _FakeTextEmbedding:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def embed(self, texts: list[str]) -> Iterator[list[float]]:
                embed_calls.append(texts)
                yield [0.0] * 1024

        _install(monkeypatch, "fastembed", TextEmbedding=_FakeTextEmbedding)
        from importlib import reload

        import jera.adapters.embedding.fastembed_embedding as _mod

        reload(_mod)
        adapter = _mod.FastEmbedEmbedding(model_name="BAAI/bge-m3")
        assert adapter.dimensions == 1024


# ===========================================================================
# 6.  FastEmbed Reranker
# ===========================================================================


class TestFastEmbedReranker:
    """Verify reordering + score update from FastEmbedReranker."""

    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[Any]]:
        rerank_calls: list[Any] = []

        class _FakeTextCrossEncoder:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def rerank(self, query: str, docs: list[str]) -> list[float]:
                rerank_calls.append({"query": query, "docs": docs})
                # Return scores in same order as docs
                return [0.3, 0.8, 0.1]

        rerank_mod = types.ModuleType("fastembed.rerank.cross_encoder")
        rerank_mod.TextCrossEncoder = _FakeTextCrossEncoder  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", rerank_mod)
        monkeypatch.setitem(sys.modules, "fastembed.rerank", types.ModuleType("fastembed.rerank"))
        monkeypatch.setitem(sys.modules, "fastembed", types.ModuleType("fastembed"))

        from jera.adapters.ranking.fastembed_reranker import FastEmbedReranker

        adapter = FastEmbedReranker(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
        return adapter, rerank_calls

    def _candidates(self) -> list[ScoredChunk]:
        return [
            ScoredChunk(chunk_id="c1", score=0.5, chunk=_make_chunk("c1", "alpha")),
            ScoredChunk(chunk_id="c2", score=0.4, chunk=_make_chunk("c2", "beta")),
            ScoredChunk(chunk_id="c3", score=0.3, chunk=_make_chunk("c3", "gamma")),
        ]

    def test_rerank_sorts_by_score_descending(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, _ = self._setup(monkeypatch)
        results = adapter.rerank("q", self._candidates(), top_k=3)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_updates_scores_from_cross_encoder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter, _ = self._setup(monkeypatch)
        results = adapter.rerank("q", self._candidates(), top_k=3)
        # c2 had score 0.8, c1 had 0.3, c3 had 0.1
        assert results[0].chunk_id == "c2"
        assert results[0].score == pytest.approx(0.8)
        assert results[0].components["rerank"] == pytest.approx(0.8)

    def test_rerank_respects_top_k(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, _ = self._setup(monkeypatch)
        results = adapter.rerank("q", self._candidates(), top_k=2)
        assert len(results) == 2

    def test_rerank_sends_correct_texts_to_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter, calls = self._setup(monkeypatch)
        adapter.rerank("my query", self._candidates(), top_k=3)
        assert calls[0]["query"] == "my query"
        assert calls[0]["docs"] == ["alpha", "beta", "gamma"]

    def test_rerank_handles_chunk_none_as_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter, calls = self._setup(monkeypatch)
        candidates = [ScoredChunk(chunk_id="c1", score=0.5)]
        # Only 1 candidate but model returns 3 scores — override the reranker to return 1 score
        rerank_calls: list[Any] = []

        class _FakeTCE:
            def __init__(self, model_name: str) -> None:
                pass

            def rerank(self, query: str, docs: list[str]) -> list[float]:
                rerank_calls.append(docs)
                return [0.7]

        rerank_mod = types.ModuleType("fastembed.rerank.cross_encoder")
        rerank_mod.TextCrossEncoder = _FakeTCE  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", rerank_mod)
        from importlib import reload

        import jera.adapters.ranking.fastembed_reranker as _mod

        reload(_mod)
        adapter2 = _mod.FastEmbedReranker()
        results = adapter2.rerank("q", candidates, top_k=1)
        assert rerank_calls[0] == [""]  # chunk is None → empty string
        assert results[0].score == pytest.approx(0.7)


# ===========================================================================
# 7.  Docling Parser
# ===========================================================================


class TestDoclingParser:
    """Verify DoclingParser maps a canned docling document into ParsedDocument."""

    def _setup(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        """Install fake docling modules and return a DoclingParser."""

        # Build a fake item for the document
        class _FakeLabel:
            def __init__(self, value: str) -> None:
                self.value = value

        class _FakeItem:
            def __init__(self, label_value: str, text: str, level: int = 0) -> None:
                self.label = _FakeLabel(label_value)
                self.text = text
                self.prov: list[Any] = []

        class _FakeDoc:
            def iterate_items(self) -> Iterator[tuple[Any, int]]:
                yield _FakeItem("title", "My Title"), 1
                yield _FakeItem("text", "Some content here."), 0
                yield _FakeItem("table", ""), 0  # will be skipped (empty text)
                yield _FakeItem("list_item", "Bullet point."), 0

        class _FakeConvertResult:
            document = _FakeDoc()

        class _FakeDocumentConverter:
            def __init__(self) -> None:
                pass

            def convert(self, stream: Any) -> _FakeConvertResult:
                return _FakeConvertResult()

        class _FakeDocumentStream:
            def __init__(self, name: str, stream: Any) -> None:
                self.name = name
                self.stream = stream

        # Install fake docling modules
        docling_mod = types.ModuleType("docling")
        docling_dc_mod = types.ModuleType("docling.document_converter")
        docling_dc_mod.DocumentConverter = _FakeDocumentConverter  # type: ignore[attr-defined]
        docling_base_mod = types.ModuleType("docling.datamodel.base_models")
        docling_base_mod.DocumentStream = _FakeDocumentStream  # type: ignore[attr-defined]
        docling_dm_mod = types.ModuleType("docling.datamodel")

        monkeypatch.setitem(sys.modules, "docling", docling_mod)
        monkeypatch.setitem(sys.modules, "docling.document_converter", docling_dc_mod)
        monkeypatch.setitem(sys.modules, "docling.datamodel", docling_dm_mod)
        monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", docling_base_mod)

        from jera.adapters.parsing.docling_parser import DoclingParser

        return DoclingParser()

    def _source_ref(self) -> Any:
        from jera.domain.document import MediaType, SourceRef

        return SourceRef(source_id="test_src", media_type=MediaType.PDF, content=b"%PDF-fake")

    def test_parse_returns_parsed_document(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = self._setup(monkeypatch)
        doc = parser.parse(self._source_ref())
        assert isinstance(doc, ParsedDocument)

    def test_parse_provenance_has_parser_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = self._setup(monkeypatch)
        doc = parser.parse(self._source_ref())
        assert doc.provenance.parser_name == "docling"
        assert doc.provenance.source_id == "test_src"

    def test_parse_maps_title_element(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jera.domain.document import ElementType

        parser = self._setup(monkeypatch)
        doc = parser.parse(self._source_ref())
        titles = [e for e in doc.elements if e.type is ElementType.TITLE]
        assert len(titles) == 1
        assert titles[0].text == "My Title"

    def test_parse_sets_document_title_from_first_heading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parser = self._setup(monkeypatch)
        doc = parser.parse(self._source_ref())
        assert doc.title == "My Title"

    def test_parse_maps_narrative_text_element(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jera.domain.document import ElementType

        parser = self._setup(monkeypatch)
        doc = parser.parse(self._source_ref())
        narr = [e for e in doc.elements if e.type is ElementType.NARRATIVE_TEXT]
        assert any("Some content" in e.text for e in narr)

    def test_parse_maps_list_item_element(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jera.domain.document import ElementType

        parser = self._setup(monkeypatch)
        doc = parser.parse(self._source_ref())
        items = [e for e in doc.elements if e.type is ElementType.LIST_ITEM]
        assert any("Bullet" in e.text for e in items)

    def test_parse_skips_empty_text_elements(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = self._setup(monkeypatch)
        doc = parser.parse(self._source_ref())
        assert all(e.text.strip() for e in doc.elements)

    def test_parse_assigns_section_path_to_content_under_heading(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parser = self._setup(monkeypatch)
        doc = parser.parse(self._source_ref())
        # content after the title heading should carry the title in section_path
        from jera.domain.document import ElementType

        narr = [e for e in doc.elements if e.type is ElementType.NARRATIVE_TEXT]
        assert any(e.section_path == ("My Title",) for e in narr)

    def test_parse_preserves_element_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = self._setup(monkeypatch)
        doc = parser.parse(self._source_ref())
        orders = [e.order for e in doc.elements]
        assert orders == list(range(len(orders)))

    def test_parse_page_span_defaults_to_page_1_when_no_prov(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        parser = self._setup(monkeypatch)
        doc = parser.parse(self._source_ref())
        # All fake items have empty prov, so all should default to PageSpan(1,1)
        assert all(e.page_span.start_page == 1 for e in doc.elements)

    def test_supports_pdf_html_markdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jera.domain.document import MediaType, SourceRef

        parser = self._setup(monkeypatch)
        for mt in (MediaType.PDF, MediaType.HTML, MediaType.MARKDOWN):
            src = SourceRef(source_id="s", media_type=mt, content=b"x")
            assert parser.supports(src)

    def test_table_exported_via_export_to_markdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When item.text is empty but export_to_markdown exists, use that."""

        class _FakeLabel:
            def __init__(self, value: str) -> None:
                self.value = value

        class _TableItem:
            label = _FakeLabel("table")
            text = ""
            prov: list[Any] = []

            def export_to_markdown(self, doc: Any) -> str:
                return "| col | val |\n| --- | --- |"

        class _FakeDoc2:
            def iterate_items(self) -> Iterator[tuple[Any, int]]:
                yield _TableItem(), 0

        class _FakeConvertResult2:
            document = _FakeDoc2()

        class _FakeDocumentConverter2:
            def convert(self, stream: Any) -> _FakeConvertResult2:
                return _FakeConvertResult2()

        class _FakeDocumentStream2:
            def __init__(self, name: str, stream: Any) -> None:
                pass

        docling_mod = types.ModuleType("docling")
        docling_dc_mod = types.ModuleType("docling.document_converter")
        docling_dc_mod.DocumentConverter = _FakeDocumentConverter2  # type: ignore[attr-defined]
        docling_base_mod = types.ModuleType("docling.datamodel.base_models")
        docling_base_mod.DocumentStream = _FakeDocumentStream2  # type: ignore[attr-defined]
        docling_dm_mod = types.ModuleType("docling.datamodel")

        monkeypatch.setitem(sys.modules, "docling", docling_mod)
        monkeypatch.setitem(sys.modules, "docling.document_converter", docling_dc_mod)
        monkeypatch.setitem(sys.modules, "docling.datamodel", docling_dm_mod)
        monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", docling_base_mod)

        from importlib import reload

        import jera.adapters.parsing.docling_parser as _mod

        reload(_mod)
        parser = _mod.DoclingParser()
        from jera.domain.document import ElementType, MediaType, SourceRef

        doc = parser.parse(SourceRef(source_id="t", media_type=MediaType.PDF, content=b"x"))
        tables = [e for e in doc.elements if e.type is ElementType.TABLE]
        assert len(tables) == 1
        assert "col" in tables[0].text


# ===========================================================================
# 8.  Postgres Store (URL building + shared SqlMetadataStore via sqlite)
# ===========================================================================


class TestPostgresStore:
    """Verify make_postgres_store builds the engine with the given DSN and
    that the shared SqlMetadataStore logic round-trips against sqlite."""

    def _patch_sqlalchemy_create_engine(
        self,
        monkeypatch: pytest.MonkeyPatch,
        captured_dsns: list[str],
    ) -> None:
        """Patch sqlalchemy.create_engine at the module level so the local import
        inside make_postgres_store picks up the fake.

        We capture the real create_engine BEFORE patching to avoid recursion.
        """
        import sqlalchemy
        from sqlalchemy.pool import StaticPool

        _real_ce = sqlalchemy.create_engine

        def _fake(url: str, **kwargs: Any) -> Any:
            captured_dsns.append(url)
            return _real_ce(
                "sqlite://",
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )

        monkeypatch.setattr(sqlalchemy, "create_engine", _fake)

    def test_make_postgres_store_calls_create_engine_with_dsn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Assert the DSN is forwarded verbatim to create_engine."""
        captured_dsns: list[str] = []
        self._patch_sqlalchemy_create_engine(monkeypatch, captured_dsns)

        import jera.adapters.metadata_store.postgres_store as pg_mod

        dsn = "postgresql+psycopg://user:pw@localhost/mydb"
        pg_mod.make_postgres_store(dsn)
        assert len(captured_dsns) == 1
        assert captured_dsns[0] == dsn

    def test_make_postgres_store_returns_sql_metadata_store(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_dsns: list[str] = []
        self._patch_sqlalchemy_create_engine(monkeypatch, captured_dsns)

        import jera.adapters.metadata_store.postgres_store as pg_mod
        from jera.adapters.metadata_store.sql_store import SqlMetadataStore

        store = pg_mod.make_postgres_store("postgresql+psycopg://h/db")
        assert isinstance(store, SqlMetadataStore)

    def test_sql_metadata_store_round_trips_document_and_chunks_on_sqlite(self) -> None:
        """Engine-agnostic SqlMetadataStore logic verified against real sqlite."""
        from jera.adapters.metadata_store.sqlite_store import make_sqlite_store
        from jera.domain.document import (
            MediaType,
            PageSpan,
            ParsedDocument,
        )

        store = make_sqlite_store(":memory:")

        doc = ParsedDocument(
            document_id="doc-pg-1",
            source_id="src-pg",
            title="Postgres Test Doc",
            elements=[],
            provenance=Provenance(
                source_id="src-pg",
                parser_name="markdown",
                parser_version="1.0",
                media_type=MediaType.MARKDOWN,
            ),
        )
        store.save_document(doc)

        chunk = Chunk(
            chunk_id="chunk-pg-1",
            document_id="doc-pg-1",
            source_id="src-pg",
            text="This is test content for postgres store.",
            page_span=PageSpan.single(1),
            section_path=("Intro", "Details"),
            element_ids=("e1", "e2"),
            char_span=(0, 40),
            token_count=8,
            chunk_strategy="heading_aware",
            chunk_version="1.0.0",
        )
        store.save_chunks([chunk])

        retrieved = store.get_chunk("chunk-pg-1")
        assert retrieved is not None
        assert retrieved.chunk_id == "chunk-pg-1"
        assert retrieved.document_id == "doc-pg-1"
        assert retrieved.text == "This is test content for postgres store."
        assert retrieved.section_path == ("Intro", "Details")
        assert retrieved.element_ids == ("e1", "e2")
        assert retrieved.char_span == (0, 40)
        assert retrieved.token_count == 8

    def test_sql_metadata_store_get_chunks_preserves_order(self) -> None:
        from jera.adapters.metadata_store.sqlite_store import make_sqlite_store

        store = make_sqlite_store(":memory:")
        chunks = [
            Chunk(
                chunk_id=f"cid-{i}",
                document_id="doc1",
                source_id="src1",
                text=f"text {i}",
                page_span=PageSpan.single(1),
                section_path=(),
                element_ids=(),
                char_span=(0, 6),
                token_count=2,
                chunk_strategy="s",
                chunk_version="1",
            )
            for i in range(5)
        ]
        store.save_chunks(chunks)
        # Request in reverse order
        ids = ["cid-4", "cid-2", "cid-0"]
        results = store.get_chunks(ids)
        assert [r.chunk_id for r in results] == ids

    def test_create_schema_flag_creates_tables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sqlalchemy import create_engine, inspect
        from sqlalchemy.pool import StaticPool

        # Build the real engine BEFORE patching so the lambda doesn't recurse.
        real_engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        import sqlalchemy

        # Patch after real_engine is already constructed — safe, no recursion.
        monkeypatch.setattr(sqlalchemy, "create_engine", lambda url, **kw: real_engine)

        import jera.adapters.metadata_store.postgres_store as pg_mod

        pg_mod.make_postgres_store("postgresql+psycopg://h/db", create_schema=True)
        names = set(inspect(real_engine).get_table_names())
        assert "chunks" in names
        assert "documents" in names
