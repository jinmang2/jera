"""Retrieval gate (Gate 4, CI/test profile — deterministic, no torch).

Exercises dense-only, sparse-only, hybrid-RRF, and rerank stages, and proves a
NON-TAUTOLOGICAL hybrid lift: a target that is #1 in neither modality alone but #1 after RRF.
The semantic-paraphrase superiority case lives under the `local` profile (skipped here).
"""

from __future__ import annotations

from jera.adapters.embedding.hash_embedding import HashEmbedding
from jera.adapters.ranking.identity_reranker import IdentityReranker
from jera.adapters.sparse.bm25_local import BM25Local
from jera.adapters.vector_store.in_memory import InMemoryVectorStore
from jera.domain.retrieval import FusionMethod, ScoredChunk
from jera.ports.vector_store import CollectionSpec, VectorRecord

# Corpus discovered to exhibit genuine fusion lift under hash-embedding + BM25 (frozen).
QUERY = "bravo india alpha"
CORPUS = {
    "T": "bravo india juliet charlie",  # target: #2 dense, #2 sparse
    "d0": "hotel india charlie alpha juliet",  # sparse leader (#1 sparse), weak dense
    "d1": "bravo bravo charlie foxtrot",
    "d2": "alpha bravo delta",  # dense leader (#1 dense), weak sparse
    "d3": "bravo charlie golf juliet",
    "d4": "alpha alpha alpha alpha alpha alpha juliet golf",
    "d5": "bravo bravo bravo foxtrot delta",
}


def _build() -> InMemoryVectorStore:
    emb = HashEmbedding(dimensions=256)
    bm = BM25Local()
    ids = list(CORPUS)
    texts = list(CORPUS.values())
    bm.fit(texts)
    store = InMemoryVectorStore()
    store.ensure_collection(CollectionSpec(name="c", dense_dim=256))
    dense = emb.embed(texts)
    sparse = bm.encode(texts)
    store.upsert(
        "c",
        [
            VectorRecord(chunk_id=i, document_id="doc", dense=d, sparse=s)
            for i, d, s in zip(ids, dense, sparse, strict=True)
        ],
    )
    return store


def _q():
    return HashEmbedding(256).embed_query(QUERY), _bm_query()


def _bm_query():
    bm = BM25Local()
    bm.fit(list(CORPUS.values()))
    return bm.encode_query(QUERY)


def test_dense_only_and_sparse_only_have_different_leaders() -> None:
    store = _build()
    dq, sq = _q()
    dense = [c.chunk_id for c in store.search("c", dense=dq, top_k=10)]
    sparse = [c.chunk_id for c in store.search("c", sparse=sq, top_k=10)]
    assert dense[0] == "d2"  # dense leader
    assert sparse[0] == "d0"  # sparse leader
    assert dense[0] != sparse[0]


def test_hybrid_rrf_produces_genuine_fusion_lift() -> None:
    store = _build()
    dq, sq = _q()
    dense = [c.chunk_id for c in store.search("c", dense=dq, top_k=10)]
    sparse = [c.chunk_id for c in store.search("c", sparse=sq, top_k=10)]
    hybrid = [
        c.chunk_id
        for c in store.search("c", dense=dq, sparse=sq, top_k=10, fusion=FusionMethod.RRF)
    ]

    # Target is #1 in NEITHER modality alone ...
    assert dense[0] != "T"
    assert sparse[0] != "T"
    assert dense.index("T") == 1 and sparse.index("T") == 1  # genuinely #2 in both
    # ... but #1 after RRF fusion.
    assert hybrid[0] == "T"


def test_dimension_guard_rejects_mismatched_query_vector() -> None:
    store = _build()
    import pytest

    with pytest.raises(ValueError, match="dim"):
        store.search("c", dense=[0.0] * 10, top_k=5)  # collection dim is 256


def test_identity_rerank_is_score_stable() -> None:
    candidates = [
        ScoredChunk(chunk_id="b", score=0.5),
        ScoredChunk(chunk_id="a", score=0.9),
        ScoredChunk(chunk_id="c", score=0.5),
    ]
    out = IdentityReranker().rerank("q", candidates, top_k=3)
    assert [c.chunk_id for c in out] == ["a", "b", "c"]  # score desc, then id asc
    assert out[0].components["rerank"] == 0.9
