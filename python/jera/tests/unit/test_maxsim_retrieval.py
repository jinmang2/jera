"""Late-interaction (ColBERT-style MaxSim) retrieval — unit tests.

NON-TAUTOLOGICAL PROPERTY PROVEN
---------------------------------
MaxSim beats naive single-vector bag-of-words (HashEmbedding) on a *partial-overlap*
case where the target document shares only ONE token with the query but that token is
highly specific.  The single-vector approach averages all tokens and is diluted by the
many unrelated tokens in the distractor documents.  MaxSim, by taking the per-query-token
MAXIMUM over document tokens, lets the single overlapping token dominate and correctly
ranks the target #1.

Specifically:
  query  = "alpha"               (one query token)
  target = "alpha zzzz"         (one exact query-token match + one unique token)
  noise  = "bravo charlie delta echo foxtrot golf"   (no overlap with query, but 6 tokens)

With HashEmbedding (single-vector bag-of-words), the noise chunk accumulates magnitude
from its six tokens and — after L2 normalisation — its dense vector may score similarly to
or higher than the two-token target.  MaxSim, by contrast, takes max cosine per query
token: the target's exact "alpha" token produces cosine ≈ 1.0, while the noise chunk's
best match to "alpha" is near 0 (no overlap).  MaxSim ranks the target #1; the
single-vector store does not always agree.

The opt-in real adapter would wrap BGE-M3's ColBERT head (not built here).
"""

from __future__ import annotations

import math

from jera.adapters.embedding.hash_embedding import HashEmbedding
from jera.adapters.embedding.hash_multivector import HashMultiVectorEmbedding
from jera.adapters.vector_store.maxsim_store import MaxSimVectorStore, _cosine, _maxsim_score
from jera.ports.multi_vector_embedding import MultiVectorEmbedding
from jera.ports.multi_vector_store import MultiVectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_store(
    corpus: dict[str, str],
    dimensions: int = 64,
) -> tuple[HashMultiVectorEmbedding, MaxSimVectorStore]:
    emb = HashMultiVectorEmbedding(dimensions=dimensions)
    store = MaxSimVectorStore()
    texts = list(corpus.values())
    ids = list(corpus.keys())
    matrices = emb.embed_multi(texts)
    store.add(list(zip(ids, matrices, strict=True)))
    return emb, store


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_hash_multivector_satisfies_protocol() -> None:
    emb = HashMultiVectorEmbedding()
    assert isinstance(emb, MultiVectorEmbedding)


def test_maxsim_store_satisfies_protocol() -> None:
    store = MaxSimVectorStore()
    assert isinstance(store, MultiVectorStore)


# ---------------------------------------------------------------------------
# MaxSim formula — hand-computed unit test
# ---------------------------------------------------------------------------


def test_maxsim_formula_hand_computed() -> None:
    """Verify _maxsim_score matches a manual calculation on a 2-D toy example."""
    # q = [[1, 0], [0, 1]]  (two query tokens, already unit vectors)
    # d = [[1, 0], [0.6, 0.8]]  (two doc tokens)
    # cosine(q0, d0) = 1.0,  cosine(q0, d1) = 0.6  →  max = 1.0
    # cosine(q1, d0) = 0.0,  cosine(q1, d1) = 0.8  →  max = 0.8
    # expected total = 1.0 + 0.8 = 1.8
    q = [[1.0, 0.0], [0.0, 1.0]]
    d = [[1.0, 0.0], [0.6, 0.8]]
    score = _maxsim_score(q, d)
    assert abs(score - 1.8) < 1e-9


def test_cosine_identical_vectors() -> None:
    v = [0.6, 0.8]
    assert abs(_cosine(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal_vectors() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(_cosine(a, b)) < 1e-9


def test_cosine_zero_vector_returns_zero() -> None:
    a = [0.0, 0.0]
    b = [1.0, 0.0]
    assert _cosine(a, b) == 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_embedding_is_deterministic() -> None:
    emb = HashMultiVectorEmbedding(dimensions=32)
    r1 = emb.embed_query_multi("neural retrieval benchmark")
    r2 = emb.embed_query_multi("neural retrieval benchmark")
    assert r1 == r2


def test_same_token_same_vector() -> None:
    """The fundamental property: same token text → identical vector regardless of context."""
    emb = HashMultiVectorEmbedding(dimensions=32)
    # "alpha" appears in two different texts; its vector must be identical in both.
    # embed_multi(["alpha bravo"]) → [[alpha_vec, bravo_vec]]  (1 text, 2 tokens)
    result_ab = emb.embed_multi(["alpha bravo"])
    v_alpha_a = result_ab[0][0]  # first token of "alpha bravo"
    result_a = emb.embed_multi(["alpha"])
    v_alpha_b = result_a[0][0]  # sole token of "alpha"
    assert v_alpha_a == v_alpha_b


def test_per_token_vectors_are_unit_length() -> None:
    emb = HashMultiVectorEmbedding(dimensions=64)
    matrices = emb.embed_multi(["hello world foo bar"])
    for tok_vec in matrices[0]:
        norm = math.sqrt(sum(v * v for v in tok_vec))
        assert abs(norm - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Core retrieval correctness — target ranks #1
# ---------------------------------------------------------------------------

# Corpus designed so that the target "T" shares a specific token ("alpha") with the query
# while the distractors share no query tokens at all.
QUERY_TEXT = "alpha"
CORPUS: dict[str, str] = {
    "T": "alpha zzzz",  # one exact query-token match + one unique filler token
    "d0": "bravo charlie delta",  # no overlap with query
    "d1": "echo foxtrot golf hotel",  # no overlap with query
    "d2": "india juliet kilo lima",  # no overlap with query
}


def test_target_ranks_first_by_maxsim() -> None:
    emb, store = _build_store(CORPUS)
    q_vecs = emb.embed_query_multi(QUERY_TEXT)
    results = store.search_maxsim(q_vecs, top_k=4)
    assert results[0].chunk_id == "T", (
        f"Expected 'T' at rank 1, got {[r.chunk_id for r in results]}"
    )


def test_maxsim_scores_stored_in_components() -> None:
    emb, store = _build_store(CORPUS)
    q_vecs = emb.embed_query_multi(QUERY_TEXT)
    results = store.search_maxsim(q_vecs, top_k=4)
    for r in results:
        assert "maxsim" in r.components
        assert abs(r.components["maxsim"] - r.score) < 1e-12


# ---------------------------------------------------------------------------
# NON-TAUTOLOGICAL: MaxSim beats naive single-vector on partial-overlap case
# ---------------------------------------------------------------------------


def test_maxsim_beats_single_vector_on_partial_overlap() -> None:
    """Prove a real MaxSim advantage over HashEmbedding (bag-of-words single vector).

    Corpus:
      target  = "alpha zzzz"               (1 query-token match out of 2 tokens)
      noise   = "bravo charlie delta echo foxtrot golf"   (0 query-token matches, 6 tokens)

    Query = "alpha".

    HashEmbedding aggregates ALL tokens into one vector; "zzzz" dilutes the target and
    the noise chunk's 6 tokens build a dense vector that may beat the 2-token target.
    MaxSim takes the maximum cosine per query token — the target's "alpha" token scores
    cosine ≈ 1.0, noise's best token scores ≈ 0, so MaxSim correctly ranks target #1.

    The test asserts:
    1. MaxSim ranks target #1 (score strictly above all distractors).
    2. The same-token identity property gives the target a MaxSim score > 0.9
       (since "alpha" → "alpha" cosine = 1.0 is the dominant contribution).
    3. The noise distractor's MaxSim score < target's score by a meaningful margin,
       proving the formula distinguishes them — not just the corpus arrangement.
    """
    dimensions = 64
    corpus = {
        "target": "alpha zzzz",
        "noise": "bravo charlie delta echo foxtrot golf",
    }
    emb_multi = HashMultiVectorEmbedding(dimensions=dimensions)
    store = MaxSimVectorStore()
    matrices = emb_multi.embed_multi(list(corpus.values()))
    store.add(list(zip(corpus.keys(), matrices, strict=True)))

    q_vecs = emb_multi.embed_query_multi("alpha")
    results = store.search_maxsim(q_vecs, top_k=2)
    maxsim_ranking = [r.chunk_id for r in results]

    # MaxSim must rank target first.
    assert maxsim_ranking[0] == "target", f"MaxSim ranking: {maxsim_ranking}"

    # The target's MaxSim score is dominated by the exact "alpha" token match → near 1.0.
    target_score = results[0].score
    noise_score = results[1].score
    assert target_score > 0.9, f"Expected target MaxSim score > 0.9, got {target_score}"

    # Meaningful gap: target beats noise by at least 0.5 (noise has cosine ≈ 0 for "alpha").
    assert target_score - noise_score > 0.5, (
        f"Gap too small: target={target_score:.4f}, noise={noise_score:.4f}"
    )

    # Now show the single-vector (bag-of-words) baseline does NOT always achieve this
    # separation with the same corpus.  We measure the gap and confirm MaxSim gap is larger.
    emb_single = HashEmbedding(dimensions=dimensions)
    q_single = emb_single.embed_query("alpha")

    doc_vecs_single = emb_single.embed(list(corpus.values()))
    qn = math.sqrt(sum(v * v for v in q_single))

    def single_cosine(q: list[float], d: list[float], qnorm: float) -> float:
        dot = sum(x * y for x, y in zip(q, d, strict=True))
        dn = math.sqrt(sum(x * x for x in d))
        if qnorm == 0.0 or dn == 0.0:
            return 0.0
        return dot / (qnorm * dn)

    single_scores = {
        cid: single_cosine(q_single, dvec, qn)
        for cid, dvec in zip(corpus.keys(), doc_vecs_single, strict=True)
    }
    single_gap = single_scores["target"] - single_scores["noise"]

    # MaxSim gap must strictly exceed the single-vector gap, proving it is a better
    # discriminator on this partial-overlap case.
    maxsim_gap = target_score - noise_score
    assert maxsim_gap > single_gap, (
        f"MaxSim gap {maxsim_gap:.4f} should exceed single-vector gap {single_gap:.4f}; "
        "if this fires the hash vectors may have unexpected collision structure — "
        "try increasing dimensions."
    )


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_removes_chunk() -> None:
    emb, store = _build_store(CORPUS)
    q_vecs = emb.embed_query_multi(QUERY_TEXT)

    # Confirm T is present before delete.
    before = [r.chunk_id for r in store.search_maxsim(q_vecs, top_k=4)]
    assert "T" in before

    store.delete(["T"])

    after = [r.chunk_id for r in store.search_maxsim(q_vecs, top_k=4)]
    assert "T" not in after


def test_delete_unknown_id_is_noop() -> None:
    _, store = _build_store(CORPUS)
    # Should not raise.
    store.delete(["nonexistent_chunk_xyz"])


def test_delete_all_returns_empty() -> None:
    emb, store = _build_store(CORPUS)
    q_vecs = emb.embed_query_multi(QUERY_TEXT)
    store.delete(list(CORPUS.keys()))
    results = store.search_maxsim(q_vecs, top_k=4)
    assert results == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty() -> None:
    _, store = _build_store(CORPUS)
    # embed_multi on empty string produces a single <empty> token; but passing an
    # explicit empty list of query vectors should return empty.
    results = store.search_maxsim([], top_k=4)
    assert results == []


def test_top_k_limits_results() -> None:
    emb, store = _build_store(CORPUS)
    q_vecs = emb.embed_query_multi(QUERY_TEXT)
    results = store.search_maxsim(q_vecs, top_k=2)
    assert len(results) <= 2


def test_upsert_semantics() -> None:
    """Adding the same chunk_id twice keeps only the latest token vectors."""
    emb = HashMultiVectorEmbedding(dimensions=32)
    store = MaxSimVectorStore()
    vecs_a = emb.embed_query_multi("alpha")
    vecs_b = emb.embed_query_multi("zeta omega")
    store.add([("c1", vecs_a)])
    store.add([("c1", vecs_b)])  # overwrite
    # Only one entry for c1 should exist.
    results = store.search_maxsim(emb.embed_query_multi("zeta"), top_k=5)
    c1_results = [r for r in results if r.chunk_id == "c1"]
    assert len(c1_results) == 1


def test_results_sorted_descending_by_score() -> None:
    emb, store = _build_store(CORPUS)
    q_vecs = emb.embed_query_multi(QUERY_TEXT)
    results = store.search_maxsim(q_vecs, top_k=4)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
