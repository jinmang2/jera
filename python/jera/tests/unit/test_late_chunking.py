"""Unit tests for LateChunkingEmbedding (arXiv:2409.04701).

Non-tautological coreference fixture
-------------------------------------
The core claim of late chunking is that a pronoun/anaphora chunk (e.g. "It will ship
next year.") gains similarity to an entity-named query ("Tesla") when its embedding is
context-mixed with its antecedent chunk ("Tesla announced a new EV.").

With HashEmbedding (bag-of-hashed-tokens, lexical):

  * chunk 1 isolated  → zero overlap with "Tesla" → low cosine similarity
  * chunk 1 late-chunked (mixed with chunk 0 which *does* contain "Tesla") →
    chunk 0's Tesla-token contribution bleeds into chunk 1's vector → higher cosine

This is a *real arithmetic consequence* of the mean-pooling formula, not a mock
assertion.  The test will fail if the implementation is wrong (e.g. alpha=0, wrong
window, missing normalisation) because cosine(query, chunk1_late) would not exceed
cosine(query, chunk1_isolated).
"""

from __future__ import annotations

import math

import pytest

from jera.adapters.embedding.hash_embedding import HashEmbedding
from jera.adapters.embedding.late_chunking import LateChunkingEmbedding

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors (assumes unit-length for speed)."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CHUNK_0 = "Tesla announced a new electric vehicle."  # entity-bearing
CHUNK_1 = "It will ship next year."  # pronoun/anaphora – no entity token
CHUNK_2 = "The weather forecast shows rain all week."  # unrelated

QUERY = "Tesla"

DOCUMENT_CHUNKS = [CHUNK_0, CHUNK_1, CHUNK_2]


@pytest.fixture()
def base() -> HashEmbedding:
    return HashEmbedding(dimensions=256)


@pytest.fixture()
def late(base: HashEmbedding) -> LateChunkingEmbedding:
    return LateChunkingEmbedding(base, alpha=0.3, window=1)


# ---------------------------------------------------------------------------
# Core coreference-lift test (NON-TAUTOLOGICAL)
# ---------------------------------------------------------------------------


def test_late_chunking_improves_pronoun_chunk_retrieval(
    base: HashEmbedding,
    late: LateChunkingEmbedding,
) -> None:
    """Late-chunked pronoun chunk is more similar to entity query than isolated embed.

    HashEmbedding is purely lexical.  CHUNK_1 contains none of the tokens in QUERY
    ("tesla"), so base cosine similarity is 0.  After late chunking, CHUNK_0's Tesla
    token contributes to CHUNK_1's vector, producing a positive cosine — a real and
    meaningful improvement.
    """
    query_vec = base.embed_query(QUERY)

    # Isolated baseline: embed CHUNK_1 alone, no context.
    chunk1_isolated = base.embed([CHUNK_1])[0]
    sim_isolated = cosine(query_vec, chunk1_isolated)

    # Late-chunked: context from CHUNK_0 bleeds into CHUNK_1's vector.
    late_vecs = late.embed_document_chunks(DOCUMENT_CHUNKS)
    chunk1_late = late_vecs[1]
    sim_late = cosine(query_vec, chunk1_late)

    # The lift must be strictly positive — not just "slightly better by luck".
    assert sim_late > sim_isolated, (
        f"Late chunking should improve pronoun-chunk retrievability "
        f"(cosine before={sim_isolated:.4f}, after={sim_late:.4f})"
    )


# ---------------------------------------------------------------------------
# Unit-length guarantee
# ---------------------------------------------------------------------------


def test_output_vectors_are_unit_length(late: LateChunkingEmbedding) -> None:
    vecs = late.embed_document_chunks(DOCUMENT_CHUNKS)
    for i, v in enumerate(vecs):
        norm = l2_norm(v)
        assert abs(norm - 1.0) < 1e-9, f"chunk {i} vector norm={norm:.6f}, expected 1.0"


# ---------------------------------------------------------------------------
# Dimensions correct
# ---------------------------------------------------------------------------


def test_output_dimensions_match_base(
    base: HashEmbedding,
    late: LateChunkingEmbedding,
) -> None:
    vecs = late.embed_document_chunks(DOCUMENT_CHUNKS)
    for i, v in enumerate(vecs):
        assert len(v) == base.dimensions, (
            f"chunk {i}: expected {base.dimensions} dims, got {len(v)}"
        )


# ---------------------------------------------------------------------------
# alpha=0 identity: late chunking reduces to base embedding
# ---------------------------------------------------------------------------


def test_alpha_zero_equals_base_embedding(base: HashEmbedding) -> None:
    """When alpha=0 embed_document_chunks must equal base.embed (up to rounding)."""
    lc_zero = LateChunkingEmbedding(base, alpha=0.0, window=1)
    base_vecs = base.embed(DOCUMENT_CHUNKS)
    late_vecs = lc_zero.embed_document_chunks(DOCUMENT_CHUNKS)

    for i, (bv, lv) in enumerate(zip(base_vecs, late_vecs, strict=True)):
        # base.embed already returns unit-length vectors; late path also normalises.
        for j, (b_val, l_val) in enumerate(zip(bv, lv, strict=True)):
            assert abs(b_val - l_val) < 1e-9, (
                f"chunk {i} dim {j}: base={b_val}, late(alpha=0)={l_val}"
            )


# ---------------------------------------------------------------------------
# model_id, dimensions, context_limit pass-through
# ---------------------------------------------------------------------------


def test_model_id_has_latechunk_suffix(
    base: HashEmbedding,
    late: LateChunkingEmbedding,
) -> None:
    assert late.model_id == base.model_id + "-latechunk"


def test_dimensions_and_context_limit_delegated(
    base: HashEmbedding,
    late: LateChunkingEmbedding,
) -> None:
    assert late.dimensions == base.dimensions
    assert late.context_limit == base.context_limit


# ---------------------------------------------------------------------------
# embed / embed_query delegate unchanged to base
# ---------------------------------------------------------------------------


def test_embed_delegates_to_base(
    base: HashEmbedding,
    late: LateChunkingEmbedding,
) -> None:
    texts = [CHUNK_0, CHUNK_1]
    assert late.embed(texts) == base.embed(texts)


def test_embed_query_delegates_to_base(
    base: HashEmbedding,
    late: LateChunkingEmbedding,
) -> None:
    assert late.embed_query(QUERY) == base.embed_query(QUERY)


# ---------------------------------------------------------------------------
# Edge: single chunk (window has nothing to mix)
# ---------------------------------------------------------------------------


def test_single_chunk_is_unit_length(late: LateChunkingEmbedding) -> None:
    vecs = late.embed_document_chunks(["Only one chunk here."])
    assert len(vecs) == 1
    assert abs(l2_norm(vecs[0]) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Edge: empty input
# ---------------------------------------------------------------------------


def test_empty_returns_empty(late: LateChunkingEmbedding) -> None:
    assert late.embed_document_chunks([]) == []


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_invalid_alpha_raises(base: HashEmbedding) -> None:
    with pytest.raises(ValueError, match="alpha"):
        LateChunkingEmbedding(base, alpha=1.5)


def test_invalid_window_raises(base: HashEmbedding) -> None:
    with pytest.raises(ValueError, match="window"):
        LateChunkingEmbedding(base, window=-1)


# ---------------------------------------------------------------------------
# EmbeddingProvider protocol satisfaction
# ---------------------------------------------------------------------------


def test_satisfies_embedding_provider_protocol(
    late: LateChunkingEmbedding,
) -> None:
    from jera.ports.embedding import EmbeddingProvider

    assert isinstance(late, EmbeddingProvider)
