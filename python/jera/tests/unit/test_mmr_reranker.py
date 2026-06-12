"""Unit tests for MMRReranker — deterministic, offline, no torch.

Fixture rationale (frozen under HashEmbedding(256)):
  query  = "machine learning training"
  dup1   = "machine learning training optimization"       rel≈0.866
  dup2   = "machine learning training optimization epoch" rel≈0.775, sim(dup1,dup2)≈0.894
  diverse= "machine learning kernel methods"              rel≈0.577, sim(dup1,diverse)≈0.500

After picking dup1 first (highest relevance), MMR with lambda_=0.5:
  MMR(dup2)   = 0.5*0.775 - 0.5*0.894 ≈ -0.060  (penalised: nearly identical to dup1)
  MMR(diverse)= 0.5*0.577 - 0.5*0.500 ≈ +0.039  (positive: sufficiently different)
  → second selection is diverse, proving real MMR diversity suppression.
"""

from __future__ import annotations

import pytest

from jera.adapters.embedding.hash_embedding import HashEmbedding
from jera.adapters.ranking.identity_reranker import IdentityReranker
from jera.adapters.ranking.mmr_reranker import MMRReranker
from jera.domain.chunk import Chunk
from jera.domain.document import PageSpan
from jera.domain.retrieval import ScoredChunk
from jera.ports.reranker import Reranker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMB = HashEmbedding(dimensions=256)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        source_id="src",
        text=text,
        page_span=PageSpan.single(1),
        section_path=(),
        element_ids=(),
        char_span=(0, len(text)),
        token_count=len(text.split()),
        chunk_strategy="t",
        chunk_version="t",
    )


def _sc(chunk_id: str, score: float, text: str) -> ScoredChunk:
    return ScoredChunk(chunk_id=chunk_id, score=score, chunk=_chunk(chunk_id, text))


# ---------------------------------------------------------------------------
# Frozen diversity fixture
# (verified empirically: see module docstring for computed MMR values)
# ---------------------------------------------------------------------------

_QUERY = "machine learning training"
_DUP1 = _sc("dup1", 0.90, "machine learning training optimization")
_DUP2 = _sc("dup2", 0.85, "machine learning training optimization epoch")
_DIVERSE = _sc("diverse", 0.80, "machine learning kernel methods")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_protocol_satisfied() -> None:
    """MMRReranker structurally satisfies the Reranker protocol."""
    assert isinstance(MMRReranker(_EMB), Reranker)


def test_diversity_suppresses_near_duplicate() -> None:
    """With lambda_=0.5, the second selected chunk is the diverse one, not the near-duplicate.

    dup1 and dup2 share ~89% cosine similarity; after dup1 is selected first, MMR penalises
    dup2 heavily. The diverse chunk (kernel methods) has a different lexical footprint and
    wins the second slot despite a lower first-stage score.
    """
    reranker = MMRReranker(_EMB, lambda_=0.5)
    result = reranker.rerank(_QUERY, [_DUP1, _DUP2, _DIVERSE], top_k=3)
    ids = [r.chunk_id for r in result]
    assert ids[0] == "dup1", f"Expected dup1 first, got {ids}"
    assert ids[1] == "diverse", f"Expected diverse second (MMR diversity), got {ids}"
    assert ids[2] == "dup2", f"Expected dup2 last (penalised), got {ids}"


def test_lambda_1_reduces_to_relevance_ordering() -> None:
    """lambda_=1.0 disables the diversity penalty; result matches pure relevance order."""
    reranker = MMRReranker(_EMB, lambda_=1.0)
    identity = IdentityReranker()
    candidates = [_DUP1, _DUP2, _DIVERSE]
    mmr_result = reranker.rerank(_QUERY, candidates, top_k=3)
    identity_result = identity.rerank(_QUERY, candidates, top_k=3)
    assert [r.chunk_id for r in mmr_result] == [r.chunk_id for r in identity_result]


def test_top_k_truncation() -> None:
    """top_k is respected: only that many results are returned."""
    reranker = MMRReranker(_EMB, lambda_=0.7)
    result = reranker.rerank(_QUERY, [_DUP1, _DUP2, _DIVERSE], top_k=2)
    assert len(result) == 2


def test_components_written() -> None:
    """Each selected ScoredChunk gets 'rerank' and 'relevance' in components."""
    reranker = MMRReranker(_EMB, lambda_=0.7)
    result = reranker.rerank(_QUERY, [_DUP1, _DUP2, _DIVERSE], top_k=3)
    for sc in result:
        assert "rerank" in sc.components, f"Missing 'rerank' in {sc.chunk_id}"
        assert "relevance" in sc.components, f"Missing 'relevance' in {sc.chunk_id}"


def test_candidates_without_chunk_appended_last() -> None:
    """Candidates with chunk=None are appended after MMR-selected ones, ordered by score desc."""
    no_chunk_high = ScoredChunk(chunk_id="nc_high", score=0.99)
    no_chunk_low = ScoredChunk(chunk_id="nc_low", score=0.10)
    reranker = MMRReranker(_EMB, lambda_=0.7)
    result = reranker.rerank(_QUERY, [_DUP1, no_chunk_high, _DIVERSE, no_chunk_low], top_k=4)
    ids = [r.chunk_id for r in result]
    # MMR-selected (those with chunks) come first
    assert "nc_high" not in ids[:2], "No-chunk candidate should not lead"
    # No-chunk candidates appear at the end, high score before low score
    no_chunk_pos = {cid: i for i, cid in enumerate(ids) if cid.startswith("nc_")}
    assert no_chunk_pos["nc_high"] < no_chunk_pos["nc_low"]


def test_all_candidates_without_chunk() -> None:
    """When all candidates lack a chunk, falls back to score-ordered list (top_k applied)."""
    reranker = MMRReranker(_EMB, lambda_=0.7)
    candidates = [
        ScoredChunk(chunk_id="b", score=0.5),
        ScoredChunk(chunk_id="a", score=0.9),
        ScoredChunk(chunk_id="c", score=0.3),
    ]
    result = reranker.rerank("query", candidates, top_k=2)
    assert len(result) == 2
    assert result[0].chunk_id == "a"
    assert result[1].chunk_id == "b"


def test_lambda_validation() -> None:
    """lambda_ outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="lambda_"):
        MMRReranker(_EMB, lambda_=-0.1)
    with pytest.raises(ValueError, match="lambda_"):
        MMRReranker(_EMB, lambda_=1.1)


def test_lambda_boundary_values_accepted() -> None:
    """lambda_=0.0 and lambda_=1.0 are valid boundary values."""
    MMRReranker(_EMB, lambda_=0.0)
    MMRReranker(_EMB, lambda_=1.0)


def test_model_id_default_and_custom() -> None:
    """model_id defaults to 'mmr-rerank-v1' and can be overridden."""
    assert MMRReranker(_EMB).model_id == "mmr-rerank-v1"
    assert MMRReranker(_EMB, model_id="custom-mmr").model_id == "custom-mmr"


def test_empty_candidates() -> None:
    """Empty candidate list returns empty list without error."""
    reranker = MMRReranker(_EMB, lambda_=0.7)
    result = reranker.rerank(_QUERY, [], top_k=5)
    assert result == []


def test_top_k_larger_than_candidates() -> None:
    """top_k larger than the number of candidates returns all candidates."""
    reranker = MMRReranker(_EMB, lambda_=0.7)
    result = reranker.rerank(_QUERY, [_DUP1, _DIVERSE], top_k=10)
    assert len(result) == 2
