"""Golden-file fusion contract (Gate 4b).

Pins the five determinism rules: 1-based ranks, RRF k=60, explicit-0 missing modality,
DBSF 3-sigma-normalize+sum (canonical Qdrant DBSF), tie-break by chunk_id ascending.
Hand-built rankings → asserted output.
"""

from __future__ import annotations

import math

from jera.adapters.vector_store.fusion import (
    RRF_K,
    distribution_based_score_fusion,
    reciprocal_rank_fusion,
)


def test_rrf_k_constant_is_60() -> None:
    assert RRF_K == 60


def test_rrf_one_based_ranks_and_missing_modality_zero() -> None:
    # 'a' is rank1 dense only; 'b' rank1 sparse only; 'c' rank2 in both.
    fused = reciprocal_rank_fusion({"dense": ["a", "c"], "sparse": ["b", "c"]})
    scores = dict(fused)
    # rule 1+2: 1-based ranks, k=60
    assert math.isclose(scores["a"], 1 / 61)  # dense rank1, missing in sparse -> +0
    assert math.isclose(scores["b"], 1 / 61)
    assert math.isclose(scores["c"], 1 / 62 + 1 / 62)
    # 'c' present in both beats single-modality 'a'/'b'
    assert fused[0][0] == "c"


def test_rrf_tie_break_by_chunk_id_ascending() -> None:
    # 'a' and 'b' both rank1 in one modality each -> identical scores -> id tie-break.
    fused = reciprocal_rank_fusion({"dense": ["b"], "sparse": ["a"]})
    assert [cid for cid, _ in fused] == ["a", "b"]


def test_dbsf_three_sigma_then_sum() -> None:
    # Canonical Qdrant DBSF: normalize each modality by (s-(mu-3sigma))/(6sigma), sample sigma.
    # dense {10,6,2}: mu=6, sigma=4 -> denom=24, lo=-6 => a=16/24, b=12/24, c=8/24
    # sparse {4,2}:   mu=3, sigma=sqrt(2) -> denom=6*sqrt(2), lo=3-3*sqrt(2)
    dense = {"a": 10.0, "b": 6.0, "c": 2.0}
    sparse = {"c": 4.0, "b": 2.0}
    fused = dict(distribution_based_score_fusion({"dense": dense, "sparse": sparse}))
    s_denom = 6.0 * math.sqrt(2.0)
    s_c = (4.0 - (3.0 - 3.0 * math.sqrt(2.0))) / s_denom
    s_b = (2.0 - (3.0 - 3.0 * math.sqrt(2.0))) / s_denom
    assert math.isclose(fused["a"], 16.0 / 24.0)  # dense only (missing sparse -> +0)
    assert math.isclose(fused["b"], 12.0 / 24.0 + s_b)
    assert math.isclose(fused["c"], 8.0 / 24.0 + s_c)


def test_dbsf_equal_scores_contribute_half() -> None:
    # A modality with zero variance (all scores equal) emits 0.5 per point (Qdrant guard),
    # NOT 0. Here dense is flat (both 5.0) -> each contributes 0.5; sparse drives the spread.
    fused = dict(
        distribution_based_score_fusion(
            {"dense": {"a": 5.0, "b": 5.0}, "sparse": {"a": 9.0, "b": 3.0}}
        )
    )
    # sparse {9,3}: mu=6, sigma=sqrt(18)=3*sqrt(2) -> denom=18*sqrt(2), lo=6-9*sqrt(2)
    s_denom = 18.0 * math.sqrt(2.0)
    s_a = (9.0 - (6.0 - 9.0 * math.sqrt(2.0))) / s_denom
    s_b = (3.0 - (6.0 - 9.0 * math.sqrt(2.0))) / s_denom
    assert math.isclose(fused["a"], 0.5 + s_a)  # flat dense 0.5 + sparse high
    assert math.isclose(fused["b"], 0.5 + s_b)  # flat dense 0.5 + sparse low
    assert fused["a"] > fused["b"]  # sparse spread still drives ordering
