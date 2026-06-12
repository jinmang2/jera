"""Golden-file fusion contract (Gate 4b).

Pins the five determinism rules: 1-based ranks, RRF k=60, explicit-0 missing modality,
DBSF min-max+sum, tie-break by chunk_id ascending. Hand-built rankings → asserted output.
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


def test_dbsf_min_max_then_sum() -> None:
    dense = {"a": 10.0, "b": 6.0, "c": 2.0}  # min-max -> a=1, b=0.5, c=0
    sparse = {"c": 4.0, "b": 2.0}  # min-max -> c=1, b=0
    fused = dict(distribution_based_score_fusion({"dense": dense, "sparse": sparse}))
    assert math.isclose(fused["a"], 1.0)  # 1.0 (dense) + 0 (missing sparse)
    assert math.isclose(fused["b"], 0.5)  # 0.5 + 0
    assert math.isclose(fused["c"], 1.0)  # 0 + 1.0


def test_dbsf_equal_scores_contribute_zero() -> None:
    # A modality whose scores are all equal -> min-max undefined -> contributes 0 deterministically.
    # Here dense is flat (both 0) while sparse has spread (a=1, b=0) and drives the result.
    fused = dict(
        distribution_based_score_fusion(
            {"dense": {"a": 5.0, "b": 5.0}, "sparse": {"a": 9.0, "b": 3.0}}
        )
    )
    assert math.isclose(fused["a"], 1.0)  # 0 (flat dense) + 1.0 (sparse max)
    assert math.isclose(fused["b"], 0.0)  # 0 (flat dense) + 0.0 (sparse min)
