"""Evaluation metric contracts."""

from __future__ import annotations

import math

from jera.evaluation_contracts import (
    GoldChunk,
    citation_faithfulness,
    mrr,
    ndcg_at_k,
    numeric_accuracy,
    recall_at_k,
)

GOLD = [GoldChunk(chunk_id="a"), GoldChunk(chunk_id="b")]


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "x", "b"], GOLD, k=3) == 1.0
    assert recall_at_k(["a", "x", "y"], GOLD, k=3) == 0.5
    assert recall_at_k(["x"], GOLD, k=1) == 0.0


def test_mrr() -> None:
    assert mrr(["x", "a"], GOLD) == 0.5
    assert mrr(["a"], GOLD) == 1.0
    assert mrr(["z"], GOLD) == 0.0


def test_ndcg_perfect_is_one() -> None:
    assert math.isclose(ndcg_at_k(["a", "b"], GOLD, k=2), 1.0)


def test_citation_faithfulness() -> None:
    assert citation_faithfulness(["a", "b"], ["a", "b", "c"]) == 1.0
    assert citation_faithfulness(["a", "z"], ["a", "b"]) == 0.5
    assert citation_faithfulness([], ["a"]) == 1.0  # no citations = vacuously faithful


# ---------------------------------------------------------------------------
# numeric_accuracy
# ---------------------------------------------------------------------------


def test_numeric_accuracy_exact_match() -> None:
    # Exact match is always 1.0
    assert numeric_accuracy(42.0, 42.0) == 1.0


def test_numeric_accuracy_within_tolerance() -> None:
    # 0.001 * max(100, 1) = 0.1 — answer within that window
    assert numeric_accuracy(100.05, 100.0) == 1.0
    assert numeric_accuracy(99.95, 100.0) == 1.0


def test_numeric_accuracy_outside_tolerance() -> None:
    # 0.001 * max(100, 1) = 0.1 — answer outside that window
    assert numeric_accuracy(100.2, 100.0) == 0.0
    assert numeric_accuracy(99.8, 100.0) == 0.0


def test_numeric_accuracy_none_answer() -> None:
    # None answer always returns 0.0 (generator produced no value)
    assert numeric_accuracy(None, 42.0) == 0.0


def test_numeric_accuracy_large_number_relative() -> None:
    # For a large expected value the tolerance scales with the magnitude
    # 0.001 * max(1_000_000, 1) = 1000 — within
    assert numeric_accuracy(1_000_500.0, 1_000_000.0) == 1.0
    # 0.001 * 1_000_000 = 1000 — outside (diff = 1001)
    assert numeric_accuracy(1_001_001.0, 1_000_000.0) == 0.0


def test_numeric_accuracy_zero_expected_floor() -> None:
    # max(|0|, 1) = 1 — tolerance = 0.001 * 1 = 0.001
    # answer 0.0005 is within 0.001 of 0
    assert numeric_accuracy(0.0005, 0.0) == 1.0
    # answer 0.002 is outside 0.001 of 0
    assert numeric_accuracy(0.002, 0.0) == 0.0


def test_numeric_accuracy_custom_tolerance() -> None:
    # tolerance=0.05 → threshold = 0.05 * max(200, 1) = 10
    assert numeric_accuracy(205.0, 200.0, tolerance=0.05) == 1.0
    assert numeric_accuracy(215.0, 200.0, tolerance=0.05) == 0.0
