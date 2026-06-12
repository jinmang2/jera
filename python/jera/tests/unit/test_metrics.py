"""Evaluation metric contracts."""

from __future__ import annotations

import math

from jera.evaluation_contracts import (
    GoldChunk,
    citation_faithfulness,
    mrr,
    ndcg_at_k,
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
