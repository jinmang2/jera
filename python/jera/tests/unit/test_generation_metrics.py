"""RAGAS-lite generation-quality metric contracts (pure, deterministic)."""

from __future__ import annotations

import math

import pytest

from jera.evaluation_contracts.dataset import GoldChunk
from jera.evaluation_contracts.generation_metrics import (
    answer_correctness,
    answer_relevance,
    context_precision,
    faithfulness,
)

# --- faithfulness -------------------------------------------------------------------------


def test_faithfulness_fully_grounded_answer_scores_one() -> None:
    contexts = ["Reciprocal rank fusion merges dense and sparse rankings."]
    answer = "Reciprocal rank fusion merges dense and sparse rankings."
    assert faithfulness(answer, contexts) == 1.0


def test_faithfulness_hallucinated_sentence_lowers_score() -> None:
    contexts = ["The capital of France is Paris."]
    answer = "The capital of France is Paris. The moon is made of cheese."
    score = faithfulness(answer, contexts)
    assert 0.0 < score < 1.0
    assert math.isclose(score, 0.5)  # one of two sentences grounded


def test_faithfulness_empty_answer_is_vacuously_faithful() -> None:
    assert faithfulness("", ["anything"]) == 1.0


def test_faithfulness_with_no_context_is_zero() -> None:
    assert faithfulness("A claim.", []) == 0.0


# --- answer_relevance ---------------------------------------------------------------------


def test_answer_relevance_identical_vectors_is_one() -> None:
    v = [0.1, 0.2, 0.3]
    assert math.isclose(answer_relevance(v, v), 1.0)


def test_answer_relevance_orthogonal_vectors_is_zero() -> None:
    assert answer_relevance([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_answer_relevance_opposite_vectors_clamped_to_zero() -> None:
    assert answer_relevance([1.0, 0.0], [-1.0, 0.0]) == 0.0


def test_answer_relevance_dimension_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="dim"):
        answer_relevance([1.0], [1.0, 2.0])


def test_answer_relevance_zero_vector_is_zero() -> None:
    assert answer_relevance([0.0, 0.0], [1.0, 1.0]) == 0.0


# --- answer_correctness -------------------------------------------------------------------


def test_answer_correctness_exact_match_is_one() -> None:
    assert answer_correctness("hybrid retrieval", "hybrid retrieval") == 1.0


def test_answer_correctness_no_overlap_is_zero() -> None:
    assert answer_correctness("apples oranges", "quantum chromodynamics") == 0.0


def test_answer_correctness_partial_overlap_is_token_f1() -> None:
    # answer={a,b,c}, reference={a,b} → precision 2/3, recall 2/2, F1 = 0.8
    assert math.isclose(answer_correctness("a b c", "a b"), 0.8)


def test_answer_correctness_padding_cannot_inflate_via_multiset() -> None:
    # Padding the answer with repeats of a correct token can't reach 1.0: precision drops.
    # answer={a,a,a}, reference={a} → precision 1/3, recall 1/1, F1 = 0.5 (not 1.0).
    assert math.isclose(answer_correctness("a a a", "a"), 0.5)


def test_answer_correctness_both_empty_is_one() -> None:
    assert answer_correctness("", "") == 1.0


# --- context_precision --------------------------------------------------------------------


def _gold(*ids: str) -> list[GoldChunk]:
    return [GoldChunk(chunk_id=i) for i in ids]


def test_context_precision_relevant_first_is_one() -> None:
    ranked = ["a", "x", "y"]
    assert context_precision(ranked, _gold("a"), k=3) == 1.0


def test_context_precision_rewards_higher_ranking() -> None:
    gold = _gold("a")
    high = context_precision(["a", "x", "y"], gold, k=3)
    low = context_precision(["x", "y", "a"], gold, k=3)
    assert high > low


def test_context_precision_average_precision_two_relevant() -> None:
    # relevant at ranks 1 and 3 → (1/1 + 2/3) / 2 = 0.8333...
    ap = context_precision(["a", "x", "b"], _gold("a", "b"), k=3)
    assert math.isclose(ap, (1.0 + 2.0 / 3.0) / 2.0)


def test_context_precision_no_relevant_retrieved_is_zero() -> None:
    assert context_precision(["x", "y"], _gold("a"), k=2) == 0.0


def test_context_precision_empty_gold_is_zero() -> None:
    assert context_precision(["a"], [], k=1) == 0.0
