"""NON-TAUTOLOGICAL tests for RAGChecker-style claim-level metrics.

Each test asserts a *grounding property* of the metric — a relationship between
inputs with known structural differences (grounded vs. hallucinated, cited vs.
uncited, abstaining vs. assertive) — not a hard-coded constant plucked from thin
air.  The fixture values are chosen so that token-containment arithmetic yields
verifiable, human-checkable results.
"""

from __future__ import annotations

import math

import pytest

from jera.evaluation_contracts.ragchecker_metrics import (
    abstention_score,
    citation_precision,
    citation_recall,
    claim_precision,
    claim_recall,
    noise_sensitivity,
)

# ===========================================================================
# 1. claim_precision — anti-hallucination at claim granularity
# ===========================================================================


def test_claim_precision_fully_grounded_is_one() -> None:
    """All answer claims appear in context → precision = 1.0."""
    context = ["The Eiffel Tower is located in Paris France."]
    answer = "The Eiffel Tower is located in Paris France."
    assert claim_precision(answer, context) == 1.0


def test_claim_precision_half_grounded_is_half() -> None:
    """One grounded claim + one hallucinated claim → precision = 0.5.

    Grounding property: introducing a claim with NO context tokens
    cuts the score by exactly 1/N where N = number of claims.
    """
    context = ["The Eiffel Tower is located in Paris."]
    # Sentence 1 is grounded; sentence 2 has zero overlap with context.
    answer = "The Eiffel Tower is located in Paris. Unicorns roam Antarctica."
    score = claim_precision(answer, context)
    assert math.isclose(score, 0.5), f"expected 0.5, got {score}"


def test_claim_precision_empty_answer_is_vacuously_one() -> None:
    assert claim_precision("", ["some context"]) == 1.0


def test_claim_precision_no_context_is_zero() -> None:
    assert claim_precision("A claim.", []) == 0.0


def test_claim_precision_grounded_beats_hallucinated() -> None:
    """A fully-grounded answer must score strictly higher than a hallucinated one."""
    context = ["Paris is the capital of France."]
    good = "Paris is the capital of France."
    bad = "London is the capital of Australia. Dragons breathe fire underwater."
    assert claim_precision(good, context) > claim_precision(bad, context)


# ===========================================================================
# 2. claim_recall — completeness relative to gold answer
# ===========================================================================


def test_claim_recall_all_gold_covered_is_one() -> None:
    """Answer reproduces both gold claims → recall = 1.0."""
    gold = "Paris is the capital. France is in Europe."
    answer = "Paris is the capital of France which is in Europe."
    assert claim_recall(answer, gold) == 1.0


def test_claim_recall_half_gold_covered_is_half() -> None:
    """Answer covers only the first of two gold claims → recall = 0.5.

    Grounding property: omitting a gold claim has a measurable recall penalty
    proportional to 1/(number of gold claims).
    """
    gold = "The Eiffel Tower is in Paris. The Louvre is a famous museum."
    # Answer only mentions the Eiffel Tower, not the Louvre.
    answer = "The Eiffel Tower is located in Paris."
    score = claim_recall(answer, gold)
    assert math.isclose(score, 0.5), f"expected 0.5, got {score}"


def test_claim_recall_empty_gold_is_vacuously_one() -> None:
    assert claim_recall("some answer", "") == 1.0


def test_claim_recall_empty_answer_is_zero() -> None:
    assert claim_recall("", "The capital of France is Paris.") == 0.0


def test_claim_recall_increases_with_coverage() -> None:
    """More gold claims reproduced → higher recall."""
    gold = "Paris is the capital. France is in Europe. The Seine flows through Paris."
    partial = "Paris is the capital."
    full = "Paris is the capital of France which is in Europe and the Seine flows through Paris."
    assert claim_recall(full, gold) >= claim_recall(partial, gold)


# ===========================================================================
# 3. noise_sensitivity — drift caused by noisy context chunks
# ===========================================================================
#
# Fixture from m12-eval.md PICK 1:
#   gold       = "The Eiffel Tower is in Paris."
#   noise_chunk = "The Louvre has 8 million visitors and is in Paris."
#   answer_clean   = grounded in gold, ignores noise → 0.0
#   answer_drifted = introduces Louvre content NOT in gold → > 0.0


_GOLD_NS = "The Eiffel Tower is in Paris."
_NOISE_CHUNK = "The Louvre has 8 million visitors and is in Paris."


def test_noise_sensitivity_clean_answer_is_zero() -> None:
    """Generator ignored the noisy chunk → noise_sensitivity = 0.0."""
    answer = "The Eiffel Tower is in Paris."
    score = noise_sensitivity(answer, [_NOISE_CHUNK], _GOLD_NS)
    assert score == 0.0, f"expected 0.0, got {score}"


def test_noise_sensitivity_drifted_answer_is_positive() -> None:
    """Generator parroted the noisy chunk's content → noise_sensitivity > 0.0.

    Grounding property: the answer contains a claim that is (a) not in gold AND
    (b) shares tokens with the noisy chunk — the metric must detect this.
    """
    # Second sentence is not in gold but closely mirrors the noise chunk.
    answer = "The Eiffel Tower is in Paris. The Louvre has 8 million visitors."
    score = noise_sensitivity(answer, [_NOISE_CHUNK], _GOLD_NS)
    assert score > 0.0, f"expected > 0.0, got {score}"


def test_noise_sensitivity_no_noisy_chunks_is_zero() -> None:
    """All context chunks are grounded in gold → no noisy chunks → score = 0.0."""
    relevant_chunk = "The Eiffel Tower is a famous landmark in Paris."
    # Even a drifted answer cannot be attributed to a noisy chunk if none exist.
    answer = "The Eiffel Tower is in Paris. Something unrelated happened."
    score = noise_sensitivity(answer, [relevant_chunk], _GOLD_NS)
    assert score == 0.0, f"expected 0.0 when no noisy chunks, got {score}"


def test_noise_sensitivity_clean_beats_drifted() -> None:
    """A grounded answer must have strictly lower noise sensitivity than a drifted one."""
    clean = "The Eiffel Tower is in Paris."
    drifted = "The Eiffel Tower is in Paris. The Louvre has 8 million visitors."
    s_clean = noise_sensitivity(clean, [_NOISE_CHUNK], _GOLD_NS)
    s_drifted = noise_sensitivity(drifted, [_NOISE_CHUNK], _GOLD_NS)
    assert s_clean < s_drifted


def test_noise_sensitivity_empty_answer_is_zero() -> None:
    assert noise_sensitivity("", [_NOISE_CHUNK], _GOLD_NS) == 0.0


# ===========================================================================
# 4. citation_precision & citation_recall (TREC 2025 weighted support)
# ===========================================================================


def test_citation_precision_fully_supported_is_one() -> None:
    """All sentences have a supporting citation → precision = 1.0."""
    mapping = {
        "Paris is the capital of France.": ["Paris is the capital of France."],
    }
    assert citation_precision(mapping) == 1.0


def test_citation_precision_one_supported_one_unsupported_is_half() -> None:
    """One cited sentence + one uncited sentence → precision = 0.5.

    Grounding property: removing a valid citation for a sentence halves the
    precision when there are exactly two sentences.
    """
    mapping = {
        "Paris is the capital of France.": ["Paris is the capital of France."],
        "Dragons breathe fire.": [],  # no citation
    }
    score = citation_precision(mapping)
    assert math.isclose(score, 0.5), f"expected 0.5, got {score}"


def test_citation_precision_empty_mapping_is_one() -> None:
    assert citation_precision({}) == 1.0


def test_citation_precision_wrong_chunk_lowers_score() -> None:
    """A cited chunk that shares few tokens with the sentence lowers precision."""
    # Fully supporting citation
    perfect = citation_precision(
        {
            "Paris is the capital of France.": ["Paris is the capital of France."],
        }
    )
    # Mismatched citation (entirely unrelated chunk)
    bad = citation_precision(
        {
            "Paris is the capital of France.": ["Bananas grow in tropical climates."],
        }
    )
    assert perfect > bad


def test_citation_precision_partial_support_between_zero_and_one() -> None:
    """A sentence with partial token overlap with its citation scores between 0 and 1."""
    # "Paris is the capital of France and home to 12 million people." has many
    # tokens NOT in the cited chunk "Paris is the capital of France." → Partial
    mapping = {
        "Paris is the capital of France and home to twelve million people.": [
            "Paris is the capital of France."
        ],
    }
    score = citation_precision(mapping)
    assert 0.0 < score < 1.0


def test_citation_recall_both_cited_is_one() -> None:
    """Both sentences have supporting citations → recall = 1.0."""
    mapping = {
        "Paris is the capital.": ["Paris is the capital of France."],
        "Mars has two moons.": ["Mars has two moons named Phobos and Deimos."],
    }
    assert citation_recall(mapping) == 1.0


def test_citation_recall_one_of_two_cited_is_half() -> None:
    """One cited, one uncited → recall = 0.5.

    Grounding property: every uncited sentence uniformly penalises recall by 1/N.
    """
    mapping = {
        "Paris is the capital.": ["Paris is the capital of France."],
        "The moon is made of cheese.": [],
    }
    score = citation_recall(mapping)
    assert math.isclose(score, 0.5), f"expected 0.5, got {score}"


def test_citation_recall_no_citations_is_zero() -> None:
    mapping = {
        "Paris is the capital.": [],
        "Mars has two moons.": [],
    }
    assert citation_recall(mapping) == 0.0


def test_citation_recall_empty_mapping_is_one() -> None:
    assert citation_recall({}) == 1.0


def test_citation_recall_increases_with_more_citations() -> None:
    """Adding a supporting citation for the second sentence must raise recall."""
    both_cited = {
        "Paris is the capital.": ["Paris is the capital of France."],
        "Mars has two moons.": ["Mars has two moons named Phobos and Deimos."],
    }
    only_one = {
        "Paris is the capital.": ["Paris is the capital of France."],
        "Mars has two moons.": [],
    }
    assert citation_recall(both_cited) > citation_recall(only_one)


# ===========================================================================
# 5. abstention_score (RGB — negative rejection)
# ===========================================================================


def test_abstention_score_i_dont_know_is_one() -> None:
    """Classic hedge phrase → 1.0."""
    assert abstention_score("I don't know based on the context.") == 1.0


def test_abstention_score_substantive_answer_is_zero() -> None:
    """A confident factual answer → 0.0."""
    assert abstention_score("The answer is definitely Paris, France.") == 0.0


def test_abstention_score_cannot_determine_is_one() -> None:
    assert abstention_score("I cannot determine the answer from the provided context.") == 1.0


def test_abstention_score_no_information_is_one() -> None:
    assert abstention_score("There is no information about this topic.") == 1.0


def test_abstention_score_insufficient_context_is_one() -> None:
    assert abstention_score("The context is insufficient to answer this question.") == 1.0


def test_abstention_score_korean_hedge_is_one() -> None:
    """Korean abstention phrase → 1.0 (bilingual coverage)."""
    assert abstention_score("모르겠습니다.") == 1.0


def test_abstention_score_case_insensitive() -> None:
    """Hedge phrase detection is case-insensitive."""
    assert abstention_score("I DON'T KNOW the answer.") == 1.0
    assert abstention_score("Cannot Determine from context.") == 1.0


def test_abstention_score_hedge_beats_assertion() -> None:
    """Grounding property: abstaining answer must score strictly higher than assertive one."""
    hedge = "I don't have enough information to answer this."
    assert_answer = "The capital of France is Paris."
    assert abstention_score(hedge) > abstention_score(assert_answer)


@pytest.mark.parametrize(
    "text",
    [
        "The study was conducted in 2023.",
        "Results showed a 15% improvement.",
        "Paris is the capital of France.",
        "The population is approximately 67 million.",
    ],
)
def test_abstention_score_factual_answers_are_zero(text: str) -> None:
    """Factual assertions without any hedge phrase must score 0.0."""
    assert abstention_score(text) == 0.0
