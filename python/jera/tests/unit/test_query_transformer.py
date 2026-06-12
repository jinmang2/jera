"""Query-transformation adapters: deterministic rule-based expansion + HyDE wiring (fake LLM)."""

from __future__ import annotations

from jera.adapters.query.hyde import HydeTransformer
from jera.adapters.query.rule_based_expander import RuleBasedExpander
from jera.ports.query_transformer import QueryTransformer


def test_rule_based_is_a_query_transformer() -> None:
    assert isinstance(RuleBasedExpander(), QueryTransformer)


def test_rule_based_splits_english_conjunctions() -> None:
    out = RuleBasedExpander().transform("dense and sparse retrieval")
    assert out[0] == "dense and sparse retrieval"  # original always first
    assert "dense" in out and "sparse retrieval" in out


def test_rule_based_splits_korean_conjunctions() -> None:
    out = RuleBasedExpander().transform("벡터 검색 그리고 재순위화")
    assert out[0] == "벡터 검색 그리고 재순위화"
    assert "벡터 검색" in out and "재순위화" in out


def test_rule_based_non_compound_is_noop() -> None:
    assert RuleBasedExpander().transform("how are rankings merged?") == ["how are rankings merged?"]


def test_rule_based_drops_short_noise_clauses() -> None:
    # single-char clauses are punctuation noise, not real sub-questions
    assert RuleBasedExpander().transform("a, b, c") == ["a, b, c"]


def test_rule_based_dedups_and_is_never_empty() -> None:
    out = RuleBasedExpander().transform("backup and backup")
    assert out == ["backup and backup", "backup"]  # duplicate clause collapsed
    assert RuleBasedExpander().transform("   ") == ["   "]  # never returns []


class _FakeHyDE:
    model_id = "fake-hyde"

    def hypothesize(self, query: str) -> str:
        return f"A hypothetical answer about {query.split()[-1]}."


def test_hyde_appends_hypothetical_answer() -> None:
    out = HydeTransformer(_FakeHyDE()).transform("storage backend")
    assert out == ["storage backend", "A hypothetical answer about backend."]


def test_hyde_dedups_when_hypothesis_equals_query() -> None:
    class Echo:
        model_id = "echo"

        def hypothesize(self, query: str) -> str:
            return query

    assert HydeTransformer(Echo()).transform("same") == ["same"]


def test_hyde_is_a_query_transformer() -> None:
    assert isinstance(HydeTransformer(_FakeHyDE()), QueryTransformer)
