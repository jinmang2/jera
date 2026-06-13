"""Tests for ListwiseReranker and ClaudeListwiseReranker.

Non-tautological property under test
-------------------------------------
The first-stage ``score`` order is deliberately WRONG:

- chunk A  (chunk_id="a", score=0.9) covers only the common query term "revenue"
  which appears in ALL three candidates — so its IDF weight is low.
- chunk B  (chunk_id="b", score=0.1) covers "revenue" AND the rare query term
  "amortisation" which appears in NO other candidate — highest IDF weight.
- chunk C  (chunk_id="c", score=0.5) covers "revenue" only.

First-stage order:  A > C > B  (by score).
Listwise order:     B > A = C  (B has full coverage; A and C tie on coverage
                                 then chunk_id asc → A before C).

The test asserts ListwiseReranker promotes B to #1, beating the first-stage
leader A.  A secondary test confirms the whole-list dependency: removing B
from the pool changes A and C's weights (the rare token is gone, all remaining
query tokens become equally common) but B's advantage depends on B being
in the pool that defines rarity — i.e. B's presence raises IDF(amortisation).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from jera.domain.chunk import Chunk
from jera.domain.document import PageSpan
from jera.domain.retrieval import ScoredChunk
from jera.ports.reranker import Reranker  # noqa: F401 (used in isinstance check)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        source_id="src1",
        text=text,
        page_span=PageSpan(start_page=1, end_page=1),
        section_path=("Section 1",),
        element_ids=("e1",),
        char_span=(0, len(text)),
        token_count=len(text.split()),
        chunk_strategy="heading_aware",
        chunk_version="v1",
    )


def _make_scored(chunk_id: str, score: float, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        score=score,
        chunk=_make_chunk(chunk_id, text),
    )


# Query used for the non-tautological scenario:
#   content tokens → {"revenue", "amortisation"}
QUERY = "revenue amortisation"

# chunk A: first-stage leader (score=0.9) but only covers common term "revenue"
CAND_A = _make_scored("a", 0.9, "The revenue figures were strong this quarter.")
# chunk B: first-stage loser (score=0.1) but covers both terms including rare "amortisation"
CAND_B = _make_scored("b", 0.1, "Revenue increased; amortisation costs were also reported.")
# chunk C: middle first-stage score (score=0.5), covers only "revenue"
CAND_C = _make_scored("c", 0.5, "Revenue declined due to market conditions.")

ALL_CANDIDATES = [CAND_A, CAND_B, CAND_C]


# ---------------------------------------------------------------------------
# SDK-boundary mock helpers (mirrors test_cloud_anthropic_adapters.py style)
# ---------------------------------------------------------------------------


def _make_text_block(text: str) -> Any:
    return types.SimpleNamespace(type="text", text=text)


def _make_response(content: list[Any], stop_reason: str = "end_turn") -> Any:
    return types.SimpleNamespace(content=content, stop_reason=stop_reason)


def _install_fake_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
    responses: list[Any],
) -> None:
    """Inject a fake ``anthropic`` module that records ``messages.create`` kwargs."""
    call_index: list[int] = [0]

    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            captured.setdefault("calls", []).append(dict(kwargs))
            idx = call_index[0]
            call_index[0] += 1
            return responses[idx]

    class Anthropic:
        def __init__(self, **kwargs: Any) -> None:
            self.messages = _Messages()

    mod.Anthropic = Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", mod)


# ===========================================================================
# 1. ListwiseReranker — protocol conformance
# ===========================================================================


class TestListwiseRerankerProtocol:
    def test_is_reranker_protocol(self) -> None:
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        assert isinstance(ListwiseReranker(), Reranker)

    def test_model_id(self) -> None:
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        assert ListwiseReranker().model_id == "listwise-rerank-v1"


# ===========================================================================
# 2. ListwiseReranker — NON-TAUTOLOGICAL: listwise beats pointwise order
# ===========================================================================


class TestListwiseRerankerCoreProperty:
    """The key non-tautological property: full-coverage rare-term chunk wins."""

    def test_full_coverage_chunk_promoted_to_first(self) -> None:
        """B (first-stage rank 3) must be promoted to rank 1 by listwise scoring."""
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        result = ListwiseReranker().rerank(QUERY, ALL_CANDIDATES, top_k=3)

        assert len(result) == 3
        assert result[0].chunk_id == "b", (
            f"Expected chunk 'b' at rank 1 (rare-term coverage), got '{result[0].chunk_id}'. "
            f"Full ranking: {[r.chunk_id for r in result]}"
        )

    def test_first_stage_leader_is_not_rank_one(self) -> None:
        """Chunk 'a' (highest first-stage score) must NOT be ranked first."""
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        result = ListwiseReranker().rerank(QUERY, ALL_CANDIDATES, top_k=3)

        assert result[0].chunk_id != "a", (
            "ListwiseReranker must reorder; first-stage leader 'a' should not be rank 1 "
            "because it misses the rare query token 'amortisation'."
        )

    def test_components_key_is_listwise(self) -> None:
        """Returned ScoredChunks must carry components={'listwise': <float>}."""
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        result = ListwiseReranker().rerank(QUERY, ALL_CANDIDATES, top_k=3)

        for sc in result:
            assert "listwise" in sc.components
            assert isinstance(sc.components["listwise"], float)

    def test_listwise_score_differs_from_first_stage_score(self) -> None:
        """Listwise scores must not equal the original first-stage scores."""
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        result = ListwiseReranker().rerank(QUERY, ALL_CANDIDATES, top_k=3)

        original_scores = {sc.chunk_id: sc.score for sc in ALL_CANDIDATES}
        for sc in result:
            assert sc.components["listwise"] != original_scores[sc.chunk_id], (
                f"Listwise score for chunk '{sc.chunk_id}' equals first-stage score — "
                "that would be tautological."
            )


# ===========================================================================
# 3. ListwiseReranker — whole-list dependency (genuinely listwise)
# ===========================================================================


class TestListwiseWholeListDependency:
    """Removing chunk B from the pool should change A and C's scores because
    'amortisation' is no longer a rare discriminating term — with only A and C
    in the pool, both cover the same set of query tokens ('revenue' only) and
    IDF('amortisation') becomes irrelevant (df=0 → weight=1, but neither A nor
    C contains it).  Scores therefore change from the three-candidate scenario.
    """

    def test_scores_change_when_rare_term_carrier_removed(self) -> None:
        """A and C's listwise scores must differ between the 3-cand and 2-cand pools."""
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        reranker = ListwiseReranker()

        result_full = reranker.rerank(QUERY, ALL_CANDIDATES, top_k=3)
        result_without_b = reranker.rerank(QUERY, [CAND_A, CAND_C], top_k=2)

        score_a_full = next(r.components["listwise"] for r in result_full if r.chunk_id == "a")
        score_a_no_b = next(r.components["listwise"] for r in result_without_b if r.chunk_id == "a")

        assert score_a_full != score_a_no_b, (
            "Chunk A's listwise score must change when chunk B (the only carrier of "
            "'amortisation') is removed from the candidate pool. "
            f"Got {score_a_full} both times — reranker is not genuinely listwise."
        )


# ===========================================================================
# 4. ListwiseReranker — top_k truncation and determinism
# ===========================================================================


class TestListwiseRerankerBehaviours:
    def test_top_k_truncation(self) -> None:
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        result = ListwiseReranker().rerank(QUERY, ALL_CANDIDATES, top_k=2)

        assert len(result) == 2

    def test_top_k_larger_than_candidates(self) -> None:
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        result = ListwiseReranker().rerank(QUERY, ALL_CANDIDATES, top_k=10)

        assert len(result) == len(ALL_CANDIDATES)

    def test_deterministic(self) -> None:
        """Two calls with same inputs must return identical results."""
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        reranker = ListwiseReranker()
        r1 = reranker.rerank(QUERY, ALL_CANDIDATES, top_k=3)
        r2 = reranker.rerank(QUERY, ALL_CANDIDATES, top_k=3)

        assert [sc.chunk_id for sc in r1] == [sc.chunk_id for sc in r2]

    def test_tie_break_chunk_id_asc(self) -> None:
        """Chunks with equal coverage must be ordered by chunk_id ascending."""
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        # Both chunks cover "revenue" only — equal coverage, tie-break by chunk_id.
        cand_z = _make_scored("z", 0.8, "Revenue is high.")
        cand_m = _make_scored("m", 0.2, "Revenue is low.")
        result = ListwiseReranker().rerank("revenue", [cand_z, cand_m], top_k=2)

        assert result[0].chunk_id == "m"
        assert result[1].chunk_id == "z"

    def test_empty_candidates_returns_empty(self) -> None:
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        result = ListwiseReranker().rerank(QUERY, [], top_k=5)

        assert result == []

    def test_no_chunk_object_scores_zero_coverage(self) -> None:
        """ScoredChunks without an attached Chunk score 0 (no text to match)."""
        from jera.adapters.ranking.listwise_reranker import ListwiseReranker

        no_chunk = ScoredChunk(chunk_id="x", score=1.0)
        with_chunk = _make_scored("y", 0.0, "Revenue and amortisation details.")

        result = ListwiseReranker().rerank(QUERY, [no_chunk, with_chunk], top_k=2)

        assert result[0].chunk_id == "y", (
            "Chunk with text coverage must rank above chunk without text."
        )


# ===========================================================================
# 5. ClaudeListwiseReranker — disabled-by-default guard
# ===========================================================================


class TestClaudeListwiseRerankerDisabled:
    def test_disabled_raises_runtime_error(self) -> None:
        from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker

        with pytest.raises(RuntimeError, match="disabled by default"):
            ClaudeListwiseReranker()

    def test_enabled_without_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_anthropic(monkeypatch, {}, [])

        from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker

        with pytest.raises(RuntimeError, match="api_key"):
            ClaudeListwiseReranker(enabled=True)


# ===========================================================================
# 6. ClaudeListwiseReranker — SDK-boundary tests (prompt + permutation)
# ===========================================================================


class TestClaudeListwiseRerankerRequest:
    """Verify the RankGPT-style prompt structure sent to messages.create."""

    def test_model_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("[2] > [1] > [3]")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker

        r = ClaudeListwiseReranker(model="claude-test-model", api_key="key", enabled=True)
        r.rerank(QUERY, ALL_CANDIDATES, top_k=3)

        assert captured["model"] == "claude-test-model"

    def test_max_tokens_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("[1] > [2] > [3]")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker

        r = ClaudeListwiseReranker(api_key="key", enabled=True, max_tokens=128)
        r.rerank(QUERY, ALL_CANDIDATES, top_k=3)

        assert captured["max_tokens"] == 128

    def test_prompt_is_single_user_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("[1] > [2] > [3]")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker

        r = ClaudeListwiseReranker(api_key="key", enabled=True)
        r.rerank(QUERY, ALL_CANDIDATES, top_k=3)

        msgs = captured["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_prompt_contains_numbered_passages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The prompt must include [1], [2], [3] markers for the candidates."""
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("[1] > [2] > [3]")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker

        r = ClaudeListwiseReranker(api_key="key", enabled=True)
        r.rerank(QUERY, ALL_CANDIDATES, top_k=3)

        prompt = captured["messages"][0]["content"]
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt

    def test_prompt_contains_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("[1] > [2] > [3]")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker

        r = ClaudeListwiseReranker(api_key="key", enabled=True)
        r.rerank("MY_DISTINCT_QUERY", ALL_CANDIDATES, top_k=3)

        prompt = captured["messages"][0]["content"]
        assert "MY_DISTINCT_QUERY" in prompt

    def test_prompt_contains_chunk_texts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each candidate's chunk text must appear in the prompt."""
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("[1] > [2] > [3]")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker

        cand = _make_scored("x", 0.5, "UNIQUE_CHUNK_TEXT_XYZ")
        r = ClaudeListwiseReranker(api_key="key", enabled=True)
        r.rerank("query", [cand], top_k=1)

        prompt = captured["messages"][0]["content"]
        assert "UNIQUE_CHUNK_TEXT_XYZ" in prompt


class TestClaudeListwiseRerankerPermutation:
    """Verify the permutation is correctly applied to reorder candidates."""

    def test_permutation_applied_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Model says [2] > [3] > [1] — candidates should be reordered accordingly."""
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("[2] > [3] > [1]")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker

        r = ClaudeListwiseReranker(api_key="key", enabled=True)
        result = r.rerank(QUERY, ALL_CANDIDATES, top_k=3)

        # CAND_A=index0, CAND_B=index1, CAND_C=index2
        # [2]→B, [3]→C, [1]→A
        assert result[0].chunk_id == "b"
        assert result[1].chunk_id == "c"
        assert result[2].chunk_id == "a"

    def test_top_k_truncates_permutation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """top_k=2 must return only the first 2 entries from the permutation."""
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("[3] > [1] > [2]")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker

        r = ClaudeListwiseReranker(api_key="key", enabled=True)
        result = r.rerank(QUERY, ALL_CANDIDATES, top_k=2)

        assert len(result) == 2
        assert result[0].chunk_id == "c"
        assert result[1].chunk_id == "a"

    def test_components_contain_listwise_rank(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each returned ScoredChunk must have components={'listwise_rank': float}."""
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("[1] > [2] > [3]")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker

        r = ClaudeListwiseReranker(api_key="key", enabled=True)
        result = r.rerank(QUERY, ALL_CANDIDATES, top_k=3)

        for sc in result:
            assert "listwise_rank" in sc.components

    def test_partial_permutation_appends_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If the model returns only [2], missing candidates are appended at the end."""
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("[2]")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker

        r = ClaudeListwiseReranker(api_key="key", enabled=True)
        result = r.rerank(QUERY, ALL_CANDIDATES, top_k=3)

        assert len(result) == 3
        assert result[0].chunk_id == "b"  # [2] → B is first
        # Remaining two appear in original order (A=0, C=2)
        remaining_ids = {result[1].chunk_id, result[2].chunk_id}
        assert remaining_ids == {"a", "c"}

    def test_empty_candidates_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        _install_fake_anthropic(monkeypatch, captured, [])

        from jera.adapters.ranking.listwise_reranker import ClaudeListwiseReranker

        r = ClaudeListwiseReranker(api_key="key", enabled=True)
        result = r.rerank(QUERY, [], top_k=5)

        assert result == []
        assert captured == {}  # no API call made
