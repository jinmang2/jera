"""Iterative multi-hop retrieval — NON-TAUTOLOGICAL unit tests.

IRCoT / Search-R1 inference pattern (sequential think→search→stop loop).

References
----------
* Search-R1 (Jin et al., 2025): arXiv:2503.09516
* A-RAG hierarchical retrieval (Li et al., 2026): arXiv:2602.03442
* IRCoT (Trivedi et al., 2022): https://arxiv.org/abs/2212.10509

NON-TAUTOLOGICAL DESIGN
-----------------------
The 2-hop bridge property requires a corpus where the hop-2 answer chunk:
  (a) shares NO content-length tokens with the original query, and
  (b) is reachable ONLY because hop-1 retrieves a chunk that introduces a
      BRIDGE ENTITY which the controller appends to the next query.

Corpus
------
  HOP1 — "alphacorp merged with zetacorp in 2020"
           BM25 overlap with "who did alphacorp merge with":
             matched tokens: {alphacorp, merged/merge, with} → high score.
           Bridge entity introduced: "zetacorp" (absent from original query).

  HOP2 — "zetacorp specializes in quantum semiconductor manufacturing"
           ZERO content-token overlap with the original query
           {"who","did","alphacorp","merge","with"}.
           Only retrievable once "zetacorp" is appended to the query.

  NOISE — "delta inc acquired gamma systems for two hundred million dollars"
           Financial distractor; no overlap with original query or zetacorp.

Original query: "who did alphacorp merge with"
  * BM25 top_k=2: HOP1 (score 2.14), NOISE (score 0.0).
    HOP2 scores 0.0 — NOT retrieved in round-0.

After hop-0 (retrieval of HOP1):
  Controller: HOP1 text contains "zetacorp" which is NOT in issued vocab.
  df("zetacorp") = 1 → top bridge term.
  Follow-up query: "who did alphacorp merge with zetacorp"

Hop-1 retrieval of "who did alphacorp merge with zetacorp" (top_k=2):
  HOP2 matches "zetacorp" → score 0.51 → retrieved at rank 2.
  HOP2 IS accumulated for the first time.

Result: IterativeRetrievalPipeline issues ≥ 2 queries, rounds ≥ 2,
and HOP2 appears in accumulated contexts after iterative retrieval.

Self-containment / STOP tests
------------------------------
Query: "zetacorp specializes quantum semiconductor" (exact tokens of HOP2).
With HOP2 as the only corpus chunk, coverage after hop-0 reaches 1.0 ≥ 0.85
→ controller returns None → loop terminates in 1 round.

Max-hops test
-------------
A _NeverStopController always returns a new query string.  The pipeline's
hard max_hops cap stops the loop after exactly max_hops rounds regardless.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from jera.adapters.query.bridge_followup_controller import BridgeFollowupController
from jera.config.registry import RagSystem, build_system
from jera.config.settings import Profile, Settings
from jera.domain.chunk import Chunk
from jera.domain.document import MediaType, PageSpan, SourceRef
from jera.domain.retrieval import Query, RetrievalMode
from jera.pipeline.iterative import IterativeResult, IterativeRetrievalPipeline
from jera.ports.followup_controller import FollowupController

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

# HOP1: matches original query on {alphacorp, merged≈merge, with}.
#        Introduces "zetacorp" as bridge entity absent from the original query.
_HOP1_SID = "hop1_alphacorp"
_HOP1_TEXT = "alphacorp merged with zetacorp in 2020"

# HOP2: ZERO token overlap with "who did alphacorp merge with".
#        Only reachable via the bridge "zetacorp".
_HOP2_SID = "hop2_zetacorp"
_HOP2_TEXT = "zetacorp specializes in quantum semiconductor manufacturing"

# NOISE: financial distractor — no overlap with query or bridge term.
_NOISE_SID = "noise_delta"
_NOISE_TEXT = "delta inc acquired gamma systems for two hundred million dollars"

# Original query must NOT contain any content token from _HOP2_TEXT.
# Confirmed:
#   {"who","did","alphacorp","merge","with"} ∩
#   {"zetacorp","specializes","quantum","semiconductor","manufacturing"} = ∅
_QUERY = "who did alphacorp merge with"

# Top-k for retrieval.  We use 2 so that:
# - round-0: HOP1 (score 2.14) + NOISE (score 0.0) → HOP2 NOT accumulated.
# - round-1: HOP1 (score 2.65) + HOP2 (score 0.51) → HOP2 IS accumulated.
_TOP_K = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_system(corpus: dict[str, str]) -> RagSystem:
    system = build_system(Settings(profile=Profile.TEST))
    system.ingest.ingest_many(
        [
            SourceRef(source_id=sid, media_type=MediaType.MARKDOWN, content=text.encode())
            for sid, text in corpus.items()
        ]
    )
    return system


def _source_ids(result: IterativeResult) -> set[str]:
    return {c.source_id for c in result.contexts}


def _make_chunk(chunk_id: str, source_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=source_id,
        source_id=source_id,
        text=text,
        page_span=PageSpan(start_page=1, end_page=1),
        section_path=(),
        element_ids=(),
        char_span=(0, len(text)),
        token_count=len(text.split()),
        chunk_strategy="test",
        chunk_version="0",
    )


# ---------------------------------------------------------------------------
# Prerequisite: single retrieve misses HOP2
# ---------------------------------------------------------------------------


class TestSingleRetrieveMissesHop2:
    """Baseline: without iterative retrieval, HOP2 cannot be reached.

    This is the prerequisite that makes the 2-hop test non-tautological:
    if single-step retrieval already surfaced HOP2, the iterative pipeline
    would provide no genuine lift.
    """

    def setup_method(self) -> None:
        self.system = _build_system(
            {_HOP1_SID: _HOP1_TEXT, _HOP2_SID: _HOP2_TEXT, _NOISE_SID: _NOISE_TEXT}
        )

    def test_hop2_text_has_no_content_tokens_in_common_with_query(self) -> None:
        """Structural check: the bridge property requires zero content-token overlap."""
        query_tokens = {t for t in _QUERY.lower().split() if len(t) >= 3}
        hop2_tokens = {t for t in _HOP2_TEXT.lower().split() if len(t) >= 3}
        shared = query_tokens & hop2_tokens
        assert not shared, (
            f"HOP2 must share no content tokens (len>=3) with the query; shared: {shared}"
        )

    def test_single_retrieve_top_k_returns_hop1_not_hop2(self) -> None:
        """BM25 top-{k} on the original query must include HOP1 but NOT HOP2.

        HOP2 has zero lexical overlap with the original query so its BM25
        score is 0.  HOP1 matches {alphacorp, merged, with} and wins.
        Verified empirically: round-0 top_k=2 → [HOP1 (2.14), NOISE (0.0)].
        """
        q = Query(text=_QUERY, top_k=_TOP_K, mode=RetrievalMode.SPARSE)
        result = self.system.query.retrieve(q)
        source_ids = {sc.chunk.source_id for sc in result.results if sc.chunk}
        assert _HOP1_SID in source_ids, f"HOP1 must be in round-0 top-{_TOP_K}; got {source_ids}"
        assert _HOP2_SID not in source_ids, (
            f"HOP2 must NOT appear in round-0 top-{_TOP_K}; got {source_ids}.  "
            "If this fails, BM25 scoring changed — adjust top_k or the corpus."
        )

    def test_bridge_term_zetacorp_is_in_hop1_text(self) -> None:
        """The bridge entity 'zetacorp' must appear in HOP1's text.

        The controller scans new chunks for bridge terms; it must find 'zetacorp' here.
        """
        assert "zetacorp" in _HOP1_TEXT.lower()

    def test_bridge_term_zetacorp_not_in_original_query(self) -> None:
        """The bridge entity must be absent from the original query (otherwise no bridge needed)."""
        assert "zetacorp" not in _QUERY.lower()


# ---------------------------------------------------------------------------
# Core 2-hop bridge property
# ---------------------------------------------------------------------------


class TestTwoHopBridgeProperty:
    """Core NON-TAUTOLOGICAL test: HOP2 is only reachable via the bridge entity.

    The IterativeRetrievalPipeline must:
    1. Issue >= 2 queries.
    2. Report rounds >= 2.
    3. Accumulate the HOP2 chunk (not reachable in a single round-0 retrieve).

    The reachability is a genuine BM25 consequence: appending "zetacorp" to
    the follow-up query gives HOP2 a non-zero BM25 score (0.51) so it appears
    in top-2 of round-1.  Without the bridge term, HOP2 scores 0.0 and is
    never returned.
    """

    def setup_method(self) -> None:
        self.system = _build_system(
            {_HOP1_SID: _HOP1_TEXT, _HOP2_SID: _HOP2_TEXT, _NOISE_SID: _NOISE_TEXT}
        )
        # coverage_threshold=1.0 disables early-stop so we always attempt hop-1.
        controller = BridgeFollowupController(
            max_hops=3,
            coverage_threshold=1.0,
            max_bridge_terms=6,
            min_token_length=3,
        )
        self.pipe = IterativeRetrievalPipeline(
            self.system.query,
            controller,
            self.system.generator,
            max_hops=3,
        )

    def test_pipeline_issues_at_least_two_queries(self) -> None:
        result = self.pipe.answer(_QUERY, top_k=_TOP_K, mode=RetrievalMode.SPARSE)
        assert len(result.queries) >= 2, (
            f"IterativeRetrievalPipeline must issue >= 2 queries for a 2-hop bridge; "
            f"got queries: {result.queries}"
        )

    def test_pipeline_rounds_at_least_two(self) -> None:
        result = self.pipe.answer(_QUERY, top_k=_TOP_K, mode=RetrievalMode.SPARSE)
        assert result.rounds >= 2, f"rounds must be >= 2 for the 2-hop case; got {result.rounds}"

    def test_pipeline_accumulates_hop2_chunk(self) -> None:
        """The real bridge test: HOP2 is in accumulated contexts after iterative retrieval.

        This is non-tautological because:
        - Round-0 with the original query ("who did alphacorp merge with") returns
          HOP1 and NOISE; HOP2 has BM25 score 0.0 and is not returned.
        - The controller extracts "zetacorp" from HOP1's text (a bridge entity not
          in the original query vocabulary) and appends it to the next query.
        - Round-1 with "who did alphacorp merge with zetacorp" gives HOP2 a
          non-zero BM25 score (≈0.51) and retrieves it at rank 2 of top_k=2.
        - HOP2 is then accumulated into contexts.
        """
        result = self.pipe.answer(_QUERY, top_k=_TOP_K, mode=RetrievalMode.SPARSE)
        assert _HOP2_SID in _source_ids(result), (
            f"HOP2 chunk ({_HOP2_SID!r}) must be accumulated after iterative retrieval.  "
            f"Contexts found: {_source_ids(result)!r}.  Queries issued: {result.queries}"
        )

    def test_hop1_also_accumulated(self) -> None:
        """HOP1 (the bridge chunk from round 0) is also in the accumulated contexts."""
        result = self.pipe.answer(_QUERY, top_k=_TOP_K, mode=RetrievalMode.SPARSE)
        assert _HOP1_SID in _source_ids(result), (
            f"HOP1 chunk must be in accumulated contexts; got {_source_ids(result)!r}"
        )

    def test_second_query_contains_bridge_term(self) -> None:
        """The second query issued must contain 'zetacorp' (the bridge entity from HOP1)."""
        result = self.pipe.answer(_QUERY, top_k=_TOP_K, mode=RetrievalMode.SPARSE)
        assert len(result.queries) >= 2, "need at least 2 queries to check the bridge term"
        second_q = result.queries[1].lower()
        assert "zetacorp" in second_q, (
            f"The second query must contain the bridge entity 'zetacorp'; got: {second_q!r}"
        )

    def test_contexts_are_unique(self) -> None:
        """Accumulated contexts must have no duplicate chunk_ids."""
        result = self.pipe.answer(_QUERY, top_k=_TOP_K, mode=RetrievalMode.SPARSE)
        chunk_ids = [c.chunk_id for c in result.contexts]
        assert len(chunk_ids) == len(set(chunk_ids)), (
            f"duplicate chunk_ids in accumulated contexts: {chunk_ids}"
        )

    def test_result_has_answer(self) -> None:
        result = self.pipe.answer(_QUERY, top_k=_TOP_K, mode=RetrievalMode.SPARSE)
        assert result.answer is not None
        assert hasattr(result.answer, "text")


# ---------------------------------------------------------------------------
# STOP condition: self-contained query terminates in 1 round
# ---------------------------------------------------------------------------


class TestStopConditionSelfContained:
    """The controller stops in 1 round when the query is self-contained.

    Corpus: only HOP2 chunk (zetacorp text).
    Query: uses tokens that are all present in HOP2 → high coverage after hop-0.

    After hop-0, coverage of query content-tokens (len>=3) by the retrieved
    chunk reaches the coverage_threshold (0.85), so the controller returns None
    and the loop terminates in 1 round.
    """

    def setup_method(self) -> None:
        # Corpus contains only the HOP2 chunk.
        self.system = _build_system({_HOP2_SID: _HOP2_TEXT})
        self.controller = BridgeFollowupController(
            max_hops=5,  # permissive cap — stop must come from coverage, not cap
            coverage_threshold=0.85,
            max_bridge_terms=6,
            min_token_length=3,
        )
        self.pipe = IterativeRetrievalPipeline(
            self.system.query,
            self.controller,
            self.system.generator,
            max_hops=5,
        )

    def test_self_contained_query_terminates_in_one_round(self) -> None:
        """A query whose content tokens are covered by the top chunk stops after round 0."""
        # Use a query whose content tokens are a subset of HOP2's tokens.
        self_contained_query = "zetacorp specializes quantum semiconductor"
        result = self.pipe.answer(self_contained_query, top_k=_TOP_K, mode=RetrievalMode.SPARSE)
        assert result.rounds == 1, (
            f"Self-contained query should stop in 1 round (coverage-based); "
            f"got rounds={result.rounds}, queries={result.queries}"
        )


# ---------------------------------------------------------------------------
# STOP condition: max_hops is respected (no infinite loop)
# ---------------------------------------------------------------------------


class _NeverStopController:
    """Test-only controller that never returns None — exercises the max_hops hard cap."""

    def next_query(
        self,
        original_query: str,
        accumulated_chunks: Sequence[Chunk],
        round_index: int,
    ) -> str | None:
        # Always request another hop — relies on the pipeline's max_hops cap to stop.
        return original_query + f" hop{round_index + 1}"


# Verify _NeverStopController satisfies the FollowupController protocol.
_check: FollowupController = _NeverStopController()  # type: ignore[assignment]


class TestMaxHopsRespected:
    """max_hops is a hard cap that terminates the loop regardless of controller output."""

    def setup_method(self) -> None:
        self.system = _build_system({_HOP1_SID: _HOP1_TEXT})

    @pytest.mark.parametrize("max_hops", [1, 2, 3])
    def test_rounds_never_exceed_max_hops(self, max_hops: int) -> None:
        controller = _NeverStopController()
        pipe = IterativeRetrievalPipeline(
            self.system.query,
            controller,
            self.system.generator,
            max_hops=max_hops,
        )
        result = pipe.answer(_QUERY, top_k=1, mode=RetrievalMode.SPARSE)
        assert result.rounds <= max_hops, (
            f"rounds ({result.rounds}) must not exceed max_hops ({max_hops})"
        )
        assert len(result.queries) <= max_hops, (
            f"number of queries ({len(result.queries)}) must not exceed max_hops ({max_hops})"
        )

    def test_max_hops_1_terminates_after_single_round(self) -> None:
        controller = _NeverStopController()
        pipe = IterativeRetrievalPipeline(
            self.system.query,
            controller,
            self.system.generator,
            max_hops=1,
        )
        result = pipe.answer(_QUERY, top_k=1, mode=RetrievalMode.SPARSE)
        assert result.rounds == 1
        assert len(result.queries) == 1


# ---------------------------------------------------------------------------
# BridgeFollowupController unit tests (deterministic, no retrieval system)
# ---------------------------------------------------------------------------


class TestBridgeFollowupController:
    """Unit tests for BridgeFollowupController without any retrieval system."""

    def test_returns_none_when_max_hops_reached(self) -> None:
        ctrl = BridgeFollowupController(max_hops=2)
        chunk = _make_chunk("c1", "s1", "zetacorp revenue billion")
        # round_index=1 >= max_hops-1=1 → STOP
        result = ctrl.next_query("who did alphacorp merge with", [chunk], round_index=1)
        assert result is None

    def test_returns_next_query_on_first_round(self) -> None:
        ctrl = BridgeFollowupController(max_hops=3, coverage_threshold=1.0)
        chunk = _make_chunk("c1", "s1", "zetacorp revenue exceeded billion")
        result = ctrl.next_query("who did alphacorp merge with", [chunk], round_index=0)
        assert result is not None
        assert "zetacorp" in result.lower()

    def test_bridge_terms_not_duplicated_across_rounds(self) -> None:
        ctrl = BridgeFollowupController(max_hops=5, coverage_threshold=1.0, max_bridge_terms=10)
        chunk = _make_chunk("c1", "s1", "zetacorp revenue exceeded billion")
        q1 = ctrl.next_query("who did alphacorp merge with", [chunk], round_index=0)
        assert q1 is not None
        # Second round: zetacorp is now in issued_vocab; should not be re-added as bridge.
        chunk2 = _make_chunk("c2", "s2", "zetacorp produces software widgets")
        q2 = ctrl.next_query("who did alphacorp merge with", [chunk, chunk2], round_index=1)
        if q2 is not None:
            # "zetacorp" should not be added again (already in issued_vocab).
            # Count: appears once from q1's append; q2 should NOT add it again.
            assert q2.lower().count("zetacorp") <= 1

    def test_returns_none_when_no_new_bridge_terms(self) -> None:
        ctrl = BridgeFollowupController(max_hops=5, coverage_threshold=1.0)
        # Chunk text has only tokens already in the query — no new bridges.
        chunk = _make_chunk("c1", "s1", "who did alphacorp merge with")
        result = ctrl.next_query("who did alphacorp merge with", [chunk], round_index=0)
        assert result is None

    def test_reset_clears_issued_vocab(self) -> None:
        ctrl = BridgeFollowupController(max_hops=5, coverage_threshold=1.0)
        chunk = _make_chunk("c1", "s1", "zetacorp revenue exceeded billion")
        ctrl.next_query("who did alphacorp merge with", [chunk], round_index=0)
        ctrl.reset()
        # After reset, issued_vocab is empty; next call re-seeds from original_query.
        q = ctrl.next_query("who did alphacorp merge with", [chunk], round_index=0)
        assert q is not None
        assert "zetacorp" in q.lower()

    def test_constructor_rejects_max_hops_below_1(self) -> None:
        with pytest.raises(ValueError, match="max_hops"):
            BridgeFollowupController(max_hops=0)

    def test_coverage_based_stop(self) -> None:
        ctrl = BridgeFollowupController(max_hops=5, coverage_threshold=0.5)
        # Chunk covers all content tokens of query → coverage=1.0 >= 0.5 → STOP.
        chunk = _make_chunk("c1", "s1", "who did alphacorp merge with zetacorp")
        result = ctrl.next_query("who did alphacorp merge with", [chunk], round_index=0)
        assert result is None

    def test_returns_none_when_no_chunks(self) -> None:
        ctrl = BridgeFollowupController(max_hops=5, coverage_threshold=1.0)
        # No accumulated chunks → no bridge terms → STOP.
        result = ctrl.next_query("who did alphacorp merge with", [], round_index=0)
        assert result is None


# ---------------------------------------------------------------------------
# IterativeResult dataclass contract
# ---------------------------------------------------------------------------


class TestIterativeResultContract:
    """Structural contract tests for IterativeResult fields."""

    def setup_method(self) -> None:
        self.system = _build_system({_HOP1_SID: _HOP1_TEXT})
        controller = BridgeFollowupController(max_hops=2, coverage_threshold=1.0)
        self.pipe = IterativeRetrievalPipeline(
            self.system.query,
            controller,
            self.system.generator,
            max_hops=2,
        )

    def test_result_has_required_fields(self) -> None:
        result = self.pipe.answer(_QUERY, top_k=1, mode=RetrievalMode.SPARSE)
        assert hasattr(result, "answer")
        assert hasattr(result, "queries")
        assert hasattr(result, "contexts")
        assert hasattr(result, "rounds")

    def test_queries_list_length_equals_rounds(self) -> None:
        result = self.pipe.answer(_QUERY, top_k=1, mode=RetrievalMode.SPARSE)
        assert len(result.queries) == result.rounds

    def test_first_query_is_normalized_original(self) -> None:
        result = self.pipe.answer(
            "  who did  alphacorp  merge  with  ", top_k=1, mode=RetrievalMode.SPARSE
        )
        # QueryPipeline.analyze normalizes whitespace.
        assert result.queries[0] == "who did alphacorp merge with"

    def test_pipeline_rejects_max_hops_below_1(self) -> None:
        ctrl = BridgeFollowupController(max_hops=1)
        with pytest.raises(ValueError, match="max_hops"):
            IterativeRetrievalPipeline(
                self.system.query,
                ctrl,
                self.system.generator,
                max_hops=0,
            )
