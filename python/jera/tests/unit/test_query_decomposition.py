"""Sub-question decomposition for multi-hop RAG — NON-TAUTOLOGICAL coverage test.

Multi-hop setup (Pereira et al., ACL-SRW 2025, arXiv:2507.00355)
-----------------------------------------------------------------
The claim: splitting a multi-hop query into ordered sub-questions, retrieving each hop
independently, and accumulating contexts beats single-step retrieval on bridge-entity queries.

Corpus (4 chunks):
  A — "The Eiffel Tower is located in Paris."         (hop-1 answer)
  B — "Paris is the capital of France."               (hop-2 bridge answer)
  C — "The Louvre Museum houses thousands of artworks."  (distractor)
  D — "The Seine river flows through Paris and Normandy." (distractor with "Paris")

Query: "Where is the Eiffel Tower located, and what is Paris the capital of?"

The ConnectiveDecomposer matches R5 (comma + "and") and produces:
  sub-q1 → "Where is the Eiffel Tower located?"   BM25 tokens: eiffel, tower, located  → hits A
  sub-q2 → "what is Paris the capital of?"        BM25 tokens: paris, capital           → hits B

Single-step retrieve(top_k=1):
  BM25 on the full query overlaps chunk A on {eiffel, tower, located, paris} (4 tokens) and
  chunk B on {paris, capital} (2 tokens) → chunk A wins; chunk B is NOT in top-1.

Decomposed retrieve(top_k=1 per sub-question):
  sub-q1 → [A],  sub-q2 → [B]  → accumulated = [A, B].

Coverage gain assertion (non-tautological):
  The set of source_ids in single-step top-1 is a strict SUBSET of decomposed contexts — the
  decomposed pipeline surfaces a genuinely new chunk (B) that single-step retrieval misses.
  This is a real retrieval difference, not a score manipulation.
"""

from __future__ import annotations

from jera.adapters.query.connective_decomposer import ConnectiveDecomposer
from jera.config.registry import RagSystem, build_system
from jera.config.settings import Profile, Settings
from jera.domain.document import MediaType, SourceRef
from jera.domain.retrieval import Query, RetrievalMode
from jera.pipeline.decompositional import DecompositionalQueryPipeline

# ── Corpus ─────────────────────────────────────────────────────────────────────
# Each chunk is ingested as a one-section Markdown document so the heading-aware
# chunker produces exactly one retrievable chunk per source_id.

_CORPUS: dict[str, str] = {
    "eiffel_location": "# Eiffel Tower\n\nThe Eiffel Tower is located in Paris.\n",
    "paris_capital": "# Paris\n\nParis is the capital of France.\n",
    "louvre": "# Louvre\n\nThe Louvre Museum houses thousands of artworks.\n",
    "seine": "# Seine\n\nThe Seine river flows through Paris and Normandy.\n",
}

# The compound query the decomposer splits into two independent sub-questions.
_QUERY = "Where is the Eiffel Tower located, and what is Paris the capital of?"

# Sub-question 1 targets chunk A (Eiffel Tower / Paris).
_SUB_Q1_TOKENS = {"eiffel", "tower", "located"}
# Sub-question 2 targets chunk B (Paris / capital / France).
_SUB_Q2_TOKENS = {"paris", "capital"}

# Source ids for the two bridge chunks.
_HOP1_SOURCE = "eiffel_location"
_HOP2_SOURCE = "paris_capital"


def _build_ingested_system() -> RagSystem:
    system = build_system(Settings(profile=Profile.TEST))
    system.ingest.ingest_many(
        [
            SourceRef(source_id=sid, media_type=MediaType.MARKDOWN, content=md.encode())
            for sid, md in _CORPUS.items()
        ]
    )
    return system


# ── Decomposer unit tests ──────────────────────────────────────────────────────


class TestConnectiveDecomposer:
    """Unit tests for ConnectiveDecomposer rule-matching — no retrieval involved."""

    def setup_method(self) -> None:
        self.dec = ConnectiveDecomposer()

    def test_comma_and_splits_into_two_subquestions(self) -> None:
        subs = self.dec.decompose(_QUERY)
        assert len(subs) == 2, f"expected 2 sub-questions, got {subs}"

    def test_comma_and_sub_q1_targets_eiffel(self) -> None:
        sub_q1 = self.dec.decompose(_QUERY)[0].lower()
        assert "eiffel" in sub_q1 or "tower" in sub_q1 or "located" in sub_q1, (
            f"sub-q1 should mention Eiffel Tower / located: {sub_q1!r}"
        )

    def test_comma_and_sub_q2_targets_paris_capital(self) -> None:
        sub_q2 = self.dec.decompose(_QUERY)[1].lower()
        assert "paris" in sub_q2 or "capital" in sub_q2, (
            f"sub-q2 should mention Paris / capital: {sub_q2!r}"
        )

    def test_simple_query_is_unchanged(self) -> None:
        simple = "What is the capital of France?"
        subs = self.dec.decompose(simple)
        assert subs == [simple], f"simple query should pass through unchanged, got {subs}"

    def test_compare_pattern(self) -> None:
        q = "Compare dense retrieval with sparse retrieval"
        subs = self.dec.decompose(q)
        assert len(subs) == 2
        assert all(len(s) >= 5 for s in subs)

    def test_binary_and_pattern(self) -> None:
        q = "What is retrieval augmented generation and how does it work?"
        subs = self.dec.decompose(q)
        # R4 should fire and split into 2 sub-questions
        assert len(subs) == 2

    def test_empty_string_returns_original(self) -> None:
        subs = self.dec.decompose("")
        assert subs == [""]

    def test_both_and_pattern(self) -> None:
        q = "Tell me about both sparse retrieval and dense retrieval"
        subs = self.dec.decompose(q)
        assert len(subs) == 2


# ── Multi-hop coverage tests ───────────────────────────────────────────────────


class TestMultiHopCoverageGain:
    """NON-TAUTOLOGICAL multi-hop test: decomposed pipeline surfaces both bridge chunks;
    single-step top-1 retrieval misses chunk B (the second hop).

    The coverage gain is structural, not score-rigged:
    - BM25 on the FULL query gives chunk A more tokens in common than chunk B (4 vs 2).
    - Decomposed sub-q2 is a focused query whose lexical overlap specifically targets chunk B.
    """

    def setup_method(self) -> None:
        self.system = _build_ingested_system()
        self.dec = ConnectiveDecomposer()

    # ── Prerequisite: the decomposer does split our query ──────────────────────

    def test_decomposer_produces_two_subquestions(self) -> None:
        subs = self.dec.decompose(_QUERY)
        assert len(subs) == 2

    # ── Single-step baseline ────────────────────────────────────────────────────

    def test_single_step_top1_returns_eiffel_chunk(self) -> None:
        """Single-step BM25 retrieve with top_k=1 returns the Eiffel Tower chunk (hop-1)."""
        q = Query(text=_QUERY, top_k=1, mode=RetrievalMode.SPARSE)
        result = self.system.query.retrieve(q)
        sources = [sc.chunk.source_id for sc in result.results if sc.chunk]
        assert _HOP1_SOURCE in sources, (
            f"expected {_HOP1_SOURCE!r} in single-step top-1, got {sources}"
        )

    def test_single_step_top1_misses_paris_capital_chunk(self) -> None:
        """Single-step BM25 top-1 does NOT return the Paris/capital chunk (hop-2 bridge).

        This is the key non-tautological property: the second bridge chunk (B) is NOT
        reachable in a single top-1 retrieve because chunk A has strictly more lexical
        overlap with the full compound query.
        """
        q = Query(text=_QUERY, top_k=1, mode=RetrievalMode.SPARSE)
        result = self.system.query.retrieve(q)
        sources = [sc.chunk.source_id for sc in result.results if sc.chunk]
        assert _HOP2_SOURCE not in sources, (
            f"single-step top-1 should miss {_HOP2_SOURCE!r}; "
            f"if it appears, the BM25 tie-break changed — adjust top_k. Got: {sources}"
        )

    # ── Decomposed pipeline ────────────────────────────────────────────────────

    def test_decomposed_accumulates_both_bridge_chunks(self) -> None:
        """Decomposed pipeline accumulates BOTH hop-1 (A) and hop-2 (B) chunks."""
        pipe = DecompositionalQueryPipeline(
            pipeline=self.system.query,
            decomposer=self.dec,
            generator=self.system.generator,
        )
        res = pipe.answer(_QUERY, top_k=1, mode=RetrievalMode.SPARSE)
        source_ids = {c.source_id for c in res.contexts}
        assert _HOP1_SOURCE in source_ids, (
            f"decomposed should include hop-1 chunk {_HOP1_SOURCE!r}; got {source_ids}"
        )
        assert _HOP2_SOURCE in source_ids, (
            f"decomposed should include hop-2 chunk {_HOP2_SOURCE!r}; got {source_ids}"
        )

    # ── Coverage gain (the headline assertion) ─────────────────────────────────

    def test_decomposed_coverage_strictly_greater_than_single_step_top1(self) -> None:
        """The decomposed pipeline covers strictly MORE bridge chunks than single-step top-1.

        Single-step top-1 source set ⊂ decomposed source set (strict subset).
        This is the non-tautological multi-hop coverage gain.
        """
        # Single-step top-1 sources
        q = Query(text=_QUERY, top_k=1, mode=RetrievalMode.SPARSE)
        single_sources = {
            sc.chunk.source_id for sc in self.system.query.retrieve(q).results if sc.chunk
        }

        # Decomposed sources
        pipe = DecompositionalQueryPipeline(
            pipeline=self.system.query,
            decomposer=self.dec,
            generator=self.system.generator,
        )
        res = pipe.answer(_QUERY, top_k=1, mode=RetrievalMode.SPARSE)
        decomposed_sources = {c.source_id for c in res.contexts}

        # Strict superset: decomposed covers everything single-step does, plus more.
        assert single_sources < decomposed_sources, (
            f"decomposed ({decomposed_sources}) must strictly cover more than "
            f"single-step top-1 ({single_sources})"
        )

    # ── Result contract ────────────────────────────────────────────────────────

    def test_result_exposes_sub_questions(self) -> None:
        pipe = DecompositionalQueryPipeline(
            pipeline=self.system.query,
            decomposer=self.dec,
            generator=self.system.generator,
        )
        res = pipe.answer(_QUERY, top_k=1, mode=RetrievalMode.SPARSE)
        assert len(res.sub_questions) == 2
        assert res.answer is not None

    def test_simple_query_degrades_to_single_step(self) -> None:
        """A non-compound query returns [query] from the decomposer → single retrieval hop."""
        pipe = DecompositionalQueryPipeline(
            pipeline=self.system.query,
            decomposer=self.dec,
            generator=self.system.generator,
        )
        simple = "What is the capital of France?"
        res = pipe.answer(simple, top_k=3, mode=RetrievalMode.SPARSE)
        assert res.sub_questions == [simple], (
            f"simple query should not be decomposed: {res.sub_questions}"
        )

    def test_contexts_are_unique(self) -> None:
        """Accumulated contexts must have no duplicate chunk_ids."""
        pipe = DecompositionalQueryPipeline(
            pipeline=self.system.query,
            decomposer=self.dec,
            generator=self.system.generator,
        )
        res = pipe.answer(_QUERY, top_k=2, mode=RetrievalMode.SPARSE)
        chunk_ids = [c.chunk_id for c in res.contexts]
        assert len(chunk_ids) == len(set(chunk_ids)), "duplicate chunk_ids in accumulated contexts"
