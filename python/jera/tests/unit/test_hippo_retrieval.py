"""Non-tautological unit tests for HippoRAG graph retrieval.

Multi-hop property under test
------------------------------
The test corpus has three chunks:

  c1: "Alice works at Acme."
      Entities: {alice, acme}

  c2: "Acme has offices in Paris."
      Entities: {acme, paris}

  c3 (distractor): "Zeta Omega Lambda Delta."
      Entities: {zeta, omega, lambda, delta}  — entirely disconnected from
      the Alice-Acme-Paris subgraph.  The text is also longer / lexically
      padded so a naive keyword match might rank it highly — we verify it
      does NOT beat c2.

Query: "Alice"
  Seed node: alice (present in graph, connected only to acme).
  Expected multi-hop path: alice → acme → paris → c2.
  PPR seeded on *alice* flows across the Acme bridge into c2, which shares
  no direct mention of Alice.  c3 is disconnected, so it receives zero PPR
  signal and must rank below c2 (or be absent entirely).

Additional unit tests
---------------------
* PageRank distribution sums to ≈ 1.
* Changing the seed changes the distribution.
* RegexEntityExtractor returns expected entities.
* Empty graph / missing query entities → empty results.
"""

from __future__ import annotations

import math

from jera.adapters.graph.hippo_retriever import (
    HippoGraphRetriever,
    _personalized_pagerank,
)
from jera.adapters.graph.regex_entity_extractor import RegexEntityExtractor
from jera.domain.chunk import Chunk
from jera.domain.document import PageSpan

# ---------------------------------------------------------------------------
# Helpers
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
        chunk_version="1.0.0",
    )


# ---------------------------------------------------------------------------
# RegexEntityExtractor unit tests
# ---------------------------------------------------------------------------


class TestRegexEntityExtractor:
    def test_extracts_proper_nouns(self) -> None:
        ext = RegexEntityExtractor()
        entities = ext.extract("Alice works at Acme Corp in Paris.")
        # "Alice" is sentence-initial → skipped; Acme, Corp, Paris are non-initial
        assert "acme" in entities or "acme corp" in entities
        assert "paris" in entities

    def test_output_is_lowercase(self) -> None:
        ext = RegexEntityExtractor()
        entities = ext.extract("Bob visited London and Berlin.")
        for e in entities:
            assert e == e.lower(), f"Entity not lowercased: {e!r}"

    def test_deduplication(self) -> None:
        ext = RegexEntityExtractor()
        entities = ext.extract("Paris is great. I love Paris.")
        count = sum(1 for e in entities if e == "paris")
        assert count == 1, "Duplicate entities should be deduplicated"

    def test_empty_text(self) -> None:
        ext = RegexEntityExtractor()
        assert ext.extract("") == []

    def test_no_entities_in_lowercase_text(self) -> None:
        ext = RegexEntityExtractor()
        # All lowercase → nothing after position 0 is capitalized
        result = ext.extract("the quick brown fox jumps over the lazy dog.")
        assert result == []


# ---------------------------------------------------------------------------
# PPR unit tests
# ---------------------------------------------------------------------------


class TestPersonalizedPageRank:
    def _simple_graph(self) -> dict[str, dict[str, float]]:
        """A → B → C → A triangle plus isolated D."""
        return {
            "A": {"B": 1.0},
            "B": {"C": 1.0},
            "C": {"A": 1.0},
            "D": {},
        }

    def test_distribution_sums_to_one(self) -> None:
        adj = self._simple_graph()
        rank = _personalized_pagerank(adj, {"A"})
        total = sum(rank.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6), f"Sum={total}"

    def test_seed_node_gets_highest_score(self) -> None:
        adj = self._simple_graph()
        rank_a = _personalized_pagerank(adj, {"A"})
        rank_d = _personalized_pagerank(adj, {"D"})
        # Seeding on A should boost A; seeding on D should boost D
        assert rank_a["A"] > rank_a["D"]
        assert rank_d["D"] > rank_d["A"]

    def test_seeding_changes_distribution(self) -> None:
        adj = self._simple_graph()
        rank_a = _personalized_pagerank(adj, {"A"})
        rank_b = _personalized_pagerank(adj, {"B"})
        # Distributions must differ when seeds differ
        assert rank_a != rank_b

    def test_empty_graph_returns_empty(self) -> None:
        assert _personalized_pagerank({}, {"A"}) == {}

    def test_unknown_seed_returns_empty(self) -> None:
        adj = {"X": {"Y": 1.0}, "Y": {}}
        assert _personalized_pagerank(adj, {"NOTHERE"}) == {}

    def test_all_nodes_present_in_result(self) -> None:
        adj = self._simple_graph()
        rank = _personalized_pagerank(adj, {"A"})
        assert set(rank.keys()) == {"A", "B", "C", "D"}


# ---------------------------------------------------------------------------
# HippoGraphRetriever — multi-hop property (the core non-tautological test)
# ---------------------------------------------------------------------------


class TestHippoGraphRetrieverMultiHop:
    """Prove the Acme-bridge multi-hop property.

    c1: Alice ↔ Acme
    c2: Acme ↔ Paris
    c3: Zeta/Omega/Lambda/Delta (disconnected distractor)

    Query seed: "Alice"
    Expected: PPR flows alice → acme → paris, so c2 gets signal through the
    bridge.  c3 is disconnected → zero PPR → must rank below c2.
    """

    def _build_retriever(self) -> HippoGraphRetriever:
        ext = RegexEntityExtractor()
        return HippoGraphRetriever(ext)

    def _corpus(self) -> list[Chunk]:
        return [
            _make_chunk("c1", "Bob said Alice works at Acme."),
            _make_chunk("c2", "The company Acme has offices in Paris."),
            # Distractor: disconnected from Alice/Acme/Paris; padded with more
            # tokens to avoid trivial ranking advantages.
            _make_chunk(
                "c3",
                "Zeta Omega Lambda Delta Epsilon Theta Xi Psi.",
            ),
        ]

    def test_c2_ranks_above_c3_via_acme_bridge(self) -> None:
        """Multi-hop: query 'Alice' reaches c2 via the Acme bridge."""
        retriever = self._build_retriever()
        retriever.index(self._corpus())
        results = retriever.retrieve("Who is Alice?", top_k=3)

        chunk_ids = [r.chunk_id for r in results]

        # c2 must appear in results (reachable via Alice→Acme→Paris path)
        assert "c2" in chunk_ids, (
            f"c2 missing from results {chunk_ids}; "
            "Acme bridge should propagate PPR signal from Alice to c2"
        )

        # c3 must be absent or rank strictly below c2
        if "c3" in chunk_ids:
            idx_c2 = chunk_ids.index("c2")
            idx_c3 = chunk_ids.index("c3")
            assert idx_c2 < idx_c3, (
                f"c3 (idx={idx_c3}) must rank below c2 (idx={idx_c2}); "
                "disconnected distractor should score zero PPR"
            )

    def test_c3_score_is_zero_or_absent(self) -> None:
        """Disconnected chunk must receive zero PPR signal."""
        retriever = self._build_retriever()
        retriever.index(self._corpus())
        results = retriever.retrieve("Who is Alice?", top_k=10)
        scores = {r.chunk_id: r.score for r in results}
        c3_score = scores.get("c3", 0.0)
        assert c3_score == 0.0, f"c3 should score 0.0, got {c3_score}"

    def test_ppr_component_recorded(self) -> None:
        """Each ScoredChunk must carry components['ppr'] == score."""
        retriever = self._build_retriever()
        retriever.index(self._corpus())
        results = retriever.retrieve("Who is Alice?", top_k=3)
        for r in results:
            assert "ppr" in r.components
            assert math.isclose(r.components["ppr"], r.score, rel_tol=1e-9)

    def test_empty_graph_returns_empty(self) -> None:
        retriever = self._build_retriever()
        # No indexing — graph is empty
        assert retriever.retrieve("Alice", top_k=5) == []

    def test_query_entity_not_in_graph_returns_empty(self) -> None:
        retriever = self._build_retriever()
        retriever.index(self._corpus())
        # "Zzzyxw" is not in any chunk
        results = retriever.retrieve("Zzzyxw", top_k=5)
        assert results == []

    def test_results_respect_top_k(self) -> None:
        retriever = self._build_retriever()
        retriever.index(self._corpus())
        results = retriever.retrieve("Who is Alice?", top_k=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# HippoGraphRetriever — incremental indexing
# ---------------------------------------------------------------------------


class TestHippoGraphRetrieverIndexing:
    def test_incremental_index_accumulates_edges(self) -> None:
        """Indexing in two batches gives the same graph as indexing all at once."""
        ext = RegexEntityExtractor()
        r_batch = HippoGraphRetriever(ext)
        r_incremental = HippoGraphRetriever(ext)

        c1 = _make_chunk("c1", "Bob said Alice works at Acme.")
        c2 = _make_chunk("c2", "The company Acme has offices in Paris.")

        r_batch.index([c1, c2])

        r_incremental.index([c1])
        r_incremental.index([c2])

        results_batch = r_batch.retrieve("Who is Alice?", top_k=5)
        results_inc = r_incremental.retrieve("Who is Alice?", top_k=5)

        ids_batch = [r.chunk_id for r in results_batch]
        ids_inc = [r.chunk_id for r in results_inc]
        assert ids_batch == ids_inc, "Incremental indexing must match batch indexing"
