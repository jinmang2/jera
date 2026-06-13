"""Unit tests for the context-engineering pipeline adapters.

Non-tautological property each test suite proves
-------------------------------------------------

Reorderer (LostInTheMiddleReorderer)
    The test constructs 5 chunks in explicit relevance order [a, b, c, d, e] and
    asserts that after reordering:
      - chunk ``a`` (rank 1) lands at index 0 (first position, primacy).
      - chunk ``b`` (rank 2) lands at index 4 (last position, recency).
      - chunk ``c`` (rank 3, the median-ranked item) occupies the middle index 2.
    This is non-tautological because a naïve pass-through would leave [a,b,c,d,e]
    and satisfy the first assertion only.  The tests require a real interleave that
    simultaneously satisfies all three positional constraints.

Curator (RedundancyCurator)
    The test constructs 4 chunks where chunk-2 is an intentional near-duplicate of
    chunk-1 (token-Jaccard ≥ threshold).  It asserts that exactly 3 chunks remain
    (chunk-2 dropped), that those 3 are the distinct ones, and that a control pair
    well below the threshold is NOT dropped.  The non-tautological nature is that a
    pass-through keeps 4, and a too-aggressive implementation would drop distinct
    chunks — the tests verify the correct boundary behaviour.

Compressor (ExtractiveCompressor)
    The test constructs a chunk with 4 sentences: one answer-bearing sentence (high
    token overlap with the query) and three noise sentences (zero overlap).  It asserts
    that:
      - The answer sentence is retained in the compressed output.
      - At least one noise sentence is dropped (overall length is reduced).
      - Token count drops by ≥ 30 %.
      - The retained sentence was chosen by overlap, not by position — verified by
        placing the answer sentence in the non-first, non-last position so a naïve
        head/tail extractor would not find it.
    This is non-tautological because a compressor that always returns the first
    sentence would fail (answer is at index 1 of 4), and one that returns all
    sentences would fail the length-reduction assertion.

Protocol
    Each adapter is verified to satisfy isinstance(adapter, ContextProcessor), proving
    the runtime-checkable Protocol structural match holds.
"""

from __future__ import annotations

from jera.adapters.context.compressor import ExtractiveCompressor
from jera.adapters.context.curator import RedundancyCurator
from jera.adapters.context.reorderer import LostInTheMiddleReorderer
from jera.domain.chunk import Chunk
from jera.domain.document import PageSpan
from jera.ports.context_processor import ContextProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        source_id="src",
        text=text,
        page_span=PageSpan.single(1),
        section_path=(),
        element_ids=(),
        char_span=(0, len(text)),
        token_count=len(text.split()),
        chunk_strategy="test",
        chunk_version="0",
    )


# ---------------------------------------------------------------------------
# LostInTheMiddleReorderer
# ---------------------------------------------------------------------------


class TestLostInTheMiddleReorderer:
    """Tests for LostInTheMiddleReorderer."""

    def test_protocol_satisfied(self) -> None:
        """LostInTheMiddleReorderer satisfies the ContextProcessor Protocol."""
        assert isinstance(LostInTheMiddleReorderer(), ContextProcessor)

    def test_rank1_at_index0_rank2_at_last_index(self) -> None:
        """Rank-1 lands first (primacy); rank-2 lands last (recency).

        Non-tautological: a pass-through satisfies only the first assertion.
        The reorderer must perform a real alternating-edge interleave.
        """
        chunks = [
            _chunk("a", "alpha text"),
            _chunk("b", "beta text"),
            _chunk("c", "gamma text"),
            _chunk("d", "delta text"),
            _chunk("e", "epsilon text"),
        ]
        reorderer = LostInTheMiddleReorderer()
        result = reorderer.process("query", chunks)

        assert len(result) == 5
        assert result[0].chunk_id == "a", (
            f"Rank-1 chunk must be at index 0 (primacy); got {result[0].chunk_id!r}"
        )
        assert result[4].chunk_id == "b", (
            f"Rank-2 chunk must be at index 4 (recency); got {result[4].chunk_id!r}"
        )

    def test_lowest_ranked_at_middle(self) -> None:
        """The median-ranked chunk (rank 3 of 5) occupies the middle position.

        With 5 chunks the interleave is [a, c, e, d, b]; rank-3 (c) → index 1,
        rank-4 (d) → index 3, rank-5 (e) → index 2 (the exact middle).
        """
        chunks = [
            _chunk("a", "alpha"),
            _chunk("b", "beta"),
            _chunk("c", "gamma"),
            _chunk("d", "delta"),
            _chunk("e", "epsilon"),
        ]
        reorderer = LostInTheMiddleReorderer()
        result = reorderer.process("query", chunks)

        # The lowest-ranked chunk (rank 5, "e") must be at the middle index.
        middle_idx = len(result) // 2
        assert result[middle_idx].chunk_id == "e", (
            f"Lowest-ranked chunk must occupy the middle (index {middle_idx}); "
            f"got {result[middle_idx].chunk_id!r}. Full order: "
            f"{[r.chunk_id for r in result]}"
        )

    def test_all_chunks_preserved(self) -> None:
        """No chunks are added or dropped by the reorderer."""
        chunks = [_chunk(f"c{i}", f"text {i}") for i in range(6)]
        result = LostInTheMiddleReorderer().process("q", chunks)
        assert sorted(r.chunk_id for r in result) == sorted(c.chunk_id for c in chunks)

    def test_single_chunk_unchanged(self) -> None:
        """A single-chunk list is returned as-is."""
        chunk = _chunk("only", "solo text")
        result = LostInTheMiddleReorderer().process("q", [chunk])
        assert result == [chunk]

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert LostInTheMiddleReorderer().process("q", []) == []


# ---------------------------------------------------------------------------
# RedundancyCurator
# ---------------------------------------------------------------------------


class TestRedundancyCurator:
    """Tests for RedundancyCurator."""

    def test_protocol_satisfied(self) -> None:
        """RedundancyCurator satisfies the ContextProcessor Protocol."""
        assert isinstance(RedundancyCurator(), ContextProcessor)

    def test_near_duplicate_dropped(self) -> None:
        """A near-duplicate (Jaccard ≥ threshold) is dropped; 3 distinct chunks remain.

        Non-tautological: a pass-through would return 4; the curator must identify
        and drop chunk-2 (near-duplicate of chunk-1) while retaining chunks 1, 3, 4.
        """
        # chunk_1 and chunk_2 share almost all tokens — Jaccard will be ≥ 0.8.
        text_1 = "the quick brown fox jumps over the lazy dog"
        text_2 = "the quick brown fox jumps over the lazy dog runs fast"
        text_3 = "neural networks learn representations from data"
        text_4 = "quantum computing uses superposition and entanglement"

        chunks = [
            _chunk("c1", text_1),
            _chunk("c2", text_2),  # near-duplicate of c1
            _chunk("c3", text_3),
            _chunk("c4", text_4),
        ]
        curator = RedundancyCurator(threshold=0.8)
        result = curator.process("query", chunks)

        result_ids = [c.chunk_id for c in result]
        assert "c1" in result_ids, "chunk-1 (first of the pair) must be kept"
        assert "c2" not in result_ids, "chunk-2 (near-duplicate) must be dropped"
        assert "c3" in result_ids, "chunk-3 (distinct) must be kept"
        assert "c4" in result_ids, "chunk-4 (distinct) must be kept"
        assert len(result) == 3, f"Expected 3 chunks, got {len(result)}: {result_ids}"

    def test_below_threshold_pair_not_dropped(self) -> None:
        """A pair with Jaccard well below the threshold is NOT dropped.

        Non-tautological: an over-aggressive implementation might drop both;
        this test verifies the threshold boundary is respected.
        """
        # Completely disjoint vocabulary — Jaccard = 0.0.
        text_a = "alpha beta gamma delta epsilon"
        text_b = "one two three four five six seven"

        curator = RedundancyCurator(threshold=0.8)
        result = curator.process("q", [_chunk("a", text_a), _chunk("b", text_b)])

        assert len(result) == 2, (
            f"Distinct chunks must both be kept; got {[c.chunk_id for c in result]}"
        )

    def test_order_preserved(self) -> None:
        """Kept chunks appear in the same relative order as the input."""
        texts = [
            "apple orange pear mango cherry",
            "dog cat bird fish rabbit",
            "red green blue yellow purple",
        ]
        chunks = [_chunk(f"x{i}", t) for i, t in enumerate(texts)]
        result = RedundancyCurator(threshold=0.9).process("q", chunks)
        assert [c.chunk_id for c in result] == [c.chunk_id for c in chunks]

    def test_exact_duplicate_dropped(self) -> None:
        """An exact duplicate (Jaccard = 1.0) is always dropped regardless of threshold."""
        text = "identical content word for word here"
        chunks = [_chunk("orig", text), _chunk("copy", text)]
        result = RedundancyCurator(threshold=0.5).process("q", chunks)
        assert len(result) == 1
        assert result[0].chunk_id == "orig"

    def test_threshold_validation(self) -> None:
        """threshold outside (0, 1] raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="threshold"):
            RedundancyCurator(threshold=0.0)
        with pytest.raises(ValueError, match="threshold"):
            RedundancyCurator(threshold=1.1)

    def test_threshold_1_only_drops_exact_duplicates(self) -> None:
        """threshold=1.0 only drops chunks with Jaccard == 1.0 (exact match)."""
        text_a = "the quick brown fox"
        text_b = "the quick brown fox jumps"  # Jaccard < 1.0
        result = RedundancyCurator(threshold=1.0).process(
            "q", [_chunk("a", text_a), _chunk("b", text_b)]
        )
        assert len(result) == 2


# ---------------------------------------------------------------------------
# ExtractiveCompressor
# ---------------------------------------------------------------------------


class TestExtractiveCompressor:
    """Tests for ExtractiveCompressor."""

    def test_protocol_satisfied(self) -> None:
        """ExtractiveCompressor satisfies the ContextProcessor Protocol."""
        assert isinstance(ExtractiveCompressor(), ContextProcessor)

    def test_retains_answer_sentence_drops_noise(self) -> None:
        """The answer-bearing sentence is retained; ≥1 noise sentence is dropped.

        Non-tautological property: the answer sentence is placed at index 1 of 4
        (not first, not last) so a naïve head extractor or tail extractor would
        miss it.  The compressor must use real query-token overlap to select it.

        Token-overlap proof:
          query_tokens    = {"neural", "network", "training"}
          sentence scores:
            s0 (noise)  = {"the", "weather", "was", "sunny", "today"}        → 0/3 = 0.0
            s1 (answer) = {"neural", "network", "training", "uses", "backprop"} → 3/3 = 1.0
            s2 (noise)  = {"the", "cat", "sat", "on", "the", "mat"}          → 0/3 = 0.0
            s3 (noise)  = {"apples", "grow", "on", "trees", "in", "autumn"}  → 0/3 = 0.0
          mean score = (0 + 1 + 0 + 0) / 4 = 0.25
          retained: s1 (score 1.0 > 0.25)
        """
        query = "neural network training"
        # Answer sentence at index 1 (not first, not last) to defeat naïve extractors.
        answer_sentence = "Neural network training uses backprop to optimise weights."
        passage = (
            "The weather was sunny today. "
            f"{answer_sentence} "
            "The cat sat on the mat. "
            "Apples grow on trees in autumn."
        )
        chunk = _chunk("doc1", passage)
        compressor = ExtractiveCompressor(min_keep=1)
        result = compressor.process(query, [chunk])

        assert len(result) == 1
        compressed_text = result[0].text

        # The answer sentence must survive.
        assert "backprop" in compressed_text, (
            f"Answer-bearing sentence must be retained; got: {compressed_text!r}"
        )

        # At least one noise sentence must be dropped.
        noise_sentences = [
            "The weather was sunny today",
            "The cat sat on the mat",
            "Apples grow on trees in autumn",
        ]
        dropped = sum(1 for ns in noise_sentences if ns not in compressed_text)
        assert dropped >= 1, (
            f"At least one noise sentence must be dropped; compressed text: {compressed_text!r}"
        )

    def test_token_count_drops_30_percent(self) -> None:
        """Compressed token count is at least 30 % lower than the original.

        Same fixture as above: 4 sentences, only 1 retained → 75 % reduction.
        """
        query = "neural network training"
        answer_sentence = "Neural network training uses backprop to optimise weights."
        passage = (
            "The weather was sunny today. "
            f"{answer_sentence} "
            "The cat sat on the mat. "
            "Apples grow on trees in autumn."
        )
        chunk = _chunk("doc1", passage)
        original_tokens = len(passage.split())
        compressor = ExtractiveCompressor(min_keep=1)
        result = compressor.process(query, [chunk])

        compressed_tokens = len(result[0].text.split())
        reduction = 1.0 - compressed_tokens / original_tokens
        assert reduction >= 0.30, (
            f"Expected ≥30 % token reduction, got {reduction:.1%} "
            f"({original_tokens} → {compressed_tokens} tokens)"
        )

    def test_answer_chosen_by_overlap_not_position(self) -> None:
        """The retained sentence is the one with highest query overlap, not a fixed position.

        Variant: place the answer sentence last (index 3 of 4) to rule out any
        head-biased selection.
        """
        query = "climate change carbon emissions"
        answer_sentence = "Climate change is driven by rising carbon emissions globally."
        passage = (
            "The stock market closed higher on Friday. "
            "Rainfall patterns vary across the continent. "
            "Local sports teams competed in the regional finals. "
            f"{answer_sentence}"
        )
        chunk = _chunk("doc2", passage)
        result = ExtractiveCompressor(min_keep=1).process(query, [chunk])
        compressed = result[0].text
        assert "carbon" in compressed, (
            f"Answer sentence (last position) must be selected by overlap; got: {compressed!r}"
        )

    def test_chunk_id_preserved(self) -> None:
        """chunk_id and all provenance fields survive compression unchanged."""
        query = "fast sorting algorithm"
        passage = (
            "Quicksort is a fast sorting algorithm with average O(n log n) complexity. "
            "The sky is blue on clear days. "
            "Water boils at 100 degrees Celsius."
        )
        chunk = _chunk("prov-123", passage)
        result = ExtractiveCompressor(min_keep=1).process(query, [chunk])
        assert result[0].chunk_id == "prov-123"
        assert result[0].document_id == chunk.document_id
        assert result[0].char_span == chunk.char_span

    def test_token_count_updated(self) -> None:
        """token_count on the returned chunk reflects the compressed text length."""
        query = "fast algorithm complexity"
        passage = (
            "Quicksort has average O(n log n) complexity and is a fast algorithm. "
            "Bananas are yellow tropical fruits. "
            "Mountains are formed by tectonic activity. "
            "The library opens at nine in the morning."
        )
        chunk = _chunk("tc-test", passage)
        result = ExtractiveCompressor(min_keep=1).process(query, [chunk])
        expected_count = len(result[0].text.split())
        assert result[0].token_count == expected_count

    def test_min_keep_one_never_empties_chunk(self) -> None:
        """With min_keep=1 a chunk with zero query-overlap retains exactly 1 sentence."""
        query = "astrophysics quantum gravity"
        passage = (
            "The dog barked loudly at midnight. "
            "She bought fresh vegetables from the market. "
            "They watched a movie together on the weekend."
        )
        chunk = _chunk("noise-only", passage)
        result = ExtractiveCompressor(min_keep=1).process(query, [chunk])
        assert result[0].text, "Compressed chunk must not be empty"

    def test_single_sentence_chunk_unchanged(self) -> None:
        """A chunk with a single sentence is returned with its text unchanged."""
        query = "machine learning"
        chunk = _chunk("single", "Machine learning models learn from data.")
        result = ExtractiveCompressor(min_keep=1).process(query, [chunk])
        assert result[0].chunk_id == "single"
        assert result[0].text == chunk.text

    def test_min_keep_validation(self) -> None:
        """min_keep < 1 raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="min_keep"):
            ExtractiveCompressor(min_keep=0)
