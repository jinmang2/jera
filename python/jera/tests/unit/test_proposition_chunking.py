"""Proposition chunker — non-tautological precision test (M12).

Design note (why this test is non-tautological)
------------------------------------------------
A *tautological* test would merely verify that the chunker runs and returns
something.  This test verifies the *semantic property* that gives proposition
chunking its value: **fact isolation**.

With passage-level chunking (HeadingAwareChunker, max_tokens large enough to
keep both sentences in one chunk), a query about fact1 retrieves a chunk that
also contains fact2 — low precision.  With PropositionChunker, each sentence is
its own chunk: retrieving fact1 does NOT drag in fact2 — higher precision.

The test exercises this directly:
(a) PropositionChunker yields ≥ 2 chunks for a section with 2 sentences (one
    per fact), while HeadingAwareChunker yields 1 chunk (both facts merged).
(b) The chunk carrying fact1 does NOT contain fact2's text.
(c) The chunk carrying fact2 does NOT contain fact1's text.
(d) Provenance is fully populated: char_spans are within the section text,
    section_path is carried, element_ids is non-empty, embedding_text carries
    the heading prefix so each unit is self-contained for dense retrieval.
(e) PropositionChunker satisfies the Chunker Protocol (isinstance check).
(f) chunk_ids are deterministic across two independent runs.
"""

from __future__ import annotations

from jera.adapters.chunking.heading_aware import HeadingAwareChunker
from jera.adapters.chunking.proposition import PropositionChunker
from jera.domain.chunk import Chunk
from jera.domain.document import (
    DocumentElement,
    ElementType,
    MediaType,
    PageSpan,
    ParsedDocument,
    Provenance,
)
from jera.ports.chunker import Chunker

# ---------------------------------------------------------------------------
# Two facts about DIFFERENT topics so mixing them is a genuine precision error.
# ---------------------------------------------------------------------------
FACT1 = "The Eiffel Tower is 330 metres tall."
FACT2 = "The Seine river flows through Paris."
HEADING = "Paris Landmarks"

# Section text as assembled by group_sections.
# The Title element (section_key → (HEADING,)) and the two NarrativeText elements
# (section_path = (HEADING,)) share the same section key, so all three land in one
# section whose text is heading + "\n\n" + fact1 + "\n\n" + fact2.
SECTION_TEXT = f"{HEADING}\n\n{FACT1}\n\n{FACT2}"


def _make_document() -> ParsedDocument:
    """Build a minimal ParsedDocument inline — no parser involved."""
    provenance = Provenance(
        source_id="test-src",
        parser_name="inline",
        parser_version="0.0.0",
        media_type=MediaType.PLAIN,
    )
    page = PageSpan.single(1)
    title_el = DocumentElement(
        element_id="el-0",
        type=ElementType.TITLE,
        text=HEADING,
        page_span=page,
        order=0,
        section_path=(),  # top-level heading
    )
    # Both NarrativeText elements share the same section_path (heading breadcrumb).
    section_path: tuple[str, ...] = (HEADING,)
    fact1_el = DocumentElement(
        element_id="el-1",
        type=ElementType.NARRATIVE_TEXT,
        text=FACT1,
        page_span=page,
        order=1,
        section_path=section_path,
    )
    fact2_el = DocumentElement(
        element_id="el-2",
        type=ElementType.NARRATIVE_TEXT,
        text=FACT2,
        page_span=page,
        order=2,
        section_path=section_path,
    )
    return ParsedDocument(
        document_id="doc-proposition-test",
        source_id="test-src",
        title=HEADING,
        elements=[title_el, fact1_el, fact2_el],
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_proposition_chunker_satisfies_chunker_protocol() -> None:
    """PropositionChunker must satisfy the Chunker Protocol (runtime check)."""
    assert isinstance(PropositionChunker(), Chunker)


def test_proposition_yields_one_chunk_per_sentence() -> None:
    """(a) Proposition chunker emits ≥ 2 chunks for a 2-sentence section."""
    doc = _make_document()
    chunks = PropositionChunker().chunk(doc)
    # Filter to only the content section (skip title-only section if any)
    content_chunks = [c for c in chunks if HEADING in c.section_path]
    assert len(content_chunks) >= 2, (
        f"Expected ≥ 2 proposition chunks for 2 sentences, got {len(content_chunks)}: "
        f"{[c.text for c in content_chunks]}"
    )


def test_heading_aware_merges_both_facts_into_one_chunk() -> None:
    """Baseline: HeadingAwareChunker with large max_tokens merges both facts."""
    doc = _make_document()
    # Use a large max_tokens so both sentences land in a single window.
    chunks = HeadingAwareChunker(max_tokens=500, overlap_tokens=0).chunk(doc)
    content_chunks = [c for c in chunks if HEADING in c.section_path]
    assert len(content_chunks) == 1, (
        "HeadingAwareChunker (large window) should merge both sentences into one chunk "
        f"but got {len(content_chunks)} chunks"
    )
    merged_text = content_chunks[0].text
    assert FACT1 in merged_text and FACT2 in merged_text, (
        "The merged chunk must contain both facts — this is exactly the low-precision "
        "behaviour that proposition chunking avoids."
    )


def test_fact_isolation_precision_property() -> None:
    """(b+c) Each proposition chunk contains ONLY its own fact — no cross-contamination.

    This is the core non-tautological assertion: retrieving the fact1 chunk must
    NOT surface fact2 text, and vice versa.  With passage-level chunking both
    facts would appear together.
    """
    doc = _make_document()
    chunks = PropositionChunker().chunk(doc)
    content_chunks = [c for c in chunks if HEADING in c.section_path]

    fact1_chunks = [c for c in content_chunks if FACT1 in c.text]
    fact2_chunks = [c for c in content_chunks if FACT2 in c.text]

    assert fact1_chunks, f"No chunk found containing fact1 text '{FACT1}'"
    assert fact2_chunks, f"No chunk found containing fact2 text '{FACT2}'"

    # Fact1 chunk must not drag in fact2.
    for c in fact1_chunks:
        assert FACT2 not in c.text, (
            f"Fact1 chunk contains fact2 text — precision failure.\nchunk.text = {c.text!r}"
        )

    # Fact2 chunk must not drag in fact1.
    for c in fact2_chunks:
        assert FACT1 not in c.text, (
            f"Fact2 chunk contains fact1 text — precision failure.\nchunk.text = {c.text!r}"
        )


def test_provenance_is_valid() -> None:
    """(d) Every proposition chunk carries correct provenance.

    - char_span[start] <= char_span[end] and the span maps back to chunk.text
      within SECTION_TEXT.
    - section_path matches the heading.
    - element_ids is non-empty.
    - embedding_text includes the heading breadcrumb (self-containment).
    - token_count > 0.
    - chunk_strategy == "proposition".

    Note: the Title element ("Paris Landmarks") also forms a section whose
    section_path is (HEADING,) — group_sections assigns titles to a section
    keyed by their breadcrumb + own text.  We filter to fact-bearing chunks
    only (those whose text is FACT1 or FACT2) so the char_span assertion
    references the correct section text (SECTION_TEXT).
    """
    doc = _make_document()
    chunks = PropositionChunker().chunk(doc)
    # Only the two fact sentences — the title chunk is a different section.
    fact_chunks = [c for c in chunks if c.text in (FACT1, FACT2)]
    assert fact_chunks, "Expected proposition chunks for the two fact sentences"

    for c in fact_chunks:
        # chunk_strategy / chunk_version
        assert c.chunk_strategy == "proposition"
        assert c.chunk_version == "1.0.0"

        # section_path — NarrativeText elements inherit the heading breadcrumb
        assert c.section_path == (HEADING,), f"section_path mismatch: {c.section_path}"

        # char_span validity
        start, end = c.char_span
        assert start <= end, f"char_span inverted: {c.char_span}"

        # char_span maps back to chunk.text within the section's concatenated text
        extracted = SECTION_TEXT[start:end]
        assert extracted == c.text, (
            f"char_span [{start}:{end}] extracts {extracted!r} but chunk.text is {c.text!r}"
        )

        # element_ids
        assert c.element_ids, "element_ids must not be empty"

        # token_count
        assert c.token_count > 0

        # embedding_text includes heading prefix (self-containment for retrieval)
        assert HEADING in c.embedding_text, (
            f"embedding_text does not carry the heading prefix.\n"
            f"embedding_text = {c.embedding_text!r}"
        )

        # text itself is just the sentence — no heading contamination in citation form
        assert c.text in (FACT1, FACT2)


def test_chunk_ids_are_deterministic() -> None:
    """(f) Two independent runs over the same document must yield identical chunk_ids."""
    doc = _make_document()
    ids_a = [c.chunk_id for c in PropositionChunker().chunk(doc)]
    ids_b = [c.chunk_id for c in PropositionChunker().chunk(doc)]
    assert ids_a == ids_b, "chunk_ids are not deterministic across runs"
    # Also verify all ids are unique within a single run
    assert len(ids_a) == len(set(ids_a)), "Duplicate chunk_ids within a single run"


def test_proposition_chunks_satisfy_chunk_schema() -> None:
    """All emitted chunks must have the same field set as the Chunk model."""
    doc = _make_document()
    chunks = PropositionChunker().chunk(doc)
    assert chunks
    expected_fields = set(Chunk.model_fields)
    for c in chunks:
        assert set(c.model_dump()) == expected_fields, (
            f"Chunk field set mismatch: {set(c.model_dump())} != {expected_fields}"
        )
