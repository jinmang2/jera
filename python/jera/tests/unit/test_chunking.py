"""Chunking gate (Gate 3): heading-aware vs semantic shape; id/section/page stability."""

from __future__ import annotations

from jera.adapters.chunking.heading_aware import HeadingAwareChunker
from jera.adapters.chunking.semantic import SemanticChunker
from jera.adapters.embedding.hash_embedding import HashEmbedding
from jera.adapters.parsing import MarkdownParser
from jera.domain.document import MediaType, SourceRef


def _parse(md: str):
    return MarkdownParser().parse(
        SourceRef(source_id="md1", media_type=MediaType.MARKDOWN, content=md.encode())
    )


def test_heading_aware_chunks_carry_full_provenance(sample_markdown: str) -> None:
    doc = _parse(sample_markdown)
    chunks = HeadingAwareChunker().chunk(doc)
    assert chunks
    for c in chunks:
        assert c.chunk_strategy == "heading_aware"
        assert c.chunk_version == "1.0.0"
        assert c.document_id == doc.document_id
        assert c.token_count > 0
        assert c.char_span[0] <= c.char_span[1]
        assert c.section_path  # every chunk knows its section


def test_chunk_ids_and_metadata_are_stable_across_runs(sample_markdown: str) -> None:
    doc = _parse(sample_markdown)
    a = HeadingAwareChunker().chunk(doc)
    b = HeadingAwareChunker().chunk(doc)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.section_path for c in a] == [c.section_path for c in b]
    assert [c.page_span.model_dump() for c in a] == [c.page_span.model_dump() for c in b]


def test_semantic_and_heading_aware_share_chunk_shape(sample_markdown: str) -> None:
    doc = _parse(sample_markdown)
    heading = HeadingAwareChunker().chunk(doc)
    semantic = SemanticChunker(HashEmbedding()).chunk(doc)
    # Both strategies emit Chunks with the same provenance contract (shape comparison).
    from jera.domain.chunk import Chunk

    assert set(Chunk.model_fields) == set(heading[0].model_dump()) == set(semantic[0].model_dump())
    assert all(c.chunk_strategy == "semantic" for c in semantic)
    # Both produce non-empty chunk sets over the same structured document; the semantic
    # splitter may emit finer or coarser boundaries (count is not constrained either way).
    assert heading and semantic
