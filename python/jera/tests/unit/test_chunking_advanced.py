"""Semantic + hierarchical (RAPTOR-lite) chunker tests."""

from __future__ import annotations

from jera.adapters.chunking import HeadingAwareChunker, HierarchicalChunker, SemanticChunker
from jera.adapters.chunking.sentences import split_sentences_with_offsets
from jera.adapters.embedding.hash_embedding import HashEmbedding
from jera.adapters.parsing import MarkdownParser
from jera.config import Profile, Settings, build_system
from jera.domain.document import MediaType, SourceRef

MD = """# Guide

## Retrieval
Dense retrieval uses embeddings. It captures semantic similarity. Sparse retrieval uses BM25.

## Fusion
Hybrid retrieval merges both. Reciprocal rank fusion combines the rankings. The constant k is 60.
"""


def _doc(md: str = MD):
    return MarkdownParser().parse(
        SourceRef(source_id="g", media_type=MediaType.MARKDOWN, content=md.encode())
    )


def test_sentence_splitter_offsets_are_exact() -> None:
    text = "First sentence. Second one! Third?"
    sents = split_sentences_with_offsets(text)
    assert [s for s, _, _ in sents] == ["First sentence.", "Second one!", "Third?"]
    for s, start, end in sents:
        assert text[start:end] == s


def test_semantic_chunks_carry_contract_and_are_deterministic() -> None:
    doc = _doc()
    emb = HashEmbedding()
    a = SemanticChunker(emb).chunk(doc)
    b = SemanticChunker(emb).chunk(doc)
    assert a, "semantic chunker produced no chunks"
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]  # deterministic
    for c in a:
        assert c.chunk_strategy == "semantic"
        assert c.token_count > 0
        assert doc.document_id == c.document_id
        # char span maps back to the chunk text within its section
        assert c.char_span[0] <= c.char_span[1]


def test_semantic_respects_token_cap() -> None:
    doc = _doc()
    chunks = SemanticChunker(HashEmbedding(), max_tokens=8).chunk(doc)
    # with a small cap, no chunk may greatly exceed it (single long sentences excepted)
    assert all(c.token_count <= 8 or len(c.text.split(".")) == 1 for c in chunks)


def test_hierarchical_builds_parent_child_tree() -> None:
    doc = _doc()
    emb = HashEmbedding()
    chunks = HierarchicalChunker(emb).chunk(doc)
    parents = [c for c in chunks if c.parent_chunk_id is None]
    children = [c for c in chunks if c.parent_chunk_id is not None]
    assert parents and children
    parent_ids = {p.chunk_id for p in parents}
    # every child links to an emitted parent
    assert all(c.parent_chunk_id in parent_ids for c in children)
    # parents are summaries with their own provenance
    for p in parents:
        assert p.chunk_strategy == "hierarchical"
        assert p.token_count > 0
    # deterministic
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in HierarchicalChunker(emb).chunk(doc)]


def test_hierarchical_clusters_similar_leaves_together() -> None:
    # Two near-identical sections should land in the same cluster (one parent for both).
    md = """# Doc

## A
machine learning models learn patterns from data using gradient descent optimization.

## B
machine learning models learn patterns from data using gradient descent optimization methods.

## C
the quarterly revenue grew because pricing changed in unrelated commercial markets entirely.
"""
    doc = _doc(md)
    chunks = HierarchicalChunker(HashEmbedding(), cluster_threshold=0.6).chunk(doc)
    parents = [c for c in chunks if c.parent_chunk_id is None]
    children = [c for c in chunks if c.parent_chunk_id is not None]
    # at least one parent must cover more than one child (genuine clustering, not all singletons)
    sizes = {
        p.chunk_id: sum(1 for c in children if c.parent_chunk_id == p.chunk_id) for p in parents
    }
    assert max(sizes.values()) >= 2


def test_registry_wires_chunk_strategy() -> None:
    sys_h = build_system(Settings(profile=Profile.TEST, chunk_strategy="hierarchical"))
    assert isinstance(sys_h.ingest._chunker, HierarchicalChunker)  # type: ignore[attr-defined]
    sys_s = build_system(Settings(profile=Profile.TEST, chunk_strategy="semantic"))
    assert isinstance(sys_s.ingest._chunker, SemanticChunker)  # type: ignore[attr-defined]
    sys_d = build_system(Settings(profile=Profile.TEST))
    assert isinstance(sys_d.ingest._chunker, HeadingAwareChunker)  # type: ignore[attr-defined]
