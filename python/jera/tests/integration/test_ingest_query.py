"""Integration: ingest→store→retrieve round-trip, citation correctness, empty-result path."""

from __future__ import annotations

from jera.config.registry import RagSystem
from jera.domain.document import MediaType, SourceRef
from jera.domain.jobs import JobStatus
from jera.domain.retrieval import Query, RetrievalMode
from jera.evaluation_contracts import citation_faithfulness


def _ingest(system: RagSystem, md: str, source_id: str = "doc1") -> None:
    system.ingest.ingest(
        SourceRef(source_id=source_id, media_type=MediaType.MARKDOWN, content=md.encode())
    )


def test_ingest_persists_document_chunks_and_job(system: RagSystem, sample_markdown: str) -> None:
    job = system.ingest.ingest(
        SourceRef(source_id="doc1", media_type=MediaType.MARKDOWN, content=sample_markdown.encode())
    )
    assert job.status is JobStatus.SUCCEEDED
    assert job.chunk_count >= 1
    assert system.metadata_store.get_job(job.job_id) is not None


def test_provenance_fields_have_correct_values(system: RagSystem, sample_markdown: str) -> None:
    job = system.ingest.ingest(
        SourceRef(source_id="doc1", media_type=MediaType.MARKDOWN, content=sample_markdown.encode())
    )
    # fetch a chunk back and assert provenance correctness (not just stability)
    result = system.query.retrieve(
        Query(text="hybrid retrieval", mode=RetrievalMode.HYBRID, top_k=5)
    )
    chunk = next(c.chunk for c in result.results if c.chunk is not None)
    assert chunk.chunk_strategy == "heading_aware"
    assert chunk.chunk_version == "1.0.0"
    assert chunk.document_id == job.document_id


def test_citations_resolve_to_retrieved_chunks(system: RagSystem, sample_markdown: str) -> None:
    _ingest(system, sample_markdown)
    answer = system.query.answer("What fusion does hybrid retrieval use?", top_k=3)
    assert answer.citations
    cited = [c.chunk_id for c in answer.citations]
    # every citation resolves to a real stored chunk
    assert all(system.metadata_store.get_chunk(cid) is not None for cid in cited)
    # and is fully grounded in retrieval
    result = system.query.retrieve(Query(text="What fusion does hybrid retrieval use?", top_k=3))
    retrieved = [c.chunk_id for c in result.results]
    assert citation_faithfulness(cited, retrieved) == 1.0


def test_empty_result_returns_empty_answer_not_error(system: RagSystem) -> None:
    # query an empty index: defined behavior is an empty answer, no citations, no exception.
    answer = system.query.answer("anything at all", top_k=3)
    assert answer.citations == []
    assert answer.is_empty


def test_sparse_finds_exact_identifier(system: RagSystem, sample_markdown: str) -> None:
    _ingest(system, sample_markdown)
    result = system.query.retrieve(Query(text="ZX9000", mode=RetrievalMode.SPARSE, top_k=3))
    top_chunk = result.results[0].chunk
    assert top_chunk is not None and "ZX9000" in top_chunk.text
