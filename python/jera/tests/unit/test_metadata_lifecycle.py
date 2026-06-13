"""Document-lifecycle methods on SqlMetadataStore: list/get/delete + helper queries."""

from __future__ import annotations

from jera.adapters.metadata_store.sqlite_store import make_sqlite_store
from jera.domain.chunk import Chunk
from jera.domain.document import MediaType, PageSpan, ParsedDocument, Provenance

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _parsed_doc(document_id: str, source_id: str, title: str | None = None) -> ParsedDocument:
    return ParsedDocument(
        document_id=document_id,
        source_id=source_id,
        title=title,
        elements=[],
        provenance=Provenance(
            source_id=source_id,
            parser_name="test-parser",
            parser_version="0.0.1",
            media_type=MediaType.PLAIN,
        ),
    )


def _chunk(chunk_id: str, document_id: str, source_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_id=source_id,
        text="sample text",
        page_span=PageSpan.single(1),
        section_path=(),
        element_ids=(),
        char_span=(0, 11),
        token_count=2,
        chunk_strategy="heading_aware",
        chunk_version="1.0.0",
    )


def _store_with_doc_and_chunks(
    document_id: str = "doc1",
    source_id: str = "src1",
    chunk_ids: list[str] | None = None,
    title: str | None = "My Doc",
):
    """Return a fresh in-memory store with one document and its chunks."""
    if chunk_ids is None:
        chunk_ids = ["c1", "c2", "c3"]
    store = make_sqlite_store(":memory:")
    store.save_document(_parsed_doc(document_id, source_id, title=title))
    store.save_chunks([_chunk(cid, document_id, source_id) for cid in chunk_ids])
    return store


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------


def test_list_documents_empty() -> None:
    store = make_sqlite_store(":memory:")
    assert store.list_documents() == []


def test_list_documents_returns_correct_chunk_count() -> None:
    store = _store_with_doc_and_chunks(chunk_ids=["c1", "c2", "c3"])
    docs = store.list_documents()
    assert len(docs) == 1
    doc = docs[0]
    assert doc.document_id == "doc1"
    assert doc.source_id == "src1"
    assert doc.title == "My Doc"
    assert doc.chunk_count == 3


def test_list_documents_ordered_by_document_id() -> None:
    store = make_sqlite_store(":memory:")
    store.save_document(_parsed_doc("doc_b", "src_b"))
    store.save_document(_parsed_doc("doc_a", "src_a"))
    docs = store.list_documents()
    assert [d.document_id for d in docs] == ["doc_a", "doc_b"]


def test_list_documents_no_chunks_shows_zero() -> None:
    store = make_sqlite_store(":memory:")
    store.save_document(_parsed_doc("doc1", "src1"))
    docs = store.list_documents()
    assert docs[0].chunk_count == 0


def test_list_documents_multiple_docs_independent_counts() -> None:
    store = make_sqlite_store(":memory:")
    store.save_document(_parsed_doc("doc1", "src1"))
    store.save_document(_parsed_doc("doc2", "src2"))
    store.save_chunks([_chunk("c1", "doc1", "src1"), _chunk("c2", "doc1", "src1")])
    store.save_chunks([_chunk("c3", "doc2", "src2")])
    docs = {d.document_id: d for d in store.list_documents()}
    assert docs["doc1"].chunk_count == 2
    assert docs["doc2"].chunk_count == 1


# ---------------------------------------------------------------------------
# get_document_info
# ---------------------------------------------------------------------------


def test_get_document_info_returns_correct_data() -> None:
    store = _store_with_doc_and_chunks(chunk_ids=["c1", "c2"])
    info = store.get_document_info("doc1")
    assert info is not None
    assert info.document_id == "doc1"
    assert info.source_id == "src1"
    assert info.title == "My Doc"
    assert info.chunk_count == 2


def test_get_document_info_unknown_returns_none() -> None:
    store = make_sqlite_store(":memory:")
    assert store.get_document_info("nonexistent") is None


def test_get_document_info_no_chunks_shows_zero() -> None:
    store = make_sqlite_store(":memory:")
    store.save_document(_parsed_doc("doc1", "src1"))
    info = store.get_document_info("doc1")
    assert info is not None
    assert info.chunk_count == 0


def test_get_document_info_null_title() -> None:
    store = make_sqlite_store(":memory:")
    store.save_document(_parsed_doc("doc1", "src1", title=None))
    info = store.get_document_info("doc1")
    assert info is not None
    assert info.title is None


# ---------------------------------------------------------------------------
# delete_document
# ---------------------------------------------------------------------------


def test_delete_document_returns_chunk_ids() -> None:
    store = _store_with_doc_and_chunks(chunk_ids=["c1", "c2", "c3"])
    deleted = store.delete_document("doc1")
    assert sorted(deleted) == ["c1", "c2", "c3"]


def test_delete_document_removes_doc_and_chunks() -> None:
    store = _store_with_doc_and_chunks(chunk_ids=["c1", "c2"])
    store.delete_document("doc1")
    assert store.list_documents() == []
    assert store.get_chunk("c1") is None
    assert store.get_chunk("c2") is None


def test_delete_document_idempotent_unknown_id() -> None:
    store = make_sqlite_store(":memory:")
    result = store.delete_document("does_not_exist")
    assert result == []


def test_delete_document_only_removes_target() -> None:
    store = make_sqlite_store(":memory:")
    store.save_document(_parsed_doc("doc1", "src1"))
    store.save_document(_parsed_doc("doc2", "src2"))
    store.save_chunks([_chunk("c1", "doc1", "src1")])
    store.save_chunks([_chunk("c2", "doc2", "src2")])

    store.delete_document("doc1")

    remaining = store.list_documents()
    assert len(remaining) == 1
    assert remaining[0].document_id == "doc2"
    assert store.get_chunk("c2") is not None


def test_delete_document_second_call_idempotent() -> None:
    store = _store_with_doc_and_chunks(chunk_ids=["c1"])
    store.delete_document("doc1")
    result = store.delete_document("doc1")
    assert result == []


# ---------------------------------------------------------------------------
# chunk_ids_for_document
# ---------------------------------------------------------------------------


def test_chunk_ids_for_document_returns_sorted() -> None:
    store = _store_with_doc_and_chunks(chunk_ids=["c3", "c1", "c2"])
    ids = store.chunk_ids_for_document("doc1")
    assert ids == ["c1", "c2", "c3"]


def test_chunk_ids_for_document_unknown_returns_empty() -> None:
    store = make_sqlite_store(":memory:")
    assert store.chunk_ids_for_document("no_such_doc") == []


def test_chunk_ids_for_document_no_chunks_returns_empty() -> None:
    store = make_sqlite_store(":memory:")
    store.save_document(_parsed_doc("doc1", "src1"))
    assert store.chunk_ids_for_document("doc1") == []


# ---------------------------------------------------------------------------
# document_id_for_source
# ---------------------------------------------------------------------------


def test_document_id_for_source_returns_correct_id() -> None:
    store = _store_with_doc_and_chunks(document_id="doc1", source_id="my_source")
    assert store.document_id_for_source("my_source") == "doc1"


def test_document_id_for_source_unknown_returns_none() -> None:
    store = make_sqlite_store(":memory:")
    assert store.document_id_for_source("unknown_source") is None


def test_document_id_for_source_after_delete_returns_none() -> None:
    store = _store_with_doc_and_chunks(document_id="doc1", source_id="src1")
    store.delete_document("doc1")
    assert store.document_id_for_source("src1") is None
