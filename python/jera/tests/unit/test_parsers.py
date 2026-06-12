"""Parser gate (Gate 2): typed elements + provenance, not flat text. Markdown + PDF + table."""

from __future__ import annotations

from jera.adapters.parsing import MarkdownParser, PyMuPDFParser
from jera.domain.document import ElementType, MediaType, SourceRef


def test_markdown_produces_typed_elements_with_section_paths(sample_markdown: str) -> None:
    parser = MarkdownParser()
    src = SourceRef(
        source_id="md1", media_type=MediaType.MARKDOWN, content=sample_markdown.encode()
    )
    doc = parser.parse(src)

    types = {e.type for e in doc.elements}
    assert ElementType.TITLE in types
    assert ElementType.NARRATIVE_TEXT in types
    assert doc.title == "Jera Overview"
    # provenance present and correct
    assert doc.provenance.parser_name == "markdown"
    assert doc.provenance.media_type is MediaType.MARKDOWN
    # section-path breadcrumb is populated for nested content
    retrieval_content = [
        e for e in doc.elements if e.section_path and e.section_path[-1] == "Retrieval"
    ]
    assert retrieval_content, "expected content under the 'Retrieval' heading"


def test_markdown_table_is_typed_as_table(sample_table_markdown: str) -> None:
    parser = MarkdownParser()
    src = SourceRef(
        source_id="tbl", media_type=MediaType.MARKDOWN, content=sample_table_markdown.encode()
    )
    doc = parser.parse(src)
    assert any(e.type is ElementType.TABLE for e in doc.elements)


def test_pdf_parser_extracts_text_with_page_spans(text_pdf_bytes: bytes) -> None:
    parser = PyMuPDFParser()
    src = SourceRef(source_id="pdf1", media_type=MediaType.PDF, content=text_pdf_bytes)
    doc = parser.parse(src)
    assert doc.elements, "expected at least one element from the text PDF"
    assert all(e.page_span.start_page >= 1 for e in doc.elements)
    joined = " ".join(e.text for e in doc.elements)
    assert "ZX9000" in joined
    assert doc.provenance.parser_name == "pymupdf"


def test_ids_are_deterministic(sample_markdown: str) -> None:
    parser = MarkdownParser()
    src = SourceRef(
        source_id="md1", media_type=MediaType.MARKDOWN, content=sample_markdown.encode()
    )
    a = parser.parse(src)
    b = parser.parse(src)
    assert [e.element_id for e in a.elements] == [e.element_id for e in b.elements]
    assert a.document_id == b.document_id
