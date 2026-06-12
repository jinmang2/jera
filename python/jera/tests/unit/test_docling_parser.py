"""Docling parser gate — runs only when the `docling` extra is installed (skipped in CI).

Marked ``requires_extra``: default CI uses `uv sync` (no extras) so docling is absent and this
is skipped; `uv sync --extra docling` makes it run and verifies the typed-element mapping.
"""

from __future__ import annotations

import importlib.util

import pytest

pytestmark = pytest.mark.requires_extra

_HAS_DOCLING = importlib.util.find_spec("docling") is not None


def _text_pdf() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Jera Technical Note", fontsize=18)
    page.insert_text((72, 110), "The ranking module identifier is ZX9000.", fontsize=11)
    data = doc.tobytes()
    doc.close()
    return bytes(data)


@pytest.mark.skipif(not _HAS_DOCLING, reason="docling extra not installed")
def test_docling_parses_pdf_into_typed_elements_with_provenance() -> None:
    from jera.adapters.parsing.docling_parser import DoclingParser
    from jera.domain.document import ElementType, MediaType, SourceRef

    parser = DoclingParser()
    doc = parser.parse(SourceRef(source_id="t", media_type=MediaType.PDF, content=_text_pdf()))

    assert doc.provenance.parser_name == "docling"
    assert doc.elements, "expected typed elements from docling"
    assert any(e.type is ElementType.TITLE for e in doc.elements)
    joined = " ".join(e.text for e in doc.elements)
    assert "ZX9000" in joined
    # content under the title carries the section-path breadcrumb
    assert any(e.section_path for e in doc.elements if e.type is ElementType.NARRATIVE_TEXT)


@pytest.mark.skipif(not _HAS_DOCLING, reason="docling extra not installed")
def test_registry_uses_docling_when_enabled() -> None:
    from jera.config import Profile, Settings, build_system
    from jera.domain.document import MediaType, SourceRef

    system = build_system(Settings(profile=Profile.TEST, use_docling=True))
    job = system.ingest.ingest(
        SourceRef(source_id="d", media_type=MediaType.PDF, content=_text_pdf())
    )
    assert job.chunk_count >= 1
