"""SDK-boundary tests for OpenDataLoaderParser.

Injects a fake ``opendataloader_pdf`` module into sys.modules so no Java 11+
runtime or pip install is required.  The fake ``convert()`` writes a canned
JSON file to ``output_dir``, exactly mimicking what the real CLI would produce.

A ``@pytest.mark.requires_extra`` guard wraps the live-import test that needs
the real package installed.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from jera.domain.document import (
    METADATA_OCR_ENGINE,
    METADATA_ROUTE,
    ElementType,
    MediaType,
    ParsedDocument,
    SourceRef,
)

# ---------------------------------------------------------------------------
# Canned JSON output that the fake convert() writes to disk.
# Covers: heading (→ TITLE), paragraph (→ NARRATIVE_TEXT), table (→ TABLE),
#         image (→ FIGURE), formula (→ FORMULA).
# ---------------------------------------------------------------------------

_CANNED_ELEMENTS: list[dict[str, Any]] = [
    {
        "type": "heading",
        "id": 1,
        "page_number": 1,
        "bounding_box": [72.0, 700.0, 540.0, 730.0],
        "heading_level": 1,
        "font": "Helvetica-Bold",
        "font_size": 18.0,
        "text_color": "#000000",
        "content": "Executive Summary",
    },
    {
        "type": "paragraph",
        "id": 2,
        "page_number": 1,
        "bounding_box": [72.0, 660.0, 540.0, 695.0],
        "heading_level": None,
        "font": "Helvetica",
        "font_size": 11.0,
        "text_color": "#000000",
        "content": "This document describes the ZX9000 ranking module.",
    },
    {
        "type": "table",
        "id": 3,
        "page_number": 2,
        "bounding_box": [72.0, 400.0, 540.0, 600.0],
        "heading_level": None,
        "content": "| Strategy | Score |\n| dense | 0.91 |",
    },
    {
        "type": "image",
        "id": 4,
        "page_number": 2,
        "bounding_box": [72.0, 200.0, 300.0, 380.0],
        "heading_level": None,
        "content": "Architecture diagram",
    },
    {
        "type": "formula",
        "id": 5,
        "page_number": 3,
        "bounding_box": [100.0, 100.0, 400.0, 130.0],
        "heading_level": None,
        "content": "E = mc^2",
    },
    {
        "type": "paragraph",
        "id": 6,
        "page_number": 3,
        "bounding_box": [72.0, 50.0, 540.0, 90.0],
        "heading_level": None,
        "content": "",  # empty — must be skipped
    },
]


# ---------------------------------------------------------------------------
# Fixture: install fake opendataloader_pdf into sys.modules
# ---------------------------------------------------------------------------


def _install_fake_odl(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Register a fake opendataloader_pdf; return a list that records convert() calls."""
    calls: list[dict[str, Any]] = []

    def _fake_convert(
        input_path: list[str] | str,
        output_dir: str,
        format: str = "json",
        **kwargs: Any,
    ) -> None:
        calls.append({"input_path": input_path, "output_dir": output_dir, "format": format})
        # Derive output stem from the first input path (mirrors real behaviour).
        first = input_path[0] if isinstance(input_path, list) else input_path
        stem = Path(first).stem
        out_file = Path(output_dir) / f"{stem}.json"
        out_file.write_text(json.dumps(_CANNED_ELEMENTS), encoding="utf-8")

    odl_mod = types.ModuleType("opendataloader_pdf")
    odl_mod.convert = _fake_convert  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opendataloader_pdf", odl_mod)
    return calls


def _make_source(source_id: str = "test_doc") -> SourceRef:
    return SourceRef(source_id=source_id, media_type=MediaType.PDF, content=b"%PDF-fake")


def _make_parser(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install fake module, (re-)import the parser, return a fresh instance."""
    _install_fake_odl(monkeypatch)
    # Force reimport so the lazy-import inside __init__ resolves to our fake.
    mod_name = "jera.adapters.parsing.opendataloader_parser"
    if mod_name in sys.modules:
        monkeypatch.delitem(sys.modules, mod_name)
    mod = importlib.import_module(mod_name)
    return mod.OpenDataLoaderParser()


# ===========================================================================
# 1.  supports()
# ===========================================================================


class TestSupports:
    def test_supports_pdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = _make_parser(monkeypatch)
        assert parser.supports(_make_source())

    def test_does_not_support_markdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = _make_parser(monkeypatch)
        src = SourceRef(source_id="s", media_type=MediaType.MARKDOWN, content=b"# hi")
        assert not parser.supports(src)

    def test_does_not_support_html(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parser = _make_parser(monkeypatch)
        src = SourceRef(source_id="s", media_type=MediaType.HTML, content=b"<html/>")
        assert not parser.supports(src)


# ===========================================================================
# 2.  convert() invocation — correct arguments forwarded
# ===========================================================================


class TestConvertInvocation:
    def test_convert_called_with_json_format(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _install_fake_odl(monkeypatch)
        mod_name = "jera.adapters.parsing.opendataloader_parser"
        if mod_name in sys.modules:
            monkeypatch.delitem(sys.modules, mod_name)
        parser = importlib.import_module(mod_name).OpenDataLoaderParser()

        parser.parse(_make_source("my_doc"))

        assert len(calls) == 1
        assert calls[0]["format"] == "json"

    def test_convert_input_path_is_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _install_fake_odl(monkeypatch)
        mod_name = "jera.adapters.parsing.opendataloader_parser"
        if mod_name in sys.modules:
            monkeypatch.delitem(sys.modules, mod_name)
        parser = importlib.import_module(mod_name).OpenDataLoaderParser()

        parser.parse(_make_source("my_doc"))

        assert isinstance(calls[0]["input_path"], list)
        assert calls[0]["input_path"][0].endswith("my_doc.pdf")

    def test_convert_output_dir_matches_input_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = _install_fake_odl(monkeypatch)
        mod_name = "jera.adapters.parsing.opendataloader_parser"
        if mod_name in sys.modules:
            monkeypatch.delitem(sys.modules, mod_name)
        parser = importlib.import_module(mod_name).OpenDataLoaderParser()

        parser.parse(_make_source("my_doc"))

        input_dir = str(Path(calls[0]["input_path"][0]).parent)
        assert calls[0]["output_dir"] == input_dir


# ===========================================================================
# 3.  ParsedDocument structure
# ===========================================================================


class TestParsedDocument:
    def _parse(self, monkeypatch: pytest.MonkeyPatch) -> ParsedDocument:
        return _make_parser(monkeypatch).parse(_make_source())

    def test_returns_parsed_document(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        assert isinstance(doc, ParsedDocument)

    def test_provenance_parser_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        assert doc.provenance.parser_name == "opendataloader"

    def test_provenance_source_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        assert doc.provenance.source_id == "test_doc"

    def test_provenance_media_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        assert doc.provenance.media_type is MediaType.PDF

    def test_document_title_from_first_heading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        assert doc.title == "Executive Summary"

    def test_document_id_is_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc1 = self._parse(monkeypatch)
        doc2 = _make_parser(monkeypatch).parse(_make_source())
        assert doc1.document_id == doc2.document_id

    def test_empty_content_elements_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        assert all(e.text.strip() for e in doc.elements)

    def test_element_order_is_sequential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        assert [e.order for e in doc.elements] == list(range(len(doc.elements)))


# ===========================================================================
# 4.  Element type mapping
# ===========================================================================


class TestElementTypeMapping:
    def _parse(self, monkeypatch: pytest.MonkeyPatch) -> ParsedDocument:
        return _make_parser(monkeypatch).parse(_make_source())

    def test_heading_maps_to_title(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        titles = [e for e in doc.elements if e.type is ElementType.TITLE]
        assert len(titles) == 1
        assert titles[0].text == "Executive Summary"

    def test_paragraph_maps_to_narrative_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        narr = [e for e in doc.elements if e.type is ElementType.NARRATIVE_TEXT]
        assert any("ZX9000" in e.text for e in narr)

    def test_table_maps_to_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        tables = [e for e in doc.elements if e.type is ElementType.TABLE]
        assert len(tables) == 1
        assert "dense" in tables[0].text

    def test_image_maps_to_figure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        figs = [e for e in doc.elements if e.type is ElementType.FIGURE]
        assert len(figs) == 1
        assert "diagram" in figs[0].text

    def test_formula_maps_to_formula(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        formulas = [e for e in doc.elements if e.type is ElementType.FORMULA]
        assert len(formulas) == 1
        assert "mc" in formulas[0].text


# ===========================================================================
# 5.  section_path (heading breadcrumb)
# ===========================================================================


class TestSectionPath:
    def _parse(self, monkeypatch: pytest.MonkeyPatch) -> ParsedDocument:
        return _make_parser(monkeypatch).parse(_make_source())

    def test_paragraph_under_heading_carries_section_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc = self._parse(monkeypatch)
        narr = [e for e in doc.elements if e.type is ElementType.NARRATIVE_TEXT]
        assert any(e.section_path == ("Executive Summary",) for e in narr)

    def test_heading_itself_has_empty_section_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        titles = [e for e in doc.elements if e.type is ElementType.TITLE]
        # heading has no *parent* heading → empty section_path before it pushes itself
        assert titles[0].section_path == ()

    def test_nested_headings_build_breadcrumb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """H1 → H2 → paragraph should give section_path = (H1_text, H2_text)."""
        nested: list[dict[str, Any]] = [
            {
                "type": "heading",
                "id": 1,
                "page_number": 1,
                "heading_level": 1,
                "content": "Chapter One",
            },
            {
                "type": "heading",
                "id": 2,
                "page_number": 1,
                "heading_level": 2,
                "content": "Section 1.1",
            },
            {
                "type": "paragraph",
                "id": 3,
                "page_number": 1,
                "heading_level": None,
                "content": "Body text.",
            },
        ]

        calls: list[dict[str, Any]] = []

        def _fake_convert(
            input_path: list[str] | str,
            output_dir: str,
            format: str = "json",
            **kwargs: Any,
        ) -> None:
            calls.append({})
            first = input_path[0] if isinstance(input_path, list) else input_path
            stem = Path(first).stem
            (Path(output_dir) / f"{stem}.json").write_text(json.dumps(nested), encoding="utf-8")

        odl_mod = types.ModuleType("opendataloader_pdf")
        odl_mod.convert = _fake_convert  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "opendataloader_pdf", odl_mod)

        mod_name = "jera.adapters.parsing.opendataloader_parser"
        if mod_name in sys.modules:
            monkeypatch.delitem(sys.modules, mod_name)
        parser = importlib.import_module(mod_name).OpenDataLoaderParser()
        doc = parser.parse(_make_source("nested_doc"))

        body = [e for e in doc.elements if e.type is ElementType.NARRATIVE_TEXT]
        assert len(body) == 1
        assert body[0].section_path == ("Chapter One", "Section 1.1")


# ===========================================================================
# 6.  page_span
# ===========================================================================


class TestPageSpan:
    def _parse(self, monkeypatch: pytest.MonkeyPatch) -> ParsedDocument:
        return _make_parser(monkeypatch).parse(_make_source())

    def test_heading_page_span_is_page_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        title = next(e for e in doc.elements if e.type is ElementType.TITLE)
        assert title.page_span.start_page == 1
        assert title.page_span.end_page == 1

    def test_table_page_span_is_page_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        table = next(e for e in doc.elements if e.type is ElementType.TABLE)
        assert table.page_span.start_page == 2

    def test_formula_page_span_is_page_3(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        formula = next(e for e in doc.elements if e.type is ElementType.FORMULA)
        assert formula.page_span.start_page == 3


# ===========================================================================
# 7.  metadata / provenance keys
# ===========================================================================


class TestMetadata:
    def _parse(self, monkeypatch: pytest.MonkeyPatch) -> ParsedDocument:
        return _make_parser(monkeypatch).parse(_make_source())

    def test_paragraph_has_route_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        narr = next(e for e in doc.elements if e.type is ElementType.NARRATIVE_TEXT)
        assert narr.metadata[METADATA_ROUTE] == "text"

    def test_image_has_route_vlm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        fig = next(e for e in doc.elements if e.type is ElementType.FIGURE)
        assert fig.metadata[METADATA_ROUTE] == "vlm"

    def test_bounding_box_stored_in_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = self._parse(monkeypatch)
        title = next(e for e in doc.elements if e.type is ElementType.TITLE)
        assert "bounding_box" in title.metadata
        assert title.metadata["bounding_box"] == [72.0, 700.0, 540.0, 730.0]

    def test_ocr_engine_sets_route_ocr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When an element carries ocr_engine, route should become 'ocr'."""
        ocr_elements: list[dict[str, Any]] = [
            {
                "type": "paragraph",
                "id": 1,
                "page_number": 1,
                "heading_level": None,
                "content": "Scanned text via OCR.",
                "ocr_engine": "tesseract",
            },
        ]

        def _fake_convert(
            input_path: list[str] | str,
            output_dir: str,
            format: str = "json",
            **kwargs: Any,
        ) -> None:
            first = input_path[0] if isinstance(input_path, list) else input_path
            stem = Path(first).stem
            (Path(output_dir) / f"{stem}.json").write_text(
                json.dumps(ocr_elements), encoding="utf-8"
            )

        odl_mod = types.ModuleType("opendataloader_pdf")
        odl_mod.convert = _fake_convert  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "opendataloader_pdf", odl_mod)

        mod_name = "jera.adapters.parsing.opendataloader_parser"
        if mod_name in sys.modules:
            monkeypatch.delitem(sys.modules, mod_name)
        parser = importlib.import_module(mod_name).OpenDataLoaderParser()
        doc = parser.parse(_make_source("ocr_doc"))

        elem = doc.elements[0]
        assert elem.metadata[METADATA_ROUTE] == "ocr"
        assert elem.metadata[METADATA_OCR_ENGINE] == "tesseract"


# ===========================================================================
# 8.  ImportError path (no opendataloader_pdf installed)
# ===========================================================================


class TestImportError:
    # The import is lazy (in parse(), like DoclingParser), so construction succeeds without the
    # extra; the ImportError surfaces when parse() actually runs without opendataloader_pdf.
    @staticmethod
    def _pdf_source() -> SourceRef:
        return SourceRef(source_id="odl-x", media_type=MediaType.PDF, content=b"%PDF-1.4 fake")

    def test_missing_library_raises_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "opendataloader_pdf", raising=False)
        parser = importlib.import_module(
            "jera.adapters.parsing.opendataloader_parser"
        ).OpenDataLoaderParser()
        with pytest.raises(ImportError, match="opendataloader"):
            parser.parse(self._pdf_source())

    def test_import_error_message_mentions_java(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(sys.modules, "opendataloader_pdf", raising=False)
        parser = importlib.import_module(
            "jera.adapters.parsing.opendataloader_parser"
        ).OpenDataLoaderParser()
        with pytest.raises(ImportError, match="Java 11"):
            parser.parse(self._pdf_source())


# ===========================================================================
# 9.  requires_extra: real-library smoke test (skipped without Java/package)
# ===========================================================================

pytestmark_real = pytest.mark.requires_extra

_HAS_ODL = importlib.util.find_spec("opendataloader_pdf") is not None


@pytest.mark.requires_extra
@pytest.mark.skipif(not _HAS_ODL, reason="opendataloader extra not installed")
def test_opendataloader_parses_real_pdf_into_typed_elements() -> None:
    """Live smoke test: requires ``pip install opendataloader-pdf`` + Java 11+."""
    import pymupdf

    from jera.adapters.parsing.opendataloader_parser import OpenDataLoaderParser

    pdf_doc = pymupdf.open()
    page = pdf_doc.new_page()
    page.insert_text((72, 72), "OpenDataLoader Test Heading", fontsize=18)
    page.insert_text((72, 110), "The ranking module identifier is ZX9000.", fontsize=11)
    pdf_bytes = bytes(pdf_doc.tobytes())
    pdf_doc.close()

    parser = OpenDataLoaderParser()
    src = SourceRef(source_id="odl_live_test", media_type=MediaType.PDF, content=pdf_bytes)
    doc = parser.parse(src)

    assert doc.provenance.parser_name == "opendataloader"
    assert doc.elements, "expected typed elements"
    joined = " ".join(e.text for e in doc.elements)
    assert "ZX9000" in joined
    assert any(e.section_path for e in doc.elements if e.type is ElementType.NARRATIVE_TEXT)
