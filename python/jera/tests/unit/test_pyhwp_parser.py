"""SDK-boundary tests for PyHwpParser (legacy .hwp OLE binary adapter).

Strategy:
- Inject a fake ``hwp5`` module into ``sys.modules`` via monkeypatch so no real
  pyhwp install is required (mirrors the cloud-vendor-adapter test pattern).
- The fake exposes ``hwp5.filestructure.Hwp5File`` and
  ``hwp5.filestructure.InvalidHwp5FileError`` — the exact surface ``PyHwpParser``
  calls.
- A canned ``preview_text.text`` string with ``\\r\\n``-separated paragraphs feeds
  the parser, and we assert the resulting ``ParsedDocument`` structure.
- One ``@pytest.mark.requires_extra`` test guards the real-library path.

Real API targeted:
    ``from hwp5.filestructure import Hwp5File, InvalidHwp5FileError``
    ``Hwp5File(path_str).preview_text.text``  → plain str (UTF-16LE PrvText stream)
Source: https://github.com/mete0r/pyhwp  (hwp5/filestructure.py, Hwp5FileBase.__init__,
        Hwp5File.preview_text cached_property, PreviewText.get_text)
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from typing import Any

import pytest

from jera.domain.document import ElementType, MediaType, SourceRef

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CANNED_TEXT = "\r\n".join(
    [
        "인공지능 기술 개요",  # short → TITLE (heading heuristic)
        "인공지능은 컴퓨터 과학의 한 분야로, 기계가 인간의 지능을 모방하도록"
        " 설계된 시스템을 연구합니다.",
        "연구 방법론",  # short → TITLE
        "본 논문에서는 다양한 실험적 방법론을 통해 모델 성능을 평가하였습니다.",
        "",  # blank — must be skipped
        "결론",  # short → TITLE
        "위 실험 결과를 바탕으로 최적의 파라미터를 도출하였습니다.",
    ]
)


def _fake_hwp5_modules(
    monkeypatch: pytest.MonkeyPatch,
    preview_text: str = _CANNED_TEXT,
    raise_on_open: Exception | None = None,
) -> None:
    """Inject a minimal fake ``hwp5`` + ``hwp5.filestructure`` into sys.modules."""

    class _FakeInvalidHwp5FileError(Exception):
        pass

    class _FakePreviewText:
        def __init__(self, text: str) -> None:
            self.text = text

    class _FakeHwp5File:
        def __init__(self, path: str) -> None:
            if raise_on_open is not None:
                raise raise_on_open
            self.preview_text = _FakePreviewText(preview_text)

    fs_mod = types.ModuleType("hwp5.filestructure")
    fs_mod.Hwp5File = _FakeHwp5File  # type: ignore[attr-defined]
    fs_mod.InvalidHwp5FileError = _FakeInvalidHwp5FileError  # type: ignore[attr-defined]

    hwp5_mod = types.ModuleType("hwp5")
    hwp5_mod.filestructure = fs_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "hwp5", hwp5_mod)
    monkeypatch.setitem(sys.modules, "hwp5.filestructure", fs_mod)


def _source(content: bytes = b"\xd0\xcf\x11\xe0fake") -> SourceRef:
    """Return an in-memory HWP SourceRef (bytes path, no real file needed)."""
    return SourceRef(source_id="hwp_test_doc", media_type=MediaType.HWP, content=content)


def _parser(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> Any:
    _fake_hwp5_modules(monkeypatch, **kwargs)
    # Re-import so the monkeypatched sys.modules is picked up.
    import jera.adapters.parsing.pyhwp_parser as _mod

    importlib.reload(_mod)
    return _mod.PyHwpParser()


# ---------------------------------------------------------------------------
# supports()
# ---------------------------------------------------------------------------


class TestSupports:
    def test_supports_hwp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        p = _parser(monkeypatch)
        assert p.supports(_source()) is True

    def test_does_not_support_hwpx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        p = _parser(monkeypatch)
        src = SourceRef(source_id="x", media_type=MediaType.HWPX, content=b"PK")
        assert p.supports(src) is False

    def test_does_not_support_pdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        p = _parser(monkeypatch)
        src = SourceRef(source_id="x", media_type=MediaType.PDF, content=b"%PDF")
        assert p.supports(src) is False

    def test_does_not_support_markdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        p = _parser(monkeypatch)
        src = SourceRef(source_id="x", media_type=MediaType.MARKDOWN, content=b"# hi")
        assert p.supports(src) is False


# ---------------------------------------------------------------------------
# parse() — document-level assertions
# ---------------------------------------------------------------------------


class TestParseReturnShape:
    def test_returns_parsed_document(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from jera.domain.document import ParsedDocument

        doc = _parser(monkeypatch).parse(_source())
        assert isinstance(doc, ParsedDocument)

    def test_source_id_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        assert doc.source_id == "hwp_test_doc"

    def test_document_id_is_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        a = _parser(monkeypatch).parse(_source())
        b = _parser(monkeypatch).parse(_source())
        assert a.document_id == b.document_id

    def test_title_set_to_first_heading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        assert doc.title == "인공지능 기술 개요"

    def test_elements_non_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        assert len(doc.elements) > 0

    def test_blank_paragraphs_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        assert all(e.text.strip() for e in doc.elements)

    def test_element_order_is_sequential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        orders = [e.order for e in doc.elements]
        assert orders == list(range(len(orders)))


# ---------------------------------------------------------------------------
# parse() — element type mapping
# ---------------------------------------------------------------------------


class TestParseElementTypes:
    def test_short_text_classified_as_title(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        titles = [e for e in doc.elements if e.type is ElementType.TITLE]
        assert any(e.text == "인공지능 기술 개요" for e in titles)
        assert any(e.text == "연구 방법론" for e in titles)
        assert any(e.text == "결론" for e in titles)

    def test_long_text_classified_as_narrative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        narr = [e for e in doc.elements if e.type is ElementType.NARRATIVE_TEXT]
        assert any("인공지능은 컴퓨터" in e.text for e in narr)
        assert any("본 논문에서는" in e.text for e in narr)

    def test_has_both_title_and_narrative_elements(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        types = {e.type for e in doc.elements}
        assert ElementType.TITLE in types
        assert ElementType.NARRATIVE_TEXT in types


# ---------------------------------------------------------------------------
# parse() — section_path and provenance
# ---------------------------------------------------------------------------


class TestParseSectionPathAndProvenance:
    def test_narrative_under_heading_carries_section_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        doc = _parser(monkeypatch).parse(_source())
        narr = [e for e in doc.elements if e.type is ElementType.NARRATIVE_TEXT]
        # First narrative paragraph is under "인공지능 기술 개요"
        first_narr = narr[0]
        assert first_narr.section_path == ("인공지능 기술 개요",)

    def test_title_elements_have_no_section_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        titles = [e for e in doc.elements if e.type is ElementType.TITLE]
        assert all(e.section_path == () for e in titles)

    def test_provenance_parser_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        assert doc.provenance.parser_name == "pyhwp"

    def test_provenance_parser_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        assert doc.provenance.parser_version == "1.0.0"

    def test_provenance_source_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        assert doc.provenance.source_id == "hwp_test_doc"

    def test_provenance_media_type_is_hwp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        assert doc.provenance.media_type is MediaType.HWP

    def test_page_span_defaults_to_page_1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        assert all(e.page_span.start_page == 1 for e in doc.elements)
        assert all(e.page_span.end_page == 1 for e in doc.elements)


# ---------------------------------------------------------------------------
# parse() — element_id stability
# ---------------------------------------------------------------------------


class TestElementIdStability:
    def test_element_ids_are_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        a = _parser(monkeypatch).parse(_source())
        b = _parser(monkeypatch).parse(_source())
        assert [e.element_id for e in a.elements] == [e.element_id for e in b.elements]

    def test_element_ids_are_unique(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch).parse(_source())
        ids = [e.element_id for e in doc.elements]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# parse() — paragraph separator variants
# ---------------------------------------------------------------------------


class TestParagraphSeparators:
    def _count_elements(self, monkeypatch: pytest.MonkeyPatch, sep: str) -> int:
        text = sep.join(["제목", "내용입니다.", ""])
        doc = _parser(monkeypatch, preview_text=text).parse(_source())
        return len(doc.elements)

    def test_crlf_separator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._count_elements(monkeypatch, "\r\n") == 2

    def test_lf_separator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._count_elements(monkeypatch, "\n") == 2

    def test_cr_separator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert self._count_elements(monkeypatch, "\r") == 2


# ---------------------------------------------------------------------------
# parse() — empty document
# ---------------------------------------------------------------------------


class TestEmptyDocument:
    def test_all_blank_text_yields_no_elements(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch, preview_text="\r\n\r\n  \r\n").parse(_source())
        assert doc.elements == []
        assert doc.title is None

    def test_empty_string_yields_no_elements(self, monkeypatch: pytest.MonkeyPatch) -> None:
        doc = _parser(monkeypatch, preview_text="").parse(_source())
        assert doc.elements == []


# ---------------------------------------------------------------------------
# parse() — path-based SourceRef (real file path, not in-memory bytes)
# ---------------------------------------------------------------------------


class TestPathBasedSourceRef:
    def test_parse_with_path_source_ref(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
    ) -> None:
        import pathlib

        fake_file = tmp_path / "test.hwp"  # type: ignore[operator]
        fake_file.write_bytes(b"\xd0\xcf\x11\xe0fake")

        doc = _parser(monkeypatch).parse(
            SourceRef(
                source_id="path_src",
                media_type=MediaType.HWP,
                path=pathlib.Path(str(fake_file)),
            )
        )
        assert doc.source_id == "path_src"
        assert len(doc.elements) > 0


# ---------------------------------------------------------------------------
# ImportError guard — missing hwp5 module
# ---------------------------------------------------------------------------


class TestImportErrorGuard:
    def test_missing_hwp5_raises_import_error_with_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Remove hwp5 from sys.modules so the lazy import fails.
        monkeypatch.delitem(sys.modules, "hwp5", raising=False)
        monkeypatch.delitem(sys.modules, "hwp5.filestructure", raising=False)

        import jera.adapters.parsing.pyhwp_parser as _mod

        importlib.reload(_mod)
        parser = _mod.PyHwpParser()

        with pytest.raises(ImportError, match="pyhwp"):
            parser.parse(_source())


# ---------------------------------------------------------------------------
# requires_extra: real pyhwp integration test (skipped without lib)
# ---------------------------------------------------------------------------

_HAS_PYHWP = importlib.util.find_spec("hwp5") is not None


@pytest.mark.requires_extra
@pytest.mark.skipif(not _HAS_PYHWP, reason="hwp extra (pyhwp) not installed")
def test_pyhwp_parser_real_lib_import_succeeds() -> None:
    """Smoke-test that the real hwp5 import path works when pyhwp is installed."""
    # We don't have a real .hwp fixture, so we just verify the class loads and
    # supports() returns True for HWP without crashing.
    from jera.adapters.parsing.pyhwp_parser import PyHwpParser

    parser = PyHwpParser()
    src = SourceRef(source_id="smoke", media_type=MediaType.HWP, content=b"x")
    assert parser.supports(src) is True
    assert parser.name == "pyhwp"
    assert parser.version == "1.0.0"
