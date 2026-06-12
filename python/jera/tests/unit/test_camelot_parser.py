"""SDK-boundary tests for CamelotTableParser.

Injects a fake ``camelot`` module into ``sys.modules`` so no real camelot-py
or Ghostscript installation is required.  The fake exposes exactly the surface
that the adapter calls:

* ``camelot.read_pdf(filepath, pages, flavor, suppress_stdout)``
* ``Table.df``  — a fake DataFrame with ``.columns`` and ``.iterrows()``
* ``Table.page`` — int page number

A ``@pytest.mark.requires_extra`` real integration test is also included; it
is auto-skipped when camelot-py is not installed.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from typing import Any

import pytest

from jera.domain.document import ElementType, MediaType, PageSpan, SourceRef

# ---------------------------------------------------------------------------
# Helpers: fake camelot surface
# ---------------------------------------------------------------------------


class _FakeDataFrame:
    """Minimal pandas-DataFrame-shaped object covering the adapter's usage."""

    def __init__(self, rows: list[list[str]], columns: list[str]) -> None:
        self._rows = rows
        self.columns = columns

    def iterrows(self) -> Any:
        yield from enumerate(self._rows)


class _FakeTable:
    def __init__(
        self,
        df: _FakeDataFrame,
        page: int = 1,
        accuracy: float = 99.0,
    ) -> None:
        self.df = df
        self.page = page
        self.accuracy = accuracy


def _make_fake_camelot(
    tables: list[_FakeTable],
    captured: list[dict[str, Any]],
) -> types.ModuleType:
    """Return a fake ``camelot`` module whose ``read_pdf`` records its call."""

    def read_pdf(
        filepath: str,
        pages: str = "1",
        flavor: str = "lattice",
        suppress_stdout: bool = False,
        **_kwargs: Any,
    ) -> list[_FakeTable]:
        captured.append(
            {
                "filepath": filepath,
                "pages": pages,
                "flavor": flavor,
                "suppress_stdout": suppress_stdout,
            }
        )
        return tables

    mod = types.ModuleType("camelot")
    mod.read_pdf = read_pdf  # type: ignore[attr-defined]
    return mod


def _source(content: bytes = b"%PDF-fake") -> SourceRef:
    return SourceRef(source_id="test_src", media_type=MediaType.PDF, content=content)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_table_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, list[dict[str, Any]]]:
    """Install fake camelot with two tables; return (parser, captured_calls)."""
    df1 = _FakeDataFrame(
        rows=[["서울", "0.82"], ["부산", "0.75"]],
        columns=["지역", "점수"],
    )
    df2 = _FakeDataFrame(
        rows=[["dense", "0.71"], ["sparse", "0.65"]],
        columns=["Strategy", "Recall"],
    )
    table1 = _FakeTable(df=df1, page=1, accuracy=98.5)
    table2 = _FakeTable(df=df2, page=2, accuracy=95.0)
    captured: list[dict[str, Any]] = []
    fake_mod = _make_fake_camelot([table1, table2], captured)
    monkeypatch.setitem(sys.modules, "camelot", fake_mod)

    # Force fresh import of adapter so the monkeypatched sys.modules is used.
    if "jera.adapters.parsing.camelot_parser" in sys.modules:
        del sys.modules["jera.adapters.parsing.camelot_parser"]

    from jera.adapters.parsing.camelot_parser import CamelotTableParser

    parser = CamelotTableParser(flavor="lattice", pages="all")
    return parser, captured


# ---------------------------------------------------------------------------
# 1. supports()
# ---------------------------------------------------------------------------


class TestCamelotParserSupports:
    def test_supports_pdf(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[dict[str, Any]] = []
        monkeypatch.setitem(sys.modules, "camelot", _make_fake_camelot([], captured))
        if "jera.adapters.parsing.camelot_parser" in sys.modules:
            del sys.modules["jera.adapters.parsing.camelot_parser"]
        from jera.adapters.parsing.camelot_parser import CamelotTableParser

        parser = CamelotTableParser()
        assert parser.supports(SourceRef(source_id="s", media_type=MediaType.PDF, content=b"x"))

    def test_does_not_support_markdown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[dict[str, Any]] = []
        monkeypatch.setitem(sys.modules, "camelot", _make_fake_camelot([], captured))
        if "jera.adapters.parsing.camelot_parser" in sys.modules:
            del sys.modules["jera.adapters.parsing.camelot_parser"]
        from jera.adapters.parsing.camelot_parser import CamelotTableParser

        parser = CamelotTableParser()
        assert not parser.supports(
            SourceRef(source_id="s", media_type=MediaType.MARKDOWN, content=b"x")
        )


# ---------------------------------------------------------------------------
# 2. read_pdf called with correct arguments
# ---------------------------------------------------------------------------


class TestCamelotReadPdfArguments:
    def test_read_pdf_called_once(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, captured = two_table_setup
        parser.parse(_source())
        assert len(captured) == 1

    def test_read_pdf_flavor_forwarded(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, captured = two_table_setup
        parser.parse(_source())
        assert captured[0]["flavor"] == "lattice"

    def test_read_pdf_pages_forwarded(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, captured = two_table_setup
        parser.parse(_source())
        assert captured[0]["pages"] == "all"

    def test_read_pdf_suppress_stdout_forwarded(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, captured = two_table_setup
        parser.parse(_source())
        assert captured[0]["suppress_stdout"] is True

    def test_read_pdf_filepath_is_string(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        """Adapter must pass a string path (temp file) not raw bytes."""
        parser, captured = two_table_setup
        parser.parse(_source())
        assert isinstance(captured[0]["filepath"], str)

    def test_stream_flavor_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[dict[str, Any]] = []
        fake_mod = _make_fake_camelot([], captured)
        monkeypatch.setitem(sys.modules, "camelot", fake_mod)
        if "jera.adapters.parsing.camelot_parser" in sys.modules:
            del sys.modules["jera.adapters.parsing.camelot_parser"]
        from jera.adapters.parsing.camelot_parser import CamelotTableParser

        parser = CamelotTableParser(flavor="stream", pages="1,2")
        parser.parse(_source())
        assert captured[0]["flavor"] == "stream"
        assert captured[0]["pages"] == "1,2"


# ---------------------------------------------------------------------------
# 3. Returned tables → TABLE elements
# ---------------------------------------------------------------------------


class TestCamelotElementMapping:
    def test_returns_parsed_document(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        from jera.domain.document import ParsedDocument

        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert isinstance(result, ParsedDocument)

    def test_element_count_matches_table_count(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert len(result.elements) == 2

    def test_elements_are_table_type(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert all(e.type is ElementType.TABLE for e in result.elements)

    def test_first_element_page_span(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert result.elements[0].page_span == PageSpan.single(1)

    def test_second_element_page_span(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert result.elements[1].page_span == PageSpan.single(2)

    def test_element_text_contains_header_columns(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, _ = two_table_setup
        result = parser.parse(_source())
        # First table columns: 지역, 점수
        assert "지역" in result.elements[0].text
        assert "점수" in result.elements[0].text

    def test_element_text_contains_data_rows(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert "서울" in result.elements[0].text
        assert "부산" in result.elements[0].text

    def test_markdown_has_separator_row(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        """Table text must include a ``---`` separator between header and data."""
        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert "---" in result.elements[0].text

    def test_element_text_is_pipe_delimited(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert " | " in result.elements[0].text

    def test_element_order_is_sequential(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert [e.order for e in result.elements] == [0, 1]

    def test_element_section_path_is_empty(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert all(e.section_path == () for e in result.elements)

    def test_empty_table_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A table whose DataFrame produces only whitespace is excluded."""
        empty_df = _FakeDataFrame(rows=[], columns=[])
        captured: list[dict[str, Any]] = []
        fake_mod = _make_fake_camelot([_FakeTable(df=empty_df, page=1)], captured)
        monkeypatch.setitem(sys.modules, "camelot", fake_mod)
        if "jera.adapters.parsing.camelot_parser" in sys.modules:
            del sys.modules["jera.adapters.parsing.camelot_parser"]
        from jera.adapters.parsing.camelot_parser import CamelotTableParser

        result = CamelotTableParser().parse(_source())
        assert result.elements == []


# ---------------------------------------------------------------------------
# 4. Provenance
# ---------------------------------------------------------------------------


class TestCamelotProvenance:
    def test_parser_name_in_provenance(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert result.provenance.parser_name == "camelot"

    def test_source_id_in_provenance(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert result.provenance.source_id == "test_src"

    def test_media_type_in_provenance(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert result.provenance.media_type is MediaType.PDF

    def test_document_id_is_stable(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        """Parsing the same source twice must yield the same document_id."""
        parser, _ = two_table_setup
        r1 = parser.parse(_source())
        r2 = parser.parse(_source())
        assert r1.document_id == r2.document_id

    def test_source_id_in_result(
        self,
        two_table_setup: tuple[Any, list[dict[str, Any]]],
    ) -> None:
        parser, _ = two_table_setup
        result = parser.parse(_source())
        assert result.source_id == "test_src"


# ---------------------------------------------------------------------------
# 5. ImportError path (camelot absent)
# ---------------------------------------------------------------------------


class TestCamelotImportError:
    def test_parse_raises_import_error_when_camelot_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Ensure camelot is NOT in sys.modules
        monkeypatch.delitem(sys.modules, "camelot", raising=False)
        if "jera.adapters.parsing.camelot_parser" in sys.modules:
            del sys.modules["jera.adapters.parsing.camelot_parser"]

        # Make `import camelot` raise ImportError
        import builtins

        _real_import = builtins.__import__

        def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "camelot":
                raise ImportError("No module named 'camelot'")
            return _real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        from jera.adapters.parsing.camelot_parser import CamelotTableParser

        parser = CamelotTableParser()
        with pytest.raises(ImportError, match="camelot-py"):
            parser.parse(_source())


# ---------------------------------------------------------------------------
# 6. Real integration test (requires camelot-py installed)
# ---------------------------------------------------------------------------

pytestmark_requires = pytest.mark.requires_extra

_HAS_CAMELOT = importlib.util.find_spec("camelot") is not None


@pytest.mark.requires_extra
@pytest.mark.skipif(not _HAS_CAMELOT, reason="camelot extra not installed")
def test_camelot_parser_real_supports_pdf() -> None:
    """Smoke-test: real CamelotTableParser.supports() works without a live PDF."""
    from jera.adapters.parsing.camelot_parser import CamelotTableParser

    parser = CamelotTableParser()
    src = SourceRef(source_id="real", media_type=MediaType.PDF, content=b"%PDF-1.4")
    assert parser.supports(src)
    assert parser.name == "camelot"
    assert parser.version == "1.0.0"
