"""HwpxParser (stdlib OWPML) on the committed fixture, and registry dispatch for .hwpx."""

from __future__ import annotations

import pathlib

from jera.adapters.parsing.hwpx_parser import HwpxParser
from jera.config import Profile, Settings, build_system
from jera.domain.document import ElementType, MediaType, SourceRef

_FIXTURE = pathlib.Path(__file__).parents[1] / "fixtures" / "hwpx" / "sample.hwpx"


def _source() -> SourceRef:
    return SourceRef(source_id="hwpx1", media_type=MediaType.HWPX, content=_FIXTURE.read_bytes())


def test_hwpx_parses_typed_elements_with_table_and_sections() -> None:
    doc = HwpxParser().parse(_source())
    types = [e.type for e in doc.elements]
    assert ElementType.TITLE in types
    assert ElementType.NARRATIVE_TEXT in types
    assert ElementType.TABLE in types
    assert doc.title == "개요"
    assert doc.provenance.parser_name == "hwpx"
    # table cell content recovered
    table = next(e for e in doc.elements if e.type is ElementType.TABLE)
    assert "항목" in table.text and "1000" in table.text
    # section breadcrumb from heading heuristic
    narrative = next(e for e in doc.elements if e.type is ElementType.NARRATIVE_TEXT)
    assert narrative.section_path == ("개요",)


def test_hwpx_deterministic() -> None:
    a = HwpxParser().parse(_source())
    b = HwpxParser().parse(_source())
    assert [e.element_id for e in a.elements] == [e.element_id for e in b.elements]


def test_registry_dispatches_hwpx_by_default() -> None:
    system = build_system(Settings(profile=Profile.TEST))
    job = system.ingest.ingest(_source())
    assert job.status.value == "succeeded"
    assert job.chunk_count >= 1
