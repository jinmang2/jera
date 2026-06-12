"""RoutingPdfParser VLM route — now functional via an injected vision-language engine.

Previously the VLM branch raised NotImplementedError (an M5b placeholder). It now delegates to
an injected ``vlm`` engine (an OCREngine: image -> text) and writes route="vlm" provenance. The
default HeuristicRouter still never emits VLM, so default parsing is unchanged.
"""

from __future__ import annotations

import pytest

from jera.adapters.parsing.routing import OcrResult, PageFeatures, Route, RouteDecision
from jera.adapters.parsing.routing_pdf_parser import RoutingPdfParser
from jera.domain.document import (
    METADATA_OCR_CONFIDENCE,
    METADATA_OCR_ENGINE,
    METADATA_ROUTE,
    MediaType,
    SourceRef,
)


def _text_pdf(text: str = "anything") -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return bytes(data)


class _AlwaysVlmRouter:
    """A router that forces every page onto the VLM route (a VLM-capable router)."""

    def route(self, features: PageFeatures) -> RouteDecision:
        return RouteDecision(route=Route.VLM, reason="forced vlm for test")


class _FakeVlm:
    """A vision-language engine shaped as an OCREngine: image bytes -> recognized text."""

    engine_id = "fake-vlm-v1"

    def recognize(self, image: bytes, lang: str = "kor+eng") -> OcrResult:
        return OcrResult(text="VLM-read content", confidence=0.83, engine_id=self.engine_id)


def _source() -> SourceRef:
    return SourceRef(source_id="vlm-doc", media_type=MediaType.PDF, content=_text_pdf())


def test_vlm_route_delegates_to_injected_engine_with_provenance() -> None:
    parser = RoutingPdfParser(router=_AlwaysVlmRouter(), vlm=_FakeVlm())
    doc = parser.parse(_source())
    assert doc.elements, "expected a VLM-produced element"
    el = doc.elements[0]
    assert el.text == "VLM-read content"
    assert el.metadata[METADATA_ROUTE] == Route.VLM.value
    assert el.metadata[METADATA_OCR_ENGINE] == "fake-vlm-v1"
    assert el.metadata[METADATA_OCR_CONFIDENCE] == 0.83


def test_vlm_route_without_engine_raises_clear_error() -> None:
    parser = RoutingPdfParser(router=_AlwaysVlmRouter())  # no vlm engine
    with pytest.raises(RuntimeError, match="no vlm engine"):
        parser.parse(_source())
