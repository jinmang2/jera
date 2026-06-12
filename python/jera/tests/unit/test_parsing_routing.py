"""Routing internals (HeuristicRouter/FakeOCR/PageFeatures) + RoutingPdfParser end-to-end."""

from __future__ import annotations

from jera.adapters.parsing.routing import (
    FakeOCR,
    HeuristicRouter,
    PageFeatures,
    Route,
)
from jera.adapters.parsing.routing_pdf_parser import RoutingPdfParser
from jera.config import Profile, Settings, build_system
from jera.domain.document import (
    METADATA_OCR_CONFIDENCE,
    METADATA_OCR_ENGINE,
    METADATA_ROUTE,
    MediaType,
    SourceRef,
)


def _text_pdf(text: str = "Routing PDF text layer. Identifier ZZ42.") -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=12)
    data = doc.tobytes()
    doc.close()
    return bytes(data)


# --- routing.py internals ---


def test_heuristic_router_branches() -> None:
    r = HeuristicRouter()
    assert (
        r.route(
            PageFeatures(
                page_number=1, has_text_layer=True, image_area_ratio=0.0, text_char_count=100
            )
        ).route
        is Route.TEXT
    )
    assert (
        r.route(
            PageFeatures(
                page_number=1, has_text_layer=False, image_area_ratio=0.9, text_char_count=0
            )
        ).route
        is Route.OCR
    )
    # text-poor, low-image → OCR fallback
    assert (
        r.route(
            PageFeatures(
                page_number=1, has_text_layer=False, image_area_ratio=0.1, text_char_count=0
            )
        ).route
        is Route.OCR
    )


def test_fake_ocr_fixture_and_default() -> None:
    ocr = FakeOCR(fixture={b"img": "hello"}, default="def")
    assert ocr.recognize(b"img").text == "hello"
    assert ocr.recognize(b"img").confidence == 1.0
    miss = ocr.recognize(b"other")
    assert miss.text == "def"
    assert miss.engine_id == "fake-ocr-v1"


# --- RoutingPdfParser ---


def test_routing_pdf_text_route_writes_provenance() -> None:
    parser = RoutingPdfParser()  # default HeuristicRouter → text layer present → TEXT
    doc = parser.parse(SourceRef(source_id="p", media_type=MediaType.PDF, content=_text_pdf()))
    assert doc.elements
    assert all(e.metadata.get(METADATA_ROUTE) == "text" for e in doc.elements)
    assert any("ZZ42" in e.text for e in doc.elements)


def test_routing_pdf_ocr_route_uses_engine_and_writes_provenance() -> None:
    # Force OCR for every page; FakeOCR returns a constant.
    class AlwaysOcr:
        def route(self, features):  # type: ignore[no-untyped-def]
            from jera.adapters.parsing.routing import RouteDecision

            return RouteDecision(route=Route.OCR, reason="forced")

    parser = RoutingPdfParser(router=AlwaysOcr(), ocr=FakeOCR(default="OCR EXTRACTED 명세"))
    doc = parser.parse(SourceRef(source_id="p", media_type=MediaType.PDF, content=_text_pdf()))
    assert len(doc.elements) == 1
    el = doc.elements[0]
    assert el.text == "OCR EXTRACTED 명세"
    assert el.metadata[METADATA_ROUTE] == "ocr"
    assert el.metadata[METADATA_OCR_ENGINE] == "fake-ocr-v1"
    assert el.metadata[METADATA_OCR_CONFIDENCE] == 1.0


def test_routing_pdf_supports_only_pdf() -> None:
    p = RoutingPdfParser()
    assert p.supports(SourceRef(source_id="x", media_type=MediaType.PDF, content=b"%PDF"))
    assert not p.supports(SourceRef(source_id="x", media_type=MediaType.MARKDOWN, content=b"#"))


def test_routing_pdf_wired_into_ingest_pipeline() -> None:
    # AC2: RoutingPdfParser has a real consumer end-to-end via build_system → IngestPipeline.
    system = build_system(Settings(profile=Profile.TEST, use_routing_pdf=True))
    job = system.ingest.ingest(
        SourceRef(source_id="doc-pdf", media_type=MediaType.PDF, content=_text_pdf())
    )
    assert job.status.value == "succeeded"
    assert job.chunk_count >= 1


def test_default_pdf_unchanged_without_flag() -> None:
    # use_routing_pdf default off → PyMuPDFParser handles PDF (no regression).
    system = build_system(Settings(profile=Profile.TEST))
    job = system.ingest.ingest(
        SourceRef(source_id="doc-pdf2", media_type=MediaType.PDF, content=_text_pdf())
    )
    assert job.status.value == "succeeded"
