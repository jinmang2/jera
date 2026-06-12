"""RoutingPdfParser — the one parser that uses jera's internal router + OCR.

Implements the ``DocumentParser`` port. Per page it computes cheap ``PageFeatures``, asks the
``PageRouter`` for a route, then either extracts the text layer (pymupdf) or runs the
``OCREngine`` on a rendered page image. Route/OCR provenance is written into
``DocumentElement.metadata`` so downstream consumers see *how* each element was produced.

Rich parsers (docling/opendataloader) route+OCR internally and do NOT use this; this is the
composition point for jera-owned routing. Wired into the registry behind ``use_routing_pdf``
(default off) so default PDF ingestion is unchanged.
"""

from __future__ import annotations

from typing import Any

from jera.adapters.parsing.routing import (
    FakeOCR,
    HeuristicRouter,
    OCREngine,
    PageFeatures,
    PageRouter,
    Route,
)
from jera.domain.document import (
    METADATA_OCR_CONFIDENCE,
    METADATA_OCR_ENGINE,
    METADATA_ROUTE,
    DocumentElement,
    ElementType,
    MediaType,
    PageSpan,
    ParsedDocument,
    Provenance,
    SourceRef,
)
from jera.domain.ids import stable_id


class RoutingPdfParser:
    name = "routing-pdf"
    version = "1.0.0"

    def __init__(
        self,
        router: PageRouter | None = None,
        ocr: OCREngine | None = None,
        vlm: OCREngine | None = None,
    ) -> None:
        self._router: PageRouter = router or HeuristicRouter()
        self._ocr: OCREngine = ocr or FakeOCR()
        # A vision-language engine for the VLM route (also an OCREngine: image -> text). Optional:
        # the default HeuristicRouter never emits VLM, so this stays None unless a VLM-capable
        # router + engine are injected together.
        self._vlm: OCREngine | None = vlm

    def supports(self, source: SourceRef) -> bool:
        return source.media_type is MediaType.PDF

    def parse(self, source: SourceRef) -> ParsedDocument:
        import pymupdf

        pdf: Any = pymupdf
        document_id = stable_id(source.source_id, source.media_type.value, self.name)
        doc = pdf.open(stream=source.read_bytes(), filetype="pdf")
        elements: list[DocumentElement] = []
        title: str | None = None
        order = 0
        try:
            for page_index in range(doc.page_count):
                page = doc[page_index]
                features = self._page_features(page, page_index + 1)
                decision = self._router.route(features)
                page_span = PageSpan.single(page_index + 1)

                if decision.route is Route.OCR:
                    image = page.get_pixmap().tobytes("png")
                    result = self._ocr.recognize(bytes(image))
                    body = result.text.strip()
                    if body:
                        elements.append(
                            DocumentElement(
                                element_id=stable_id(document_id, str(order), "ocr"),
                                type=ElementType.NARRATIVE_TEXT,
                                text=body,
                                page_span=page_span,
                                order=order,
                                metadata={
                                    METADATA_ROUTE: Route.OCR.value,
                                    METADATA_OCR_ENGINE: result.engine_id,
                                    METADATA_OCR_CONFIDENCE: result.confidence,
                                },
                            )
                        )
                        order += 1
                elif decision.route is Route.TEXT:
                    for block in sorted(
                        page.get_text("blocks"), key=lambda b: (round(b[1], 1), round(b[0], 1))
                    ):
                        body = (block[4] or "").strip()
                        if not body:
                            continue
                        etype = ElementType.NARRATIVE_TEXT
                        if (
                            len(body) <= 80
                            and "\n" not in body
                            and block[1] < page.rect.height * 0.25
                        ):
                            etype = ElementType.TITLE
                            if title is None and page_index == 0:
                                title = body
                        elements.append(
                            DocumentElement(
                                element_id=stable_id(document_id, str(order), etype.value),
                                type=etype,
                                text=body,
                                page_span=page_span,
                                order=order,
                                metadata={METADATA_ROUTE: Route.TEXT.value},
                            )
                        )
                        order += 1
                else:  # Route.VLM — delegate to the injected vision-language engine
                    if self._vlm is None:
                        raise RuntimeError(
                            "router emitted VLM but no vlm engine was configured; pass "
                            "RoutingPdfParser(vlm=<OCREngine>) or use a router that omits VLM."
                        )
                    image = page.get_pixmap().tobytes("png")
                    result = self._vlm.recognize(bytes(image))
                    body = result.text.strip()
                    if body:
                        elements.append(
                            DocumentElement(
                                element_id=stable_id(document_id, str(order), "vlm"),
                                type=ElementType.NARRATIVE_TEXT,
                                text=body,
                                page_span=page_span,
                                order=order,
                                metadata={
                                    METADATA_ROUTE: Route.VLM.value,
                                    METADATA_OCR_ENGINE: result.engine_id,
                                    METADATA_OCR_CONFIDENCE: result.confidence,
                                },
                            )
                        )
                        order += 1
        finally:
            doc.close()

        return ParsedDocument(
            document_id=document_id,
            source_id=source.source_id,
            title=title,
            elements=elements,
            provenance=Provenance(
                source_id=source.source_id,
                parser_name=self.name,
                parser_version=self.version,
                media_type=source.media_type,
            ),
        )

    @staticmethod
    def _page_features(page: Any, page_number: int) -> PageFeatures:
        text = page.get_text("text")
        page_area = float(page.rect.width) * float(page.rect.height)
        image_area = 0.0
        if page_area > 0:
            for info in page.get_image_info():
                bbox = info.get("bbox")
                if bbox:
                    image_area += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        ratio = min(1.0, image_area / page_area) if page_area > 0 else 0.0
        return PageFeatures(
            page_number=page_number,
            has_text_layer=bool(text.strip()),
            image_area_ratio=ratio,
            text_char_count=len(text),
        )
