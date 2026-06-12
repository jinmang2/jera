"""Adapter-internal routing/OCR collaborators for ``RoutingPdfParser``.

These are NOT top-level domain ports — ``DocumentParser`` is the only domain port. A rich
parser (docling/opendataloader) routes+OCRs internally, so a top-level OCR/router port would
be dead code there. Routing/OCR live here as Protocols composed *inside* ``RoutingPdfParser``.

The default in-CI implementations (`HeuristicRouter`, `FakeOCR`) are deterministic and need no
extras; real OCR engines (tesseract/rapidocr) and a VLM router land in M5b behind extras.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Route(StrEnum):
    TEXT = "text"  # page has a usable text layer → extract text directly
    OCR = "ocr"  # scanned/image page → run an OCR engine
    VLM = "vlm"  # complex layout/figures → route to a vision-language model (M5b only)


class PageFeatures(BaseModel):
    """Cheap, deterministic signals extracted from a PDF page to decide its route."""

    model_config = {"frozen": True}

    page_number: int  # 1-based
    has_text_layer: bool
    image_area_ratio: float = Field(ge=0.0, le=1.0)  # covered image area / page area
    text_char_count: int


class RouteDecision(BaseModel):
    model_config = {"frozen": True}

    route: Route
    reason: str


class OcrResult(BaseModel):
    """Output of an OCR engine for one page/image."""

    model_config = {"frozen": True}

    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    engine_id: str


@runtime_checkable
class PageRouter(Protocol):
    """Decides how each page should be turned into text."""

    def route(self, features: PageFeatures) -> RouteDecision: ...


@runtime_checkable
class OCREngine(Protocol):
    """Recognizes text from a page image. ``model_id``/``engine_id`` identify the engine."""

    engine_id: str

    def recognize(self, image: bytes, lang: str = "kor+eng") -> OcrResult: ...


class HeuristicRouter:
    """Deterministic rule-based router (CI default).

    Pinned threshold table (see plan AC8):
      - no text layer AND image_area_ratio > 0.5  → OCR (scanned page)
      - has text layer                            → TEXT
      - otherwise (text-poor, low-image)          → OCR (nothing to extract)
    Never emits VLM in M5a (the VLM branch is an opt-in M5b adapter).
    """

    def __init__(self, image_ratio_threshold: float = 0.5) -> None:
        self._image_ratio_threshold = image_ratio_threshold

    def route(self, features: PageFeatures) -> RouteDecision:
        if not features.has_text_layer and features.image_area_ratio > self._image_ratio_threshold:
            return RouteDecision(route=Route.OCR, reason="no text layer + image-heavy → scanned")
        if features.has_text_layer:
            return RouteDecision(route=Route.TEXT, reason="usable text layer")
        return RouteDecision(route=Route.OCR, reason="no text layer → OCR fallback")


class FakeOCR:
    """Deterministic OCR engine for CI: returns text from a fixture map keyed by image bytes.

    Real engines (tesseract/rapidocr) arrive in M5b. This proves the routing→OCR composition
    offline with zero deps and zero models.
    """

    engine_id = "fake-ocr-v1"

    def __init__(self, fixture: dict[bytes, str] | None = None, default: str = "") -> None:
        self._fixture = fixture or {}
        self._default = default

    def recognize(self, image: bytes, lang: str = "kor+eng") -> OcrResult:
        text = self._fixture.get(image, self._default)
        return OcrResult(text=text, confidence=1.0 if text else 0.0, engine_id=self.engine_id)
