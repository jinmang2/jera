"""Real OCR engine adapters for :class:`jera.adapters.parsing.routing.OCREngine`.

Engines
-------
TesseractOCREngine
    Wraps ``pytesseract`` / Google Tesseract.  Requires ``pytesseract`` and
    ``Pillow`` (``ocr`` extra).
RapidOcrOCREngine
    Wraps ``rapidocr_onnxruntime``.  Requires ``rapidocr_onnxruntime``
    (``ocr`` extra).
ClovaOCREngine
    Calls the NAVER CLOVA OCR REST API.  **Disabled by default** — pass
    ``enabled=True`` with a valid ``invoke_url`` and ``secret``.
"""

from __future__ import annotations

from jera.adapters.parsing.ocr.clova_ocr import ClovaOCREngine
from jera.adapters.parsing.ocr.rapidocr_ocr import RapidOcrOCREngine
from jera.adapters.parsing.ocr.tesseract_ocr import TesseractOCREngine

__all__ = [
    "ClovaOCREngine",
    "RapidOcrOCREngine",
    "TesseractOCREngine",
]
