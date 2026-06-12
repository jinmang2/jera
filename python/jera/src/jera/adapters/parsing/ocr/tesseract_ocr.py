"""Tesseract OCR engine adapter — requires ``pytesseract`` + ``Pillow`` (ocr extra).

Wraps ``pytesseract.image_to_string`` (text) and ``pytesseract.image_to_data``
(per-word confidence) to produce a single ``OcrResult``.  Image bytes are decoded
via ``PIL.Image.open(io.BytesIO(bytes))`` before being passed to pytesseract.

Confidence is derived from ``image_to_data(output_type=Output.DICT)`` → ``conf``
column: values of ``-1`` (non-text rows) are excluded; the remaining word-level
scores (0–100) are averaged and normalised to [0, 1].  Falls back to 0.0 when no
positive-confidence words are found.
"""

from __future__ import annotations

import io
from typing import Any

from jera.adapters.parsing.routing import OcrResult


class TesseractOCREngine:
    """OCR engine backed by ``pytesseract`` / Google Tesseract.

    Parameters
    ----------
    lang:
        Default Tesseract language string (e.g. ``"kor+eng"``).  Can be
        overridden per-call via :meth:`recognize`.
    """

    engine_id = "tesseract"

    def __init__(self, lang: str = "kor+eng") -> None:
        self._default_lang = lang

    def recognize(self, image: bytes, lang: str = "kor+eng") -> OcrResult:
        """Recognise text in *image* bytes and return an :class:`OcrResult`.

        Parameters
        ----------
        image:
            Raw image bytes (PNG, JPEG, …).
        lang:
            Tesseract language string; overrides the instance default.
        """
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TesseractOCREngine requires 'pytesseract' and 'Pillow': `uv sync --extra ocr`."
            ) from exc

        pil_image = Image.open(io.BytesIO(image))

        text: str = pytesseract.image_to_string(pil_image, lang=lang)

        data: dict[str, list[Any]] = pytesseract.image_to_data(
            pil_image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )
        conf_values = [float(c) for c in data.get("conf", []) if float(c) >= 0]
        confidence = (sum(conf_values) / len(conf_values) / 100.0) if conf_values else 0.0

        return OcrResult(text=text, confidence=confidence, engine_id=self.engine_id)
