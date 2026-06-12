"""NAVER CLOVA OCR engine adapter — DISABLED by default (paid cloud service).

Calls the NAVER Cloud Platform General OCR REST API:

* **Endpoint**: the ``invoke_url`` issued when you register a CLOVA OCR app
  (e.g. ``https://…apigw.ntruss.com/custom/v1/…/general``).
* **Headers**: ``Content-Type: application/json``, ``X-OCR-SECRET: <secret>``.
* **Request body** (JSON)::

    {
      "version": "V2",
      "requestId": "<uuid>",
      "timestamp": <unix-ms>,
      "lang": "ko",
      "images": [{"format": "png", "name": "page", "data": "<base64>"}]
    }

* **Response**: ``images[0].fields[*].inferText`` / ``inferConfidence``.

The adapter joins all ``inferText`` values with ``"\\n"`` and averages the
``inferConfidence`` values (already in [0, 1]) to produce a single
``OcrResult``.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any

from jera.adapters.parsing.routing import OcrResult

_LANG_MAP: dict[str, str] = {
    "kor": "ko",
    "kor+eng": "ko",
    "eng": "en",
}


def _tesseract_lang_to_clova(lang: str) -> str:
    """Convert a Tesseract-style lang string to a CLOVA lang code."""
    return _LANG_MAP.get(lang, "ko")


class ClovaOCREngine:
    """OCR engine backed by the NAVER CLOVA OCR REST API.

    Disabled by default — live calls require a paid NAVER Cloud account.

    Parameters
    ----------
    invoke_url:
        The CLOVA OCR invoke URL (issued in the NAVER Cloud console).
    secret:
        The ``X-OCR-SECRET`` key for the registered CLOVA OCR app.
    enabled:
        Must be set to ``True`` explicitly to allow instantiation.
        Prevents accidental live calls in tests or CI.
    image_format:
        Image format hint sent in the request body (default ``"png"``).
    """

    engine_id = "clova"

    def __init__(
        self,
        invoke_url: str = "",
        secret: str = "",
        enabled: bool = False,
        image_format: str = "png",
    ) -> None:
        if not enabled:
            raise RuntimeError(
                "ClovaOCREngine is disabled by default. Pass enabled=True, invoke_url, "
                "and secret (paid live calls; never enabled in automated tests)."
            )
        if not invoke_url:
            raise RuntimeError("ClovaOCREngine requires invoke_url when enabled.")
        if not secret:
            raise RuntimeError("ClovaOCREngine requires secret when enabled.")
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ClovaOCREngine requires 'requests': `uv sync --extra ocr`.") from exc
        self._requests = requests
        self._invoke_url = invoke_url
        self._secret = secret
        self._image_format = image_format

    def recognize(self, image: bytes, lang: str = "kor+eng") -> OcrResult:
        """Call the CLOVA OCR API and return an :class:`OcrResult`.

        Parameters
        ----------
        image:
            Raw image bytes; encoded as base64 in the request body.
        lang:
            Tesseract-style language string; converted to CLOVA lang code.
        """
        clova_lang = _tesseract_lang_to_clova(lang)
        payload = {
            "version": "V2",
            "requestId": str(uuid.uuid4()),
            "timestamp": int(time.time() * 1000),
            "lang": clova_lang,
            "images": [
                {
                    "format": self._image_format,
                    "name": "page",
                    "data": base64.b64encode(image).decode("ascii"),
                }
            ],
        }
        headers = {
            "Content-Type": "application/json",
            "X-OCR-SECRET": self._secret,
        }
        resp = self._requests.post(
            self._invoke_url,
            headers=headers,
            data=json.dumps(payload),
        )
        resp.raise_for_status()
        body: dict[str, Any] = resp.json()

        images: list[Any] = body.get("images", [])
        fields: list[dict[str, Any]] = []
        if images:
            fields = images[0].get("fields", [])

        if not fields:
            return OcrResult(text="", confidence=0.0, engine_id=self.engine_id)

        texts = [str(f.get("inferText", "")) for f in fields]
        confs = [float(f.get("inferConfidence", 0.0)) for f in fields]

        text = "\n".join(texts)
        confidence = sum(confs) / len(confs) if confs else 0.0

        return OcrResult(text=text, confidence=confidence, engine_id=self.engine_id)
