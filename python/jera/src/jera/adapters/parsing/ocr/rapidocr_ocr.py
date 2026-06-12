"""RapidOCR engine adapter — requires ``rapidocr_onnxruntime`` (ocr extra).

Wraps the ``rapidocr_onnxruntime.RapidOCR`` engine.  The engine accepts a
variety of image inputs (numpy array, file path, URL); we pass raw bytes
directly, which the library converts internally.

Call pattern (``rapidocr_onnxruntime`` ≤ 1.x / 2.x legacy API)::

    engine = RapidOCR()
    result, elapse = engine(img)

``result`` is either ``None`` (nothing detected) or a list of rows where each
row is ``[box, text, score]``:

* ``box``  – list of 4 corner coordinates (not used here)
* ``text`` – recognised string for this text region
* ``score``– float in [0, 1] (recognition confidence)

The adapter joins all ``text`` values with ``"\\n"`` and averages the per-row
``score`` values to produce a single ``confidence`` in [0, 1].
"""

from __future__ import annotations

from typing import Any

from jera.adapters.parsing.routing import OcrResult


class RapidOcrOCREngine:
    """OCR engine backed by ``rapidocr_onnxruntime``.

    The ONNX model is loaded lazily on first ``recognize`` (and cached), so constructing the
    engine — e.g. when the registry wires it — never requires the ``ocr`` extra; only actual
    recognition does. This matches the lazy-import discipline of the parser adapters. ``lang`` is
    kept for protocol symmetry; rapidocr handles language internally and takes no ``lang`` arg.
    """

    engine_id = "rapidocr"

    def __init__(self) -> None:
        self._engine: Any | None = None

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "RapidOcrOCREngine requires 'rapidocr_onnxruntime': `uv sync --extra ocr`."
                ) from exc
            self._engine = RapidOCR()
        return self._engine

    def recognize(self, image: bytes, lang: str = "kor+eng") -> OcrResult:
        """Recognise text in *image* bytes (lazily loading the model on first call).

        ``lang`` is accepted for protocol compliance; not forwarded to the engine.
        """
        result, _elapse = self._ensure_engine()(image)

        if not result:
            return OcrResult(text="", confidence=0.0, engine_id=self.engine_id)

        texts: list[str] = []
        scores: list[float] = []
        for row in result:
            # row: [box, text, score]
            texts.append(str(row[1]))
            scores.append(float(row[2]))

        text = "\n".join(texts)
        confidence = sum(scores) / len(scores) if scores else 0.0

        return OcrResult(text=text, confidence=confidence, engine_id=self.engine_id)
