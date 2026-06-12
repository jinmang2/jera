"""SDK-boundary tests for real OCR engine adapters.

Each test injects fake library modules into ``sys.modules`` before constructing
the adapter, capturing the exact calls made and feeding canned responses — no
live OCR engines, no model files, no network calls required.

Covered engines
---------------
TesseractOCREngine  — mocks ``pytesseract`` + ``PIL``
RapidOcrOCREngine   — mocks ``rapidocr_onnxruntime``
ClovaOCREngine      — mocks ``requests``; disabled-by-default guard tested

Real-library smoke tests (``@pytest.mark.requires_extra``) are included at the
bottom and are auto-skipped when the optional extras are absent.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from typing import Any

import pytest

from jera.adapters.parsing.routing import OCREngine, OcrResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> types.ModuleType:
    """Register a fake module in sys.modules for the duration of the test."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _reload_engine(module_path: str) -> types.ModuleType:
    """Reload an adapter module so it picks up the freshly injected fakes."""
    mod = importlib.import_module(module_path)
    return importlib.reload(mod)


# ---------------------------------------------------------------------------
# Fake PIL.Image helper used by Tesseract tests
# ---------------------------------------------------------------------------


class _FakePilImage:
    """Minimal stand-in for a ``PIL.Image.Image`` object."""

    def __init__(self, data: bytes) -> None:
        self._data = data


def _make_pil_mod(captured_open: list[bytes]) -> types.ModuleType:
    """Build a fake ``PIL`` package + ``PIL.Image`` sub-module."""
    pil_mod = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")

    def _open(fp: Any) -> _FakePilImage:
        captured_open.append(fp.read())
        return _FakePilImage(b"img-data")

    image_mod.open = _open  # type: ignore[attr-defined]
    pil_mod.Image = image_mod  # type: ignore[attr-defined]
    return pil_mod


# ===========================================================================
# 1.  TesseractOCREngine
# ===========================================================================


class TestTesseractOCREngine:
    """Verify call shape and response parsing for TesseractOCREngine."""

    def _setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        image_to_string_result: str = "Hello World",
        conf_values: list[int] | None = None,
    ) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
        """Install fake pytesseract + PIL; return (engine, str_calls, data_calls)."""
        if conf_values is None:
            conf_values = [80, 90, 70]

        str_calls: list[dict[str, Any]] = []
        data_calls: list[dict[str, Any]] = []

        class _FakeOutput:
            DICT = "dict"
            STRING = "string"

        def _image_to_string(image: Any, lang: str = "eng", **kwargs: Any) -> str:
            str_calls.append({"image": image, "lang": lang, **kwargs})
            return image_to_string_result

        def _image_to_data(
            image: Any, lang: str = "eng", output_type: str = "string", **kwargs: Any
        ) -> dict[str, Any]:
            data_calls.append({"image": image, "lang": lang, "output_type": output_type, **kwargs})
            # Simulate Tesseract TSV output: mix of -1 (non-text) and real conf values
            return {"conf": [-1, *conf_values], "text": ["", *["word"] * len(conf_values)]}

        # Build fake pytesseract module
        _install(
            monkeypatch,
            "pytesseract",
            image_to_string=_image_to_string,
            image_to_data=_image_to_data,
            Output=_FakeOutput,
        )

        # Build fake PIL.Image module
        captured_open: list[bytes] = []
        pil_mod = _make_pil_mod(captured_open)
        monkeypatch.setitem(sys.modules, "PIL", pil_mod)
        monkeypatch.setitem(sys.modules, "PIL.Image", pil_mod.Image)

        import jera.adapters.parsing.ocr.tesseract_ocr as _tess_mod

        importlib.reload(_tess_mod)
        engine = _tess_mod.TesseractOCREngine(lang="kor+eng")
        return engine, str_calls, data_calls

    # --- Protocol conformance ---

    def test_implements_ocr_engine_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _, _ = self._setup(monkeypatch)
        assert isinstance(engine, OCREngine)

    def test_engine_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _, _ = self._setup(monkeypatch)
        assert engine.engine_id == "tesseract"

    # --- Request building ---

    def test_image_to_string_called_with_lang(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, str_calls, _ = self._setup(monkeypatch)
        engine.recognize(b"\x89PNG", lang="kor+eng")
        assert len(str_calls) == 1
        assert str_calls[0]["lang"] == "kor+eng"

    def test_image_to_data_called_with_dict_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _, data_calls = self._setup(monkeypatch)
        engine.recognize(b"\x89PNG", lang="kor+eng")
        assert len(data_calls) == 1
        assert data_calls[0]["output_type"] == "dict"
        assert data_calls[0]["lang"] == "kor+eng"

    def test_image_bytes_decoded_via_pil(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """image_to_string must receive a PIL image object, not raw bytes."""
        engine, str_calls, _ = self._setup(monkeypatch)
        engine.recognize(b"\x89PNG\r\nfake-image-data", lang="kor+eng")
        # The argument passed to image_to_string must be a _FakePilImage
        assert isinstance(str_calls[0]["image"], _FakePilImage)

    # --- Response parsing ---

    def test_recognize_returns_ocr_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _, _ = self._setup(monkeypatch)
        result = engine.recognize(b"\x89PNG")
        assert isinstance(result, OcrResult)

    def test_recognize_text_matches_image_to_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _, _ = self._setup(monkeypatch, image_to_string_result="안녕 World")
        result = engine.recognize(b"\x89PNG")
        assert result.text == "안녕 World"

    def test_confidence_averages_positive_conf_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # conf_values=[80,90,70] → mean=80 → /100 = 0.8
        engine, _, _ = self._setup(monkeypatch, conf_values=[80, 90, 70])
        result = engine.recognize(b"\x89PNG")
        assert result.confidence == pytest.approx(0.8)

    def test_negative_one_conf_excluded_from_average(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # -1 rows must be excluded; only [100] counts → confidence=1.0
        engine, _, _ = self._setup(monkeypatch, conf_values=[100])
        result = engine.recognize(b"\x89PNG")
        assert result.confidence == pytest.approx(1.0)

    def test_confidence_zero_when_all_conf_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _, _ = self._setup(monkeypatch, conf_values=[])
        result = engine.recognize(b"\x89PNG")
        assert result.confidence == pytest.approx(0.0)

    def test_confidence_in_zero_one_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _, _ = self._setup(monkeypatch, conf_values=[50, 60])
        result = engine.recognize(b"\x89PNG")
        assert 0.0 <= result.confidence <= 1.0

    def test_engine_id_in_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _, _ = self._setup(monkeypatch)
        result = engine.recognize(b"\x89PNG")
        assert result.engine_id == "tesseract"


# ===========================================================================
# 2.  RapidOcrOCREngine
# ===========================================================================


class TestRapidOcrOCREngine:
    """Verify call shape and response parsing for RapidOcrOCREngine."""

    def _setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        result_rows: list[list[Any]] | None = None,
    ) -> tuple[Any, list[Any]]:
        """Install fake rapidocr_onnxruntime; return (engine, captured_calls)."""
        if result_rows is None:
            result_rows = [
                [[[0, 0], [100, 0], [100, 20], [0, 20]], "Hello", 0.95],
                [[[0, 25], [100, 25], [100, 45], [0, 25]], "World", 0.85],
            ]

        captured_calls: list[Any] = []

        class _FakeRapidOCR:
            def __init__(self) -> None:
                pass

            def __call__(self, img: Any) -> tuple[list[list[Any]] | None, float]:
                captured_calls.append(img)
                return result_rows, 0.042

        _install(monkeypatch, "rapidocr_onnxruntime", RapidOCR=_FakeRapidOCR)

        import jera.adapters.parsing.ocr.rapidocr_ocr as _rapid_mod

        importlib.reload(_rapid_mod)
        engine = _rapid_mod.RapidOcrOCREngine()
        return engine, captured_calls

    # --- Protocol conformance ---

    def test_implements_ocr_engine_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(monkeypatch)
        assert isinstance(engine, OCREngine)

    def test_engine_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(monkeypatch)
        assert engine.engine_id == "rapidocr"

    # --- Request building ---

    def test_raw_bytes_passed_to_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, calls = self._setup(monkeypatch)
        img_bytes = b"\xff\xd8\xff\xe0fake-jpeg"
        engine.recognize(img_bytes)
        assert calls[0] is img_bytes

    def test_lang_not_forwarded_to_engine(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """RapidOCR does not take a lang param; verify no error raised."""
        engine, calls = self._setup(monkeypatch)
        engine.recognize(b"img", lang="kor+eng")
        assert len(calls) == 1  # called exactly once, no crash

    # --- Response parsing ---

    def test_recognize_returns_ocr_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(monkeypatch)
        result = engine.recognize(b"img")
        assert isinstance(result, OcrResult)

    def test_text_is_joined_lines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(
            monkeypatch,
            result_rows=[
                [None, "Line one", 0.9],
                [None, "Line two", 0.8],
            ],
        )
        result = engine.recognize(b"img")
        assert result.text == "Line one\nLine two"

    def test_confidence_is_average_of_scores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(
            monkeypatch,
            result_rows=[
                [None, "A", 0.9],
                [None, "B", 0.7],
            ],
        )
        result = engine.recognize(b"img")
        assert result.confidence == pytest.approx(0.8)

    def test_confidence_in_zero_one_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(monkeypatch)
        result = engine.recognize(b"img")
        assert 0.0 <= result.confidence <= 1.0

    def test_none_result_returns_empty_ocr_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(monkeypatch, result_rows=None)

        # Override internal engine to return None result
        captured: list[Any] = []

        class _FakeRapidOCRNone:
            def __init__(self) -> None:
                pass

            def __call__(self, img: Any) -> tuple[None, float]:
                captured.append(img)
                return None, 0.0

        _install(monkeypatch, "rapidocr_onnxruntime", RapidOCR=_FakeRapidOCRNone)

        import jera.adapters.parsing.ocr.rapidocr_ocr as _rapid_mod

        importlib.reload(_rapid_mod)
        engine2 = _rapid_mod.RapidOcrOCREngine()
        r = engine2.recognize(b"img")
        assert r.text == ""
        assert r.confidence == pytest.approx(0.0)
        assert r.engine_id == "rapidocr"

    def test_engine_id_in_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(monkeypatch)
        result = engine.recognize(b"img")
        assert result.engine_id == "rapidocr"

    def test_single_row_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(monkeypatch, result_rows=[[None, "Only line", 0.75]])
        result = engine.recognize(b"img")
        assert result.text == "Only line"
        assert result.confidence == pytest.approx(0.75)


# ===========================================================================
# 3.  ClovaOCREngine
# ===========================================================================


class TestClovaOCREngine:
    """Verify disabled-by-default guard, request building, and response parsing."""

    def _setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        fields: list[dict[str, Any]] | None = None,
        invoke_url: str = "https://ocr.example.com/api",
        secret: str = "my-secret-key",
    ) -> tuple[Any, list[dict[str, Any]]]:
        """Install fake requests; return (engine, captured_post_calls)."""
        if fields is None:
            fields = [
                {"inferText": "안녕하세요", "inferConfidence": 0.99},
                {"inferText": "World", "inferConfidence": 0.95},
            ]

        post_calls: list[dict[str, Any]] = []

        class _FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, Any]:
                return {
                    "version": "V2",
                    "images": [
                        {
                            "inferResult": "SUCCESS",
                            "fields": fields,
                        }
                    ],
                }

        class _FakeRequests:
            @staticmethod
            def post(url: str, headers: dict[str, str], data: str) -> _FakeResponse:
                import json as _json

                post_calls.append({"url": url, "headers": headers, "body": _json.loads(data)})
                return _FakeResponse()

        _install(monkeypatch, "requests", post=_FakeRequests.post)

        import jera.adapters.parsing.ocr.clova_ocr as _clova_mod

        importlib.reload(_clova_mod)
        engine = _clova_mod.ClovaOCREngine(
            invoke_url=invoke_url,
            secret=secret,
            enabled=True,
        )
        return engine, post_calls

    # --- Disabled-by-default guard ---

    def test_disabled_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, "requests")

        import jera.adapters.parsing.ocr.clova_ocr as _clova_mod

        importlib.reload(_clova_mod)
        with pytest.raises(RuntimeError, match="disabled by default"):
            _clova_mod.ClovaOCREngine()

    def test_missing_invoke_url_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, "requests")

        import jera.adapters.parsing.ocr.clova_ocr as _clova_mod

        importlib.reload(_clova_mod)
        with pytest.raises(RuntimeError, match="invoke_url"):
            _clova_mod.ClovaOCREngine(enabled=True, secret="s")

    def test_missing_secret_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, "requests")

        import jera.adapters.parsing.ocr.clova_ocr as _clova_mod

        importlib.reload(_clova_mod)
        with pytest.raises(RuntimeError, match="secret"):
            _clova_mod.ClovaOCREngine(enabled=True, invoke_url="https://x.example.com")

    # --- Protocol conformance ---

    def test_implements_ocr_engine_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(monkeypatch)
        assert isinstance(engine, OCREngine)

    def test_engine_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(monkeypatch)
        assert engine.engine_id == "clova"

    # --- Request building ---

    def test_post_sent_to_invoke_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, calls = self._setup(monkeypatch, invoke_url="https://ocr.ntruss.com/custom/v1/123")
        engine.recognize(b"\x89PNG")
        assert calls[0]["url"] == "https://ocr.ntruss.com/custom/v1/123"

    def test_x_ocr_secret_header_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, calls = self._setup(monkeypatch, secret="top-secret-key")
        engine.recognize(b"\x89PNG")
        assert calls[0]["headers"]["X-OCR-SECRET"] == "top-secret-key"

    def test_content_type_header_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, calls = self._setup(monkeypatch)
        engine.recognize(b"\x89PNG")
        assert calls[0]["headers"]["Content-Type"] == "application/json"

    def test_request_body_version_v2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, calls = self._setup(monkeypatch)
        engine.recognize(b"\x89PNG")
        assert calls[0]["body"]["version"] == "V2"

    def test_request_body_has_images_array(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, calls = self._setup(monkeypatch)
        engine.recognize(b"\x89PNG")
        body = calls[0]["body"]
        assert "images" in body
        assert len(body["images"]) == 1

    def test_request_body_image_data_is_base64(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import base64

        engine, calls = self._setup(monkeypatch)
        raw = b"\x89PNG\r\nfake-image-bytes"
        engine.recognize(raw)
        b64_sent = calls[0]["body"]["images"][0]["data"]
        assert base64.b64decode(b64_sent) == raw

    def test_lang_kor_plus_eng_maps_to_ko(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, calls = self._setup(monkeypatch)
        engine.recognize(b"\x89PNG", lang="kor+eng")
        assert calls[0]["body"]["lang"] == "ko"

    def test_lang_eng_maps_to_en(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, calls = self._setup(monkeypatch)
        engine.recognize(b"\x89PNG", lang="eng")
        assert calls[0]["body"]["lang"] == "en"

    def test_request_body_has_request_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, calls = self._setup(monkeypatch)
        engine.recognize(b"\x89PNG")
        body = calls[0]["body"]
        assert "requestId" in body
        assert isinstance(body["requestId"], str)

    def test_request_body_has_timestamp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, calls = self._setup(monkeypatch)
        engine.recognize(b"\x89PNG")
        body = calls[0]["body"]
        assert "timestamp" in body
        assert isinstance(body["timestamp"], int)

    # --- Response parsing ---

    def test_recognize_returns_ocr_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(monkeypatch)
        result = engine.recognize(b"\x89PNG")
        assert isinstance(result, OcrResult)

    def test_infer_texts_joined_with_newline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(
            monkeypatch,
            fields=[
                {"inferText": "안녕", "inferConfidence": 0.9},
                {"inferText": "하세요", "inferConfidence": 0.8},
            ],
        )
        result = engine.recognize(b"\x89PNG")
        assert result.text == "안녕\n하세요"

    def test_confidence_is_average_of_infer_confidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engine, _ = self._setup(
            monkeypatch,
            fields=[
                {"inferText": "A", "inferConfidence": 0.9},
                {"inferText": "B", "inferConfidence": 0.7},
            ],
        )
        result = engine.recognize(b"\x89PNG")
        assert result.confidence == pytest.approx(0.8)

    def test_confidence_in_zero_one_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(monkeypatch)
        result = engine.recognize(b"\x89PNG")
        assert 0.0 <= result.confidence <= 1.0

    def test_empty_fields_returns_empty_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(monkeypatch, fields=[])
        result = engine.recognize(b"\x89PNG")
        assert result.text == ""
        assert result.confidence == pytest.approx(0.0)
        assert result.engine_id == "clova"

    def test_engine_id_in_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = self._setup(monkeypatch)
        result = engine.recognize(b"\x89PNG")
        assert result.engine_id == "clova"


# ===========================================================================
# 4.  Real-library smoke tests (skipped without extras)
# ===========================================================================

_HAS_TESSERACT = (
    importlib.util.find_spec("pytesseract") is not None
    and importlib.util.find_spec("PIL") is not None
)
_HAS_RAPIDOCR = importlib.util.find_spec("rapidocr_onnxruntime") is not None


@pytest.mark.requires_extra
@pytest.mark.skipif(not _HAS_TESSERACT, reason="pytesseract/Pillow extras not installed")
def test_tesseract_engine_real_instantiates() -> None:
    """Smoke-test: TesseractOCREngine can be constructed with real libraries."""
    from jera.adapters.parsing.ocr.tesseract_ocr import TesseractOCREngine

    engine = TesseractOCREngine(lang="eng")
    assert engine.engine_id == "tesseract"
    assert isinstance(engine, OCREngine)


@pytest.mark.requires_extra
@pytest.mark.skipif(not _HAS_RAPIDOCR, reason="rapidocr_onnxruntime extra not installed")
def test_rapidocr_engine_real_instantiates() -> None:
    """Smoke-test: RapidOcrOCREngine can be constructed with real libraries."""
    from jera.adapters.parsing.ocr.rapidocr_ocr import RapidOcrOCREngine

    engine = RapidOcrOCREngine()
    assert engine.engine_id == "rapidocr"
    assert isinstance(engine, OCREngine)
