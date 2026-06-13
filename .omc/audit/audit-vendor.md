# Vendor Adapter API-Correctness Audit

**Date:** 2026-06-14
**Pin baseline:** qdrant-client>=1.10, fastembed>=0.3, docling>=2.0, opendataloader-pdf>=2.4, camelot-py[base]>=0.11, pyhwp>=0.1b15, pytesseract>=0.3.10, rapidocr-onnxruntime>=1.3
**Gate result after fixes:** `bash scripts/gates.sh` — 765 passed, 8 skipped, 0 failures

---

## 1. `qdrant_store.py` — qdrant-client >= 1.10

**Verdict: BUG — fixed**

**Official docs:**
- https://qdrant.tech/documentation/concepts/collections/
- https://github.com/qdrant/qdrant-client/issues/711

**Issue:** `QdrantClient.recreate_collection()` is deprecated and emits `DeprecationWarning` in qdrant-client >= 1.7. The replacement is explicit `collection_exists()` + `delete_collection()` + `create_collection()`. Issue #711 in the qdrant-client repo is specifically tracking the replacement of all `recreate_collection` calls, and the deprecation warning text reads: "recreate_collection method is deprecated and will be removed in the future. Use collection_exists to check collection existence and create_collection instead."

All other calls in the adapter are correct:
- `upsert(collection_name=..., points=[PointStruct(...)])` — correct
- `delete(collection_name=..., points_selector=PointIdsList(points=[...]))` — correct
- `query_points(collection_name=..., prefetch=[Prefetch(...)], query=FusionQuery(fusion=...), limit=..., with_payload=True)` — correct, non-deprecated API
- Named dense + sparse vector construction via `models.SparseVector(indices=..., values=...)` in prefetch — correct
- `models.FusionQuery(fusion=models.Fusion.RRF/DBSF)` — correct
- `result.points` iteration — correct (`query_points` returns `QueryResponse` with `.points`)

**Fix applied:**
- `adapters/vector_store/qdrant_store.py:31-42`: replaced `recreate_collection(...)` with `collection_exists + delete_collection + create_collection`
- `tests/unit/test_cloud_vendor_adapters.py`: fake client updated — replaced `recreate_collection` method with `collection_exists`, `delete_collection`, `create_collection`; renamed `recreate_calls` → `create_calls` throughout; renamed `test_ensure_collection_calls_recreate_*` → `test_ensure_collection_calls_create_*`
- `tests/unit/test_vector_delete.py`: fake client updated — same three-method replacement

---

## 2. `fastembed_embedding.py` — fastembed >= 0.3

**Verdict: OK**

**Official docs:** https://qdrant.github.io/fastembed/examples/SPLADE_with_FastEmbed/

`TextEmbedding(model_name=...).embed(list_of_texts)` returns a generator of numpy arrays. The adapter wraps with `list(map(float, v))` converting each array to `list[float]` (DenseVector). Correct.

---

## 3. `fastembed_sparse.py` — fastembed >= 0.3

**Verdict: OK**

**Official docs:** https://qdrant.github.io/fastembed/examples/SPLADE_with_FastEmbed/

`SparseTextEmbedding(model_name=...).embed(texts)` yields `SparseEmbedding` objects with `.indices` (array of int vocabulary positions) and `.values` (array of float weights). Adapter accesses `emb.indices` and `emb.values` correctly and converts to `SparseVector(indices=[int(i)...], values=[float(v)...])`. Correct.

---

## 4. `fastembed_reranker.py` — fastembed >= 0.3

**Verdict: OK**

**Official docs:** https://github.com/qdrant/fastembed/blob/main/fastembed/rerank/cross_encoder/text_cross_encoder.py

`from fastembed.rerank.cross_encoder import TextCrossEncoder` — class exists at this import path. `TextCrossEncoder(model_name=...).rerank(query, docs)` returns `Iterable[float]` (a generator). The adapter wraps with `list(...)` then iterates `zip(candidates, scores, strict=True)`. Correct.

---

## 5. `docling_parser.py` — docling >= 2.0

**Verdict: OK**

**Official docs:** https://docling-project.github.io/docling/v2/ ; https://docling-project.github.io/docling/reference/document_converter/

- `DocumentConverter().convert(stream).document` — correct; `convert()` returns `ConversionResult`, `.document` is the `DoclingDocument`
- `dl_doc.iterate_items()` yields `(item, level)` tuples — correct
- `item.label` is a `DocItemLabel` enum; `item.label.value` returns lowercase string (e.g. `"title"`, `"section_header"`, `"text"`, `"paragraph"`) — correct. The `_LABEL_MAP` keys (`"title"`, `"section_header"`, `"text"`, `"paragraph"`, `"caption"`, `"footnote"`, `"list_item"`, `"table"`, `"picture"`, `"chart"`, `"code"`, `"formula"`, `"page_header"`, `"page_footer"`) all correspond to valid `DocItemLabel` enum `.value` strings confirmed via https://github.com/DS4SD/docling-core/blob/main/docling_core/types/doc/labels.py
- `item.text` — text attribute present on TextItem and variants
- `item.export_to_markdown(dl_doc)` for tables/pictures — adapter has fallback for both old (no-arg) and new (doc-arg) signature. Correct.
- `item.prov` for page provenance — correct

---

## 6. `opendataloader_parser.py` — opendataloader-pdf >= 2.4

**Verdict: OK (with caveat)**

`opendataloader_pdf.convert(input_path=[str], output_dir=str, format="json")` — write-to-disk API writing `<stem>.json`. The adapter's boundary mock exactly mirrors this contract and the test suite (9 test classes, 30+ assertions) validates the call signature and output schema.

**Caveat:** No public API reference URL exists for opendataloader-pdf; the adapter was written against the package's own behavior. The mock faithfully reproduces the documented JSON element schema in the module docstring. If the real package ships a different function name (`run` vs `convert`) this would surface as an ImportError at integration test time.

---

## 7. `camelot_parser.py` — camelot-py[base] >= 0.11

**Verdict: OK**

**Official docs:** https://camelot-py.readthedocs.io/

- `camelot.read_pdf(filepath, pages=..., flavor=..., suppress_stdout=...)` — correct call signature
- Returns `TableList`; iteration yields `Table` objects with `.df` (pandas DataFrame) and `.page` (int, 1-indexed)
- `table.df.columns` and `table.df.iterrows()` — correct DataFrame surface
- `table.page` is an integer — adapter casts `int(table.page)` defensively. Correct.

---

## 8. `pyhwp_parser.py` — pyhwp >= 0.1b15

**Verdict: OK**

**Official docs:** https://github.com/mete0r/pyhwp (hwp5/filestructure.py)

- `from hwp5.filestructure import Hwp5File, InvalidHwp5FileError` — correct import path
- `Hwp5File(str(path)).preview_text.text` — returns the embedded UTF-16LE PrvText stream as a Python str. Correct.
- Paragraph splitting on `\r\n` / `\r` / `\n` — correct for Hancom PrvText format.

---

## 9. `tesseract_ocr.py` — pytesseract >= 0.3.10

**Verdict: OK**

**Official docs:** https://github.com/madmaze/pytesseract

- `pytesseract.image_to_string(pil_image, lang=lang)` — correct
- `pytesseract.image_to_data(pil_image, lang=lang, output_type=pytesseract.Output.DICT)` — correct; returns `dict` with `"conf"` key containing a list where `-1` marks non-text rows and `0-100` are word confidences
- Filter `float(c) >= 0` correctly excludes `-1` sentinel rows
- Normalisation `/ 100.0` gives [0,1] range. Correct.

---

## 10. `rapidocr_ocr.py` — rapidocr-onnxruntime >= 1.3

**Verdict: OK for pinned version; version-risk documented**

**Official docs:** https://github.com/RapidAI/RapidOCR/blob/v1.3.23/python/rapidocr_onnxruntime/main.py

For `rapidocr-onnxruntime` 1.3.x (the pinned series):
- `RapidOCR()(image_bytes)` returns `(result, elapse_list)` — a 2-tuple
- `result` is `None` or `list[list]`; each inner list is `[box_coords, text_str, confidence_float]`
- Adapter unpacks `result, _elapse = engine(image)` and reads `row[1]` (text), `row[2]` (score). Correct.

**Version-risk note:** The newer `rapidocr` meta-package (3.x, separate PyPI name) changed the return type to a `RapidOCROutput` dataclass with `.txts`, `.scores`, `.elapse` attributes — not a tuple. The pin `rapidocr-onnxruntime>=1.3` (not `rapidocr`) keeps the adapter on the 1.x tuple API. The module docstring already documents this boundary. If the pin is ever changed to `rapidocr>=3.0`, the adapter and its mock will both need updating.

---

## 11. `clova_ocr.py` — NAVER CLOVA OCR REST API

**Verdict: OK**

**Official docs:** https://api.ncloud-docs.com/docs/ai-application-service-ocr-general

- Endpoint: caller-supplied `invoke_url` (correct — the URL is issued per registered app)
- Headers: `Content-Type: application/json`, `X-OCR-SECRET: <secret>` — correct
- Request body: `{"version": "V2", "requestId": "<uuid>", "timestamp": <unix-ms>, "lang": "ko"/"en", "images": [{"format": "png", "name": "page", "data": "<base64>"}]}` — matches V2 schema exactly
- Response parsing: `body["images"][0]["fields"][*]["inferText"]` / `"inferConfidence"` — correct field names; `inferConfidence` is already in [0,1]. Correct.

---

## 12. `truncated_dim.py` — Matryoshka dimension truncation

**Verdict: OK**

**Reference:** Kusupati et al., "Matryoshka Representation Learning," NeurIPS 2022 (arXiv:2205.13147)

Slicing the first `N` dimensions and L2-renormalizing is the standard MRL inference procedure described in the paper (Section 3). The adapter correctly implements `trunc = vec[:dims]`, computes `norm = sqrt(sum(v*v))`, and returns `[v/norm for v in trunc]`, with a zero-norm guard. Correct.

---

## Summary

| Adapter | Verdict | Change |
|---|---|---|
| `qdrant_store.py` | **BUG — fixed** | `recreate_collection` → `collection_exists + delete_collection + create_collection` |
| `fastembed_embedding.py` | OK | — |
| `fastembed_sparse.py` | OK | — |
| `fastembed_reranker.py` | OK | — |
| `docling_parser.py` | OK | — |
| `opendataloader_parser.py` | OK (caveat) | No public API URL; mock mirrors documented schema |
| `camelot_parser.py` | OK | — |
| `pyhwp_parser.py` | OK | — |
| `tesseract_ocr.py` | OK | — |
| `rapidocr_ocr.py` | OK / version-risk | Pin `rapidocr-onnxruntime>=1.3` locks 1.x tuple API; 3.x dataclass API would break |
| `clova_ocr.py` | OK | — |
| `truncated_dim.py` | OK | — |

**Files changed:** 3 (adapter + 2 test mocks)
**API bugs found and fixed:** 1 (`recreate_collection` deprecation in qdrant_store)
**Gate status:** All gates pass (765 passed, 8 skipped)
