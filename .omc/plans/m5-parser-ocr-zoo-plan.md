# Implementation Plan: M5 — Parser/OCR Zoo + Page Routing + Bench (local-first)

Status: **pending approval**
Mode: consensus (ralplan)
Date: 2026-06-12
Builds on: M4 (`1793bf6`). Extends the existing `DocumentParser` port + `ParserRegistry`.

## Requirements Summary
Expand Jera's document-understanding layer with a **pluggable parser/OCR zoo**, a **page router** (the open-source "VLM routing" pattern, à la opendataloader-pdf / Hancom), and a **parser/OCR benchmark harness** — **local-offline-first, no paid by default**. CLOVA OCR is a disabled-by-default *comparison* adapter (not run, "not yet"). The whole framework + at least one genuinely-light real adapter must run in CI with no new system deps; heavy engines are opt-in extras (same discipline as docling/qdrant in M1–M4).

References (user-supplied): opendataloader-pdf (Apache-2.0, Java11+ with `pip install opendataloader-pdf` → JSON/markdown, layout+table+OCR+VLM routing), rhwp (Rust/WASM HWP viewer — reference only; Python path is `.hwpx` zip+lxml / pyhwp).

## RALPLAN-DR Summary

### Principles
1. **One top-level port; routing/OCR are parser internals.** `DocumentParser` stays the only domain port. `PageRouter`/`OCREngine` are adapter-internal Protocols composed inside `RoutingPdfParser` — never sibling top-level ports (rich parsers route+OCR internally, so top-level OCR/router ports would be dead code = leaky). Every engine is still a swappable adapter; no domain code imports a vendor lib directly.
2. **CI-real light core, heavy opt-in.** At least one real parser runs in CI with deps already present (`.hwpx` via `zipfile`+`lxml`). Everything torch/Java/system-binary/paid is behind an extra with `requires_extra`/`skipif` tests (docling pattern).
3. **Local-first, paid-not-yet.** No paid call in any default path. CLOVA OCR ships as a disabled-by-default adapter interface (mapping written, never invoked) so a future comparison is *possible* without enabling it now.
4. **Measure, don't assert.** A benchmark harness compares parsers/OCR on a small labeled fixture set (descriptive table, like M4's retrieval matrix) — not a hard CI gate.
5. **Router is heuristic-first.** Page routing decisions use cheap deterministic heuristics (text-coverage / image-area ratios from PyMuPDF) in CI; the VLM branch is an opt-in adapter, never required.

### Decision Drivers
1. CPU-only WSL2, Java 1.8, no tesseract → most real engines can't run here; the slice must be valuable while keeping heavy/paid runs user-triggered.
2. Korean corpus (M4 target) → `.hwpx` and Korean-capable OCR matter most; HWPX is light and pure-Python = the CI-real win.
3. Anti-scope-sprawl → a focused vertical slice (framework + ports + bench + 1 real adapter + opt-in mappings), not the entire zoo at once.

### Viable Options (first-slice scope)
- **Option A (CHOSEN): framework + 2 ports + bench harness + HWPX real adapter + opt-in mappings (opendataloader, camelot, OCR engines, CLOVA-disabled).** Pros: one CI-real adapter proves the framework end-to-end; heavy zoo lands as opt-in code without CPU/Java/paid cost. Cons: most adapters unverified-until-extra (documented, docling-style).
- **Option B: implement the whole zoo (Surya/GOT/PaddleOCR/marker/MinerU/VLM models) now.** Rejected: CPU-only can't run them, would bloat the slice with unrunnable code and no CI proof.
- **Option C: parser adapters only, no OCR/router abstraction.** Rejected: OCR + routing is the user's explicit ask ("ocr 여러 모델 테스팅", "vlm routing").

### Viable Options (HWP)
- **Option A (CHOSEN): `.hwpx` (modern, zip+XML) via stdlib `zipfile`+`lxml` — CI-real default; `.hwp` (legacy OLE) via `pyhwp` opt-in.** Pros: hwpx is light/present; covers modern Hangul docs.
- Option B: rhwp (Rust/WASM) — rejected as a Python adapter (outputs SVG, not structured text); kept as a reference.

## Architecture (extends existing ports/adapters)

**Key architect fix:** `DocumentParser` stays the **ONLY top-level port**. OCR and routing are NOT sibling ports — docling/opendataloader already route+OCR internally, so top-level OCR/router ports would be dead code on those paths (leaky abstraction). Instead they are **adapter-internal Protocols** composed inside one new `RoutingPdfParser: DocumentParser`. This makes the composition story explicit: rich parsers (docling/opendataloader) own their routing; `RoutingPdfParser` is the one parser that uses jera's router+OCR.

### Phase M5a (this build — CI-real vertical slice)
```text
python/jera/src/jera/
  domain/document.py        # + MediaType.HWPX, HWP; metadata provenance keys: route, ocr_engine, ocr_confidence
  adapters/parsing/
    routing.py              # NEW (adapter-internal Protocols, NOT ports/): PageFeatures dataclass + PageRouter.route(PageFeatures)
                            #     ->Route{TEXT|OCR|VLM,reason}; OCREngine.recognize(image,lang)->OcrResult{text,blocks,confidence,engine_id};
                            #     HeuristicRouter (deterministic threshold table, CI); FakeOCR (deterministic fixture map, CI)
    routing_pdf_parser.py   # NEW (CI via fakes): RoutingPdfParser(router, ocr, text_extractor) implements DocumentParser.parse;
                            #     supports()=PDF. Owns `_page_features(page)->PageFeatures` (pymupdf get_text/get_image_info — exact formulas
                            #     in AC8). Per-page router.route(features)->{TEXT: pymupdf text, OCR: ocr.recognize}; writes route/ocr
                            #     provenance into DocumentElement.metadata. Wired into _build_parsers behind `use_routing_pdf` (default False).
    hwpx_parser.py          # NEW (CI-REAL, ZERO deps): .hwpx (OWPML zip) via stdlib `zipfile` + `xml.etree.ElementTree`
                            #     (NOT lxml — lxml is only reachable via the docling extra, not base) -> typed elements + section paths + provenance
  evaluation_contracts/parsing_metrics.py   # NEW: cer, table_f1, element_type_accuracy, reading_order_score(Kendall tau) — pure funcs
  evaluation/parser_bench.py # NEW: PLUMBING/DETERMINISM check in CI — grades parser output vs a SEPARATE gold.json (not the input fixture)
  config/settings.py / registry.py          # + use_routing_pdf flag; register HwpxParser for .hwpx by default
python/jera/tests/fixtures/hwpx/sample.hwpx + sample.gold.json   # tiny self-authored fixture + INDEPENDENT gold labels
```

### Phase M5b (named follow-up increments — the opt-in zoo, each its own slice)
```text
adapters/parsing/opendataloader_parser.py  # extra: opendataloader (Java11+) — JSON->elements, carry route/ocr provenance into metadata
adapters/parsing/camelot_parser.py         # extra: camelot — PDF tables -> Table elements
adapters/parsing/hwp_parser.py             # extra: hwp (pyhwp) — legacy .hwp
adapters/parsing/ocr_engines/{tesseract,rapidocr}.py  # extra: ocr — real OCREngine impls (plug into RoutingPdfParser)
adapters/parsing/ocr_engines/clova.py      # extra: cloud, DISABLED-by-default — CLOVA comparison ("not yet")
adapters/parsing/vlm_router.py             # extra: vlm — local VLM (Qwen2-VL/GOT) PageRouter/OCR branch
scripts/parser_bench.py                    # opt-in REAL benchmark over real engines -> docs/eval/parser_bench_results.md
```
Each M5b adapter is opt-in (lazy import, raises "requires extra X", `requires_extra`/`skipif` tests) — the docling precedent. They are committed follow-ups with selection criteria, NOT vague deferrals.

## Acceptance Criteria (M5a — this build)
- [ ] **AC1** `PageRouter` + `OCREngine` are **adapter-internal Protocols** (in `adapters/parsing/routing.py`, NOT `ports/`); `HeuristicRouter` + `FakeOCR` are deterministic and CI-tested.
- [ ] **AC2** `RoutingPdfParser(router, ocr, text_extractor)` implements the `DocumentParser` port (`supports()` = `MediaType.PDF`) and is **wired into `_build_parsers` behind `use_routing_pdf: bool = False`** (mirrors `use_docling`). **Dispatch precedence:** when the flag is on, `RoutingPdfParser` is inserted BEFORE `PyMuPDFParser` for PDF (first-match-wins in `ParserRegistry`); default off so PDF ingestion is unchanged (no regression). A CI test ingests a PDF fixture through it (FakeOCR + HeuristicRouter) via `IngestPipeline.ingest` → `ParserRegistry.parse`, proving router→OCR/text composition end-to-end — the internals have a real consumer, not isolated scaffolding.
- [ ] **AC3** `RoutingPdfParser` writes route provenance into `DocumentElement.metadata` (`route`, `ocr_engine`, `ocr_confidence`); keys defined on the domain model. CI test asserts an OCR-routed element carries them.
- [ ] **AC4** `HwpxParser` parses a tiny self-authored `.hwpx` (OWPML zip) fixture into typed elements (sections/paras + at least one table) with provenance + section paths — runs in CI using **stdlib `zipfile` + `xml.etree.ElementTree` only** (NO lxml, NO extras; lxml is not in jera's base deps). Fixture committed as a binary blob alongside a `make_fixture.py` helper so reviewers can regenerate it. Docstring honestly scopes "proves the parser handles this OWPML shape, not all real Hancom output" (pymupdf-style honesty).
- [ ] **AC5** `MediaType.HWPX` (+ `HWP`) added; `ParserRegistry`/`registry` dispatch `.hwpx` to `HwpxParser` by default.
- [ ] **AC6** `parsing_metrics.py`: `cer` (char error rate, Levenshtein/len(ref)), `table_f1` (precision/recall over a set of (row,col,cell-text) triples), `element_type_accuracy` (fraction of aligned elements whose type matches gold), `reading_order_score` (Kendall tau over the **common-id intersection**; defined as 1.0 if ≤1 shared element, never throws on count mismatch) — pure functions, unit-tested on **hand-constructed input/gold pairs** (NOT on HwpxParser output, to stay honest).
- [ ] **AC7** `parser_bench.py`: CI **plumbing/determinism check** — grades parser output against a **SEPARATE** `sample.gold.json` (independent of the `.hwpx` input, not the fake's own output), renders a parser×metric table via `to_markdown()`. The word "benchmark" is reserved for the M5b opt-in real-engine script.
- [ ] **AC8** `PageFeatures` is a frozen dataclass produced by `_page_features(page) -> PageFeatures` inside `routing_pdf_parser.py`, with **exact formulas**: `has_text_layer = bool(page.get_text("text").strip())`; `image_area_ratio = min(1.0, sum(w*h for each img bbox from page.get_image_info()) / (page.rect.width*page.rect.height))`; `text_char_count = len(page.get_text("text"))`. `HeuristicRouter` routes deterministically with a **pinned threshold table**: `not has_text_layer and image_area_ratio>0.5 → OCR`; `has_text_layer → TEXT`; else (text-poor, low-image) → `OCR`. (VLM branch is opt-in/M5b; the heuristic router never emits VLM in M5a unless a `vlm` adapter is injected.) Unit-tested across all branches with synthetic `PageFeatures`.
- [ ] **AC9** Default `bash scripts/gates.sh` green with NO new extras, NO Java>8, NO tesseract, NO paid keys. **Add a base-only import smoke test** (`tests/unit/test_base_imports.py`) that imports `jera.adapters.parsing.hwpx_parser` + `routing_pdf_parser` + `routing` — catches any accidental non-base dep (e.g. the lxml class of leak) at CI time. Hexagonal boundary intact (no `app.*` domain imports); `import jera.rag` works.

## Acceptance Criteria (M5b — follow-up increments)
- [ ] **AC10** Opt-in parser/OCR adapters (opendataloader, camelot, hwp, tesseract, rapidocr, vlm) implement their port/Protocol, import lazily, raise a clear "requires extra X" without the extra; tests `requires_extra`/`skipif`. Real mapping code (opendataloader JSON→elements carrying route/ocr provenance into metadata, camelot tables→Table elements, rapidocr→OcrResult) — not bare `raise NotImplementedError`. New pyproject extras: `opendataloader`, `camelot`, `hwp`, `ocr`, `vlm`.
- [ ] **AC11** `ClovaOCR` (under `cloud` extra) is **disabled-by-default** (raises unless `enabled=True`+key); no default path calls it; test asserts the disabled default ("not yet" — comparison only).
- [ ] **AC12** No-evasion: genuinely-future engines (Surya/GOT-OCR2/PaddleOCR, marker/MinerU/open-parse, real local VLM model) are explicit follow-ups with a selection criterion each, not silent omissions.

## Implementation Steps — M5a only (this build; small, ~sequential)
- **S1 (CI-core foundation):** domain `MediaType.HWPX/HWP` + `DocumentElement.metadata` provenance keys; `adapters/parsing/routing.py` (PageRouter/OCREngine internal Protocols + HeuristicRouter + FakeOCR); `evaluation_contracts/parsing_metrics.py` (cer/table_f1/element_type_accuracy/reading_order) + tests. (land first)
- **S2 (RoutingPdfParser):** `adapters/parsing/routing_pdf_parser.py` composing router+OCR+pymupdf text-extract → DocumentParser; write route/ocr provenance into metadata; wire into `_build_parsers` behind `use_routing_pdf`; CI test through `IngestPipeline` with fakes. (depends S1)
- **S3 (HWPX real):** `adapters/parsing/hwpx_parser.py` + `tests/fixtures/hwpx/sample.hwpx` + `sample.gold.json` (independent labels) + tests; register `.hwpx` dispatch; honesty docstring. (depends S1)
- **S4 (bench plumbing + config):** `evaluation/parser_bench.py` (grades vs separate gold.json) + settings/registry wiring + tests. (depends S1, S2, S3)

**M5b** (opt-in zoo: opendataloader/camelot/hwp/tesseract/rapidocr/clova/vlm + real bench script) is a **separate follow-up build** — not in this slice (architect rec.2: deferring loses no CI value since they're unverified-until-extra, and removes ~9 files of review surface from this slice).

## Risks and Mitigations
| Risk | Mitigation |
|---|---|
| opendataloader needs Java 11+ (env has 1.8) | opt-in extra; adapter raises clear "Java 11+ required"; tests skipif; user upgrades JDK to run |
| OCR/VLM models are heavy/torch, can't run in CI | OCREngine port + FakeOCR deterministic CI default; real engines opt-in scripts; bench runs on fakes in CI |
| `lxml` not in base deps (only via docling extra) → red CI | HwpxParser uses **stdlib `xml.etree.ElementTree` + `zipfile`** (zero deps); base-only import smoke test (AC9) guards against recurrence |
| `PageFeatures` ambiguity (two devs compute image-area differently) | exact formulas + pinned threshold table in AC8; `_page_features` owned by `routing_pdf_parser.py` |
| `RoutingPdfParser` vs PyMuPDF/Docling both claim PDF | `use_routing_pdf` default False; when on, inserted before PyMuPDF in the registry (documented precedence, AC2) |
| `MediaType.HWP` enum added but no `.hwp` parser in M5a | documented: `.hwp` source raises "no parser supports" until M5b `hwp_parser` (pyhwp) — expected, not a regression |
| HWPX format variance (sections/tables) | start with the documented OWPML structure (zip + `Contents/section*.xml`, namespaced `<hp:p>/<hp:tbl>`); tiny self-authored fixture + `make_fixture.py`; honesty docstring on coverage limits |
| Accidental paid call (CLOVA) | disabled-by-default (enabled=True+key required), tested; never in default registry path |
| Scope sprawl | first slice = framework + HWPX-real + opt-in mappings; Surya/GOT/MinerU/marker/local-VLM are criterion-gated follow-ups |
| `.hwpx` fixture authoring | generate a minimal valid hwpx (zip of mimetype + minimal section XML) in a small committed fixture or a build helper |

## Verification Steps
1. `bash scripts/gates.sh` green with no extras/keys/Java>8/tesseract (AC8).
2. CI tests: HWPX parses the fixture into typed elements (AC2); FakeOCR + HeuristicRouter deterministic (AC1/AC7); parsing metrics + bench table (AC6).
3. `requires_extra` adapters: import-without-extra raises the documented error; tests skipped in default CI (AC4); CLOVA disabled default (AC5).
4. Boundary gate (no `app.*`), `import jera.rag` (AC9).
5. Opt-in real runs (manual): `uv sync --extra ocr --extra camelot --extra opendataloader` + `scripts/parser_bench.py` → `docs/eval/parser_bench_results.md` (not CI).

## ADR
- **Decision:** Add a local-first document-understanding expansion. Keep `DocumentParser` the only top-level port; introduce routing/OCR as **adapter-internal Protocols** composed in one `RoutingPdfParser`. **M5a** (this build) ships the framework + `RoutingPdfParser` (CI-tested via fakes through `IngestPipeline`) + `.hwpx` CI-real parser + parsing metrics + a bench *plumbing check* graded against independent gold. **M5b** lands the opt-in zoo (opendataloader, camelot, hwp, tesseract/rapidocr/CLOVA-disabled, local VLM) as separate follow-up increments behind extras.
- **Drivers:** CPU-only/Java-1.8/no-tesseract env; Korean corpus (HWPX matters); local-first + paid-not-yet; anti-sprawl.
- **Alternatives considered:** whole-zoo-now (rejected: unrunnable on CPU); rhwp for HWP (rejected as Python adapter: SVG output; reference only); parsers-without-OCR/router (rejected: misses the explicit ask); CLOVA enabled (rejected: paid "not yet" — ships disabled).
- **Why chosen:** maximizes CI-real value (HWPX) while keeping the heavy/paid surface opt-in and measurable, reusing the proven docling/qdrant extra pattern and the M4 benchmark-harness shape.
- **Consequences:** many adapters are unverified-until-extra (documented); bench is a script, not a gate; HWPX coverage starts minimal.
- **Follow-ups (criterion-gated):** all of M5b (opendataloader/camelot/hwp/tesseract/rapidocr/CLOVA/VLM); then Surya/GOT-OCR2/PaddleOCR, marker/MinerU/open-parse, a real local VLM (Qwen2-VL/GOT) behind `vlm`, CLOVA enablement for a paid comparison run, `.hwp` legacy hardening, real labeled HWPX/PDF bench fixtures.

## Changelog (consensus improvements applied)
Architect pass (SOUND-WITH-CHANGES) — all 5 folded in:
1. **No leaky top-level ports:** `OCREngine`/`PageRouter` demoted from `ports/` to adapter-internal Protocols composed inside `RoutingPdfParser: DocumentParser` (rich parsers route+OCR internally → top-level ports would be dead code). `DocumentParser` remains the sole domain port.
2. **Slice cut:** first build = M5a (framework + RoutingPdfParser + HWPX-real + metrics + bench-plumbing); the opt-in zoo (opendataloader/camelot/hwp/tesseract/rapidocr/clova/vlm, ~9 files) deferred to M5b follow-up — no CI value lost (unverified-until-extra anyway).
3. **Provenance contract:** `DocumentElement.metadata` keys `route`/`ocr_engine`/`ocr_confidence` defined so OCR/routed elements (and the M5b opendataloader mapping) don't discard routing signal.
4. **Bench honesty:** CI bench grades against a SEPARATE `sample.gold.json` (not the input fixture / fake's own output — avoids tautology); relabeled a plumbing/determinism check; "benchmark" reserved for the M5b opt-in real-engine script. `reading_order_score` defined as Kendall tau (or dropped).
5. **HWPX honesty docstring:** `HwpxParser` explicitly scopes what the synthetic OWPML fixture proves vs real Hancom output (pymupdf precedent); `table_f1` claims gated on real nested-cell extraction.

Also: `RoutingPdfParser` wired into `_build_parsers` behind `use_routing_pdf` so the new internals have a real `IngestPipeline` consumer (not isolated scaffolding) — addresses the integration-evidence gap.

Critic pass (ITERATE → APPROVE) — BLOCKER + MAJOR + minors folded in:
- **BLOCKER (lxml):** `lxml` is NOT in jera base deps (only via the `docling` extra), so HWPX would fail clean CI. Switched HwpxParser to **stdlib `xml.etree.ElementTree` + `zipfile`** (zero deps); added a base-only import smoke test (AC9) to prevent recurrence. AC4/AC9/architecture/risks all updated; no lxml references remain.
- **MAJOR (PageFeatures):** defined `PageFeatures` dataclass + `_page_features(page)` producer with exact pymupdf formulas + pinned `HeuristicRouter` threshold table (AC8).
- **Precedence gap:** AC2 now states `use_routing_pdf` default False + RoutingPdfParser-before-PyMuPDF dispatch order (no PDF regression).
- **MINORs:** fixture committed + `make_fixture.py`; `reading_order_score` defined over common-id intersection (never throws); `table_f1`/metrics unit-tested on hand-constructed pairs not parser output; `.hwp` no-handler-in-M5a documented.
