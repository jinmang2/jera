---
name: jera-rag-architecture
description: Jera is a greenfield hexagonal offline-first RAG system at /home/jinmang2/jera
metadata: 
  node_type: memory
  type: project
  originSessionId: 81378544-4aea-4f92-8f99-8348882543e3
---

Jera (`/home/jinmang2/jera`) is a greenfield **hexagonal, offline-first RAG** system built 2026-06-12 via `/ralplan` consensus (Architect→Critic→Critic APPROVE). Plan/ADR: `.omc/plans/rag-redesign-plan.md`.

**Structure:** uv workspace. Domain in `python/jera/src/jera` (src-layout, package `jera`); FastAPI adapter-only in `apps/api/app` (package `jera-api`). Public facade: `import jera.rag`. Ports/adapters for parser, chunker, embedding, sparse, vector_store, metadata_store, reranker, generator.

**Profiles** (`JERA_PROFILE`, default `test`): `test` = fully deterministic offline (hash embedding, local BM25, in-memory vector store w/ RRF+DBSF fusion, SQLite, extractive generator — no Docker/torch/keys). `local` = fastembed ONNX (extra `local`). `prod` = Qdrant + Postgres + cloud, cloud disabled unless `JERA_ENABLE_CLOUD=1` + key.

**Env constraints:** Python 3.11 + uv present; Docker, psql, pnpm, ripgrep NOT available (so gates use pure-Python scans, not `rg`; storage abstracted behind ports).

**Gates:** `bash scripts/gates.sh` = ruff + ruff format + mypy strict (`mypy -p jera -p app`) + pytest. Fusion determinism frozen in `adapters/vector_store/fusion.py` (RRF k=60, 1-based ranks, missing=0, DBSF min-max, tie-break chunk_id asc) with golden test. Non-tautological hybrid-lift fixture frozen in `tests/unit/test_retrieval.py` (query "bravo india alpha").

**Milestones shipped (all on `main`):** M1 vertical slice (73ad12b) · M2 eval harness (800a8d0) · M3 semantic/hierarchical chunking + real Docling parser (209f8c3) · M4 Korean research RAG eval track + fastembed multilingual local + tool-use numeric QA (1793bf6, 155 tests). M4 built via `/deep-interview` → consensus plan → `/team` (3 workers). New in M4: `jera/tooluse/` (turn-based tool-use runtime + AST-safe CalculatorTool + FakeToolUseLLM + ToolAugmentedGenerator), `evaluation/{computation,matrix,gold_builder}.py`, EvalCase computation/table kinds. Plan `.omc/plans/rag-korean-research-toolqa-plan.md`, spec `.omc/specs/deep-interview-rag-korean-research-toolqa.md`, `QA_REPORT.md`.

**M4 corpus target:** Korean public-institution research reports (한국은행/KDI/자본시장연구원), freely-distributable only (copyright avoidance — sell-side analyst PDFs excluded; corpus PDFs git-ignored). Real ~20–30-case dataset is the user's opt-in build (`scripts/build_eval_dataset.py` + key).

**M5a shipped (b156623):** parser/OCR routing framework, LOCAL-first. Decisions: `DocumentParser` stays the only top-level port; `PageRouter`/`OCREngine` are adapter-internal Protocols (`adapters/parsing/routing.py`) composed inside `RoutingPdfParser` (per-page text|OCR routing, writes route/ocr provenance to element metadata, wired behind `use_routing_pdf` default off). `HwpxParser` (Hancom HWPX/OWPML) uses **stdlib zipfile+ElementTree, NOT lxml** (lxml is only in the docling extra, not base) — CI-real with a self-authored fixture. `parsing_metrics` (cer/table_f1/element_type_accuracy/reading_order Kendall-tau) + `parser_bench` graded vs a SEPARATE gold.json. Plan `.omc/plans/m5-parser-ocr-zoo-plan.md`. 171 tests.

User refs for M5: "한컴 그거" = **opendataloader-pdf** (Apache-2.0, Java11+ w/ pip binding; layout+table+OCR+VLM routing); HWP legacy via pyhwp; **rhwp** (Rust/WASM) is reference-only (SVG output, not a Python adapter). CLOVA OCR = disabled-by-default comparison only ("not yet" on paid). Env: Java 1.8 (opendataloader needs 11+), no tesseract → all heavy engines opt-in.

**M6 shipped (autopilot, 2026-06-13):** **Contextual Retrieval** (Anthropic 2024) + **RAGAS-lite gen-eval**, both offline/CI-real, default-off. Self-researched feature choice (user delegated). Decisions: new `Contextualizer` port (`ports/contextualizer.py`, batch `contextualize(document, chunks)->list[str]`); `Chunk` gained `context: str|None` + `embedding_text` property (`context + "\n\n" + text`) — `Chunk.text` NEVER mutated (citations/char_span intact), only indexing uses `embedding_text` (Contextual Embeddings + **Contextual BM25** — sparse `fit` + dense/sparse index both switched to `embedding_text`). `context` persisted via new nullable `ChunkRow.context` column (shared sql_store → SQLite+Postgres). Adapters: `HeuristicContextualizer` (title+section_path, deterministic, CI default-on when enabled) and `LlmContextualizer`(`SituateLLM` protocol; offline fake-tested) + `ClaudeSituateLLM` (opt-in, prompt-cached document block; `pragma: no cover`). Wired behind `use_contextual_retrieval` (default off) + `contextualizer_kind` (heuristic|llm; llm fails loud without cloud key). Non-tautological lift test (`test_contextual_retrieval.py`): answer chunk that never names "Acme" is WORST of 3 rival "outlook" chunks without context, BEST with it (idf-driven Contextual BM25). RAGAS-lite pure fns in `evaluation_contracts/generation_metrics.py`: `faithfulness`(containment≥0.6 sentence grounding), `answer_relevance`(cosine over supplied vectors), `answer_correctness`(multiset token-F1), `context_precision`(average-precision). Plan `.omc/plans/m6-contextual-retrieval-plan.md`. **198 tests** (was 171), all gates green offline.

**Next — M5b (opt-in zoo, follow-up increments):** opendataloader/camelot/pyhwp parser adapters; tesseract/rapidocr `OCREngine` impls (plug into RoutingPdfParser); `ClovaOCR` disabled-by-default; local VLM router (Qwen2-VL/GOT) behind `vlm` extra; real `scripts/parser_bench.py`. Each opt-in (lazy import + requires_extra/skipif, docling precedent). Then Surya/GOT/PaddleOCR/marker/MinerU as criterion-gated. See [[jera-build-conventions]].
