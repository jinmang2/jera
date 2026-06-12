# Implementation Plan: M6 — Contextual Retrieval + RAGAS-lite generation metrics

Status: **in progress (autopilot)**
Date: 2026-06-13
Builds on: M5a (`b156623`). Extends `Chunk`, `IngestPipeline`, `evaluation_contracts`.

## Why (self-research)

Two of the highest-leverage, paper-backed, *CI-realizable* gaps in Jera after M5a:

1. **Contextual Retrieval** (Anthropic, "Introducing Contextual Retrieval", Sept 2024).
   Chunks lose the document context they were embedded out of — a chunk reading
   "매출이 3% 증가했다" is unfindable by "Acme 매출" because it never names the company.
   The fix: prepend a short *situating context* (title + section breadcrumb, or an LLM
   one-liner) to each chunk **before embedding AND BM25 indexing** (Contextual Embeddings +
   Contextual BM25). Anthropic reports ~35% (embeddings) / ~49% (+ BM25 + rerank) fewer
   failed retrievals. This fits Jera's ports cleanly and is provable **offline-deterministic**
   with a non-tautological lift fixture (the M1 hybrid-lift discipline).

2. **RAGAS-lite generation metrics.** Jera evaluates *retrieval* (recall/MRR/nDCG) and
   citation-faithfulness, but not generated-answer quality. Add pure-function RAGAS-style
   contracts: `faithfulness`, `answer_relevance`, `answer_correctness`, `context_precision`.
   "Measure, don't assert" — deterministic, no IO, CI-testable.

## Principles (inherited)

- **One new port, swappable adapters.** `Contextualizer` is a port; `HeuristicContextualizer`
  (CI-real, deterministic, no LLM) is the default-on adapter when the feature is enabled;
  `LlmContextualizer` (Anthropic situate prompt) is opt-in and proven offline with a fake LLM
  (FakeToolUseLLM discipline). No domain code imports a vendor SDK.
- **Provenance preserved.** `Chunk.text` stays the *original* chunk text (citations/snippets
  unchanged). The situating context lives in a new optional `Chunk.context` field; indexing
  uses `Chunk.embedding_text` = `context + "\n\n" + text`. Char spans / element ids untouched.
- **Off by default.** `use_contextual_retrieval=False` → ingestion byte-for-byte unchanged.
- **CI = offline.** No paid call in any default path; the LLM contextualizer is opt-in.

## Scope

### Part A — Contextual Retrieval
- `domain/chunk.py`: `context: str | None = None` + `embedding_text` property.
- `ports/contextualizer.py`: `Contextualizer` Protocol — `strategy`, `version`,
  `contextualize(document, chunks) -> list[str]` (one context per chunk, order-parallel).
- `adapters/contextual/heuristic_contextualizer.py`: deterministic context from
  `document.title` + `chunk.section_path` (multilingual-neutral). CI-real.
- `adapters/contextual/llm_contextualizer.py`: Anthropic situate prompt over an injected
  `SituateLLM` callable; offline-testable with a fake; real Claude path opt-in.
- `pipeline/ingest.py`: optional `contextualizer`; attach context per chunk; **index and fit
  sparse over `embedding_text`** (Contextual Embeddings + Contextual BM25).
- `config/settings.py` + `config/registry.py`: `use_contextual_retrieval`,
  `contextualizer_kind` ("heuristic"); `_build_contextualizer`.
- `tests/unit/test_contextual_retrieval.py`: **non-tautological** — a target chunk that is NOT
  retrieved (or ranked below cutoff) without context, but IS with context. Genuine lift.
- `tests/unit/test_contextualizer.py`: heuristic determinism + llm fake-LLM wiring.

### Part B — RAGAS-lite generation metrics
- `evaluation_contracts/generation_metrics.py` pure functions:
  - `faithfulness(answer, contexts)` — sentence-level grounding (Jaccard ≥ threshold).
  - `answer_relevance(answer_vec, query_vec)` — cosine (vectors → pure/deterministic).
  - `answer_correctness(answer, reference)` — token-level F1.
  - `context_precision(ranked_ids, gold)` — average-precision (position-weighted).
- `tests/unit/test_generation_metrics.py`.

## Acceptance
- All gates green offline (ruff + format + mypy --strict + pytest), no new system deps.
- Contextual lift test proves a genuine, non-tautological retrieval improvement.
- Feature default-off; existing 171 tests unaffected.
- README milestone row + project memory updated.
