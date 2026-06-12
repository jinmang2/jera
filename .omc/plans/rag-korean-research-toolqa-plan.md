# Implementation Plan: Korean Research-Report RAG — Eval Dataset, fastembed Local, Tool-Use Numeric QA

Status: **pending approval**
Mode: consensus (--direct), non-interactive
Source spec: `.omc/specs/deep-interview-rag-korean-research-toolqa.md` (ambiguity 19.5%, PASSED)
Date: 2026-06-12

## Requirements Summary
Deliver three components on top of the existing Jera RAG:
- **C1** — a small, legally-safe Korean public-institution research-report eval dataset (retrieval + table + computation cases) with LLM-assisted gold and deterministic expected values.
- **C2** — run and measure the `local` (fastembed multilingual) profile on that corpus across dense/sparse/hybrid × heading/semantic/hierarchical, producing a descriptive results report.
- **C3** — a transparent, model-native tool-use runtime (calculator tool, Program-of-Thoughts style) that answers table-grounded numeric questions, offline-testable with a deterministic FakeToolUseLLM, with Claude as the opt-in real engine; FinQA-style numeric-accuracy eval drives it.

## RALPLAN-DR Summary

### Principles
1. **Offline-deterministic core, paid opt-in edge** — retrieval/embedding and the tool-use *loop mechanics* must run and be tested with zero paid calls; Claude is used only at the edges (gold generation, real C3 answers) behind keys.
2. **Legal avoidance over legal argument** — only freely-distributable reports are committed; copyrighted PDFs are git-ignored; never scrape/bulk-collect.
3. **Ground truth independent of the model** — computation gold (`expected_value`) is computed by our own calculator over cited numbers, never trusted from the LLM.
4. **Ports, not forks** — C3 plugs into the existing `GeneratorLLM` port; the local profile changes are config/model-id, not new pipelines.
5. **Transparent mechanism** — the tool-use loop (detect `tool_use` → dispatch → `tool_result` → re-invoke → finalize) is small, commented, and documented for learning, not hidden in an SDK abstraction.

### Decision Drivers
1. CPU-only WSL2 + Korean → multilingual ONNX models are slow → corpus/eval must be small and cached.
2. The user wants to *learn* tool-call mechanics → the loop must be legible and offline-testable, not a black box.
3. CI must stay green with no extras/keys → every new capability needs a deterministic offline test path.

### Viable Options (C3 answer/compute engine)
- **Option A (CHOSEN): model-native tool-use loop, our own thin runtime, Claude as opt-in LLM, FakeToolUseLLM for tests.** Pros: transparent/learnable, offline-testable, reuses GeneratorLLM port, PoT-style exact compute. Cons: we own the loop code (small surface).
- **Option B: rely on the Anthropic SDK's higher-level agent/tool helpers.** Rejected: hides the mechanism the user explicitly wants to learn; couples tests to the SDK; harder to run offline.
- **Option C: deterministic compute-only (no LLM tool use).** Rejected as the *answer engine* (the user wants genuine tool-calling), but **adopted for gold generation** (expected_value) per Principle 3.

### Viable Options (C2 multilingual model)
- **Option A (CHOSEN): `BAAI/bge-m3` dense (1024-dim, fastembed-supported, strong Korean) + `BAAI/bge-reranker-v2-m3` + existing `bm25_local` sparse.** Pros: one multilingual family, fastembed-native. Cons: 1024-dim, heavier than bge-small.
- **Option B: `intfloat/multilingual-e5-large`.** Kept as fallback if bge-m3 isn't in the installed fastembed's supported list (verified at execution step S0).

## Acceptance Criteria
Inherited verbatim from the spec AC1–AC13 (see `.omc/specs/deep-interview-rag-korean-research-toolqa.md` §Acceptance Criteria). Each implementation step below maps to specific ACs. Summary of the testable gates:
- Offline CI (`ruff`, `ruff format`, `mypy --strict`, `pytest`) stays green with no extras/keys (AC13).
- Tool-use loop proven offline via `FakeToolUseLLM` (AC11); numeric-accuracy metric on ComputationQ (AC12).
- Local-profile matrix report produced over the Korean dataset (AC7–AC8).
- Dataset cases of all three kinds with deterministic expected values (AC3–AC4).

## Implementation Steps

### S0. Pre-flight model verification (AC6) — gating, ~15 min
- `uv sync --extra local`; in a scratch run, `from fastembed import TextEmbedding, TextCrossEncoder` and print `TextEmbedding.list_supported_models()` / `TextCrossEncoder.list_supported_models()`.
- Confirm `BAAI/bge-m3` (dense) + `BAAI/bge-reranker-v2-m3` (rerank) exist; else fall back to `intfloat/multilingual-e5-large` / `BAAI/bge-reranker-base`. Record chosen ids + dims in the plan changelog.
- Files: none committed; result recorded.

### S1. Corpus intake + manifest (C1 / AC1–AC2)
- Add `data/corpus/` (git-ignored PDFs) and committed `data/corpus/manifest.json`: `[{inst, title, url, license, sha256, lang:"ko"}]` for 5–8 한국은행/KDI/자본시장연구원 reports (manually downloaded; no scraping).
- `.gitignore`: ignore `data/corpus/*.pdf`; keep manifest + `data/eval/`.
- `scripts/fetch_corpus.py` (optional helper): reads manifest, verifies sha256 of locally-present PDFs (does NOT auto-download in bulk — prints instructions). Files: `data/corpus/manifest.json`, `scripts/fetch_corpus.py`, `.gitignore`.
- **Done-check (critic Missing): `data/` is currently untracked (clean slate) — no `git rm --cached` needed; verify with `git ls-files data/` before committing the ignore rules.**
- **Docling table done-check (critic Missing, AC2): before authoring any TableQ case (S3), confirm at least one corpus PDF yields a `Table` element under `JERA_USE_DOCLING=1` ingestion** (so TableQ gold is satisfiable and not rejected like an unsatisfiable substring label).

### S2. EvalCase extension for table/computation (C1 / AC3–AC4)
- Extend `python/jera/src/jera/evaluation_contracts/dataset.py`: add `CaseKind` enum (`retrieval|table|computation`), fields on `EvalCase`: `kind`, optional `expected_value: float|None`, `tolerance: float`, `formula: str|None`, optional `cited_numbers: list[float]`, keep `gold: list[GoldChunk]` as supporting chunks. Backward-compatible defaults (`kind=retrieval`). **Attribution (architect rec.5):** add `source_inst`, `source_url`, `license` to each `EvalCase` (carried from the manifest) so committed verbatim excerpts of attribution-licensed reports ship *with* their required attribution.
- Add `numeric_accuracy(answer_value, expected_value, tolerance)` to `evaluation_contracts/metrics.py`. **Tolerance semantics (critic Missing):** FinQA-style **relative** tolerance — `|a-e| <= tolerance * max(|e|, 1.0)` (the `max(..,1)` floor keeps small/zero expected values sane); default `tolerance=0.001`. Documented in the docstring + unit-tested. Files: `dataset.py`, `metrics.py`, tests in `tests/unit/test_metrics.py`.

### S3. LLM-assisted dataset builder (C1 / AC3–AC5)
- `python/jera/src/jera/evaluation/gold_builder.py`: `ClaudeGoldGenerator` (opt-in, `cloud` extra + key) that, per report's chunks, prompts Claude to emit `{kind, query (incl. paraphrase), supporting_chunk_ids, (for computation) cited_numbers + operation}`.
- **Deterministic ground truth (Principle 3):** for computation cases, our code computes `expected_value` from `cited_numbers` + `operation` via the S5 calculator — the LLM never supplies the final number.
- **Operand provenance guard (architect rec.4 + critic Missing):** before committing a computation case, assert **each `cited_number` actually appears in the referenced supporting chunk text**, matching against a **normalized** form of both the chunk text and the operand — strip thousands separators (`1,234`→`1234`), normalize Korean units/scale (`억`/`조`/`%`/`pp`) and full/half-width digits, and allow a small float tolerance. Without this normalization the guard false-rejects every real Korean-formatted number; with it, hallucinated operands are still caught. Reject the case otherwise.
- **Attribution propagation (architect rec.5):** each emitted case carries `source_inst`/`source_url`/`license` from the manifest into the committed JSON.
- `scripts/build_eval_dataset.py`: runs the builder once, writes cached `data/eval/korean_research.json` (committed). CI never calls Claude — it reads the cached JSON. Files: `gold_builder.py`, `scripts/build_eval_dataset.py`, `data/eval/korean_research.json` (committed artifact).

### S4. Local-profile multilingual wiring + matrix eval (C2 / AC6–AC8)
- **Hard prerequisite (architect rec.2):** `_build_embedding` in `config/registry.py:132-136` currently constructs `FastEmbedEmbedding()` **bare**, ignoring settings. Fix it to pass `settings.embedding_model`, AND add the **S0-chosen** dense id → dims to the map in `fastembed_embedding.py:15` (`bge-m3`=1024; if S0 falls back to `multilingual-e5-large` that is also 1024). If the dims map is wrong, the `InMemoryVectorStore` dim guard (in_memory.py:57-61) raises on first search — this is a gate, not a nicety. `FastEmbedReranker.__init__` already accepts `model_name` (confirmed) — wire `settings.reranker_model` through `_build_reranker`; the S0-chosen reranker (`bge-reranker-v2-m3`, fallback `bge-reranker-base`) is set there.
- `config/settings.py`: add `embedding_model: str|None`, `reranker_model: str|None` (override defaults); `FastEmbedEmbedding`/`FastEmbedReranker` accept `model_name` — wire through `registry._build_embedding/_build_reranker`.
- `python/jera/src/jera/evaluation/matrix.py`: `run_matrix(dataset, *, strategies, modes, profile)` builds a system per chunk strategy (re-ingesting the corpus per strategy) and returns a nested `EvalReport`-like matrix; `to_markdown()` renders recall@k/MRR/nDCG for dense/sparse/hybrid × {heading_aware, semantic, hierarchical}.
  - **Caching honesty (architect rec.2):** chunk strategy changes chunk boundaries → chunk text → embeddings, so embeddings **cannot** be reused across strategies. Feasibility rests on **small corpus (5–8 docs)**, not cross-strategy caching. The only sharing is the in-memory ONNX model weights (one `FastEmbedEmbedding` instance reused across strategies within a single `run_matrix` call). No content-hash embedding cache is introduced; the earlier "cache embeddings per (strategy,model)" claim is **dropped** as inexpressible against the re-embedding `IngestPipeline`.
- `scripts/eval_local_matrix.py`: runs the matrix under `JERA_PROFILE=local`, writes `docs/eval/korean_research_results.md` with ≥1 narrative observation. Files: `settings.py`, `registry.py`, `matrix.py`, `scripts/eval_local_matrix.py`, `docs/eval/korean_research_results.md`.

### S5. Tool-use runtime + calculator (C3 / AC9–AC10) — **turn-based contract (architect rec.1)**
- New package `python/jera/src/jera/tooluse/`:
  - `tools.py`: `Tool` protocol (`name`, `json_schema`, `run(args)->str`); `CalculatorTool` using an **AST-based safe evaluator** (`ast.parse`, allow `Constant(num)/BinOp/UnaryOp/+−*/%**()`, no names/calls/attrs) — no `eval()`.
  - `llm.py`: `ToolUseLLM` yields a **full assistant turn, not a single step** (mirrors the real Anthropic stateful loop). `AssistantTurn = {blocks: list[TextBlock|ToolUseBlock{id,name,args}], stop: "tool_use"|"end_turn", raw: object}` where `raw` is the opaque provider payload the adapter **replays verbatim** (preserves Opus `thinking` blocks). Protocol: `step(messages: list) -> AssistantTurn`. `FakeToolUseLLM` is deterministic and emits **two fixture behaviors**: (a) trivial single `tool_use`→`end_turn` (README walkthrough), and (b) a **multi-block turn** (text + two `tool_use` blocks in one turn) so the id-correlation / parallel-result path is genuinely covered. `ClaudeToolUseGenerator` (opt-in/`cloud`) maps Anthropic `response.content` blocks → `AssistantTurn` and keeps `raw=response.content` for verbatim replay.
  - `runtime.py`: `ToolUseRuntime` owns a real mutable `messages: list`. Loop (kept ≤~20 lines, heavily commented for legibility): `turn = llm.step(messages)` → append `{"role":"assistant","content": turn.raw}` → for each `tool_use` block dispatch to the registered tool, collect `tool_result[]` keyed by `tool_use_id` → append `{"role":"user","content": tool_results}` → repeat until `stop=="end_turn"` (cap N iters). Returns `RunResult{answer_text, tool_calls[], final_value: float|None}`. `tooluse/README.md` explains trigger→dispatch→result, the verbatim-replay invariant, and contrasts model-native function-calling vs harness-side keyword/router triggers.
- Files: `tooluse/{__init__,tools,llm,runtime}.py`, `tooluse/README.md`.
- Before implementing: consult the `claude-api` skill for the canonical tool-use loop (append `assistant.content` verbatim → `tool_result` with matching `tool_use_id` → loop to `stop_reason=="end_turn"`).

### S6. Generator integration + offline test (C3 / AC11)
- `python/jera/src/jera/adapters/generator/tool_augmented_generator.py`: `ToolAugmentedGenerator(llm, tools)`.
  - **Typed numeric path (architect rec.3):** keep `generate(query, contexts) -> Answer` **port-conformant** — citations are *chunk* citations only (honors the `query.py:96-98` citation-resolution assertion; the calculator is never cited as a chunk). Do **NOT** add a float to the frozen `Answer`.
  - Expose a **separate** typed method `run(query, contexts) -> RunResult{answer, final_value: float|None, tool_calls}`. S7's computation eval calls `run()` and reads `final_value` directly — the number stays typed, never regex-extracted from prose (preserves Principle 3).
- `config/settings.py` + `registry._build_generator`: `generator_kind: "extractive"|"tooluse"`; tooluse builds `ToolAugmentedGenerator(FakeToolUseLLM(), [CalculatorTool()])` for `test`, `ClaudeToolUseGenerator` for `cloud`+key.
- Tests `tests/unit/test_tooluse.py`: FakeToolUseLLM drives a full loop on a computation prompt → asserts the tool was dispatched, `tool_result` fed back, and the final answer equals the deterministically computed value — **zero API**. Files: `tool_augmented_generator.py`, `settings.py`, `registry.py`, `tests/unit/test_tooluse.py`.

### S7. Computation eval wiring (C3 / AC12)
- `evaluation/computation.py` (new sibling) scores ComputationQ. **Generator access (critic MINOR-1):** `run()` is NOT on the `GeneratorLLM` port (`ports/generator.py` exposes only `generate()`), so the computation eval must **hold the concrete `ToolAugmentedGenerator` directly** — construct it from the registry's `generator_kind="tooluse"` branch (or a typed factory), never reach through the port-typed `QueryPipeline._generator`. It calls `gen.run(query, contexts)` (typed path, rec.3) and scores `numeric_accuracy(run_result.final_value, case.expected_value, case.tolerance)` + supporting-chunk recall@k.
- Keep the existing retrieval-only `EvalRunner.run` shape intact (runner.py:23-39) — computation scoring is a **separate** function, not a mutation of the retrieval path. `requires_extra` Claude run is opt-in; CI uses `FakeToolUseLLM`. Files: `evaluation/computation.py`, `tests/unit/test_eval_runner.py`.

### S8. Docs + gates (AC13)
- Update `README.md` (eval track + tool-use), `scripts/gates.sh` unchanged (still green offline). Run full gate set; iterate to green. Files: `README.md`.

## Risks and Mitigations
| Risk | Mitigation |
|---|---|
| bge-m3 fastembed id/dim mismatch | S0 verification gate before any C2 code; documented fallback to multilingual-e5-large |
| CPU inference too slow | small corpus (5–8 docs); single in-memory ONNX model instance reused across strategies within one `run_matrix` call; re-embed per strategy (chunk text differs across strategies — no cross-strategy embedding reuse); matrix is a script, not CI |
| Committing copyrighted report text | only commit gold snippets from freely-distributable institutions; PDFs git-ignored; manifest records license |
| `eval()` injection in calculator | AST allowlist evaluator, no names/calls/builtins; unit-tested with hostile inputs |
| Claude tool-use API shape drift | consult the `claude-api` skill before S5/S6; pin `anthropic` SDK in the `cloud` extra; FakeToolUseLLM keeps tests SDK-independent |
| Paid calls leaking into CI | cloud generators disabled-by-default; `gold_builder`/Claude run are scripts/opt-in; cached dataset JSON committed so CI never calls |
| LLM arithmetic in gold | expected_value computed by our calculator from cited numbers, not the LLM (Principle 3) |
| LLM hallucinates operands (not just arithmetic) | S3 operand-provenance guard: every cited_number must appear in the supporting chunk text or the case is rejected |
| Offline fake proves a divergent loop (fake-path) | turn-based `ToolUseLLM` contract + a multi-block FakeToolUseLLM fixture so the id-correlation/parallel path is exercised; adapter replays `raw` assistant blocks verbatim (thinking-block safe) |
| Numeric value lost/garbled through `Answer` | typed `RunResult.final_value` path; `Answer` stays frozen & port-conformant (no float field, no regex extraction) |
| Attribution-licensed excerpts committed without attribution | per-case `source_inst`/`source_url`/`license` carried into committed JSON |

## Verification Steps
1. `bash scripts/gates.sh` green with no extras/keys (AC13).
2. `pytest tests/unit/test_tooluse.py` proves the loop offline (AC11) and numeric correctness (AC12) via FakeToolUseLLM.
3. Hostile-input tests on CalculatorTool (no code execution).
4. `uv run python scripts/eval_local_matrix.py` (manual, `--extra local`) produces `docs/eval/korean_research_results.md` with the dense/sparse/hybrid × chunking table (AC7–AC8).
5. `data/eval/korean_research.json` contains ≥20 cases across all three kinds, each computation case with a deterministic `expected_value` (AC3–AC4).
6. Boundary gate still passes (no `app.*` domain imports); `import jera.rag` works.

## ADR
- **Decision:** Add a Korean research-report eval track to Jera: a small legally-safe labeled dataset (retrieval/table/computation), a multilingual `local`-profile measurement matrix, and a transparent model-native tool-use runtime (calculator, PoT-style) with Claude as the opt-in engine and a deterministic FakeToolUseLLM for offline tests.
- **Drivers:** CPU-only + Korean (small/cached); learn-the-mechanism intent (legible offline loop); CI must stay paid-free.
- **Alternatives considered:** SDK agent helpers (rejected: hides mechanism, SDK-coupled tests); compute-only answer engine (rejected as answer engine, adopted for gold ground truth); multilingual-e5 (kept as model fallback); copyrighted analyst PDFs (rejected: legal gray zone).
- **Why chosen:** maximizes offline testability and learnability, keeps paid surface tiny and opt-in, reuses the existing ports, and produces honest numeric-accuracy measurement independent of the LLM.
- **Consequences:** we own a small tool-use loop; the matrix eval is a script (not CI) due to model weight; committed dataset is small; bge-m3 is heavier (1024-dim).
- **Follow-ups:** larger dataset + hard CI gate once stable; local instruct-LLM tool-use as a future fully-offline C3 engine; live Qdrant/PG when Docker is available.

## Changelog (consensus improvements applied)
Architect pass (SOUND-WITH-CHANGES) — all 5 folded in:
1. **C3 loop = turn-based, not step-based** (S5): `ToolUseLLM.step()->AssistantTurn{blocks,stop,raw}`; runtime owns mutable `messages`, replays `raw` verbatim (Opus thinking-block safe); FakeToolUseLLM gains a multi-block fixture so AC11 proves the *real* loop, not a linearization.
2. **Embedding-cache claim dropped / made honest** (S4): no cross-strategy embedding reuse (chunk text differs); feasibility rests on small corpus; hard prereq added — `_build_embedding` must pass `settings.embedding_model` and dims map must gain `bge-m3=1024` (else in-memory dim guard raises).
3. **Typed numeric path** (S6/S7): `ToolAugmentedGenerator.generate->Answer` stays port-conformant (chunk citations only); a separate `run()->RunResult{final_value}` carries the typed number; frozen `Answer` unchanged.
4. **Operand-provenance guard** (S3): each cited_number must appear in the supporting chunk text or the case is rejected.
5. **Attribution propagation** (S1–S3): per-case `source_inst`/`source_url`/`license` committed alongside excerpts.

Critic pass (ITERATE → APPROVE) — 1 MAJOR + 3 MINOR + 4 Missing folded in:
- **MAJOR**: removed the stale "cache embeddings per (strategy,model)" mitigation in the risk table (contradicted S4's dropped-cache decision).
- **MINOR**: S7 computation eval holds the concrete `ToolAugmentedGenerator` directly (`run()` isn't on the port), not via `QueryPipeline._generator`; fixed `EvalRunner.run` line ref (runner.py:23-39); qualified `config/registry.py`.
- **Missing**: tolerance = relative w/ `max(|e|,1)` floor; operand guard normalizes commas/units/percent/width (else false-rejects Korean numbers); AC2 Docling-table done-check before TableQ authoring; `data/` clean-slate git note; S0-chosen dims/reranker propagate to S4 (e5-large=1024 fallback).
