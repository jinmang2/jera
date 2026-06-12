# Deep Interview Spec: Korean Research-Report RAG — Eval Dataset, fastembed Local Profile, Tool-Use Numeric QA

## Metadata
- Interview ID: di-2026-06-12-rag-korean-research-toolqa
- Rounds: 9 (+ Round 0 topology)
- Final Ambiguity Score: ~19.5%
- Type: brownfield (Jera repo)
- Generated: 2026-06-12
- Threshold: 0.2
- Threshold Source: default
- Initial Context Summarized: no
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.85 | 0.35 | 0.298 |
| Constraint Clarity | 0.75 | 0.25 | 0.188 |
| Success Criteria | 0.80 | 0.25 | 0.200 |
| Context Clarity | 0.80 | 0.15 | 0.120 |
| **Total Clarity** | | | **0.805** |
| **Ambiguity** | | | **0.195** |

## Topology
| Component | Status | Description | Coverage / Deferral Note |
|-----------|--------|-------------|--------------------------|
| C1 Target dataset | active | Freely-distributable Korean public-institution research reports → labeled eval dataset (retrieval + table + computation cases) | AC1–AC5 |
| C2 fastembed local profile | active | Run `local` profile (multilingual ONNX models) on the Korean corpus; measure dense/sparse/hybrid × chunking | AC6–AC8 |
| C3 Tool-use numeric/table QA | active | Transparent model-native tool-use runtime (Claude) + calculator tool; FinQA-style numeric-accuracy eval drives it | AC9–AC12 |

## Goal
Build a **legally-safe Korean research-report RAG evaluation track** for Jera that (1) assembles a small labeled eval dataset from freely-distributable Korean public-institution research reports, (2) runs and measures the `local` (fastembed, multilingual) profile's dense/sparse/hybrid retrieval and chunking strategies on it, and (3) adds a **transparent, model-native tool-use runtime** so the system answers table-grounded numeric questions by *calling a calculator tool* (Program-of-Thoughts style) instead of doing arithmetic in the LLM — with the tool-call mechanism (trigger → dispatch → tool_result → loop) deliberately exposed and documented for learning. Paid API is minimized: retrieval/embedding stay 100% local; Claude is used only for dataset gold generation and the C3 answer/compute layer, opt-in by key.

## Constraints
- **Legal**: only freely-distributable reports (한국은행 · KDI · 자본시장연구원) may be committed. Reports are **manually downloaded, never scraped / bulk-collected**. Any copyrighted (sell-side) report is local-only and git-ignored — never committed, never redistributed.
- **Language**: Korean → retrieval requires a **multilingual** embedding model (e.g., BAAI/bge-m3 or multilingual-e5) via fastembed; `bm25_local` already tokenizes Unicode `\w+` so Korean sparse works.
- **Hardware**: WSL2, **CPU only** (no GPU). bge-m3 inference is slow → keep the corpus/eval small.
- **No training / fine-tuning** ("학습 기피") — off-the-shelf models + tool-use/prompting only.
- **Paid minimized**: Claude API only for (a) one-time dataset gold generation, (b) C3 answer/compute. Cloud adapters stay disabled-by-default; automated tests must not make live paid calls.
- Must not regress the existing offline CI gates (ruff/mypy/pytest must stay green without any extra/key).

## Non-Goals
- Model training, fine-tuning, or embedding-model distillation.
- Web scraping / automated bulk collection of any reports.
- Using or redistributing copyrighted sell-side analyst reports in the repo.
- A hard CI quality gate on retrieval metrics (this track is **descriptive/measured**, not pass/fail-gated).
- Live Qdrant/Postgres (still Docker-blocked; out of scope here).
- Production multi-tenant / UI work.

## Acceptance Criteria
- [ ] **AC1** Corpus: 5–8 freely-distributable Korean public-institution research-report PDFs, manually obtained, with source URLs + license notes recorded; copyrighted reports are git-ignored.
- [ ] **AC2** Ingestion of the corpus runs through the existing pipeline (Docling preferred for table fidelity via `JERA_USE_DOCLING=1`, PyMuPDF fallback) and produces typed elements incl. tables.
- [ ] **AC3** Eval dataset = LLM-assisted (Claude) generation of ~20–30 `EvalCase`s in three kinds: **RetrievalQ** (gold chunk), **TableQ** (gold table chunk), **ComputationQ** (gold supporting chunks + `expected_value`), including paraphrase queries. Persisted as JSON under the repo (only over committable corpus).
- [ ] **AC4** `ComputationQ.expected_value` is produced **deterministically** by our own calculator over the cited table numbers (ground truth independent of the LLM).
- [ ] **AC5** Dataset build is reproducible/documented; gold generation is one-time and cached (not re-run in CI).
- [ ] **AC6** `local` profile runs end-to-end on the Korean corpus with a fastembed **multilingual** dense model + multilingual reranker + local BM25 (model ids confirmed to exist in fastembed during planning).
- [ ] **AC7** `EvalRunner` produces a descriptive metric table: recall@k / MRR / nDCG for **dense / sparse / hybrid**, crossed with **heading_aware / semantic / hierarchical** chunking, on the Korean dataset.
- [ ] **AC8** Results are recorded (markdown report) with at least one narrative observation (e.g., where multilingual dense beats BM25 on paraphrase, where hybrid changes ranking).
- [ ] **AC9** A **`ToolUseRuntime`** implements a transparent model-native loop: pass tool JSON schemas → detect model `tool_use` → dispatch to a registered `Tool` → return `tool_result` → re-invoke → finalize. Loop steps are documented (README + inline comments) so the trigger→dispatch→result mechanism is legible.
- [ ] **AC10** A `calculator` / `evaluate_expression` `Tool` (safe numeric evaluation) is registered; a `ClaudeToolUseGenerator` (GeneratorLLM port, opt-in/`cloud`) drives it.
- [ ] **AC11** The runtime is **tested offline** with a `FakeToolUseLLM` that emits deterministic `tool_use` blocks, proving dispatch/loop mechanics with **zero API** — the real Claude run is opt-in (`requires_extra` + key) and skipped in default CI.
- [ ] **AC12** On ComputationQ cases, a **numeric execution-accuracy** metric compares the runtime's computed answer to `expected_value` (exact / tolerance); retrieval recall of the supporting chunk is also reported.
- [ ] **AC13** All existing gates stay green offline: `ruff`, `ruff format`, `mypy --strict`, `pytest` (no live paid calls, no required extras).

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| "Use analyst reports" = sell-side PDFs are usable | Those carry IP clauses (무단 복제·전재·자동수집 금지); ingest = 복제, dev gray-zone | Avoid the gray zone: use freely-distributable public-institution reports of the same shape; copyrighted ones local-only |
| Target language is English (bge-small default) | User picked Korean public institutions | Multilingual embedding required (bge-m3/m-e5); couples C1→C2 |
| "Validate local profile" = full benchmark | Contrarian: bge-m3 on CPU is slow — minimal valuable version? | Full measurement chosen, but small corpus (5–8 docs, ~20–30 queries), descriptive (no hard gate) |
| substring gold labeling is fine | Can't measure semantic/paraphrase advantage | LLM-assisted gold generation (paraphrase queries) + deterministic expected values |
| RAG = retrieve + extract | User requires table + numeric computation via tool calls | New component C3: model-native tool-use runtime + calculator; FinQA-style execution-accuracy |
| Computation can be done by the local profile | fastembed has no generator | Computation/answer layer needs a real LLM (Claude tool-use); retrieval stays local |
| Paid API is unacceptable | "유료 최소화하되 필요하면 호출" | Paid only for gold-gen + C3 answer; minimized; disabled-by-default in tests |
| Goal is the metric numbers | User: wants to *learn* trigger-capture / tool-call mechanics | C3 emphasis = transparent, documented model-native tool-use loop |

## Technical Context
- Brownfield Jera (uv workspace; `python/jera` domain + `apps/api`). Relevant surfaces: `config/settings.py` (profiles, `chunk_strategy`, `use_docling`), `adapters/embedding/fastembed_embedding.py`, `adapters/sparse/fastembed_sparse.py`, `adapters/ranking/fastembed_reranker.py`, `adapters/generator/claude_generator.py` (needs tool-use extension), `evaluation/` (EvalRunner, build_gold_dataset), `evaluation_contracts/` (metrics, dataset).
- New code expected: multilingual model wiring in the `local` profile; an `EvalCase` extension for TableQ/ComputationQ (`expected_value`); a `numeric_accuracy` metric; a `ToolUseRuntime` + `Tool` registry + `calculator` tool; `ClaudeToolUseGenerator` + `FakeToolUseLLM`; a dataset-build script (Claude-assisted, cached); a results markdown report.
- Method references (current best practice): model-native **function calling / tool use**; **Program-of-Thoughts / PAL** for numeric reasoning; **FinQA / ConvFinQA / TAT-QA** as the analogous table+text numeric-QA eval paradigm (execution accuracy). Verify exact fastembed multilingual model ids during planning.

## Ontology (Key Entities)
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| ResearchReport | core domain | source_inst, url, license, lang(ko) | parsed into Chunks |
| EvalCase | core domain | case_id, kind(retrieval/table/computation), query | has gold |
| GoldChunk | supporting | chunk_id, relevance | belongs to EvalCase |
| ExpectedValue | supporting | value, tolerance, formula | for ComputationQ |
| EmbeddingModel | external | model_id(bge-m3), dims, multilingual | used by local profile |
| Reranker | external | model_id(multilingual) | reranks candidates |
| ChunkStrategy | config | heading_aware/semantic/hierarchical | produces Chunks |
| ToolUseRuntime | core domain | loop, tool_registry | invokes Tool, calls LLM |
| CalculatorTool | core domain | name, json_schema, fn | registered in runtime |
| ToolUseLLM | external | Claude (real) / Fake (test) | emits tool_use |

## Ontology Convergence
| Round | Entity Count | New | Changed | Stable | Stability |
|-------|-------------|-----|---------|--------|-----------|
| 1 | 5 | 5 | - | - | N/A |
| 2 | 5 | 1 | 1 | 3 | 80% |
| 3 | 6 | 1 | 1 | 4 | 83% |
| 5 | 6 | 1 | 0 | 5 | 85% |
| 7 | 9 | 4 | 0 | 6 | (C3 introduced) |
| 9 | 10 | 1 | 0 | 9 | ~90% |

## Interview Transcript
<details>
<summary>Full Q&A (9 rounds + topology)</summary>

- **R0 Topology:** 2 components confirmed (later expanded to 3 at R6/R7).
- **R1 (C1 goal):** corpus domain/language → "research/analyst reports, publicly disclosed".
- **R2 (C1 constraint):** legal question about an IP clause → reproduction/scraping concerns; gray zone.
- **R3 (C1 constraint):** lock source → **Korean public institutions (한국은행·KDI·자본시장연구원)** → language=Korean → multilingual model.
- **R4 (C2 goal, Contrarian):** scope of "validate" → **full measurement + semantic chunking comparison**.
- **R5 (criteria):** gold labeling → **LLM-assisted (Claude) Q+gold generation**.
- **R6 (criteria/constraint, Simplifier):** minimal scale → **small + descriptive**; NEW requirement surfaced: **table questions + numeric computation + tool-call**.
- **R7 (topology+C3):** added C3; engine → user delegated to research, intuition "predefine calc, call as skill", **no training**.
- **R8 (approval/boundary):** explained PoT/function-calling/FinQA; user: PoT unknown, **minimize paid but call when needed**, wants to learn **trigger-capture / tool-call mechanics**.
- **R9 (C3 deliverable):** **transparent model-native tool-use loop**, documented.

</details>
