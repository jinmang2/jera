<div align="center">

# 🌾 Jera

**A hexagonal, offline-first RAG system — built port-by-port, gate-by-gate.**

*Every external capability is a `Protocol` port with swappable adapters. The whole pipeline
runs and is tested end-to-end with **zero external services, zero paid keys, zero GPU**.*

`Python 3.11` · `uv workspace` · `pydantic v2` · `SQLAlchemy 2.0` · `FastAPI` · `ruff + mypy --strict` · **536 tests**

</div>

---

## Why Jera

Most RAG codebases couple domain logic to a web framework and a vendor SDK, then can't be
tested without a running database and an API key. Jera inverts that:

- **Hexagonal core.** Domain logic lives in `python/jera` (src-layout package `jera`).
  `apps/api` is a FastAPI **adapter only**. Parsing, chunking, embedding, sparse encoding,
  vector store, reranking, generation, OCR/page-routing — each is a port with adapters.
- **Offline-deterministic by default, real models by opt-in.** The `test` profile is fully
  deterministic (hash embeddings, local BM25, in-memory RRF/DBSF fusion, SQLite, extractive
  generator). Heavy/paid engines (fastembed, Qdrant, Postgres, Claude, Docling, OCR) are
  **opt-in extras, disabled by default** — CI never makes a paid call or needs a service.
- **No fake paths.** Tooling that looks real is real: the tool-use loop mirrors the actual
  Anthropic message protocol (proven offline with a multi-block fake LLM); the hybrid-retrieval
  test proves a *genuine* fusion lift, not a rigged cosine.

## Milestones (all on `main`)

| | Milestone | Highlights |
|---|---|---|
| **M1** | Hexagonal vertical slice | ports/adapters, profiles, in-memory RRF/DBSF fusion, FastAPI, golden-file determinism |
| **M2** | Evaluation harness | `EvalRunner` + gold dataset builder (recall@k / MRR / nDCG / citation-faithfulness) |
| **M3** | Chunking + parsing | semantic (embedding-breakpoint) + hierarchical (RAPTOR-lite) chunkers; **real Docling** parser |
| **M4** | Korean research RAG | computation/table eval cases; **fastembed multilingual** (bge-m3); **tool-use numeric QA** (Program-of-Thoughts, FinQA-style) |
| **M5a** | Parser/OCR routing | `RoutingPdfParser` (per-page text\|OCR + provenance); **HWPX parser (stdlib)**; parser benchmark harness |
| **M5b** | Parser/OCR zoo | opt-in adapters — **opendataloader** · **camelot** tables · **pyhwp** (legacy .hwp) · **tesseract/rapidocr/CLOVA** OCR engines; functional **VLM route**; runnable `parser_bench.py` (all SDK-boundary-verified, `requires_extra`) |
| **M6** | Contextual retrieval + gen-eval | Anthropic **Contextual Retrieval** (situate chunks → Contextual Embeddings + Contextual BM25; deterministic heuristic in CI, Claude opt-in); **RAGAS-lite** generation metrics (faithfulness / answer-relevance / answer-correctness / context-precision), wired into the eval harness |
| **M7** | Retrieval & answer quality | **MMR** diversity reranker (λ relevance/diversity); **multi-query retrieval** (rule-based decomposition + **HyDE** opt-in, RRF-fused); generation metrics in the strategy matrix |
| **M8** | Hardening (no stopgaps) | every opt-in cloud/vendor adapter **verified by SDK-boundary tests** (real request/response logic, no keys); real `ClaudeToolUseGenerator` wired + **`pause_turn`** handled; real per-model **pricing** (no `cost_metadata` placeholder); **genuine offline Korean eval dataset** (real chunks/gold, replacing the scaffold) |
| **M9** | Lifecycle + observability | **idempotent re-ingest** (no duplicates) + document **delete** (cascades to vectors); document/job API (`GET /documents`, `GET/DELETE /documents/{id}`, `GET /jobs/{id}`); per-query **timing + cost** stats on `/query` |

Built with a disciplined loop: **`/deep-interview` → consensus plan (Planner→Architect→Critic) → execution (direct or `/team`) → independent code review.** Plans/specs/QA live in `.omc/`.

## Architecture

```
python/jera/src/jera/
  domain/                 # pure models: Document, Chunk, ScoredChunk, Answer, ... (no IO)
  ports/                  # Protocols: parser, chunker, embedding, sparse, vector_store,
                          #            metadata_store, reranker, generator
  adapters/
    parsing/              # markdown, pymupdf, docling[extra], hwpx(stdlib), routing_pdf
    chunking/             # heading_aware, semantic, hierarchical
    embedding/ sparse/    # hash + bm25 (CI) · fastembed[extra] · openai[cloud]
    vector_store/         # in-memory RRF/DBSF (CI) · qdrant[extra]
    metadata_store/       # SQLite (CI) · postgres[extra]
    ranking/ generator/   # identity + extractive (CI) · cohere/claude[cloud] · tool-augmented
  tooluse/                # transparent model-native tool-use runtime + AST-safe calculator
  pipeline/               # IngestPipeline, QueryPipeline
  evaluation/             # EvalRunner, matrix, computation, gold/parser benchmarks
  evaluation_contracts/   # metric pure-functions (retrieval + parsing)
  config/                 # Settings (profiles) + ProviderRegistry
  rag/                    # public facade:  import jera.rag
apps/api/app/             # FastAPI adapter: routers / DI / schemas
```

## Quickstart

```bash
uv sync                        # install the workspace (both packages, editable)
bash scripts/gates.sh          # ruff + ruff format + mypy --strict + pytest  (171 passed, offline)
uv run python scripts/eval.py  # demo eval: retrieval table + RAGAS-lite generation table
uv run uvicorn app.main:app --reload --app-dir apps/api   # serve the API
```

### API

```bash
curl -s localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"source_id":"d1","media_type":"text/markdown","text":"# Title\n\nHybrid retrieval uses reciprocal rank fusion."}'

curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"query":"what does hybrid retrieval use?","top_k":3}'   # response includes per-stage timing + cost stats

curl -s localhost:8000/documents                  # list ingested documents (+ chunk counts)
curl -s localhost:8000/jobs/<job_id>              # poll an ingestion job
curl -s -X DELETE localhost:8000/documents/<id>   # delete a document (cascades to its vectors)
```

## Profiles & configuration

Set via `JERA_*` env vars (default profile = `test`).

| profile | embedding | sparse | vector store | metadata | rerank | generate |
|---|---|---|---|---|---|---|
| `test`  | hash (deterministic) | BM25 local | in-memory | SQLite `:memory:` | identity | extractive |
| `local` | fastembed bge-m3 `[local]` | fastembed SPLADE `[local]` | in-memory | SQLite file | bge-reranker `[local]` | extractive / tool-use |
| `prod`  | OpenAI `[cloud]`* | (sparse) | Qdrant `[qdrant]` | Postgres `[postgres]` | Cohere `[cloud]`* | Claude `[cloud]`* |

- **Chunking** — `JERA_CHUNK_STRATEGY` ∈ `heading_aware` (default) · `semantic` · `hierarchical`.
- **Parsing** — `JERA_USE_DOCLING=1` (layout/table/OCR, `[docling]`) · `JERA_USE_ROUTING_PDF=1`
  (per-page text\|OCR routing with provenance) · `JERA_USE_OPENDATALOADER=1` (`[opendataloader]`,
  Java 11+) · `JERA_USE_CAMELOT=1` (table extraction, `[tables]`). HWPX parses with **stdlib only**;
  legacy `.hwp` via `pyhwp` (`[hwp]`). OCR route engine: `JERA_OCR_ENGINE=fake|tesseract|rapidocr|clova`.
- **Generator** — `JERA_GENERATOR_KIND=tooluse` enables the calculator-tool numeric-QA path.
- **Contextual retrieval** — `JERA_USE_CONTEXTUAL_RETRIEVAL=1` situates each chunk before
  indexing (Anthropic, 2024); `JERA_CONTEXTUALIZER_KIND` ∈ `heuristic` (title+section, offline)
  · `llm` (Claude-written, `[cloud]`). `Chunk.text` is never mutated — only `embedding_text` is.
- **Multi-query retrieval** — `JERA_USE_QUERY_TRANSFORM=1` expands the query and RRF-fuses the
  per-variant rankings; `JERA_QUERY_TRANSFORM_KIND` ∈ `rule_based` (clause decomposition, offline)
  · `hyde` (HyDE hypothetical-answer, `[cloud]`).
- **Reranker** — `JERA_RERANKER_KIND=mmr` swaps the identity reranker for **MMR** diversity
  (`JERA_MMR_LAMBDA`, 1.0 = pure relevance, lower = more diverse).
- `*` cloud adapters are disabled unless `JERA_ENABLE_CLOUD=1` + the matching key.

```bash
uv sync --extra local --extra qdrant --extra postgres --extra cloud --extra docling
```

## Quality gates

`scripts/gates.sh` (CI) runs fully offline and enforces, among others: package boundary
(no `app.*` domain imports; src-layout install proof), typed-element parsing with provenance,
chunk stability, dense/sparse/hybrid retrieval with a **non-tautological fusion-lift**,
golden-file RRF/DBSF determinism (k=60, 1-based ranks, chunk_id tie-break), storage ownership,
disabled-by-default paid providers, an end-to-end cited-answer test through FastAPI, and a
base-only import smoke test so no heavy dep can sneak into the offline core.

## Project context

`.omc/plans/` (consensus plans + ADRs) · `.omc/specs/` (deep-interview specs) ·
`.omc/memory/` (durable project memory, restorable on a new machine) · `QA_REPORT.md`
(last verification). Next up: **M5b** — the opt-in parser/OCR zoo
(opendataloader-pdf, camelot, pyhwp, tesseract/rapidocr, CLOVA-disabled, local VLM router).

---

<div align="center"><sub>Built with Claude Code — deep-interview → consensus → team → review.</sub></div>
