# Jera RAG — Rewrite-Level Architecture Plan (Consensus)

Status: **approved (consensus) — Milestone 1 implemented & all gates green (2026-06-12)**
Consensus: Architect SOUND-WITH-CHANGES → Critic ITERATE → Critic APPROVE (all changes folded in).
Date: 2026-06-12
Source research: user-supplied "RAG Real Redesign Architecture" report
Confirmed decisions (user): in-memory/SQLite dev + Qdrant/PG prod via ports; E2E vertical slice first; offline-local-first providers; consensus plan → development.

---

## 0. Context & Environment Reality

- Working dir `/home/jinmang2/jera` is **greenfield** (only `.omc/` and an empty `.omx/specs/` exist). No prior `apps/api/app`, no `.omx/specs/...`. The research report is a forward-looking design, not a description of existing code — so there is **no migration**, we build clean.
- Toolchain present: **Python 3.11.15, uv 0.11.17, Node v24**.
- Toolchain MISSING: **Docker, psql, pnpm**. → Qdrant + Postgres cannot run locally now. This is *the* driving constraint behind the ports/adapters decision: the entire pipeline must run E2E with zero external services.

---

## 1. RALPLAN-DR Summary

### Principles
1. **Ports over implementations.** Every external capability (parse, embed, sparse-encode, store, rerank, generate) is a `Protocol` port; adapters are swappable by config. No domain code imports a vendor SDK directly.
2. **Local-deterministic by default, cloud by opt-in.** Tests/CI use deterministic local fakes (no network, no torch, no services). Real local models (fastembed/ONNX) for dev. Paid/cloud adapters exist but are disabled unless keys + profile are set.
3. **Structure-first documents.** Documents are typed element trees with provenance, never flat strings. Chunking and citation depend on this.
4. **Provenance everywhere.** Every chunk carries `document_id, source_id, page_span, section_path, element_ids, char_span, token_count, chunk_strategy, chunk_version, parent_chunk_id`; every answer carries citations back to chunk IDs.
5. **Domain package independent of the API.** `jera.rag` is importable and testable with no FastAPI. `apps/api/app` is adapter-only.

### Decision Drivers (top 3)
1. No Docker/Postgres/Qdrant available → must be runnable & fully testable offline.
2. "Latest tech + clean structure" is the explicit goal → modern stack (uv, pydantic v2, fastembed/ONNX, Qdrant hybrid, SQLAlchemy 2.0), hexagonal layering.
3. Anti-scaffold-failure → real wired E2E slice, not fake cosine paths; the 7 acceptance gates are mandatory, non-deferrable.

### Viable Options (package boundary)
- **Option A (CHOSEN): repo-root `python/jera` src-layout + adapter-only `apps/api/app`.** Pros: domain reusable by future worker/CLI/eval; src-layout prevents accidental cwd imports; clean. Cons: two packages, slightly more setup.
- **Option B: single `apps/api/app` package owning domain.** Rejected: reproduces the exact coupling the research report warns against; forces a second migration when workers/CLI appear.
- **Option C: flat `jera/` at repo root (no src).** Rejected: PyPA src-layout explicitly prevents accidental imports & forces install discipline; flat invites the cwd-import trap again.

### Viable Options (local vector/hybrid without Qdrant)
- **Option A (CHOSEN): InMemoryVectorStore implementing dense + sparse + RRF fusion in numpy**, mirroring Qdrant's named-vector/fusion semantics so the Qdrant adapter is a drop-in. Lets the dense/sparse/hybrid/rerank gate pass offline.
- Option B: require pgvector. Rejected for dev default (needs Postgres install); kept as a prod-candidate adapter.

---

## 2. Architecture (Hexagonal / Ports & Adapters)

```text
/home/jinmang2/jera/
  pyproject.toml                # uv WORKSPACE root (THE packaging model). members: python/jera, apps/api. `uv sync` installs both editable.
  uv.lock
  python/jera/
    pyproject.toml              # domain package: name "jera", src-layout
    src/jera/
      __init__.py
      domain/                   # pure dataclasses/pydantic models, no IO
        document.py             # Document, DocumentElement(type=Title|NarrativeText|Table|Figure|ListItem|Code|Formula), PageSpan, Provenance
        chunk.py                # Chunk + full metadata contract
        retrieval.py            # Query, ScoredChunk, RetrievalResult, FusionMethod
        answer.py               # Citation, Answer
        ids.py                  # deterministic ID helpers
      ports/                    # Protocols (the contracts)
        parser.py               # DocumentParser.parse(SourceRef) -> ParsedDocument
        chunker.py              # Chunker.chunk(ParsedDocument) -> list[Chunk]
        embedding.py            # EmbeddingProvider(.embed, .model_id, .dimensions)
        sparse.py               # SparseVectorProvider.encode(texts) -> list[SparseVector]
        vector_store.py         # VectorStore.upsert / search(dense?, sparse?, fusion?, top_k)
        metadata_store.py       # MetadataStore: documents/chunks/jobs/config snapshots
        reranker.py             # Reranker.rerank(query, chunks, top_k)
        generator.py            # GeneratorLLM.generate(query, contexts) -> Answer
      adapters/
        parsing/
          markdown_parser.py    # heading/element-aware (default, pure-python)
          pymupdf_parser.py     # PDF text+layout via PyMuPDF (extra: pdf)
          docling_parser.py     # candidate, behind extra: docling (optional)
        chunking/
          heading_aware.py      # baseline (default)
          semantic.py           # candidate (uses EmbeddingProvider) — stub+contract now
        embedding/
          hash_embedding.py     # deterministic, offline — TEST/CI default
          fastembed_embedding.py# bge-small ONNX — LOCAL dev default (extra: local)
          openai_embedding.py   # disabled-by-default (extra: cloud)
        sparse/
          bm25_local.py         # deterministic local BM25 — default
          fastembed_sparse.py   # SPLADE/bm25 ONNX (extra: local)
        vector_store/
          in_memory.py          # numpy dense + sparse + RRF/DBSF — dev/test default
          qdrant_store.py       # named dense+sparse vectors, RRF/DBSF (extra: qdrant)
        metadata_store/
          sqlite_store.py       # SQLAlchemy 2.0 / sqlite — dev/test default
          postgres_store.py     # SQLAlchemy 2.0 / pg (+pgvector optional) (extra: postgres)
        ranking/
          identity_reranker.py  # deterministic passthrough/score-stable — TEST default
          fastembed_reranker.py # ONNX cross-encoder — LOCAL dev (extra: local)
          cohere_reranker.py    # disabled-by-default (extra: cloud)
        generator/
          extractive_generator.py # deterministic, cites retrieved chunks — TEST default
          claude_generator.py     # disabled-by-default (extra: cloud)
      pipeline/
        ingest.py               # IngestPipeline: parse→chunk→embed→sparse→store(meta+vector)
        query.py                # QueryPipeline: analyze→dense+sparse→fuse→rerank→assemble citations→generate
      config/
        settings.py             # pydantic-settings; profiles: test|local|prod
        registry.py             # ProviderRegistry: builds adapters from settings
      evaluation_contracts/
        dataset.py              # EvalCase, GoldChunk
        metrics.py              # recall@k, mrr, ndcg, citation-faithfulness contracts
    tests/
      fixtures/                 # sample.md, sample_text.pdf, sample_table.{md,html}
      unit/ integration/ e2e/
  apps/api/
    pyproject.toml              # depends on jera (path/editable)
    app/
      main.py                   # FastAPI app factory
      deps.py                   # builds pipelines from jera.config.registry
      routers/ ingest.py query.py health.py
      schemas/                  # request/response pydantic models (API-only)
```

### Data flow
- **Ingest:** `SourceRef → DocumentParser → ParsedDocument(elements+provenance) → Chunker → Chunk[] → EmbeddingProvider + SparseVectorProvider → VectorStore.upsert(named dense+sparse, payload=chunk_id) + MetadataStore.save(document, chunks, job)`.
- **Query:** `Query → embed(query)+sparse(query) → VectorStore.search(dense prefetch, sparse prefetch, RRF fusion, top_k) → Reranker → top_n ScoredChunk → assemble Citations → GeneratorLLM.generate → Answer(text, citations)`.

### Hybrid fusion: golden contract, parity honestly scoped
InMemoryVectorStore implements RRF (default) and DBSF over independent dense/sparse rankings using the *same request shape* as the Qdrant adapter (named vectors + prefetch + fusion). **Parity with Qdrant is NOT claimed-as-verified in Milestone 1** — no Qdrant instance exists to diff against. Instead:
- Fusion is pinned by a **golden-file contract test** with these determinism rules frozen (two developers must produce byte-identical orderings):
  - **Rank is 1-based** from each modality's score-sorted ranking (best = rank 1).
  - **RRF**: `score = Σ_modalities 1/(k + rank_i)`, `k=60`.
  - **Missing-modality rule**: a chunk absent from a modality's prefetch set contributes **0** to the sum for that modality (it is NOT treated as `1/(k+∞)` implicitly — explicit 0).
  - **DBSF**: per-modality **min-max normalization** of scores to [0,1] then sum across modalities; chunks missing from a modality contribute normalized 0.
  - **Tie-break**: equal fused scores break by `chunk_id` **lexicographic ascending** (stable, deterministic).
  - These five rules live in the adapter docstring and are asserted by the golden file.
- The Qdrant adapter is marked `parity-unverified` until the Qdrant-integration milestone, which adds a cross-adapter equivalence test against a live Qdrant. Until then "config-only swap" is a **goal, not a guarantee**.

### Sparse ↔ fusion coupling (acknowledged non-orthogonality)
Embedding/sparse/store ports are independently swappable, **but fusion is not fully orthogonal to the sparse provider.** DBSF normalizes per-modality score *distributions*, so swapping `bm25_local` (rank-stable counts) for `fastembed_sparse` (SPLADE logits) changes DBSF output even with identical ranking inputs. Mitigation: **RRF is the default fusion** (rank-based, robust to score-scale changes); DBSF is opt-in and its sensitivity is documented. A sparse-provider swap therefore requires re-checking fusion config, not just a drop-in.

---

## 3. Technology Stack (latest, justified)

| Concern | Choice | Why / evidence |
|---|---|---|
| Packaging/venv | **uv** + src-layout | uv present; PyPA src-layout prevents cwd-import coupling |
| Models/validation | **pydantic v2 + pydantic-settings** | typed domain + profile config |
| Local embeddings/sparse/rerank | **fastembed (ONNX)** | no torch, lightweight, Qdrant-native; dense(bge-small), sparse(SPLADE/bm25), ColBERT, cross-encoder all behind one ecosystem |
| Deterministic test providers | **hash embedding + local BM25 + extractive generator** | offline, no network, reproducible — satisfies "no fake cosine only" by being *real but deterministic* |
| Vector store | **InMemory (dev/test)** / **Qdrant (prod)** | Qdrant hybrid named vectors + RRF/DBSF; in-memory mirrors semantics |
| Metadata store | **SQLAlchemy 2.0**: SQLite dev / Postgres prod | shared schema, transactional; pgvector optional in prod |
| Migrations | **alembic** (prod path) | versioned schema for pg |
| API | **FastAPI + uvicorn** | adapter-only; routers per FastAPI bigger-apps guidance |
| Lint/format/type/test | **ruff + mypy(strict) + pytest + pytest-cov** | modern gate set |
| Optional extras | `[pdf] [docling] [local] [qdrant] [postgres] [cloud]` | heavy/paid deps opt-in |

Candidates explicitly NOT chosen now, with reason + future selection criterion:
- **sentence-transformers** as local default embedder: pulls a full **torch** runtime (heavy install, slower CI, GPU-coupling temptation). fastembed (ONNX) is lighter and Qdrant-native. Promote sentence-transformers only if a required local model has no ONNX/fastembed equivalent, or if fine-tuning a custom encoder is needed. (Cost named: fastembed's model catalog is smaller than the HF/ST ecosystem — accepted trade for install weight.)
- **DBSF** as default fusion: distribution-normalizing, so coupled to sparse score scale (see §2 coupling note). **RRF is default**; DBSF opt-in when score-aware fusion is benchmarked to beat RRF on the eval set.
- **Docling** as default parser: heavier dep surface; chosen later if PyMuPDF table/layout fidelity fails fixture benchmark (table-bearing + scanned PDF F1 below threshold).
- **Semantic/RAPTOR chunking** as default: needs embedding dep + tuned thresholds; promote when heading-aware recall@k on eval set underperforms semantic by a set margin.
- **pgvector-only stack**: simpler ops; promote for prod if operating two services (Qdrant+PG) is rejected by ops constraints.
- **Cloud providers (OpenAI/Cohere/Voyage/Claude-gen)**: paid live calls excluded this pass; adapters built & disabled; selected per cost/quality eval when keys provided.

---

## 4. Milestone 1 — E2E Vertical Slice (this build)

**Definition of done:** one `pytest` runs the *entire real pipeline* with deterministic local adapters and asserts a cited answer.

Scope IN:
- Domain models + all 8 ports (parser, chunker, embedding, sparse, vector_store, metadata_store, reranker, generator). The query-`analyze` step is a **pure in-`QueryPipeline` function (no port/adapter)** — normalization/tokenization only in M1 — so the port count stays honest.
- Adapters: markdown_parser, pymupdf_parser (text fixture), heading_aware chunker, hash_embedding, bm25_local, in_memory vector store (dense+sparse+RRF), identity_reranker, extractive_generator.
- IngestPipeline + QueryPipeline + ProviderRegistry + `test`/`local` profiles.
- `apps/api` with `/health`, `POST /ingest`, `POST /query` wired to the pipelines.
- Fixtures: `sample.md`, a small generated text PDF, one table-bearing doc.
- Stubs WITH contracts + xfail-free skipped-by-extra tests for: docling, fastembed, qdrant, postgres, cloud adapters (interfaces compile, marked `requires_extra`).

Scope OUT (next milestones, with criteria): semantic/RAPTOR chunking real impl; Qdrant/PG live adapters tested against services; cloud provider eval; full eval harness datasets; frontend. **alembic/migrations: SCOPE OUT for M1** — SQLite dev/test uses `create_all`; alembic baseline is prod-only, added with the Postgres milestone.

**M1 correctness assertions (beyond happy-path wiring):**
- **Citation correctness:** every `Citation.chunk_id` in the answer ∈ the retrieved top-n set AND resolves to a real row in MetadataStore.
- **Empty-result path:** a query with no matching chunks returns an `Answer` with empty text/no citations (defined behavior), never an unhandled error.
- **Provenance correctness (not just stability):** assert `chunk_strategy == "heading_aware"` and `chunk_version` equals the adapter's declared version on produced chunks — populated with *correct* values, not merely stable.
- **Dimension guard:** `InMemoryVectorStore.search` with a query vector whose dim ≠ stored dim raises a typed error, not silent wrong results.

---

## 5. Acceptance Gates (mandatory, non-deferrable)

1. **Package boundary (exact commands):**
   - No domain imports: `rg -n "from app\.(ingestion|retrieval|evaluation)|import app\.(ingestion|retrieval|evaluation)" apps python` → no matches.
   - Positive install proof: after `uv sync` at workspace root, `uv run python -c "import jera"` exits 0.
   - Negative cwd-trap proof (exact): from a directory with no `jera` on path and using an interpreter without the editable install, `cd /tmp && python -c "import jera"` must raise `ModuleNotFoundError`. (This pins cwd so the test proves the absence of a cwd/`src`-adjacent import, not an accident of where it ran.)
   - Single packaging model = **uv workspace**; no manual `pip install -e` in docs/CI.
2. **Parser (OCR gated to extras):** fixtures include text PDF, markdown, and a table-bearing doc asserted in M1; scanned/image-PDF OCR + HTML are `requires_extra`-gated (docling/fastembed extras), so the gate name does not over-claim OCR coverage in M1. Parser returns typed elements + provenance, not flat text. Asserted on markdown + pymupdf paths now.
3. **Chunking:** test compares heading-aware vs semantic strategy output shape on same fixture (semantic via contract/stub now); chunk IDs, section_path, page_span, parent_chunk_id stable across runs.
4. **Retrieval (split by profile to stay honest):**
   - *CI/`test` profile (deterministic, no torch):* separate tests for dense-only, sparse-only, hybrid-RRF, and rerank stages. Hybrid behavior is proven at the **mechanism** level — (a) **non-tautological hybrid case**: a fixture + query where the target chunk is ranked #1 by **neither** dense-only **nor** sparse-only alone, but **#1 after RRF fusion** (genuine fusion lift, not "sparse already had it at #1"); (b) golden-file fusion vectors asserting the RRF/DBSF math + tie-break rules. We do NOT claim hash embeddings demonstrate semantic paraphrase superiority (they have no semantic geometry — claiming so would be the fake path Gate 7 forbids).
   - *`local` profile (fastembed, `requires_extra=local`, skipped in default CI):* the genuine **semantic** case — a paraphrase query with no lexical overlap where dense (bge-small) wins and sparse misses, and hybrid recovers both. This is where "dense wins on paraphrase" is asserted honestly.
5. **Storage/vector:** SQLite schema owns documents/chunks/jobs/config-snapshots; in-memory + Qdrant adapter spec define named dense+sparse vectors + payload→chunk_id; re-index semantics documented (embedding model/dimension change ⇒ new collection + migration note).
6. **Provider:** paid providers are adapter interfaces, disabled by default; provider config records `model_id, dimensions, context_limit, cost_metadata_placeholder, version_snapshot` and is persisted in the config-snapshot table.
7. **No-evasion:** ADR/this plan/test-spec do not use "roadmap/deferred" to dodge axes 1–6; every not-chosen candidate has evidence reason + future selection criterion (see §3).

CI gate command set: `ruff check`, `ruff format --check`, `mypy`, `pytest -q` (with the boundary `rg` check as a pre-step).

---

## 6. Test & Eval Strategy
- **Unit:** each adapter against its port contract (shared contract test parametrized over adapters).
- **Integration:** ingest→store→retrieve round-trips on SQLite + in-memory.
- **E2E:** the milestone-1 cited-answer test through `apps/api` TestClient.
- **Determinism:** seed-free by construction (hash embeddings, deterministic BM25, identity rerank, extractive generator).
- **Eval contracts:** `recall@k`, `mrr`, `ndcg`, `citation-faithfulness` defined as contracts now; populated datasets are a later milestone (criterion-gated, not "deferred to dodge").

---

## 7. ADR

- **Decision:** Build Jera as a hexagonal, offline-first RAG system: repo-root `python/jera` src-layout domain package with ports/adapters; `apps/api/app` adapter-only; deterministic-local default providers; Qdrant+Postgres as opt-in prod adapters; deliver an E2E vertical slice first.
- **Drivers:** no Docker/services available; "latest tech + clean structure" goal; anti-scaffold-failure gates.
- **Alternatives considered:** domain-in-API package (rejected: coupling/2nd migration); flat layout (rejected: cwd-import trap); pgvector-only (kept as prod candidate); cloud-first providers (rejected as default: paid, non-offline); Docling/semantic-chunking as defaults (deferred to benchmark criteria, not to dodge).
- **Why chosen:** maximizes testability offline, matches Qdrant semantics for a clean prod swap, isolates vendor/paid surface, and forecloses the prior `app.*` coupling failure.
- **Consequences:** two packages + adapter breadth up front; fastembed/qdrant/pg/cloud are extras; in-memory fusion parity with Qdrant is a **goal, not a Milestone-1 guarantee** — pinned by a golden-file fusion contract (RRF k=60, DBSF min-max) and only cross-verified against live Qdrant in a later milestone; sparse-provider and fusion-method are coupled (DBSF), so RRF is the safe default.
- **Follow-ups (criterion-gated):** Docling default if PyMuPDF table/scan F1 < threshold; semantic/RAPTOR if recall margin; Qdrant/PG live tests when Docker available; cloud provider cost/quality eval when keys provided; eval datasets.

---

## 8. Execution Plan (post-approval, team/ralph)
1. Scaffold `python/jera` (pyproject, src-layout, uv install -e) + tooling (ruff/mypy/pytest).
2. Domain models + ports.
3. Default offline adapters (parse/chunk/embed/sparse/store/rerank/generate).
4. IngestPipeline + QueryPipeline + registry + profiles.
5. `apps/api` adapter wiring.
6. Fixtures + unit/integration/E2E tests + the 7 gates in CI.
7. Stub+contract the opt-in adapters (docling/fastembed/qdrant/pg/cloud) behind extras.
8. Run full gate suite; iterate to green.
