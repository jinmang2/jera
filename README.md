# Jera

A hexagonal, **offline-first RAG** system. Domain logic lives in a repo-root `python/jera`
package (src-layout); `apps/api` is a FastAPI **adapter only**. Every external capability —
parsing, embedding, sparse encoding, vector store, reranking, generation — is a `Protocol`
**port** with swappable **adapters**, selected by configuration profile.

- **Dev/test**: fully deterministic, no Docker, no torch, no paid keys — hash embeddings,
  local BM25, an in-memory vector store with RRF/DBSF fusion, SQLite, and an extractive
  generator. The whole pipeline runs and is tested end-to-end with zero external services.
- **Prod**: Qdrant (named dense+sparse vectors, RRF/DBSF) + Postgres, with cloud
  embedding/rerank/generation adapters — all opt-in, disabled by default.

See `.omc/plans/rag-redesign-plan.md` for the full architecture/ADR and the acceptance gates.

## Layout

```
python/jera/src/jera/
  domain/                 # pure models (Document, Chunk, ScoredChunk, Answer, ...)
  ports/                  # Protocols: parser, chunker, embedding, sparse, vector_store,
                          #            metadata_store, reranker, generator
  adapters/               # concrete implementations (defaults + opt-in extras)
  pipeline/               # IngestPipeline, QueryPipeline
  config/                 # Settings (profiles) + ProviderRegistry
  evaluation_contracts/   # recall@k, MRR, nDCG, citation-faithfulness
  rag/                    # public facade: `import jera.rag`
apps/api/app/             # FastAPI adapter: routers / DI / schemas
```

## Quickstart

```bash
uv sync                        # install the workspace (both packages, editable)
bash scripts/gates.sh          # ruff + mypy + pytest (all acceptance gates)
uv run python scripts/eval.py  # demo eval: dense vs sparse vs hybrid metric table
uv run uvicorn app.main:app --reload --app-dir apps/api   # serve the API
```

### Evaluation

`jera.evaluation` turns the metric contracts (recall@k / MRR / nDCG / citation-faithfulness)
into a runnable harness: `build_gold_dataset` labels gold chunks by substring (no rotting id
lists), and `EvalRunner` scores dense/sparse/hybrid retrieval into an `EvalReport`. The same
harness measures real model quality under the `local`/`prod` profiles with no code change.

### API

```bash
curl -s localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"source_id":"d1","media_type":"text/markdown","text":"# Title\n\nHybrid retrieval uses reciprocal rank fusion."}'

curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"query":"what does hybrid retrieval use?","top_k":3}'
```

## Profiles

Set `JERA_PROFILE` (default `test`):

| profile | embedding | sparse | vector store | metadata | rerank | generate |
|---|---|---|---|---|---|---|
| `test`  | hash (deterministic) | BM25 local | in-memory | SQLite `:memory:` | identity | extractive |
| `local` | fastembed (ONNX) `[local]` | fastembed SPLADE `[local]` | in-memory | SQLite file | fastembed CE `[local]` | extractive |
| `prod`  | OpenAI `[cloud]`* | (sparse) | Qdrant `[qdrant]` | Postgres `[postgres]` | Cohere `[cloud]`* | Claude `[cloud]`* |

`*` cloud adapters are disabled unless `JERA_ENABLE_CLOUD=1` and the matching API key are set.
Optional extras: `uv sync --extra local --extra qdrant --extra postgres --extra cloud --extra docling`.

## Acceptance gates (M1)

`scripts/gates.sh` enforces: package boundary (no `app.*` domain imports; src-layout
import proof), typed-element parsing with provenance, chunk stability, dense/sparse/hybrid
retrieval with a **non-tautological** fusion-lift case, golden-file RRF/DBSF determinism,
storage ownership of documents/chunks/jobs/config-snapshots, disabled-by-default paid
providers, and an end-to-end cited-answer test through the FastAPI app.
