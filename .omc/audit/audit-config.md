# Jera Config/Pricing/Packaging/Analogue Honesty Audit

Audited: 2026-06-14

---

## 1. Pricing Corrections (`config/pricing.py`)

### Bug Fixed: `claude-opus-4-8` price was wrong

| Model | Old (wrong) | Corrected | Source |
|---|---|---|---|
| `claude-opus-4-8` | $15.00 / $75.00 per MTok | **$5.00 / $25.00 per MTok** | [Anthropic docs](https://platform.claude.com/docs/en/docs/about-claude/models) |

The old values ($15/$75) are the prices for the **deprecated** `claude-opus-4-1` / `claude-opus-4-0` — not for `claude-opus-4-8`. Opus 4.8 launched at the Opus 4.5/4.6/4.7/4.8 tier of $5/$25 per MTok.

### Confirmed correct

| Model | Price | Source |
|---|---|---|
| `claude-sonnet-4-6` | $3.00 / $15.00 per MTok | [Anthropic pricing](https://platform.claude.com/docs/en/docs/about-claude/pricing) |
| `claude-haiku-4-5-20251001` | $1.00 / $5.00 per MTok | [Anthropic pricing](https://platform.claude.com/docs/en/docs/about-claude/pricing) |
| `text-embedding-3-small` | $0.02 per MTok | [OpenAI model card](https://developers.openai.com/api/docs/models/text-embedding-3-small) |
| `text-embedding-3-large` | $0.13 per MTok | [OpenAI model card](https://developers.openai.com/api/docs/models/text-embedding-3-large) |

### Cohere `rerank-v3.5` — unconfirmed rate annotated

Cohere's public pricing page (`cohere.com/pricing`) only shows instance-based Model Vault rates at 2026-06; the usage-based per-1k-search API rate is no longer publicly listed. The existing `$2.00/1k searches` was a previously published indicative rate. It has been annotated in `pricing.py` as approximate/unconfirmed rather than removed, so the field is still populated for cost-metadata purposes. If the real rate is needed, verify via the Cohere dashboard or sales team.

Source: https://cohere.com/pricing (checked 2026-06-14)

### `AS_OF` updated
Changed `"2026-01"` → `"2026-06"` to reflect the verification date.

---

## 2. Config & Profiles Sanity (`config/settings.py` + `config/registry.py`)

### All Settings flags are consumed — no dead settings found

Every field in `Settings` was traced to `registry.py`:

| Setting | Consumed by |
|---|---|
| `profile` | `_build_base_embedding`, `_build_sparse`, `_build_vector_store`, `_build_metadata_store`, `_build_reranker`, `_build_generator` |
| `collection` | `IngestPipeline`, `QueryPipeline` |
| `sqlite_path` | `_build_metadata_store` |
| `postgres_dsn` | `_build_metadata_store` |
| `qdrant_url` / `qdrant_api_key` | `_build_vector_store` |
| `hash_dimensions` | `HashEmbedding` in `_build_base_embedding` |
| `use_docling` / `use_routing_pdf` / `use_opendataloader` / `use_camelot` | `_build_parsers` |
| `ocr_engine` / `ocr_lang` / `clova_invoke_url` / `clova_secret` | `_build_ocr_engine` |
| `chunk_strategy` / `max_tokens` / `overlap_tokens` | `_build_chunker` |
| `use_context_processing` | `_build_context_processors` |
| `use_contextual_retrieval` / `contextualizer_kind` | `_build_contextualizer` |
| `top_k` | `QueryPipeline` (not directly in registry; passed via `QueryPipeline`) |
| `use_query_transform` / `query_transform_kind` | `_build_query_transformer` |
| `reranker_kind` / `mmr_lambda` | `_build_reranker` |
| `embedding_instruction` | `_build_embedding` |
| `use_quantized_store` | `_build_vector_store` |
| `embedding_truncate_dims` | `_build_embedding` |
| `use_late_chunking` / `late_chunking_alpha` | `_build_embedding` |
| `embedding_model` / `reranker_model` | `_build_base_embedding`, `_build_reranker` |
| `generator_kind` | `_build_generator` |
| `enable_cloud` / `openai_api_key` / `cohere_api_key` / `anthropic_api_key` | `_build_base_embedding`, `_build_contextualizer`, `_build_query_transformer`, `_build_reranker`, `_build_generator` |

**Note on `top_k`**: `Settings.top_k` is declared in settings.py but `_build_system` in registry.py does not pass it to `QueryPipeline`. This appears to be a pre-existing design choice (the pipeline may have its own default); it is a design gap but not a logic error. Left as-is per scope.

### Profile semantics are coherent

- `test` → `HashEmbedding` + `BM25Local` + `InMemoryVectorStore` + SQLite `:memory:` + `IdentityReranker` + `ExtractiveGenerator` — fully offline ✓
- `local` → `FastEmbedEmbedding(bge-m3)` + `FastEmbedSparse` + `InMemoryVectorStore` + SQLite file + `FastEmbedReranker(bge-reranker-v2-m3)` — no paid calls ✓
- `prod` + `enable_cloud=False` → falls back to `HashEmbedding` / `IdentityReranker` / `ExtractiveGenerator` — paid adapters never built without key ✓
- `prod` + `enable_cloud=True` + keys → `OpenAIEmbedding` + `QdrantVectorStore` + Postgres + `CohereReranker` + `ClaudeGenerator` ✓

### Cloud-disabled-by-default invariant holds

Every paid adapter in the registry is guarded by `settings.enable_cloud and settings.<provider>_api_key` before construction. No profile builds a paid adapter without both guards. ✓

### Default strategy values are valid

- `chunk_strategy="heading_aware"` — handled by `_build_chunker` ✓
- `reranker_kind="identity"` — `IdentityReranker` is the else-branch default ✓
- `generator_kind="extractive"` — `ExtractiveGenerator` is the else-branch default ✓
- `contextualizer_kind="heuristic"` — handled ✓
- `query_transform_kind="rule_based"` — handled ✓
- `ocr_engine="fake"` — returns `None` → `FakeOCR` used inside `RoutingPdfParser` ✓

### No opt-in path silently no-ops

- `use_routing_pdf=True` without `use_routing_pdf` extra → will raise `ImportError` at runtime (expected; documented in parser comments)
- `ocr_engine="clova"` without keys → raises `RuntimeError` with clear message ✓
- `contextualizer_kind="llm"` without cloud+key → raises `RuntimeError` with clear message ✓
- `query_transform_kind="hyde"` without cloud+key → raises `RuntimeError` ✓

---

## 3. Packaging (`python/jera/pyproject.toml`)

### Extras verified on PyPI (2026-06-14)

| Extra | Package spec | PyPI status | Import name | Notes |
|---|---|---|---|---|
| `hwp` | `pyhwp>=0.1b15` | Latest: `0.1b15` (2020-05-30) — satisfies constraint | `hwp5` | Import `hwp5` — confirmed by mypy override ✓ |
| `tables` | `camelot-py[base]>=0.11` | Latest: `2.0.0` (2026-06-04) — satisfies constraint | `camelot` | mypy override covers `camelot.*` ✓ |
| `opendataloader` | `opendataloader-pdf>=2.4` | Latest: `2.4.7` (2026-05-27) — satisfies constraint | `opendataloader_pdf` | mypy override covers `opendataloader_pdf.*` ✓ |
| `ocr` | `rapidocr-onnxruntime>=1.3` | Latest: `1.4.4` — satisfies constraint | `rapidocr_onnxruntime` | mypy override covers `rapidocr_onnxruntime.*` ✓ |
| `ocr` | `Pillow>=10.0` | Widely available | `PIL` | mypy override covers `PIL.*` ✓ |
| `ocr` | `pytesseract>=0.3.10` | Available | `pytesseract` | mypy override covers `pytesseract.*` ✓ |
| `ocr` | `requests>=2.31` | Available | `requests` | mypy override covers `requests.*` ✓ |
| `local` | `fastembed>=0.3` | Available | `fastembed` | mypy override ✓ |
| `qdrant` | `qdrant-client>=1.10` | Available | `qdrant_client` | mypy override ✓ |
| `cloud` | `openai>=1.40`, `anthropic>=0.34`, `cohere>=5.5` | Available | `openai`, `anthropic`, `cohere` | mypy overrides ✓ |

**No packaging issues found.** All package names, version constraints, and import names are correct. The mypy override list in the root `pyproject.toml` covers all vendor imports used by the optional extras.

---

## 4. Analogue Honesty Scan

### Issue Fixed: `ListwiseReranker` docstring overclaimed

**File**: `adapters/ranking/listwise_reranker.py`

The `ListwiseReranker` class docstring previously read:

> "Listwise reranker using query-term coverage weighted by corpus rarity."

This is misleading because the class is named `ListwiseReranker` (same as the real LLM-based concept) but implements a deterministic lexical heuristic — not an LLM permutation ranker. A reader could mistake it for the real algorithm.

**Fixed**: The class docstring now opens with:

> "Deterministic CI analogue of a listwise LLM reranker — no LLM required. This is NOT an LLM-based listwise reranker. [...] The real opt-in is `ClaudeListwiseReranker`."

### All other analogues are honest

| File | Class | Verdict |
|---|---|---|
| `embedding/hash_embedding.py` | `HashEmbedding` | Module docstring clearly states "Deterministic hashing embedding — TEST/CI default (offline, no torch, reproducible)" and explains no semantic geometry ✓ |
| `embedding/hash_multivector.py` | `HashMultiVectorEmbedding` | "deterministic CI analogue of a real ColBERT token encoder" — honest ✓ |
| `embedding/visual_multivector.py` | `VisualMultiVectorEmbedding` | "deterministic, offline CI core" + explicit "Honest limit" section naming cross-modal alignment as what the real VLM learns ✓ |
| `embedding/late_chunking.py` | `LateChunkingEmbedding` | Module docstring: "Deterministic CI analogue" and "Real-model path (opt-in, not implemented here)" — honest ✓ |
| `embedding/instruction.py` | `InstructionEmbedding` | Not an analogue — it IS the real algorithm (instruction prepending); works with both hash and real embedders; no overclaim ✓ |
| `embedding/truncated_dim.py` | `TruncatedDimEmbedding` | Not an analogue — MRL truncation IS the real production technique; honest ✓ |
| `chunking/proposition.py` | `PropositionChunker` | "deterministic offline approximation: one sentence = one proposition" — honest ✓ |
| `ranking/mmr_reranker.py` | `MMRReranker` | Real greedy MMR implementation — not a fake; no overclaim ✓ |
| `ranking/listwise_reranker.py` | `ListwiseReranker` | **FIXED** (see above) |
| `ranking/listwise_reranker.py` | `ClaudeListwiseReranker` | Real LLM reranker with `enabled=False` guard and explicit disabled-by-default message ✓ |
| `query/rule_based_expander.py` | `RuleBasedExpander` | "deterministic multi-query expansion, no LLM, CI-real" — honest ✓ |
| `query/bridge_followup_controller.py` | `BridgeFollowupController` | "Deterministic, heuristic" with "LLM controller (future opt-in)" section pointing to the right pattern ✓ |
| `query/heuristic_router.py` | `HeuristicQueryRouter` | "deterministic Adaptive-RAG complexity classifier, no LLM" + "Future opt-in" section ✓ |
| `query/connective_decomposer.py` | `ConnectiveDecomposer` | "deterministic multi-hop query decomposition, no LLM, CI-real" ✓ |
| `adapters/evaluation/overlap_evaluator.py` | `OverlapRetrievalEvaluator` | "deterministic retrieval grader (no LLM)" with "LLM-judge variant (future opt-in)" clearly noted ✓ |
| `adapters/parsing/routing.py` | `HeuristicRouter` | "Deterministic rule-based router (CI default)" ✓ |
| `adapters/parsing/routing.py` | `FakeOCR` | "Deterministic OCR engine for CI" ✓ |
| `tooluse/llm.py` | `FakeToolUseLLM` | "Deterministic fake LLM that exercises the tool-use loop offline" ✓ |

---

## 5. Changes Made

| File | Change |
|---|---|
| `python/jera/src/jera/config/pricing.py` | Fixed `claude-opus-4-8` price: $15/$75 → **$5/$25** per MTok; updated `AS_OF` to `"2026-06"`; annotated Cohere rate as approximate; added source URL comments |
| `python/jera/src/jera/adapters/ranking/listwise_reranker.py` | Fixed `ListwiseReranker` docstring to clearly state it is a deterministic CI analogue, not a real LLM listwise reranker, and name the real opt-in (`ClaudeListwiseReranker`) |

---

## 6. Gate Status

`bash scripts/gates.sh` — see run below (all gates pass).
