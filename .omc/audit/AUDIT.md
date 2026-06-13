# Jera Correctness Re-Audit (대공사)

**Date:** 2026-06-14
**Trigger:** Full re-inspection of M0→M14b against official docs / canonical specs — verifying
nothing was "eyeballed into rolling along" (대충 눈대중으로 굴러만 가게) instead of matching real specs.
**Final gate:** `bash scripts/gates.sh` → **765 passed, 8 skipped, 0 failures** (GREEN).

Audit tracks, each verified against the originating paper or the vendor's **actual source code**
(not just memory). Round 1 verified formulas/APIs; round 2 traced each technique's **data flow**
to the generated answer (does the technique's work actually reach the answer, or roll along?).

| Track | Report | Bugs fixed | Honesty fixes |
|-------|--------|-----------|---------------|
| LLM SDK adapters | [audit-llm-sdk.md](audit-llm-sdk.md) | 0 | 0 |
| Config / pricing | [audit-config.md](audit-config.md) | 1 | 1 |
| Core algorithms | [audit-algos.md](audit-algos.md) | 1 | 0 |
| Vendor store/parser/OCR | [audit-vendor.md](audit-vendor.md) | 1 | 0 |
| **Advanced pipelines / techniques (round 2)** | [audit-pipelines.md](audit-pipelines.md) | **1** | 0 |

**Total: 4 real bugs fixed, 1 honesty/overclaim fix. No test weakening.**

---

## Real bugs found & fixed

### 1. Pricing — `claude-opus-4-8` mispriced (config track)
`config/pricing.py` listed Opus 4.8 at **$15/$75 per MTok** — those are the *deprecated* Opus
4.0/4.1 prices. Corrected to the real **$5/$25**. `AS_OF` bumped 2026-01 → 2026-06; source URLs
added. This is exactly the "eyeballed, looks-plausible-but-wrong" class the re-audit was looking for.

### 2. DBSF fusion — eyeballed min-max, not real DBSF (algorithm track)
`adapters/vector_store/fusion.py` implemented DBSF as plain min-max `(s−min)/(max−min)`. The
canonical Distribution-Based Score Fusion (and Qdrant's production path this offline analogue is
contractually required to mirror) is **3-sigma**: `(s−(μ−3σ))/(6σ)` with sample σ, a 0.5
single-point/zero-variance guard, and no clamping. Verified line-by-line against Qdrant's actual
`qdrant_client/hybrid/fusion.py`. Fixed; golden test re-pinned to the exact 3-sigma arithmetic.
Bonus: this **restores** the frozen "two implementations agree" contract that min-max silently broke.

### 3. Qdrant `recreate_collection` deprecated (vendor track)
`adapters/vector_store/qdrant_store.py` called `recreate_collection()`, deprecated since
qdrant-client ≥ 1.7 (issue #711). Replaced with the documented
`collection_exists() → delete_collection() → create_collection()`. Two test fakes updated to match.

### 4. CRAG answer bypassed the correction (round 2, pipelines track)
`pipeline/corrective.py` computed corrected contexts (graded → query-expanded → RRF-fused →
reranked) but generated the answer via `answer_with_contexts`, which **re-ran vanilla retrieval
and discarded the correction** — the answer was built from the uncorrected ranking. The test only
checked `contexts`, never the answer, so it stayed green while the answer cited A and `contexts`
showed B. Fixed by adding `QueryPipeline.generate_from_contexts(...)` (generate from an explicit
context set, no re-retrieval) and routing both CRAG paths through it; strengthened the test to
assert the answer reflects the corrected context. See [audit-pipelines.md](audit-pipelines.md).

## Honesty fix

### ListwiseReranker overclaim (config track)
`adapters/ranking/listwise_reranker.py` shared the "listwise LLM reranker" name without disclosing
it is the **deterministic CI analogue**; the real opt-in is `ClaudeListwiseReranker`. Docstring
corrected.

---

## Verified correct (no change needed)

- **8/8 LLM adapters** (Anthropic Messages incl. tool-use/pause_turn + prompt-caching, OpenAI
  embeddings, Cohere rerank) — checked against current official API docs. Zero mismatches.
- **RRF** (k=60, 1-based, Cormack 2009), **BM25** (Lucene non-negative IDF + tf saturation),
  **MMR** (Carbonell-Goldstein 1998), **ColBERT MaxSim** (Khattab-Zaharia 2020),
  **Personalized PageRank** (HippoRAG, mass-conserving power iteration).
- **Retrieval metrics** nDCG/MRR/recall (Järvelin-Kekäläinen linear gain), FinQA numeric accuracy.
- **RAGAS-lite** and **RAGChecker-style** metrics — honest deterministic stand-ins, each docstring
  states its approximation; no LLM-judge claims dressed up as the real thing.
- **fastembed** (dense/sparse/rerank), **docling**, **camelot**, **pyhwp**, **tesseract**,
  **rapidocr-onnxruntime**, **CLOVA OCR**, **Matryoshka truncation** — all match vendor APIs.

## Advisories (non-blocking, recorded for future work)

- `cohere.Client` works; `cohere.ClientV2` is the newer recommended pattern.
- `rapidocr-onnxruntime>=1.3` pin locks the 1.x tuple API; `rapidocr` 3.x returns a dataclass and
  would need an adapter+mock update if the pin changes.
- `opendataloader-pdf` has no public API reference URL; adapter written against package behavior.
- PPR docstring says "damping factor" where it means restart probability (math is correct).
- RAGChecker `claim_precision` checks against context (faithfulness-like) vs the paper's
  precision-against-ground-truth; internally consistent and documented.

---

**Bottom line:** the system was not "just rolling along" — but four substantive bugs were hiding
behind green tests: a wrong price, a wrong fusion algorithm, a deprecated DB call, and a CRAG
pipeline whose corrected context never reached the answer. Each test encoded the same blind spot
as the code (right intermediate, unchecked outcome). The re-audit caught them by going to the
**primary sources** — paper formulas, vendor source, and end-to-end data-flow tracing — and the
suite is green with the fixes. Round 2 specifically traced every advanced technique from its input
to the generated answer, not just "does it return a result."
