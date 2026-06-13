# Core Algorithm Correctness Audit

**Date:** 2026-06-14
**Scope:** Every canonical ranking/retrieval/eval algorithm in `python/jera/src/jera/`, verified
against its originating paper / production reference implementation.
**Gate result after fixes:** `bash scripts/gates.sh` — 765 passed, 8 skipped, 0 failures.

The audit-algos pass found **1 real spec bug** (DBSF was an eyeballed min-max, not the canonical
distribution-based fusion) and fixed it. All other algorithms match their canonical specs.

---

## 1. RRF — `adapters/vector_store/fusion.py::reciprocal_rank_fusion`

**Reference:** Cormack, Clarke & Büttcher, "Reciprocal Rank Fusion outperforms Condorcet…",
SIGIR 2009.

**Spec:** `RRF(d) = Σ_modalities 1/(k + rank_i(d))`, k = 60, rank 1-based.

| Check | Spec | Code | Result |
|-------|------|------|--------|
| Constant k | 60 | `RRF_K = 60` | PASS |
| Rank base | 1-based | `enumerate(ranked, start=1)` | PASS |
| Formula | `1/(k+rank)` | `1.0 / (k + rank)` | PASS |
| Missing modality | contributes 0 | only present ids accumulate | PASS |
| Tie-break | deterministic | `(-score, chunk_id asc)` | PASS |

**Verdict: CORRECT — unchanged.**

---

## 2. DBSF — `adapters/vector_store/fusion.py::distribution_based_score_fusion`

**Reference:** Mazzeschi, "Distribution-Based Score Fusion" (2024); **production reference:**
Qdrant `qdrant_client/hybrid/fusion.py` (verified against upstream source 2026-06-14).

**Spec (Qdrant canonical):** per modality, `ŝ = (s − (μ − 3σ)) / (6σ)` with **sample** standard
deviation σ (`Σ(s−μ)² / (n−1)`); single-point or zero-variance modality emits **0.5** per point;
scores are **not** clamped; sum across modalities.

### BUG FOUND & FIXED
The M1 implementation used plain **min-max** normalization (`(s−min)/(max−min)`, zero-spread → 0).
That is *not* DBSF — it is a different, eyeballed normalization. Critically, the offline analogue's
entire purpose is to mirror the production Qdrant DBSF path (`FusionQuery(fusion=DBSF)` in
`qdrant_store.py`); min-max would have **diverged** from what Qdrant actually computes, silently
breaking the "two implementations, byte-identical orderings" frozen-contract promise.

**Fix:** replaced with the canonical 3-sigma normalization. Verified line-by-line against Qdrant's
real source:

| Qdrant source | Jera code | Match |
|---|---|---|
| `variance = Σ(s−μ)² / (len−1)` (ddof=1, sample) | `sum((v-mean)**2) / (n-1)` | YES |
| `low = μ − 3·std`, `high = μ + 3·std` | `lo = mean − 3σ`, `denom = 6σ` | YES |
| `(s − low) / (high − low)` | `(s − lo) / denom` | YES (high−low = 6σ) |
| `len == 1 → 0.5`; `variance == 0 → 0.5` | `denom == 0.0 → 0.5` | YES |
| no clamping to [0,1] | no clamping | YES |

Golden test `test_fusion_golden.py` updated with hand-computed 3-sigma expectations (NOT a
weakening — the new asserts pin the exact Qdrant arithmetic, e.g. dense {10,6,2}→μ=6,σ=4→denom=24).

**Verdict: BUG — fixed; now faithful to the canonical/production spec.**

---

## 3. BM25 — `adapters/sparse/bm25_local.py`

**Reference:** Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"
(2009); IDF variant per Lucene/Elasticsearch `BM25Similarity`.

| Check | Spec | Code | Result |
|-------|------|------|--------|
| tf saturation | `tf·(k1+1) / (tf + k1·(1−b+b·dl/avgdl))` | exact | PASS |
| IDF | `log(1 + (N−df+0.5)/(df+0.5))` (Lucene non-negative form) | exact | PASS |
| defaults | k1∈[1.2,2.0], b=0.75 | k1=1.5, b=0.75 | PASS |
| query weighting | linear query-tf (k3→∞ limit) | `query_value = tf` | PASS (documented) |
| dot reconstruction | `doc·query == BM25(q,d)` | holds by construction | PASS |

Note: the `1 +` inside the IDF log is the Lucene/ES guard that keeps IDF ≥ 0 (vs the classic
Robertson-Sparck-Jones form that can go negative for very common terms). This is the modern
production-standard variant, not a mistake. **Verdict: CORRECT.**

---

## 4. MMR — `adapters/ranking/mmr_reranker.py`

**Reference:** Carbonell & Goldstein, "The Use of MMR…", SIGIR 1998.

**Spec:** `MMR = argmax_{d∈R\S} [ λ·sim(d,q) − (1−λ)·max_{d'∈S} sim(d,d') ]`.

Code: `mmr_score = λ·rel − (1−λ)·max_sim`, greedy selection, λ=0.7 default, cosine similarity,
first pick uses `max_sim = 0`. Tie-break (`mmr` desc, chunk_id asc) is deterministic.
**Verdict: CORRECT.**

---

## 5. ColBERT MaxSim — `adapters/vector_store/maxsim_store.py`

**Reference:** Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via
Contextualized Late Interaction over BERT", SIGIR 2020.

**Spec:** `S(q,d) = Σ_{qi∈q} max_{dj∈d} cos(qi, dj)`.

Code: `total += max(cos(q_tok, d_tok) for d_tok in doc_vecs)` summed over query tokens — exact.
Tie-break chunk_id asc. **Verdict: CORRECT.**

---

## 6. Personalized PageRank — `adapters/graph/hippo_retriever.py`

**Reference:** Gutiérrez et al., "HippoRAG", NeurIPS 2024 (arXiv:2405.14831).

**Spec:** power iteration `r_{t+1} = (1−α)·Mᵀr_t + α·e_seed`, M row-stochastic, e_seed the
personalization (seed) vector.

| Check | Spec | Code | Result |
|-------|------|------|--------|
| Transition | row-normalized | `w/total` per row | PASS |
| Dangling nodes | uniform redistribution | `1/n` to all | PASS |
| Walk term | `(1−α)·Mᵀr` | `(1−alpha)·src_score·prob` | PASS |
| Restart term | `α·e_seed` | `alpha·seed_dist[s]` | PASS |
| Mass conservation | sums to 1 each step | (1−α)+α = 1 (verified) | PASS |
| Seed init | uniform over seeds | `1/len(valid_seeds)` | PASS |

Terminology nit (non-bug): the docstring calls α the "damping factor"; in classic PageRank
"damping" usually names the edge-follow probability (1−α here). The *math* is correct
restart-to-seed PPR. **Verdict: CORRECT** (advisory: tighten the word "damping" in the docstring).

---

## 7. Retrieval metrics — `evaluation_contracts/metrics.py`

| Metric | Spec | Code | Result |
|--------|------|------|--------|
| recall@k | hits@k / |gold| | exact | PASS |
| MRR | 1/rank of first relevant | exact | PASS |
| nDCG@k | `Σ rel/log2(i+2)` ÷ ideal (Järvelin & Kekäläinen, linear gain) | exact | PASS |
| citation_faithfulness | cited∩retrieved / cited | exact | PASS |
| numeric_accuracy | FinQA relative tol w/ `max(|exp|,1)` floor | exact | PASS |

**Verdict: CORRECT.** (nDCG uses linear gain — the original JK2002 form; valid, just not the
`2^rel−1` Burges variant.)

---

## 8. RAGAS-lite — `evaluation_contracts/generation_metrics.py`

faithfulness (sentence-containment ≥0.6), answer_relevance (clamped cosine), answer_correctness
(multiset token-F1), context_precision (average precision ÷ min(|gold|,k)). All four are **honestly
documented** as deterministic offline stand-ins for the RAGAS quartet — the module docstring and
each function state the approximation explicitly. No overclaiming. **Verdict: CORRECT (honest).**

---

## 9. RAGChecker-style — `evaluation_contracts/ragchecker_metrics.py`

claim_precision/recall, noise_sensitivity, citation_precision/recall (TREC-2025 weighted
Full=1.0/Partial=0.5/None=0.0), abstention_score (RGB hedge lexicon, EN+KO). Honestly labeled
"inspired by" RAGChecker (arXiv:2408.08067) with token-containment standing in for the paper's
NLI/LLM judge; every docstring states exactly what it computes.

Advisory (fidelity, not a bug): the paper's *overall Precision* checks response claims against the
**ground-truth answer**, whereas this `claim_precision` checks against the **retrieved context**
(closer to the paper's *Faithfulness*); `claim_recall` does check against gold. The split is
internally consistent and clearly documented, but a future revision could rename/realign to the
paper's exact precision-vs-truth definition. **Verdict: CORRECT (honest; advisory noted).**

---

## Summary

| Algorithm | Verdict | Change |
|-----------|---------|--------|
| RRF | CORRECT | — |
| **DBSF** | **BUG — fixed** | min-max → canonical Qdrant 3-sigma (sample σ, 0.5 guard, no clamp) |
| BM25 | CORRECT | — |
| MMR | CORRECT | — |
| ColBERT MaxSim | CORRECT | — |
| Personalized PageRank | CORRECT | advisory: "damping" wording |
| Retrieval metrics (nDCG/MRR/recall) | CORRECT | — |
| RAGAS-lite | CORRECT (honest) | — |
| RAGChecker-style | CORRECT (honest) | advisory: precision-vs-context vs paper's precision-vs-truth |

**Spec bugs found and fixed:** 1 (DBSF).
**Files changed:** `fusion.py`, `tests/unit/test_fusion_golden.py`.
