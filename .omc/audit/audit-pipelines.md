# Advanced-Pipeline & Technique Deep Audit (round 2)

**Date:** 2026-06-14
**Scope:** Every advanced retrieval pipeline and technique adapter, read against its paper +
traced end-to-end (does the technique's output actually reach the generated answer?).
**Trigger:** User challenge — "did you *really* check everything, or just the pricing? is this
actually SOTA-faithful?" The round-1 audit verified formulas; this round traces *data flow* —
whether each technique's work reaches the answer or silently rolls along.

**Gate after fix:** `bash scripts/gates.sh` → 765 passed, 8 skipped, 0 failures.

---

## BUG FOUND & FIXED — CRAG answer bypassed the correction

`pipeline/corrective.py` (Corrective RAG, Yan et al. ICLR 2025).

**Symptom:** the pipeline graded retrieval, expanded the query, RRF-fused corrective variants,
reranked them into `contexts`, exposed `corrected=True` and corrected `retrieved_ids` — and then
generated the answer by calling `QueryPipeline.answer_with_contexts(query_text)`, which **re-runs
vanilla retrieval internally and discards the corrected contexts entirely.** The generated answer
was built from the *uncorrected* ranking. CRAG's entire purpose (answer from corrected evidence)
did not happen. A code comment even admitted it: *"answer_with_contexts runs its own full
retrieve_multi path internally — we use its Answer and stats."*

**Why it stayed green:** `test_corrective_rag.py` asserted `result.contexts[0].source_id == "B"`
(the corrected context) but never checked the **answer**. The answer cited A while `contexts`
showed B — an incoherence the test couldn't see.

**Root cause:** `QueryPipeline` had no way to generate from a caller-supplied context set; every
generation path re-retrieved.

**Fix (3 parts):**
1. Added `QueryPipeline.generate_from_contexts(query_text, contexts, *, retrieved_ids, upstream_timings_ms)`
   — generates from an explicit, already-retrieved+reranked context list: applies the configured
   context processors, enforces the citation-resolves-to-context invariant, records stats, and
   performs **no** retrieval/rerank of its own.
2. `corrective.py` now calls `generate_from_contexts(query_text, contexts, …)` in **both** the
   CORRECT (reranked-initial) and corrected (RRF-fused) paths, so the answer is built from the
   contexts the pipeline actually selected.
3. Strengthened `test_corrective_rag.py`: asserts the first citation is the corrected recall@1 (B)
   **and** that B's scientific vocabulary ("mitosis") appears in the answer text — both fail on the
   old bypass code (vanilla retrieval surfaces only A's lay vocabulary). Non-tautological.

---

## Verified CORRECT — technique output *does* reach the answer

| Pipeline | Generator call | Verdict |
|----------|----------------|---------|
| **Decomposition** (`decompositional.py`) | `generate(original, accumulated_contexts)` — line 126 | CORRECT: per-sub-question chunks reach the generator |
| **Iterative** (`iterative.py`) | `generate(original, accumulated)` — line 175 | CORRECT: multi-hop accumulated chunks reach the generator |
| **Adaptive** (`adaptive.py`) | NO_RETRIEVAL → `generate(q, [])`; else standard path | CORRECT: routing-only by design; empty-context path is a real compute saving |

(Advisory: decomposition/iterative call their injected generator directly and therefore skip the
optional context-processors + stats. Not a correctness bug — those are separate opt-in features —
but they could route through `generate_from_contexts` for consistency in a future pass.)

---

## Verified CORRECT & honestly labeled — technique adapters

| Technique | Source | Finding |
|-----------|--------|---------|
| **Contextual Retrieval prompt** (`llm_contextualizer.py`) | Anthropic cookbook 2024 | Prompt matches the official cookbook essentially **verbatim** (`<document>…</document>` + "situate this chunk… Answer only with the succinct context and nothing else"); doc sent as a cacheable block. CORRECT |
| **Late chunking** (`late_chunking.py`) | Günther et al. arXiv:2409.04701 | Docstring cleanly separates the real long-context-token-pooling path (opt-in, not built) from the deterministic neighbor-window mean-pool analogue. Honest; `alpha=0` ≡ base. No overclaim |
| **Lost-in-the-middle reorder** (`reorderer.py`) | Liu et al. 2023 | Alternating-edge interleave (rank1→pos0, rank2→last, …) = LangChain `LongContextReorder`. CORRECT |
| **int8 quantization** (`quantized_in_memory.py`) | Qdrant/ES scalar-quant rescoring | Symmetric per-vector max-abs int8 + oversized candidate set + exact float32 cosine rescore. CORRECT; approximation honestly documented |
| **Instruction embedding** (`instruction.py`) | E5-instruct / Qwen3-Embedding | `Instruct: {task}\nQuery: {text}`, query-only asymmetric — **verbatim** convention. CORRECT |
| **Proposition chunking** (`proposition.py`) | Chen et al. EMNLP 2024 | Honestly labeled: real paper uses a Flan-T5 Propositionizer; Jera's is a deterministic "one sentence = one proposition" approximation + heading-breadcrumb self-containment. CORRECT (honest) |
| **CRAG grader** (`overlap_evaluator.py`) | Yan et al. 2025 | Faithfully maps CRAG's three actions (CORRECT/AMBIGUOUS/INCORRECT) via score+Jaccard gates; paper's fine-tuned evaluator honestly noted as future opt-in. CORRECT (honest) |

---

## Summary

| | Count |
|---|---|
| Real data-flow bugs found & fixed | **1** (CRAG answer bypass) |
| Pipelines verified correct | 3 (decomposition, iterative, adaptive) |
| Technique adapters verified correct/honest | 7 |
| Tests strengthened (now catch the bug class) | 1 (`test_corrective_rag.py`) |

**Lesson (same as round 1):** the bug hid because the test asserted on the *intermediate*
(`contexts`) not the *outcome* (`answer`). Tracing each technique's data flow all the way to the
generated answer — not just checking it produces *a* result — is what surfaced it.
