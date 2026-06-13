# M12 RAG Evaluation SOTA Research (2025–2026)

> Research date: 2026-06-14  
> Scope: 2025–2026 SOTA in RAG evaluation, faithfulness, hallucination detection  
> System context: Jera — hexagonal, offline-first RAG. Deterministic CI via hash embeddings + BM25 + in-memory store.  
> Already implemented in `evaluation_contracts/`: faithfulness (sentence containment), answer_relevance (cosine), answer_correctness (token-F1), context_precision (average-precision), recall@k, MRR, nDCG, citation_faithfulness, numeric_accuracy + GenerationEvalRunner.  
> Constraint: every new metric must be provable offline with a non-tautological deterministic test.

---

## Table of Contents

1. [RAGChecker — claim-level entailment framework](#1-ragchecker)
2. [RAGAS newer metrics (Noise Sensitivity, Factual Correctness, etc.)](#2-ragas-newer-metrics)
3. [ARES — fine-tuned LM judges with PPI](#3-ares)
4. [TruLens / RAG Triad](#4-trulens--rag-triad)
5. [CRUD-RAG benchmark](#5-crud-rag)
6. [RGB benchmark](#6-rgb)
7. [RAGTruth — span-level hallucination corpus](#7-ragtruth)
8. [LettuceDetect — encoder hallucination detector](#8-lettucedetect)
9. [FActScore / OpenFActScore](#9-factscore--openfactscore)
10. [SelfCheckGPT and sampling-consistency methods](#10-selfcheckgpt--sampling-consistency)
11. [LLM-as-judge reliability & bias (2025–26)](#11-llm-as-judge-reliability--bias)
12. [Context attribution / utilization metrics](#12-context-attribution--utilization-metrics)
13. [TREC 2025 RAG Track — weighted precision/recall](#13-trec-2025-rag-track)
14. [LiveRAG / SIGIR 2025 Challenge](#14-liverag--sigir-2025)
15. [The Semantic Illusion — certified limits of embedding-based detection](#15-the-semantic-illusion)
16. [ARC-JSD — Jensen-Shannon context attribution](#16-arc-jsd)
17. [TOP 3 PICKS for Jera implementation](#top-3-picks)

---

## 1. RAGChecker

**What it is:** A fine-grained claim-level evaluation framework that decomposes both the gold answer and the generated response into atomic verifiable claims using an LLM extractor, then checks entailment relations between those claims and the retrieved context via an entailment checker. Published at NeurIPS 2024 Datasets & Benchmarks Track.

**Citation:** Ru et al., "RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation," arXiv:2408.08067, NeurIPS 2024. https://arxiv.org/abs/2408.08067  
**Repository:** https://github.com/amazon-science/RAGChecker

### Complete Metric Set

RAGChecker exposes three layers of metrics:

**Overall metrics** (cross-pipeline):
- **Precision** — fraction of generated claims entailed by the gold answer
- **Recall** — fraction of gold claims present in the generated response
- **F1** — harmonic mean of precision and recall

**Retriever diagnostics** (grades the retrieval module):
- **Claim Recall** — proportion of gold-answer claims that appear in the retrieved documents (did retrieval surface the right information?)
- **Context Precision** — percentage of retrieved chunks that actually contain relevant claims (signal-to-noise in the retrieved set)
- **Noise Sensitivity (Relevant)** — how often the generator produces wrong claims when a relevant but imperfect chunk is present
- **Noise Sensitivity (Irrelevant)** — how often the generator produces wrong claims when irrelevant chunks are in the context

**Generator diagnostics** (grades the generation module):
- **Faithfulness** — fraction of generated claims entailed by the retrieved context (not gold; grades grounding in the actual retrieved evidence)
- **Self-Knowledge** — fraction of correct generated claims NOT found in the retrieved context (measures the generator's parametric knowledge contribution)
- **Context Utilization** — fraction of retrieved claims actually used in the generation (does the generator actually read the context?)
- **Hallucination** — fraction of generated claims that are neither entailed by the retrieved context nor by the gold answer

### Methodology

1. An LLM (Llama-3-70B-Instruct or similar) extracts atomic claims from text as short declarative sentences.
2. An NLI model checks entailment between each claim and a target text.
3. All metrics derive from binary entailment decisions across claim sets.

**SOTA-vs-hype verdict:** Genuine SOTA for diagnostic granularity. Meta-evaluation shows significantly higher correlation with human judgments than prior metrics. The main limitation is the LLM claim extractor requirement — not free to run.

### Implementability for Jera

**Can it be a deterministic CI metric?** Partially. The full pipeline requires an LLM for claim extraction, but a deterministic stand-in is achievable:

- **Deterministic approximation:** Replace LLM claim extraction with sentence splitting (already in `_sentences()` in `generation_metrics.py`). Replace NLI entailment with the existing containment check (`_containment()`). This gives a sentence-level approximation of the 11 metrics with no external dependencies.
- **Opt-in LLM path:** When an LLM port is available (non-CI), use the real LLM extractor and an NLI model (e.g., a locally cached DeBERTa-MNLI).
- **Plug-in point:** New file `evaluation_contracts/ragchecker_metrics.py` — a pure-function module returning a `RAGCheckerResult` dataclass with all 11 metrics.

**Non-tautological CI test sketch:**
```python
# noise_sensitivity_irrelevant should be zero when an irrelevant chunk does NOT
# cause the generator to hallucinate (irrelevant chunk correctly ignored)
contexts_relevant = ["Paris is the capital of France."]
contexts_with_noise = ["Paris is the capital of France.", "Bananas are yellow fruit."]
answer = "Paris is the capital of France."

score_clean = noise_sensitivity_irrelevant(answer, contexts_relevant, gold="Paris is the capital of France.")
score_noisy = noise_sensitivity_irrelevant(answer, contexts_with_noise, gold="Paris is the capital of France.")
# Both should be 0: the answer is fully grounded, noise chunk did not cause drift
assert score_clean == score_noisy == 0.0

# But an answer that hallucinated during noisy context:
hallucinated_answer = "Paris is the capital of France. Bananas are the main crop of France."
score_hallucinated = noise_sensitivity_irrelevant(hallucinated_answer, contexts_with_noise, gold="Paris is the capital of France.")
assert score_hallucinated > 0.0
```

---

## 2. RAGAS Newer Metrics

**What it is:** RAGAS (arXiv:2309.15217) has grown substantially beyond its original 4 metrics. As of December 2025 documentation updates, it ships a full metric library.

**Citation:** Es et al., "Ragas: Automated Evaluation of Retrieval Augmented Generation," arXiv:2309.15217, EACL 2024. https://arxiv.org/abs/2309.15217  
**Docs:** https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/

### New Metrics Beyond the Original 4 (already in Jera)

| Metric | What it measures | Computation |
|--------|-----------------|-------------|
| **Noise Sensitivity** | How often the system makes wrong claims due to retrieved noise | LLM-based, checks claims against noisy vs. clean contexts |
| **Context Entities Recall** | Fraction of named entities in the gold answer that appear in the retrieved context | String-matching on entity sets (near-deterministic with a NER model) |
| **Factual Correctness** | Semantic-level factual match between generated answer and reference (not just token-F1) | LLM-based claim extraction + verification |
| **Non-LLM String Similarity** | ROUGE, BLEU, CHRF, exact match, string presence | Fully deterministic |
| **Summarization Score** | Quality of a generated summary vs. source document | LLM-based |
| **Tool Call Accuracy / F1** | For agentic RAG: were the right tools called? | Deterministic (exact-match on tool names/args) |
| **Topic Adherence** | Does the agent stay on-topic? | LLM-based |
| **Multimodal Faithfulness/Relevance** | Faithfulness extended to images | LLM-based (vision model) |

**SOTA-vs-hype verdict:** RAGAS is well-maintained and genuinely useful but the core metrics remain LLM-dependent in their "real" form. The non-LLM string-similarity metrics are sound deterministic baselines. Noise Sensitivity is the most novel addition relevant to Jera.

### Implementability for Jera

**Noise Sensitivity** is the highest-value new RAGAS metric. It measures whether the generator produces incorrect claims in the presence of irrelevant retrieved chunks. Deterministic approximation:

- Split generated answer into sentences.
- For each sentence, check if it is NOT grounded in the gold answer tokens but IS found in irrelevant chunks → that sentence is "noise-induced."
- `noise_sensitivity = noise_induced_sentences / total_sentences`

**Non-tautological test:**
```python
gold = "The capital of France is Paris."
irrelevant_chunk = "Germany borders France and is known for its cars."
hallucinated_answer = "The capital of France is Paris. Germany is part of France."
clean_answer = "The capital of France is Paris."

assert noise_sensitivity(clean_answer, [irrelevant_chunk], gold) == 0.0
assert noise_sensitivity(hallucinated_answer, [irrelevant_chunk], gold) > 0.0
```

**Plug-in:** `evaluation_contracts/generation_metrics.py` — add `noise_sensitivity(answer, contexts, gold_answer)` function.

---

## 3. ARES

**What it is:** Automated RAG Evaluation System. Fine-tunes lightweight LM judges on synthetic data to classify (query, passage, answer) triples as relevant/faithful/relevant-answer. Uses Prediction-Powered Inference (PPI) with a small human-annotated set to produce confidence intervals.

**Citation:** Saad-Falcon et al., "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems," arXiv:2311.09476, NAACL 2024. https://arxiv.org/abs/2311.09476  
**Docs:** https://docs.anyscale.com/rag/evaluation

### Metrics

- **Context Relevance** — is the retrieved passage relevant to the query?
- **Answer Faithfulness** — is the answer grounded in the passage?
- **Answer Relevance** — does the answer address the query?
- All three produce **confidence intervals** via PPI when even 50–150 human labels are available.

**SOTA-vs-hype verdict:** Strong for production use where you have some labeled data; PPI confidence intervals are a genuine contribution over point estimates. Fine-tuning burden makes it heavier than RAGAS. Less novel in 2025–26 given the broader evaluation ecosystem.

### Implementability for Jera

Low priority. ARES requires training a judge model, which conflicts with the offline-first/deterministic CI requirement. The PPI statistical framework is interesting but needs human labels (violates the "non-tautological deterministic test" constraint). The three metrics are functionally covered by existing Jera contracts.

---

## 4. TruLens / RAG Triad

**What it is:** TruLens implements the "RAG Triad" — three LLM-as-judge evaluations: Context Relevance, Groundedness (faithfulness), and Answer Relevance. Groundedness decomposes the answer into statements and checks each against the context. Snowflake engineering blog (2025) shows eval-guided optimization of the LLM judges.

**Citation:** Snowflake Engineering Blog, "Benchmarking LLM-as-a-Judge for the RAG Triad Metrics," 2025. https://www.snowflake.com/en/engineering-blog/benchmarking-LLM-as-a-judge-RAG-triad-metrics/  
**Docs:** https://www.trulens.org/getting_started/core_concepts/rag_triad/

**SOTA-vs-hype verdict:** TruLens groundedness (statement-level decomposition) is more robust than simple sentence-containment but requires an LLM call per answer. The RAG Triad concept is mature and well-understood. No novel contributions beyond what RAGAS/RAGChecker offer, but the observability and dashboard integration are production-friendly.

### Implementability for Jera

The Groundedness metric's statement decomposition is essentially the same operation as RAGChecker's claim extraction. No new metric contract is needed; TruLens is a framework, not a novel metric. Skip.

---

## 5. CRUD-RAG

**What it is:** A comprehensive Chinese-language RAG benchmark with four task categories mirroring database operations: Create (text continuation), Read (QA), Update (hallucination modification), Delete (open-domain multi-document summarization). Published in ACM TOIS 2025.

**Citation:** Lyu et al., "CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models," ACM TOIS, 2025. https://arxiv.org/abs/2401.17043

**SOTA-vs-hype verdict:** Valuable as a benchmark covering underserved task types (especially Update/hallucination modification and multi-document summarization). Not a metric framework — it is a dataset + task taxonomy. Genuine contribution for Korean-language RAG evaluation by analogy (Jera already has a Korean eval track from M4).

### Implementability for Jera

**Not a metric — it is a benchmark structure.** The "Update" task (hallucination modification: given a hallucinated answer and context, fix the hallucination) is an interesting evaluation format for Jera's Korean RAG track. No new `evaluation_contracts` function needed; the existing `answer_correctness` + `faithfulness` contracts apply. The structural insight (Read/Update/Delete task taxonomy) could inform gold dataset construction in `gold_builder.py`.

---

## 6. RGB

**What it is:** A bilingual (EN + ZH) RAG benchmark evaluating four fundamental LLM abilities in RAG contexts: (1) Noise Robustness — filtering irrelevant retrieved content; (2) Negative Rejection — abstaining when no reliable information is retrieved; (3) Information Integration — synthesizing multi-document answers; (4) Counterfactual Robustness — resisting false/outdated retrieved information. Published at AAAI 2024.

**Citation:** Chen et al., "Benchmarking Large Language Models in Retrieval-Augmented Generation," AAAI 2024. https://arxiv.org/abs/2309.01431

**SOTA-vs-hype verdict:** Groundbreaking when published; the four-axis taxonomy is widely cited. Findings (LLMs handle noise decently but fail at negative rejection and counterfactual robustness) have been validated by subsequent work. Less novel in 2026 but the task structure remains a useful design checklist.

### Implementability for Jera

The **Negative Rejection** axis is the highest-value underdeveloped area for Jera. It asks: when the retrieved context does NOT contain the answer, does the system correctly say "I don't know" rather than hallucinate?

**Deterministic metric sketch:** `abstention_rate` — given a set of queries where the gold context is absent from the retrieved set, what fraction of answers contain a hedge/abstention signal vs. hallucinated confident assertions.

**Non-tautological test:**
```python
# When retrieved context is empty/irrelevant, a system that hedges scores 1.0
no_context_answer = "I don't have enough information to answer this question."
hallucinated_answer = "The answer is definitely 42."

assert abstention_score(no_context_answer) > abstention_score(hallucinated_answer)
```

This requires a hedge-phrase lexicon (deterministic) as the CI approximation. The LLM-judge version checks if the claim is assertive despite missing context.

**Plug-in:** New function `evaluation_contracts/generation_metrics.py`: `abstention_score(answer: str) -> float` using a curated hedge-phrase token list.

---

## 7. RAGTruth

**What it is:** A large-scale hallucination corpus for RAG — ~18,000 naturally generated responses from diverse LLMs (GPT-4, Llama-2, Mistral), annotated at the word/span level across three tasks (QA, data-to-text, news summarization). Defines four hallucination types: Evident Conflict, Subtle Conflict, Evident Baseless Information, Subtle Baseless Information. Published at ACL 2024.

**Citation:** Wu et al., "RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models," ACL 2024. https://aclanthology.org/2024.acl-long.585/

**SOTA-vs-hype verdict:** RAGTruth is the canonical benchmark for span-level hallucination detection in RAG. The four-type taxonomy (Evident/Subtle × Conflict/Baseless) is clinically precise. Key finding: a small LLM fine-tuned on RAGTruth achieves GPT-4-comparable detection, validating the "small model, good labels" hypothesis.

### Implementability for Jera

RAGTruth itself is a corpus, not a metric function. Its value for Jera:
1. **Test fixture:** RAGTruth examples (or Jera-created analogs) make excellent non-tautological test cases for `faithfulness()` — an Evident Conflict example MUST score lower than a clean answer.
2. **Hallucination taxonomy:** The four-type taxonomy (Evident/Subtle × Conflict/Baseless) should label any future Jera test fixtures to distinguish surface errors from semantic drift.

No new `evaluation_contracts` function needed; use RAGTruth structure to strengthen existing test fixtures in `tests/unit/test_generation_metrics.py`.

---

## 8. LettuceDetect

**What it is:** A token-classification hallucination detector for RAG, built on ModernBERT (8k context window). Input: (context, question, answer) triple. Output: token-level labels of hallucinated spans. Trained on RAGTruth. Achieves F1 79.22% on RAGTruth example-level detection — 14.8% above previous encoder SOTA (Luna). ~30× smaller than GPT-4 judge. Published February 2025.

**Citation:** Kovács & Recski, "LettuceDetect: A Hallucination Detection Framework for RAG Applications," arXiv:2502.17125, 2025. https://arxiv.org/abs/2502.17125  
**HuggingFace:** https://huggingface.co/blog/adaamko/lettucedetect

**SOTA-vs-hype verdict:** Genuine SOTA for encoder-based hallucination detection. The shift from DeBERTa (512-token limit) to ModernBERT (8k) is practically important for long-context RAG. The 30×/inference-speed advantage over LLM judges is real. However, it requires GPU for the 30–60 ex/sec throughput; CPU-only inference is much slower.

### Implementability for Jera

**High practical value, but requires an opt-in model port.**

- **Deterministic CI approximation:** Use the existing `faithfulness()` containment check as the CI stand-in. The LettuceDetect model becomes an opt-in real model behind a port (like how embedding is abstracted).
- **Architecture:** New port `HallucinationDetectorPort` (Protocol) in `domain/` with a `detect_hallucinated_spans(context: str, question: str, answer: str) -> list[TextSpan]` method. The CI implementation returns `[]` (no spans) for trivially grounded answers, verified by the existing containment logic.
- **Non-tautological CI test:**
```python
# A fully-contained sentence should have no hallucinated spans
context = "The capital of France is Paris."
answer = "The capital of France is Paris."
assert detect_spans_deterministic(context, answer) == []

# An introduced unsupported claim should yield a non-empty span list
answer_with_hallucination = "The capital of France is Paris. The Eiffel Tower was built in 1850."
assert len(detect_spans_deterministic(context, answer_with_hallucination)) > 0
```
- **Plug-in:** `src/jera/domain/hallucination_detector.py` (Protocol port) + deterministic adapter in `evaluation_contracts/` using sentence-level containment to approximate span detection.

---

## 9. FActScore / OpenFActScore

**What it is:** FActScore (EMNLP 2023) decomposes a generation into atomic claims and computes the percentage supported by a reliable knowledge source (dense retrieval + NLI). OpenFActScore (arXiv July 2025) is a fully open-source reimplementation with modular claim handling and single-pass decompose+verify.

**Citation (original):** Min et al., "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation," EMNLP 2023. https://arxiv.org/abs/2305.14251  
**Citation (open):** OpenFActScore, arXiv:2507.05965, July 2025.  
**VeriFastScore:** arXiv:2505.16973, May 2025 — single-pass efficiency improvement.

**SOTA-vs-hype verdict:** FActScore established the atomic-claim paradigm. OpenFActScore modernizes it. The core idea (claim decomposition → per-claim verification) is directly embodied in RAGChecker with better RAG-specific framing. VeriFastScore (2025) shows you can do decompose+verify in a single LLM call. For long-form generation quality (not RAG-specific), FActScore remains the reference metric.

### Implementability for Jera

The claim decomposition step requires an LLM. However, the *structure* (atomic claims as the unit of evaluation) is already approximated by `faithfulness()` which uses sentence splitting as the claim proxy.

**If Jera wants a higher-fidelity claim-level metric without an LLM:**
- Use syntactic chunking (noun-phrase + verb-phrase extraction) as a deterministic claim approximation.
- Each syntactic chunk is a "claim"; check containment against context.
- This is a stronger proxy than sentence splitting because one sentence can contain multiple independent claims.

**Non-tautological test:**
```python
# "Paris is the capital. The population is 2 million." → 2 claims
# Context only supports claim 1 → claim_precision = 0.5
context = "Paris is the capital of France."
answer = "Paris is the capital. The population is 2 million."
assert 0.4 < claim_precision_approx(answer, context) < 0.6
```

**Verdict for Jera:** Skip FActScore proper; implement the syntactic-claim approximation as an upgrade to `faithfulness()` (finer granularity at same compute cost). Medium priority.

---

## 10. SelfCheckGPT & Sampling Consistency

**What it is:** SelfCheckGPT (EMNLP 2023) generates a deterministic response + N stochastic samples, then measures consistency across samples via BERTScore, NLI contradiction probability, or n-gram overlap. Inconsistency → hallucination signal. Works black-box, no reference needed. Recent work (2025) applies self-consistency to chain-of-thought reasoning and key-fact consistency.

**Citation:** Manakul et al., "SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models," EMNLP 2023. https://arxiv.org/abs/2303.08896  
**2025 follow-up:** "Consistency Is the Key: Detecting Hallucinations by Checking Inconsistencies About Key Facts," arXiv:2511.12236.

**SOTA-vs-hype verdict:** Clever zero-resource approach, AUROC 0.74–0.76 on WikiBio. The "Semantic Illusion" paper (Section 15) shows that both embedding and NLI consistency checks fail on real RLHF-aligned model hallucinations (100% FPR at 95% recall). Sampling consistency is thus a research tool, not a production-grade detector for modern aligned models.

### Implementability for Jera

**Low priority.** SelfCheckGPT requires multiple LLM calls (N samples per answer), which is expensive and incompatible with deterministic CI. The NLI variant requires a loaded NLI model. No deterministic offline approximation exists that isn't also caught by simpler metrics. Skip for CI; note as an opt-in research metric.

---

## 11. LLM-as-Judge Reliability & Bias (2025–26)

**What it is:** A large body of 2025–26 work systematically characterizing when and how LLM judges fail. Key findings across arXiv:2602.02219, arXiv:2603.01865, ACL Anthology 2025, and FutureAGI survey:

**Key biases documented:**
1. **Position bias** — LLMs favor whichever response appears first (or last) in a pairwise comparison; 10–15 point winrate swing on close calls. Mitigation: randomize order + average both permutations.
2. **Verbosity bias** — longer responses score higher independent of quality. Mitigation: length-normalized rubrics.
3. **Self-preference bias** — a model judging its own output scores it higher.
4. **Calibration drift** — off-the-shelf judges validated on chat conversations are unreliable on domain-specific tasks (RAG, code review) without recalibration. Divergence >20–25% signals recalibration need.
5. **Format bias** — responses with markdown formatting, bullet points, or bold text score artificially higher.

**Citations:**  
- "A Systematic Study of Position Bias in LLM-as-a-Judge," IJCNLP 2025. https://aclanthology.org/2025.ijcnlp-long.18.pdf  
- "CyclicJudge: Mitigating Judge Bias Efficiently in LLM-based Evaluation," arXiv:2603.01865. https://arxiv.org/pdf/2603.01865  
- "Am I More Pointwise or Pairwise? Revealing Position Bias," arXiv:2602.02219.  
- "Case-Aware LLM-as-a-Judge Evaluation for Enterprise-Scale RAG Systems," arXiv:2602.20379.  
- Snowflake Engineering Blog, 2025: https://www.snowflake.com/en/engineering-blog/benchmarking-LLM-as-a-judge-RAG-triad-metrics/

**SOTA-vs-hype verdict:** The reliability characterization work is mature and converging. The consensus: LLM judges are useful but systematically biased; bias can be partially mitigated via (1) order randomization, (2) structured JSON output, (3) evidence-constrained prompts, (4) calibration against a small human-labeled set. Frontier models achieve ~7% FPR vs. 100% for embeddings on hard hallucinations (Semantic Illusion paper).

### Can a deterministic stand-in judge exist for CI?

**Yes — with defined scope.** A deterministic judge can provably audit:
- Token-set grounding (existing `faithfulness()`) — approximates factual faithfulness.
- Hedge-phrase detection (abstention appropriateness).
- Length ratio (verbosity proxy — does response length match query complexity?).
- Citation resolution (existing `citation_faithfulness()`).

A deterministic judge CANNOT approximate:
- Semantic reasoning about multi-hop claims.
- Detecting subtle conflicts (paraphrase-level contradictions).
- Evaluating response coherence or fluency.

**Recommendation for Jera:** Document that `GenerationEvalRunner` uses deterministic judges in CI and LLM judges are opt-in. Add a `JudgeMode` enum (`DETERMINISTIC` / `LLM_JUDGE`) to the runner configuration — already present architecturally in the hexagonal design.

---

## 12. Context Attribution / Utilization Metrics

**What it is:** A cluster of 2025 metrics measuring whether the generator actually *uses* the retrieved context (distinct from whether it *could* — the context may be present but ignored):

- **Context Utilization** (RAGChecker): fraction of retrieved claims actually reflected in the generated response.
- **Citation Precision/Recall** (TREC 2025 RAG, various): fraction of generated sentences that correctly cite a supporting passage (precision) and fraction of answer sentences supported by cited passages (recall).
- **ARC-JSD** (arXiv:2505.16415, 2025): Jensen-Shannon divergence between output distributions with vs. without each context sentence — identifies which sentences causally affected the output. Deterministic given the generator's output probabilities.

**Citation:**
- TREC 2025 RAG Track, arXiv:2603.09891.
- "Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study," arXiv:2505.16415.

**SOTA-vs-hype verdict:** Context utilization is an underserved metric dimension — existing metrics grade whether the answer *is* grounded, not whether the context *was used*. Citation precision/recall (TREC 2025) is the emerging standard for attributed generation. ARC-JSD is novel and computationally efficient but requires access to the generator's output token probabilities (available if Jera's `GeneratorPort` exposes logprobs).

### Implementability for Jera

**Citation Precision and Recall** are directly implementable as deterministic pure functions:

- `citation_precision(cited_ids, context_ids, answer_sentences)` — fraction of cited chunks that actually support the corresponding sentence (approximated: cited chunk tokens overlap with sentence tokens).
- `citation_recall(answer_sentences, cited_ids, context_ids)` — fraction of answer sentences that have a supporting citation.

Jera's `citation_faithfulness()` in `metrics.py` already captures one direction (cited_ids ⊆ retrieved_ids). Extending to sentence-level attribution is the next step.

**Non-tautological test:**
```python
# Answer with uncited sentence has lower citation_recall than fully cited answer
answer_sents = ["Paris is the capital.", "The Eiffel Tower is famous."]
cited_for_sent1 = ["chunk-paris"]  # only first sentence has a citation
all_cited = ["chunk-paris", "chunk-eiffel"]

assert citation_recall(answer_sents, cited_for_sent1, ...) < citation_recall(answer_sents, all_cited, ...)
```

**Plug-in:** `evaluation_contracts/metrics.py` — add `citation_precision` and `citation_recall` as companions to existing `citation_faithfulness`.

---

## 13. TREC 2025 RAG Track

**What it is:** The official NIST RAG evaluation track (2025), with 70+ participating teams. Introduces **Weighted Support Precision** and **Weighted Support Recall** as citation-aware metrics graded at three levels: Full Support (1.0), Partial Support (0.5), No Support (0.0). Automated judge: GPT-4o/120B; manual: NIST assessors. High Kendall's τ between automated and manual assessments.

**Citation:** "Overview of the TREC 2025 Retrieval Augmented Generation (RAG) Track," arXiv:2603.09891. https://arxiv.org/pdf/2603.09891

**SOTA-vs-hype verdict:** TREC 2025 weighted precision/recall is becoming the community standard for attributed generation evaluation. The graded support levels (Full/Partial/No) are more nuanced than binary citation checks.

### Implementability for Jera

The Full/Partial/No support taxonomy translates to a deterministic approximation:
- **Full Support:** ≥80% token overlap between cited chunk and answer sentence.
- **Partial Support:** 40–80% token overlap.
- **No Support:** <40% token overlap.

`weighted_citation_precision(answer_sents, citations, contexts) -> float` and `weighted_citation_recall(answer_sents, citations, contexts) -> float` can be implemented as pure functions in `evaluation_contracts/metrics.py`.

**Non-tautological test:**
```python
# Partial support answer scores 0.5, full support scores 1.0
# Weighted precision should reflect this
fully_supported_answer = ["Paris is the capital of France."]
partially_supported_answer = ["Paris is the capital of France and home to 12 million people."]
context = "Paris is the capital of France."

assert weighted_citation_precision(fully_supported_answer, ["c1"], {"c1": context}) == 1.0
assert weighted_citation_precision(partially_supported_answer, ["c1"], {"c1": context}) == 0.5
```

---

## 14. LiveRAG / SIGIR 2025

**What it is:** The SIGIR 2025 LiveRAG Challenge — 70 teams, 500 unseen test questions, 2-hour live window. Evaluation metrics:
- **Correctness**: harmonic mean of Coverage (fraction of reference vital claims in the answer) and Relatedness (fraction of generated claims relevant to the query).
- **Faithfulness**: LLM-as-judge automated first pass; manual review of top submissions.
- **IRT-derived difficulty scores** per question — discriminability and difficulty proxy from Item Response Theory.

**Citation:** "SIGIR 2025 — LiveRAG Challenge Report," arXiv:2507.04942. https://arxiv.org/abs/2507.04942

**SOTA-vs-hype verdict:** The Coverage/Relatedness decomposition of correctness is a useful split of what token-F1 blends together. IRT-based difficulty scoring is novel for RAG benchmarks. The live/online evaluation format is operationally interesting but not metric-novel.

### Implementability for Jera

**Coverage** (vital claims in answer / total vital claims in reference) is functionally equivalent to `answer_correctness()` with a claim-level rather than token-level decomposition. **Relatedness** (generated claims relevant to query / total generated claims) parallels `answer_relevance()`.

No new metric needed; these are refinements of existing contracts. The IRT difficulty scoring is an interesting addition to Jera's gold dataset builder (`gold_builder.py`) — not a metric function.

---

## 15. The Semantic Illusion

**What it is:** A 2024 paper (December 2024, arXiv:2512.15068) that applies conformal prediction to RAG hallucination detection and finds a fundamental limitation: embedding similarity and NLI methods achieve near-perfect precision on *synthetic* hallucinations but fail catastrophically on *real* RLHF-aligned model hallucinations — 100% FPR at 95% recall on HaluEval. GPT-4 as a judge achieves 7% FPR on the same data, proving the task is solvable via reasoning but opaque to surface-level semantics.

**Citation:** "The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection in RAG Systems," arXiv:2512.15068. https://arxiv.org/abs/2512.15068

**SOTA-vs-hype verdict:** Critical negative result. Directly invalidates the assumption that token-containment or embedding-similarity faithfulness metrics are reliable detectors of *subtle* hallucination in production. Simple containment (Jera's current `faithfulness()`) is a reliable screen for *overt* unsupported claims but WILL miss subtle conflicts and paraphrase-level hallucinations.

### Implications for Jera

1. `faithfulness()` is correct and well-scoped as a deterministic CI metric for *overt* hallucination (claim not in context at all). The docstring should explicitly state this scope.
2. Subtle/semantic hallucination detection REQUIRES an LLM judge or a fine-tuned NLI model (LettuceDetect) — this should be the opt-in real-model path behind a port.
3. NLI models (DeBERTa-MNLI) are *better* than pure embedding similarity but still fail at 95% recall on real hallucinations. LettuceDetect (fine-tuned on RAGTruth) is the current best encoder-only option.

**Action item:** Add a code comment / docstring clarification to `faithfulness()` in `generation_metrics.py`: "This is a surface-level containment check. It reliably detects overt unsupported claims but will miss paraphrase-level or subtle semantic conflicts. See HallucinationDetectorPort for fine-tuned detection."

---

## 16. ARC-JSD

**What it is:** Context attribution via Jensen-Shannon Divergence (arXiv:2505.16415, May 2025). Computes the divergence between the model's output probability distribution with vs. without each context sentence. A high JSD for a context sentence = that sentence causally influenced the output. Requires no fine-tuning, no gradients, no surrogate model — just forward passes with and without each context sentence.

**Citation:** "Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation," arXiv:2505.16415. https://arxiv.org/pdf/2505.16415

**SOTA-vs-hype verdict:** Elegant and computationally efficient (O(n) forward passes where n = number of context sentences). However, requires access to model output logits — available only when Jera's GeneratorPort exposes `logprobs`. For black-box generators, inapplicable.

### Implementability for Jera

Medium-term: if `GeneratorPort` is extended to expose token-level log-probabilities, `arc_jsd_attribution` can be a pure function given the probability tensors. No LLM needed beyond the generator's own outputs.

```python
def arc_jsd_attribution(
    logprobs_with_context: list[float],   # from generator with full context
    logprobs_per_sentence: list[list[float]],  # from generator with each sentence removed
) -> list[float]:
    """Return JSD per context sentence — higher = more influential."""
    ...
```

CI test: removing the single most relevant sentence should produce higher JSD than removing an irrelevant sentence.

---

## TOP 3 PICKS

Ranked by **value-per-effort** for Jera's offline-first hexagonal architecture, CI-deterministic constraint, and existing `evaluation_contracts/` foundation.

---

### PICK 1: Noise Sensitivity (from RAGChecker + RAGAS)

**Value:** Directly measures a known RAG failure mode: the generator hallucinating because an irrelevant chunk "triggered" spurious content. This is orthogonal to the existing `faithfulness()` — faithfulness grades grounding, noise sensitivity grades *what caused* failures. It answers a diagnostic question: "is poor faithfulness caused by retrieval noise or by generator weakness?"

**Effort:** Low. Deterministic approximation reuses `_sentences()`, `_tokens()`, and `_containment()` already in `generation_metrics.py`.

**Implementation:**
```python
# evaluation_contracts/generation_metrics.py
def noise_sensitivity(
    answer_text: str,
    context_texts: Sequence[str],
    gold_text: str,
    *,
    support_threshold: float = 0.6,
) -> float:
    """Fraction of answer sentences that are unfaithful to gold AND traceable to noisy context.

    A sentence is "noise-induced" when it:
    1. Is NOT grounded in gold_text tokens (it deviates from the reference).
    2. BUT shares tokens with a context chunk that is itself not grounded in gold_text
       (i.e., the context chunk is "noisy" — it introduced off-topic content).

    Returns 0.0 when the generator ignores noisy chunks; returns > 0.0 when noisy
    chunks cause the generator to produce unfaithful content.
    """
    ...
```

**Non-tautological test:**
```python
gold = "The Eiffel Tower is in Paris."
noise_chunk = "The Louvre has 8 million visitors and is in Paris."
answer_drifted = "The Eiffel Tower is in Paris. The Louvre has 8 million visitors."
answer_clean = "The Eiffel Tower is in Paris."

assert noise_sensitivity(answer_clean, [noise_chunk], gold) == 0.0     # no drift
assert noise_sensitivity(answer_drifted, [noise_chunk], gold) > 0.0    # drift caused by noise
```

**Plug-in:** `evaluation_contracts/generation_metrics.py`; test in `tests/unit/test_generation_metrics.py`.

---

### PICK 2: Weighted Citation Precision + Citation Recall (TREC 2025 / RAGChecker-style)

**Value:** Jera already has `citation_faithfulness()` (are cited IDs in the retrieved set?). The missing dimension is *per-sentence attribution*: does each answer sentence have a supporting cited chunk? TREC 2025's weighted support score (Full=1.0, Partial=0.5, No=0.0) is the emerging community standard and maps cleanly to token-overlap thresholds.

**Effort:** Low-medium. Two new pure functions in `metrics.py`. The Full/Partial/No classification reuses token-level overlap already implemented in `_containment()`.

**Implementation:**
```python
# evaluation_contracts/metrics.py
def citation_precision(
    answer_sentences: Sequence[str],
    cited_chunks: Mapping[int, str],   # sentence_idx -> chunk_text
) -> float:
    """Weighted fraction of citations that support their corresponding sentence.
    Full support (≥0.8 containment) = 1.0, Partial (0.4–0.8) = 0.5, None = 0.0.
    """
    ...

def citation_recall(
    answer_sentences: Sequence[str],
    cited_chunks: Mapping[int, str],
) -> float:
    """Fraction of answer sentences that have at least partial citation support."""
    ...
```

**Non-tautological test:**
```python
sents = ["Paris is the capital.", "Mars has two moons."]
chunks = {0: "Paris is the capital of France.", 1: "Mars has two moons named Phobos and Deimos."}
empty_citations = {}

assert citation_recall(sents, chunks) == 1.0         # both sentences cited
assert citation_recall(sents, empty_citations) == 0.0  # no citations
assert citation_recall(["Some hallucination."], {0: "Paris is a city."}) < 1.0
```

**Plug-in:** `evaluation_contracts/metrics.py`; test in `tests/unit/test_matrix.py` or new `test_citation_metrics.py`.

---

### PICK 3: Abstention Score (from RGB / Negative Rejection)

**Value:** The highest-scoring underdeveloped dimension in RGB (AAAI 2024) — LLMs consistently fail at negative rejection (saying "I don't know" when context is absent or irrelevant). Jera's existing metrics all assume the generator *should* answer; no metric grades whether it *correctly abstains*. Adding `abstention_score` closes this gap and is directly relevant to Jera's extractive generator (which should abstain when no grounded sentence can be extracted).

**Effort:** Very low. A deterministic hedge-phrase lexicon (no model). Opt-in: LLM judge checks if the response makes an unsupported confident assertion.

**Implementation:**
```python
# evaluation_contracts/generation_metrics.py

_HEDGE_PATTERNS = re.compile(
    r"\b(i don't know|i do not know|cannot determine|not enough information|"
    r"insufficient|unable to answer|no information|unclear from|not specified|"
    r"모르겠|알 수 없|정보가 없|확인할 수 없)\b",
    re.IGNORECASE | re.UNICODE,
)

def abstention_score(answer_text: str) -> float:
    """1.0 if the answer contains a hedge/abstention signal, 0.0 otherwise.

    Deterministic CI stand-in for RGB's Negative Rejection dimension. An answer
    that appropriately abstains when context is absent should score 1.0; a
    confident assertion without context should score 0.0.
    """
    return 1.0 if _HEDGE_PATTERNS.search(answer_text) else 0.0
```

**Non-tautological test:**
```python
assert abstention_score("I don't have enough information to answer.") == 1.0
assert abstention_score("The answer is definitely Paris.") == 0.0
assert abstention_score("모르겠습니다.") == 1.0   # Korean: "I don't know"
```

**Plug-in:** `evaluation_contracts/generation_metrics.py`; test in `tests/unit/test_generation_metrics.py`. Can be used in `GenerationEvalRunner` to grade abstention on the Korean RAG track's unanswerable questions.

---

## Summary Table

| # | Technique | Year | Genuinely Novel? | Offline-CI Approximation | Effort | Jera Priority |
|---|-----------|------|-----------------|--------------------------|--------|---------------|
| 1 | RAGChecker (11 metrics) | NeurIPS 2024 | Yes — diagnostic depth | Sentence-split + containment | Medium | High (use struct, build 3 new metrics) |
| 2 | RAGAS Noise Sensitivity | 2025 | Yes — distinct from faithfulness | Token-level noise attribution | Low | **PICK 1** |
| 3 | ARES | NAACL 2024 | Moderate (PPI intervals) | No — needs training data | High | Low |
| 4 | TruLens / RAG Triad | 2023–25 | No (framework, not metric) | — | — | Skip |
| 5 | CRUD-RAG | ACM TOIS 2025 | Yes (task taxonomy) | Dataset structure only | — | Low (gold_builder insight) |
| 6 | RGB | AAAI 2024 | Yes (4-axis taxonomy) | Abstention = hedge phrases | Low | **PICK 3** |
| 7 | RAGTruth | ACL 2024 | Yes (span-level corpus) | Use as test fixtures | Low | Medium (test fixtures) |
| 8 | LettuceDetect | Feb 2025 | Yes (8k-context encoder) | Containment as CI stand-in | Medium | High (opt-in port) |
| 9 | FActScore / OpenFActScore | EMNLP 2023 / 2025 | Moderate (precursor to RAGChecker) | Syntactic-chunk proxy | Medium | Low |
| 10 | SelfCheckGPT | EMNLP 2023 / 2025 | No (sampling-based, LLM-required) | None viable | High | Skip |
| 11 | LLM-as-judge bias | 2025–26 | Yes (systematic characterization) | Deterministic judge scope doc | Low | Medium (docstring + JudgeMode) |
| 12 | Citation Precision/Recall | TREC 2025 | Yes (emerging standard) | Token-overlap thresholds | Low | **PICK 2** |
| 13 | TREC 2025 RAG Track | 2025 | Yes (weighted support) | Token-overlap weighted score | Low | High (part of PICK 2) |
| 14 | LiveRAG / SIGIR 2025 | 2025 | Moderate (Coverage/Relatedness) | Subsumed by existing metrics | Low | Low |
| 15 | Semantic Illusion | Dec 2024 | Yes (critical negative result) | Docstring scoping action | Minimal | Medium (correctness) |
| 16 | ARC-JSD | May 2025 | Yes (causal attribution) | Needs logprobs from GeneratorPort | Medium | Low (future) |

---

## Sources

- [RAGChecker: A Fine-grained Framework for Diagnosing RAG — arXiv:2408.08067](https://arxiv.org/abs/2408.08067) (Ru et al., NeurIPS 2024)
- [RAGChecker GitHub — amazon-science/RAGChecker](https://github.com/amazon-science/RAGChecker)
- [RAGChecker NeurIPS paper PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/27245589131d17368cccdfa990cbf16e-Paper-Datasets_and_Benchmarks_Track.pdf)
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation — arXiv:2309.15217](https://arxiv.org/abs/2309.15217)
- [RAGAS available metrics documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [ARES framework — Anyscale Docs](https://docs.anyscale.com/rag/evaluation)
- [TruLens RAG Triad documentation](https://www.trulens.org/getting_started/core_concepts/rag_triad/)
- [Snowflake Engineering Blog: Benchmarking LLM-as-a-Judge for the RAG Triad](https://www.snowflake.com/en/engineering-blog/benchmarking-LLM-as-a-judge-RAG-triad-metrics/)
- [CRUD-RAG: A Comprehensive Chinese Benchmark — arXiv:2401.17043](https://arxiv.org/abs/2401.17043) (Lyu et al., ACM TOIS 2025)
- [RGB: Benchmarking Large Language Models in RAG — arXiv:2309.01431](https://arxiv.org/abs/2309.01431) (Chen et al., AAAI 2024)
- [RAGTruth: A Hallucination Corpus for Trustworthy RAG — ACL 2024](https://aclanthology.org/2024.acl-long.585/)
- [LettuceDetect: A Hallucination Detection Framework for RAG — arXiv:2502.17125](https://arxiv.org/abs/2502.17125) (Kovács & Recski, 2025)
- [LettuceDetect HuggingFace Blog](https://huggingface.co/blog/adaamko/lettucedetect)
- [FActScore: Fine-grained Atomic Evaluation — arXiv:2305.14251](https://arxiv.org/abs/2305.14251) (Min et al., EMNLP 2023)
- [OpenFActScore — arXiv:2507.05965](https://arxiv.org/pdf/2507.05965) (2025)
- [VeriFastScore — arXiv:2505.16973](https://arxiv.org/pdf/2505.16973) (2025)
- [SelfCheckGPT — arXiv:2303.08896](https://arxiv.org/abs/2303.08896) (Manakul et al., EMNLP 2023)
- [Consistency Is the Key: Detecting Hallucinations by Checking Key Facts — arXiv:2511.12236](https://arxiv.org/pdf/2511.12236) (2025)
- [A Systematic Study of Position Bias in LLM-as-a-Judge — IJCNLP 2025](https://aclanthology.org/2025.ijcnlp-long.18.pdf)
- [CyclicJudge: Mitigating Judge Bias Efficiently — arXiv:2603.01865](https://arxiv.org/pdf/2603.01865) (2026)
- [Am I More Pointwise or Pairwise? Revealing Position Bias — arXiv:2602.02219](https://arxiv.org/pdf/2602.02219) (2026)
- [Case-Aware LLM-as-a-Judge for Enterprise-Scale RAG — arXiv:2602.20379](https://arxiv.org/pdf/2602.20379) (2026)
- [TREC 2025 RAG Track Overview — arXiv:2603.09891](https://arxiv.org/pdf/2603.09891)
- [SIGIR 2025 LiveRAG Challenge Report — arXiv:2507.04942](https://arxiv.org/abs/2507.04942)
- [LiveRAG: A diverse Q&A dataset — arXiv:2511.14531](https://arxiv.org/html/2511.14531)
- [The Semantic Illusion: Certified Limits of Embedding-Based Hallucination Detection — arXiv:2512.15068](https://arxiv.org/abs/2512.15068)
- [ARC-JSD: Context Attribution via Jensen-Shannon Divergence — arXiv:2505.16415](https://arxiv.org/pdf/2505.16415) (2025)
- [Benchmarking LLM Faithfulness in RAG with Evolving Challenges — EMNLP 2025 Industry](https://aclanthology.org/2025.emnlp-industry.54.pdf)
- [LLM-as-a-Judge: Why Frontier Models Fail 50%+ Bias Tests — Adaline](https://www.adaline.ai/blog/llm-as-a-judge-reliability-bias)
- [LLM-as-a-Judge Calibration — LangChain](https://www.langchain.com/resources/llm-as-a-judge)
