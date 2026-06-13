# M12 Frontier RAG Research (Oct 2025 – Jun 2026)

> **Scope**: What is genuinely new in RAG since the Jera M5 freeze, ordered by Jera-relevance.
> Already shipped in Jera and therefore **excluded**: hybrid dense+sparse+RRF/DBSF, ColBERT MaxSim, HippoRAG PPR, int8+Matryoshka, contextual retrieval, late chunking, multi-query+HyDE, MMR+listwise reranking, CRAG+Adaptive-RAG+query decomposition, RAGAS-lite, parser/OCR zoo, CRUD lifecycle, observability.

---

## Table of Contents

1. [RL-Trained Retrieval-Reasoning Agents](#1-rl-trained-retrieval-reasoning-agents)
2. [Deep Research Agents (Iterative Plan-Search-Synthesize)](#2-deep-research-agents)
3. [A-RAG: Hierarchical Retrieval Interfaces](#3-a-rag-hierarchical-retrieval-interfaces)
4. [SE-Search: Self-Evolving Search Agent](#4-se-search-self-evolving-search-agent)
5. [Visual Document RAG: ColPali / ColQwen / ColSmolVLM](#5-visual-document-rag)
6. [Qwen3 Embedding: Reasoning-Capable Instruction Embeddings](#6-qwen3-embedding)
7. [Agentic Long-Term Memory: Mem0 + A-MEM + HyperMem](#7-agentic-long-term-memory)
8. [LightRAG + LazyGraphRAG (Graph RAG, Cheaper)](#8-lightrag--lazygraphrag)
9. [Context Compression for Efficient RAG](#9-context-compression-for-efficient-rag)
10. [RAG Security: SDAG Sparse Attention Defense](#10-rag-security-sdag-sparse-attention-defense)
11. [RAG Security: Taxonomy of Attacks and Defenses](#11-rag-security-taxonomy)
12. [Citation Hallucination Detection (FACTUM)](#12-citation-hallucination-detection-factum)
13. [Tool-Schema Compression for Agentic RAG (TSCG)](#13-tool-schema-compression-tscg)
14. [New Evaluation Benchmarks 2025-2026](#14-new-evaluation-benchmarks)
15. [Entity-Event Knowledge Graphs (E²RAG / ChronoQA)](#15-entity-event-knowledge-graphs)
16. [MTEB v2 and the Embedding Benchmark Shift](#16-mteb-v2)
17. [RECON: In-Loop Context Condensation](#17-recon-in-loop-context-condensation)
18. [Top 3 Picks for Jera](#top-3-picks-for-jera)

---

## 1. RL-Trained Retrieval-Reasoning Agents

**What it is**: Language models trained with reinforcement learning to interleave free-form chain-of-thought reasoning with live search calls, producing multi-turn evidence gathering inside a single generation pass.

**Key citations**:
- Search-R1 — "Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning", arXiv:2503.09516, Mar 2025. [Link](https://arxiv.org/pdf/2503.09516)
- DeepDiver — "Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning", arXiv:2505.24332, NeurIPS 2025 Spotlight. [Link](https://arxiv.org/abs/2505.24332)
- SE-Search — "Self-Evolving Search Agent via Memory and Dense Reward", arXiv:2603.03293, Feb 2026. [Link](https://arxiv.org/abs/2603.03293)

**SOTA/adoption verdict**: This is the dominant research frontier for open-domain QA. Search-R1 (extending DeepSeek-R1) shows +26% on 7B models; SE-Search-3B beats Search-R1 by 10.8 points absolute. DeepDiver's Qwen2.5-7B reaches 671B-class performance on WebPuzzle. Adopted actively in open-source; not pure hype—results replicate on standard QA benchmarks.

**Implementability for Jera**:
- The *training loop* requires RL over a live retrieval environment — heavy, not offline-feasible.
- But the *inference pattern* (Think → Search → Memorize within a generation loop) is implementable as an agentic executor port (`AgenticRetrieverPort`) that calls existing Jera retrievers in a loop controlled by a deterministic router.
- **Deterministic CI version**: Mock the "search" step with BM25 over the in-memory store. The agent's outer loop (parse thought, emit query, collect hits, continue) is pure Python and testable without any model. A non-tautological test: given a two-hop question where hop-2 answer is only reachable after hop-1 retrieves the bridging entity, assert the router issues ≥2 retrieval calls and the final answer matches the extractive answer from hop-2's chunk.
- **Port location**: new `AgenticSearchExecutor` adapter under `src/retrieval/agentic/` implementing a `MultiTurnRetrieverPort`. The real-model opt-in wires an LLM to decide when to stop; the offline CI version uses a rule-based "stop if retrieved chunk contains the entity from the question."

---

## 2. Deep Research Agents

**What it is**: Autonomous agents that decompose a complex query into a plan, execute dozens–hundreds of targeted searches across multiple rounds, validate source quality, detect gaps, and synthesize a structured report. Distinguished from IRCoT/CRAG by multi-step *planning* before execution.

**Key citations**:
- "From Web Search towards Agentic Deep Research: Incentivizing Search with Reasoning Agents", arXiv:2506.18959, Jun 2026. [Link](https://arxiv.org/pdf/2506.18959)
- DeepPlanner — "Scaling Planning Capability for Deep Research Agents via Advantage Shaping", arXiv:2510.12979, Oct 2025. [Link](https://arxiv.org/pdf/2510.12979)
- DeepSearchQA benchmark (Google/DeepMind, Dec 2025) — 900 causal-chain tasks, evaluates agent planning capacity. [Link](https://huggingface.co/datasets/google/deepsearchqa)
- "Reasoning RAG via System 1 or System 2", arXiv:2506.10408, Jun 2026 (survey). [Link](https://arxiv.org/pdf/2506.10408)

**SOTA/adoption verdict**: OpenAI Deep Research (o3/o4-mini), Gemini Deep Research Max (up to 160 searches/task), and Perplexity Deep Research are production deployments. Open-source clones exist (HuggingFace Open Deep Research, open-deep-research). Google released DeepSearchQA as an open benchmark. This is real and shipped at scale.

**Implementability for Jera**:
- Full deep-research requires an LLM planner — not offline-pure.
- The *plan-then-retrieve* scaffold is implementable: a `DeepResearchOrchestratorPort` that takes a structured plan (list of sub-questions), fans out to existing Jera retrieval ports per sub-question, merges results with RRF, passes to generator.
- **Deterministic CI version**: Supply a hardcoded plan `[q1, q2]`, assert two separate BM25 retrievals occur, assert the aggregated result contains chunks from both sub-corpora. No LLM required for the structural test; the planner is a separate opt-in adapter.
- **Port location**: `src/orchestration/deep_research.py` — takes a `ResearchPlan` dataclass and returns a `ResearchResult`. Real-model adapter: use an LLM to generate the `ResearchPlan`. Offline adapter: caller provides the plan directly (useful for structured enterprise workflows).

---

## 3. A-RAG: Hierarchical Retrieval Interfaces

**What it is**: Exposes three retrieval tools to the model itself — keyword search, semantic search, chunk-read — so the model decides which granularity to use per step. The model *controls* the retrieval strategy mid-inference rather than following a fixed pipeline.

**Key citation**: "A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces", arXiv:2602.03442, Feb 2026. [Link](https://arxiv.org/html/2602.03442v1)

**SOTA/adoption verdict**: A clean formalization of tool-use for retrieval. Shows consistent outperformance of single-tool systems at comparable or fewer retrieved tokens. Genuine technique, not hype — straightforward to reproduce.

**Implementability for Jera**:
- Jera already has BM25 (keyword), dense embedding (semantic), and chunk read.
- All three can be exposed as distinct function signatures to an LLM via tool-use.
- **Deterministic CI version**: Provide a mock LLM tool-caller that cycles through all three tools in deterministic order; assert each tool is invoked and returns non-empty results; assert the union of results is a superset of single-tool results. This is non-tautological because it verifies all three code paths function and are composable.
- **Port location**: `src/retrieval/agentic/hierarchical_retriever.py` — a `HierarchicalRetrieverPort` with `.keyword()`, `.semantic()`, `.read()` methods. The LLM tool-dispatch adapter is opt-in; the deterministic test drives it with a hand-written dispatch sequence.
- **Effort**: Low. Three ports already exist; this is a thin wrapper exposing them as an agentic interface.

---

## 4. SE-Search: Self-Evolving Search Agent

**What it is**: A Think-Search-Memorize training framework adding three improvements to RL-based search agents: (1) *Memory Purification* to filter noisy retrieved docs, (2) *Atomic Query Training* for shorter/more-diverse queries, (3) *Dense Reward* signals for faster RL training. SE-Search-3B beats Search-R1 by 10.8 points absolute (+33.8% relative).

**Key citation**: "SE-Search: Self-Evolving Search Agent via Memory and Dense Reward", arXiv:2603.03293, Feb 2026. [Link](https://arxiv.org/abs/2603.03293)

**SOTA/adoption verdict**: Pre-publication (awaiting acceptance), code not yet released. Results look robust across benchmarks. The *Memory Purification* idea (extract salient facts after retrieval, discard noisy docs) is the most porting-ready concept for Jera independently of the RL training.

**Implementability for Jera**:
- The *Memory Purification* template is implementable as an extractive post-retrieval filter: for each retrieved chunk, score each sentence by BM25/TF-IDF similarity to the query; keep sentences above threshold; discard the rest. This is offline-pure.
- **Deterministic CI test**: Given a corpus where only 1-of-3 retrieved chunks is relevant, assert the purified output contains ≥1 sentence from the relevant chunk and 0 sentences from irrelevant noise chunks (use keyword overlap as oracle).
- **Port location**: New `MemoryPurificationFilter` adapter under `src/retrieval/post_process/` implementing a `PostRetrievalFilterPort`.
- Note: The full RL training loop is not reproducible offline. Only the inference-time purification pattern is applicable.

---

## 5. Visual Document RAG

**What it is**: Treat each PDF page as an image; encode it as a grid of patch embeddings via a Vision-Language Model (VLM); retrieve pages using MaxSim late interaction (same ColBERT pattern). Skips OCR entirely; handles tables, charts, complex layouts natively.

**Key citations**:
- ColPali original — arXiv:2407.01449, ICLR 2025. [Link](https://arxiv.org/abs/2407.01449)
- "Visual RAG Toolkit: Scaling Multi-Vector Visual Retrieval with Training-Free Pooling and Multi-Stage Search", arXiv:2602.12510, Feb 2026. [Link](https://arxiv.org/pdf/2602.12510)
- ColSmolVLM (HuggingFace SmolVLM series, runs on free-tier GPU / consumer hardware). [HF Cookbook](https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_smol_vlm)
- ColMate — "Contrastive Late Interaction and Masked Text for Multimodal Document Retrieval", arXiv:2511.00903, Nov 2025. [Link](https://arxiv.org/pdf/2511.00903)
- MarkTechPost tutorial, Feb 2026. [Link](https://www.marktechpost.com/2026/02/18/tutorial-building-a-visual-document-retrieval-pipeline-with-colpali-and-late-interaction-scoring/)

**SOTA/adoption verdict**: Production-ready. ColQwen2.5 and ColSmolVLM run on consumer GPUs; Visual RAG Toolkit provides training-free pooling alternatives (mean-row/column pooling) reducing index size. Multiple HF ONNX exports exist for the *embedding projection layer*. The VLM encoder itself (PaliGemma-3B, SmolVLM-256M) still needs GPU for practical throughput, but CPU inference is *technically possible* for offline test runs.

**CPU/ONNX path assessment**:
- Encoding throughput on CPU is ~10-50x slower than GPU (practical for indexing small corpora offline but not production-scale).
- The 128-dim MaxSim scoring after encoding is pure matrix math — fully offline/ONNX.
- ColSmolVLM (SmolVLM-256M backbone) is the smallest viable model; community ONNX exports exist.
- For Jera CI: use pre-computed patch embeddings (stored as fixtures) + the MaxSim scorer. The MaxSim scorer is already implemented in Jera (ColBERT). The only new piece is the image-to-patch pipeline, testable offline with a toy fixture of 3×4 patches × 128-dim.

**Implementability for Jera**:
- New `VisualDocumentIndexerPort` + `VisualPageRetrieverPort`.
- CI-offline core: `PatchEmbeddingFixture` (a pre-computed numpy array stored in `tests/fixtures/`) + existing `MaxSimScorer`. Non-tautological test: assert that a query embedding for "revenue chart" retrieves the page whose fixture patches were generated from a known chart image over a text page.
- Real-model opt-in: `ColPaliAdapter` / `ColSmolVLMAdapter` behind `extras/visual`.
- **Effort**: Medium-high. The MaxSim scorer is done; the new effort is the image ingestion pipeline and fixture tooling.

---

## 6. Qwen3 Embedding

**What it is**: A family of LLM-backbone embedders (0.6B, 4B, 8B) from Alibaba released June 2025. Trained on a multi-stage pipeline with Qwen3 foundation models. Key advance: per-task instruction input that shifts the embedding distribution, giving 1–5% gains across task types. Supports 100+ languages. Apache 2.0.

**Key citations**:
- "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models", arXiv:2506.05176, Jun 2025. [Link](https://arxiv.org/pdf/2506.05176)
- GitHub: [QwenLM/Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)
- ONNX 0.6B: [zhiqing/Qwen3-Embedding-0.6B-ONNX](https://huggingface.co/zhiqing/Qwen3-Embedding-0.6B-ONNX) and INT8 variant [janni-t/qwen3-embedding-0.6b-int8-tei-onnx](https://huggingface.co/janni-t/qwen3-embedding-0.6b-int8-tei-onnx)

**SOTA/adoption verdict**: Qwen3-Embedding-8B ranked #1 on MTEB Multilingual (score 70.58, June 2025). The 0.6B ONNX INT8 model is ~560MB and runs on CPU with 2–4× speedup vs FP32. Not hype — Apache 2.0, community ONNX exports validated. NVIDIA's Llama-Embed-Nemotron-8B competes but is non-permissive.

**Implementability for Jera**:
- Direct drop-in as a new `DenseRetrieverPort` adapter. The instruction-following feature is the novel add: Jera can pass a task-specific prefix (e.g., `"Represent this document for retrieval: "`) changing behavior per collection type.
- **Deterministic CI version**: The ONNX 0.6B model is small enough to include as an optional test dependency. But to keep CI hash-embedding-only, the test should verify the *adapter interface*: given a mock `OnnxEmbedder` that returns deterministic vectors for known inputs, assert the `Qwen3EmbeddingAdapter` correctly prepends the instruction prefix before calling the embedder.
- **Port location**: `src/embedding/adapters/qwen3_embedding.py` implementing `DenseEmbedderPort` with `instruction: str` parameter.
- **Value**: Strongest open multilingual embedder at small size; Korean-language excellence is directly useful for Jera's Korean research RAG track.

---

## 7. Agentic Long-Term Memory

### 7a. Mem0

**What it is**: Extracts, consolidates, and indexes *salient facts* from conversation history rather than storing raw chunks. Graph-enhanced variant models entity relationships. Achieves 91% lower latency than full-context, 90% token savings, +26% quality vs. OpenAI Memory on LOCOMO benchmark.

**Key citation**: "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory", arXiv:2504.19413, Apr 2025 (ECAI 2025). [Link](https://arxiv.org/abs/2504.19413)

**Implementability for Jera**:
- The *extraction* step needs an LLM. The *storage + retrieval* (vector store of facts + optional knowledge graph) is offline-feasible.
- New `ConversationMemoryPort` with `.memorize(turn)` and `.recall(query)`. CI offline test: given a hardcoded list of pre-extracted facts (bypassing LLM extraction), assert that `.recall("user's preference for X")` retrieves the relevant fact using BM25 similarity. Non-tautological because it validates the storage/retrieval pipeline independently of extraction.
- **Port location**: `src/memory/conversation_memory.py`.

### 7b. A-MEM (Agentic Memory / Zettelkasten)

**What it is**: Each stored memory becomes a structured note (keywords, contextual description, semantic tags). New memories trigger re-evaluation and re-linking of existing notes, forming an evolving interconnected knowledge network. Feb 2025.

**Key citation**: "A-Mem: Agentic Memory for LLM Agents", arXiv:2502.12110, Feb 2025. [Link](https://arxiv.org/pdf/2502.12110)

**Implementability for Jera**:
- The *note structuring* and *dynamic linking* need an LLM. The underlying index (keyword + embedding + graph edges) is offline-feasible.
- The zettelkasten note schema (`{id, content, keywords: [], tags: [], links: []}`) can be stored in Jera's existing in-memory store. Dynamic linking at write time is a graph edge update.
- **CI test**: Insert three notes where note-3 shares keywords with note-1; assert that after insertion the adjacency list of note-1 includes note-3's id. Fully deterministic — no LLM needed to test the linking logic when keywords are supplied directly.

### 7c. HyperMem

**What it is**: Three-level memory hierarchy (topics → episodes → facts) with hyperedges capturing multi-way dependencies (not just pairwise). Hybrid lexical-semantic index with coarse-to-fine retrieval. Achieves 92.73% LLM-as-a-judge accuracy.

**Key citation**: "HyperMem: Hypergraph Memory for Long-Term Conversations", arXiv:2604.08256, Apr 2026. [Link](https://arxiv.org/abs/2604.08256)

**Implementability for Jera**:
- The three-tier structure maps cleanly to Jera's hierarchical chunking. Hyperedges are a set of node-ids grouped by a topic; implementable as a Python dict without graph library dependencies.
- **CI test**: Build a mini hypergraph of 5 facts under 2 episodes under 1 topic. Query for the topic; assert coarse retrieval returns the topic node; fine retrieval returns all constituent facts. Fully offline.
- **Port location**: `src/memory/hypermem.py`. More complex than Mem0; consider as a future milestone after basic conversation memory lands.

---

## 8. LightRAG + LazyGraphRAG

**What it is**:
- **LightRAG** (EMNLP 2025): Dual-level graph RAG — retrieves entities/relations via vectors directly (no community traversal), uses graph only for structural context. 6,000× fewer tokens per query vs. GraphRAG; 60% lower indexing cost.
- **LazyGraphRAG** (Microsoft, Jun 2025): Defers all LLM summarization to query time; indexing cost = 0.1% of full GraphRAG. Comparable quality for global queries at 700× lower query cost.

**Key citations**:
- LightRAG GitHub: [HKUDS/LightRAG](https://github.com/hkuds/lightrag), EMNLP 2025.
- LazyGraphRAG: Microsoft Research Blog, Jun 2025. [Link](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
- E²GraphRAG — "Streamlining Graph-based RAG for High Efficiency and Effectiveness", arXiv:2505.24226. [Link](https://arxiv.org/pdf/2505.24226)

**SOTA/adoption verdict**: LightRAG is widely adopted (GitHub stars growing rapidly). LazyGraphRAG is Microsoft's own production-recommended variant. Both are solid improvements over the original 2024 GraphRAG. The graph-building step *still* needs LLM-based entity extraction, but retrieval is pure vector math afterward.

**Implementability for Jera**:
- Jera already has HippoRAG graph PPR. LightRAG's key addition is the *dual-level retrieval*: entity-level query embedding matched to entity embeddings, plus chunk-level retrieval, merged.
- **Deterministic CI version**: Build a toy entity graph (5 entities, 3 relations, pre-embedded with hash embeddings); assert that a query for entity "X" returns both the entity node and the chunk containing X, and that a relation query ("X rel Y") returns the edge chunk.
- LazyGraphRAG's deferred-summarization approach is a pure architectural choice (don't precompute community summaries; compute on demand) — implementable without LLMs for the indexing phase.
- **Port location**: Extend `src/retrieval/graph/` with `DualLevelGraphRetriever` and a `LazyCommunityIndex`.
- **Effort**: Medium. Most graph plumbing is done; the dual-level index is new.

---

## 9. Context Compression for Efficient RAG

### 9a. ACC-RAG (Adaptive Context Compression)

**What it is**: Dynamically adjusts compression rate based on query complexity; combines hierarchical compressor + context selector. 4× faster inference vs. standard RAG. Published EMNLP 2025.

**Key citation**: "Enhancing RAG Efficiency with Adaptive Context Compression", arXiv:2507.22931, ACL/EMNLP 2025. [Link](https://arxiv.org/abs/2507.22931)

### 9b. RECON (Reasoning with Condensation)

**What it is**: Adds an explicit summarization module *inside* the RL-based search-reasoning loop (e.g., Search-R1). Compresses retrieved context by 35%, boosting EM by 14.5% on 3B models while reducing inference latency.

**Key citation**: "RECON: Reasoning with Condensation for Efficient Retrieval-Augmented Generation", arXiv:2510.10448, Oct 2025. [Link](https://arxiv.org/abs/2510.10448)

### 9c. Tool-Schema Compression (TSCG) — see §13.

**SOTA/adoption verdict for compression broadly**: ACC-RAG is EMNLP 2025, solid. RECON is practical (integrates with Search-R1). These are not hype — they address a real latency/cost problem in production RAG.

**Implementability for Jera**:
- The *extractive* part of ACC-RAG (keep only top-N sentences per chunk by query-BM25 score) is fully offline.
- **Deterministic CI version**: Given a multi-sentence chunk and a query, assert the compressor retains ≥1 sentence whose BM25 score vs. query exceeds a threshold, and that the retained sentences are strictly fewer than the input (non-tautological because it tests both recall of relevant content and actual reduction).
- **Port location**: `src/retrieval/post_process/context_compressor.py` implementing `ContextCompressionPort`. The offline adapter uses extractive sentence ranking; the opt-in real-model adapter uses a small LLM summarizer.

---

## 10. RAG Security: SDAG Sparse Attention Defense

**What it is**: Block-sparse attention mechanism that prevents cross-attention between retrieved documents, eliminating the vector through which a poisoned document can corrupt a clean one's representation. Requires only an attention mask change at inference — no fine-tuning, no architecture change.

**Key citation**: "Addressing Corpus Knowledge Poisoning Attacks on RAG Using Sparse Attention", arXiv:2602.04711, Feb 2026. [Link](https://arxiv.org/abs/2602.04711)

**SOTA/adoption verdict**: Minimal-change defense with statistically significant improvement over SOTA defenses. Highly practical — just change the attention mask. The limitation is that it requires access to the LLM's attention layer (not applicable when using a black-box API).

**Implementability for Jera**:
- Directly applicable when Jera uses a local ONNX/transformers model.
- The attention mask is part of the model's forward pass; Jera's generator port can accept a `doc_boundary_mask: Optional[List[int]]` parameter.
- **Deterministic CI version**: Provide a tiny 2-layer transformer (or mock attention function) and two documents; assert that with SDAG mask, the attention weight from token in doc-1 to token in doc-2 is exactly 0.0; assert with standard causal mask it is non-zero. Non-tautological — tests the mask construction logic.
- **Port location**: `src/generation/attention_masks.py` with `build_sdag_mask(doc_boundaries: List[int], seq_len: int) -> np.ndarray`.
- **Effort**: Low for the mask builder. Medium if the generator adapter needs to be modified to accept and apply it.

---

## 11. RAG Security: Taxonomy of Attacks and Defenses

**What it is**: Comprehensive 2026 survey classifying RAG attacks into corpus poisoning (data injection, backdoor, knowledge manipulation), prompt injection (direct, indirect, query manipulation), and information leakage (membership inference, document extraction). Multi-layer defense taxonomy: input-level, retrieval-level, output-level, system-level.

**Key citation**: "Securing Retrieval-Augmented Generation: A Taxonomy of Attacks, Defenses, and Future Directions", arXiv:2604.08304, Apr 2026. [Link](https://arxiv.org/html/2604.08304v1)

Related: "Semantic Chameleon: Corpus-Dependent Poisoning Attacks and Defenses in RAG Systems", arXiv:2603.18034. [Link](https://arxiv.org/html/2603.18034v1)

**SOTA/adoption verdict**: The taxonomy is now settled enough to drive implementation. RAG Security Bench (13 poisoning methods × 5 datasets × 7 defenses) shows hybrid defenses consistently outperform single-mechanism approaches.

**Implementability for Jera**:
- Jera can implement retrieval-level defenses (diversity retrieval, anomaly detection, source authentication) as port adapters wrapping the existing `RetrieverPort`.
- **Concrete offline-feasible defenses**:
  - *Document provenance tagging*: attach `source_hash` and `ingest_timestamp` to every chunk; a `ProvenanceFilterPort` flags chunks modified after a threshold date.
  - *Retrieval diversity enforcement*: already partially done by MMR; add a `max_docs_per_source` parameter to prevent a single poisoned source dominating the result set.
  - *Output-level consistency check*: if two retrieved passages make contradictory factual claims (detected by simple entity+negation pattern matching), flag the result with `conflict_detected=True`.
- **CI test for provenance filter**: inject a chunk with a future `ingest_timestamp` into the corpus; assert the filter removes it from the result set.

---

## 12. Citation Hallucination Detection (FACTUM)

**What it is**: Mechanistic (non-LLM-judge) detection of citation hallucination using four internal model activation scores: Contextual Alignment Score (CAS), Attention Sink Usage (BAS), Parametric Force (PFS), and Pathway Alignment (PAS). Outperforms baselines by up to 37.5% AUC.

**Key citation**: "FACTUM: Mechanistic Detection of Citation Hallucination in Long-Form RAG", arXiv:2601.05866, Jan 2026. [Link](https://arxiv.org/pdf/2601.05866)

**SOTA/adoption verdict**: Novel mechanistic approach that does not require a second LLM to evaluate faithfulness — a significant operational advantage. Requires access to internal model activations (not black-box API compatible).

**Implementability for Jera**:
- Applicable only with local models (ONNX, transformers). Requires extracting attention weights and FFN activations per generated token.
- A simpler **proxy implementation** (offline-compatible without activation access): for each generated sentence, compute BM25 overlap between the sentence and its cited chunk; flag sentences with overlap below a threshold as potentially hallucinated. This is a weaker but offline-pure approximation.
- **CI test**: Generate a sentence that is *not* in the retrieved chunk; assert the proxy detector flags it as `citation_confidence < threshold`. Generate a sentence that is a direct substring of the chunk; assert it passes.
- **Port location**: `src/eval/citation_fidelity.py` implementing `CitationFidelityPort`. Two adapters: `BM25ProxyCitationChecker` (offline) and `FACTUMActivationChecker` (opt-in, needs transformers).

---

## 13. Tool-Schema Compression (TSCG)

**What it is**: Deterministic, rule-based compression of JSON Schema tool definitions for agentic LLM deployments. Achieves 44–68% token savings without accuracy loss. Critical for agentic RAG where tool schemas compete with retrieval context for context window space.

**Key citations**:
- "Tool-Schema Compression Enables Agentic RAG Under Constrained Context Budgets", arXiv:2605.26165, May 2026. [Link](https://arxiv.org/abs/2605.26165)
- "TSCG: Deterministic Tool-Schema Compilation for Agentic LLM Deployments", arXiv:2605.04107, May 2026. [Link](https://arxiv.org/abs/2605.04107)
- GitHub: [SKZL-AI/tscg](https://github.com/SKZL-AI/tscg) — 1,200-line zero-dependency TypeScript, MIT, 459 tests.

**SOTA/adoption verdict**: Directly implementable today. Zero-dependency, deterministic, MIT licensed. JSON schemas overflow at ~494 tools; compressed schemas support 800+. This is infrastructure, not research — ship it.

**Implementability for Jera**:
- Jera's Python tool schemas can be compressed with equivalent Python logic (remove `description` when redundant, collapse `anyOf` with single variant, alias long `$ref` chains).
- **Deterministic CI test**: Take a known 5-tool schema JSON, apply the compressor, assert the output validates against the same JSON Schema meta-schema, assert token count (measured by `len(json.dumps(schema).split())`) is ≤ 60% of input. Fully offline, fully deterministic.
- **Port location**: `src/agentic/schema_compression.py`.
- **Effort**: Low. Pure string/dict manipulation, no model dependency.

---

## 14. New Evaluation Benchmarks 2025-2026

| Benchmark | What it tests | Release | Why Jera should track |
|-----------|--------------|---------|----------------------|
| **DeepSearchQA** (Google/DeepMind) | 900 causal-chain tasks, 17 subjects; evaluates completeness+precision of an agent's final answer set, not retrieval trajectory. [Link](https://huggingface.co/datasets/google/deepsearchqa) | Dec 2025 | Gold standard for agentic deep-research evaluation. Open. |
| **WebPuzzle** (DeepDiver) | 24k training / 275 test; open-web QA where hops require real internet searches. [Link](https://arxiv.org/abs/2505.24332) | May 2025 | Stress-tests iterative retrieval; useful for measuring SE-Search-style agents. |
| **MTEB v2** | 41 English datasets across 7 task types; multimodal-aware (MIEB extension). [Link](https://huggingface.co/blog/isaacchung/mteb-v2) | 2026 | Standard embedding leaderboard — all Jera embedding adapters should report MTEB v2 scores. |
| **ChronoQA** | Temporal/causal/character QA over narrative documents (novels). Accompanies E²RAG. [Link](https://arxiv.org/abs/2506.05939) | Jun 2025 | Relevant for document corpora with temporal structure (e.g., meeting minutes, research timelines). |
| **MemoryCD** | Long-context cross-domain personalization benchmark for agent memory. [Link](https://arxiv.org/pdf/2603.25973) | Mar 2026 | Directly evaluates Mem0/A-MEM/HyperMem-class memory systems. |
| **RealMem** | Real-world memory-driven interaction benchmark. [Link](https://arxiv.org/pdf/2601.06966) | Jan 2026 | Practical multi-session memory evaluation. |

**Implementability for Jera**: DeepSearchQA and MTEB v2 are the two highest-priority additions to Jera's eval harness. DeepSearchQA: download the 900 tasks, implement a `DeepSearchQAEvaluator` that drives Jera's retrieval pipeline and scores precision+recall of the answer set. MTEB v2: use the official `mteb` Python package — already straightforward to integrate.

---

## 15. Entity-Event Knowledge Graphs (E²RAG)

**What it is**: Dual-graph RAG keeping separate entity and event subgraphs linked by a bipartite mapping. Preserves temporal and causal structure that single-entity-graph RAG loses. Introduced ChronoQA benchmark. Published EACL 2026.

**Key citation**: "Respecting Temporal-Causal Consistency: Entity-Event Knowledge Graphs for Retrieval-Augmented Generation", arXiv:2506.05939, Jun 2025, EACL 2026. [Link](https://arxiv.org/abs/2506.05939)

**SOTA/adoption verdict**: Niche but solid — improves causal and character-consistency queries over standard KG-RAG. Relevant for document types with inherent temporal structure.

**Implementability for Jera**:
- Extends HippoRAG. The bipartite entity↔event mapping is an additional edge type in the existing graph store.
- Entity extraction requires NLP (spaCy NER works offline). Event extraction requires a more capable extractor (SRL or LLM — opt-in).
- **Deterministic CI test**: Pre-build a 3-entity 2-event toy graph; issue a causal query ("what happened after event A?"); assert the retriever traverses the event edge from A to B and returns the chunk linked to B.
- **Port location**: `src/retrieval/graph/entity_event_graph.py`.
- **Effort**: Medium. Graph store extension + new extraction adapter.

---

## 16. MTEB v2

**What it is**: Refactored Massive Text Embedding Benchmark with unified `mteb.evaluate()` interface, new `ResultCache`, multimodal input support (text+image+audio), and a revised English suite of 41 datasets. Scores are NOT comparable to v1.

**Key citation**: [Introducing MTEB v2](https://huggingface.co/blog/isaacchung/mteb-v2), HuggingFace blog, 2026.

**Current leaderboard leaders**: NVIDIA Llama-Embed-Nemotron-8B (multilingual), Google Gemini Embedding 001 (English, 68.32 avg); Qwen3-Embedding-8B at 70.58 on multilingual (as of Jun 2025). Open-source models are within striking distance of proprietary ones.

**Implementability for Jera**: Use the `mteb` Python package. Add a `MtebEvalTask` to Jera's eval harness that runs a configured embedder against MTEB v2 tasks offline-safe (a subset can run without internet by caching datasets). Non-tautological CI test: run the `SciFact` retrieval task from MTEB v2 against Jera's hash embedder; assert NDCG@10 > 0 (proves the pipeline runs end-to-end, not that results are good — the hash embedder deliberately performs poorly, confirming the test is not tautological).

---

## 17. RECON: In-Loop Context Condensation

**What it is**: A trained summarization module inserted into the RL-based reasoning loop to compress verbose retrieval results *during* multi-step reasoning. Two-stage training: (1) relevance pretraining on QA, (2) multi-aspect distillation from large LLMs. Reduces context length 35%, improves EM 14.5% (3B), 3% (7B).

**Key citation**: "RECON: Reasoning with Condensation for Efficient Retrieval-Augmented Generation", arXiv:2510.10448, Oct 2025. [Link](https://arxiv.org/abs/2510.10448)

**SOTA/adoption verdict**: Solid improvement for RL-search pipelines. The summarizer is a trained model component — not purely offline. However the *design pattern* (compress each retrieved batch before appending to the reasoning context) is valuable and can be approximated offline.

**Implementability for Jera**:
- The offline approximation: after each retrieval step in an agentic loop, apply the extractive compressor from §9 to the newly retrieved chunks before they enter the context. This is a one-line change to the agentic executor.
- **CI test**: Combined with the agentic executor test (§1), assert that the total context accumulated over N retrieval steps is ≤ N × chunk_size × 0.7 (i.e., compression occurred in each round). Deterministic with BM25-based extraction.

---

## Top 3 Picks for Jera

Ranked by **value-per-effort** for an offline-first hexagonal RAG with deterministic CI.

---

### #1 — Qwen3-Embedding-0.6B-ONNX as Instruction-Aware Dense Embedder

**Value**: Immediate quality uplift for Korean + multilingual retrieval. The 0.6B ONNX INT8 model (~560MB) is the strongest open multilingual embedder at its size class, directly improving Jera's Korean research RAG track. Instruction-tuning (1–5% gain per task) adds a protocol-level feature: callers can specify the retrieval context type (`"academic paper"`, `"news"`, `"code"`) and get better embeddings.

**Effort**: Low. Implement `Qwen3EmbeddingAdapter` wrapping the ONNX runtime; add instruction prefix parameter to `DenseEmbedderPort`; write one CI test validating prefix injection into the model input.

**Port fit**: Clean adapter swap under `extras/qwen3`; hash embedder remains the CI default.

**Why not #1 concern**: The 0.6B model requires `onnxruntime` as a dependency. Keep it in `extras/`.

---

### #2 — Hierarchical Retrieval Interface (A-RAG Pattern)

**Value**: Exposes Jera's three already-implemented retrieval tools (BM25, dense, chunk-read) as a structured agentic interface. Enables LLM-driven adaptive retrieval granularity selection with *zero new retrieval logic* — all the logic already exists. This is the highest-leverage "unlock existing capability" improvement available.

**Effort**: Very low. A thin `HierarchicalRetrieverPort` wrapper with `.keyword()`, `.semantic()`, `.read()` methods, plus a deterministic mock dispatcher for CI. Two hours of implementation.

**Port fit**: Perfect hexagonal fit. New port, three existing adapters. The LLM dispatcher is an opt-in adapter.

**CI test**: Deterministic dispatcher calls all three methods in sequence; assert each returns non-empty results; assert the union of results is a superset of any single method's results.

---

### #3 — Context Compression Port (Extractive, Offline-Pure)

**Value**: Addresses a real production pain point: large retrieved contexts slow generation and dilute relevance. An extractive BM25-sentence-scoring compressor (no model needed) can be shipped as a composable post-retrieval adapter that works in the offline CI core. Unlocks RECON-style in-loop compression for the agentic executor. Serves as the offline stand-in for ACC-RAG.

**Effort**: Low-medium. BM25 sentence scorer + threshold logic; integrate as `ContextCompressionPort` post-retrieval step; CI test validates compression ratio and relevant-sentence retention.

**Port fit**: Clean post-retrieval adapter; composable with any retriever.

**Why #3 over security/memory**: Security (SDAG) requires generator-level access to attention masks; that's a deeper architectural change. Memory (Mem0/HyperMem) depends on LLM extraction. Context compression is purely extractive, self-contained, and immediately shippable.

---

## Honorable Mentions (Worth Watching, Not Yet Ready)

- **SDAG** (sparse attention defense): High value, low friction — but needs a local model with attention mask access. Implement when Jera's ONNX generator adapter ships.
- **HyperMem** (hypergraph memory): More powerful than Mem0 for long-horizon agents; implement after basic `ConversationMemoryPort` is stable.
- **ColSmolVLM** (visual RAG): Compelling for document-heavy workflows; defer until GPU/compute budget allows; include fixture-based MaxSim CI test now as a scaffold.
- **LazyGraphRAG**: Worth tracking for the deferred-summarization pattern; implement as a `LazyCommunityIndex` variant when the existing graph store is stable.
- **FACTUM** (citation hallucination): BM25-proxy version is shippable now; full mechanistic version needs local transformer access.
- **DeepSearchQA**: Add to eval harness in M13 as the new agentic-retrieval benchmark.
- **TSCG** (tool-schema compression): Shippable today as a pure Python utility if Jera has ≥10 tool definitions; low priority until agentic tool-use is a primary use case.

---

## Sources

- [Search-R1 paper (arXiv:2503.09516)](https://arxiv.org/pdf/2503.09516)
- [DeepDiver (arXiv:2505.24332)](https://arxiv.org/abs/2505.24332)
- [SE-Search (arXiv:2603.03293)](https://arxiv.org/abs/2603.03293)
- [A-RAG (arXiv:2602.03442)](https://arxiv.org/html/2602.03442v1)
- [Reasoning RAG Survey (arXiv:2506.10408)](https://arxiv.org/pdf/2506.10408)
- [From Web Search to Agentic Deep Research (arXiv:2506.18959)](https://arxiv.org/pdf/2506.18959)
- [DeepPlanner (arXiv:2510.12979)](https://arxiv.org/pdf/2510.12979)
- [DeepSearchQA benchmark (HuggingFace)](https://huggingface.co/datasets/google/deepsearchqa)
- [DeepSearchQA paper (PDF)](https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf)
- [ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Visual RAG Toolkit (arXiv:2602.12510)](https://arxiv.org/pdf/2602.12510)
- [ColSmolVLM HF Cookbook](https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_smol_vlm)
- [ColMate (arXiv:2511.00903)](https://arxiv.org/pdf/2511.00903)
- [Qwen3 Embedding paper (arXiv:2506.05176)](https://arxiv.org/pdf/2506.05176)
- [Qwen3-Embedding GitHub](https://github.com/QwenLM/Qwen3-Embedding)
- [Qwen3-Embedding-0.6B-ONNX (HuggingFace)](https://huggingface.co/zhiqing/Qwen3-Embedding-0.6B-ONNX)
- [Qwen3-Embedding-0.6B INT8 ONNX (HuggingFace)](https://huggingface.co/janni-t/qwen3-embedding-0.6b-int8-tei-onnx)
- [Mem0 paper (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413)
- [Mem0 State of AI Agent Memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- [A-MEM paper (arXiv:2502.12110)](https://arxiv.org/pdf/2502.12110)
- [HyperMem (arXiv:2604.08256)](https://arxiv.org/abs/2604.08256)
- [LightRAG GitHub (EMNLP 2025)](https://github.com/hkuds/lightrag)
- [LazyGraphRAG (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
- [E²GraphRAG (arXiv:2505.24226)](https://arxiv.org/pdf/2505.24226)
- [ACC-RAG (arXiv:2507.22931)](https://arxiv.org/abs/2507.22931)
- [RECON (arXiv:2510.10448)](https://arxiv.org/abs/2510.10448)
- [SDAG (arXiv:2602.04711)](https://arxiv.org/abs/2602.04711)
- [RAG Security Taxonomy (arXiv:2604.08304)](https://arxiv.org/html/2604.08304v1)
- [Semantic Chameleon poisoning (arXiv:2603.18034)](https://arxiv.org/html/2603.18034v1)
- [FACTUM (arXiv:2601.05866)](https://arxiv.org/pdf/2601.05866)
- [TSCG paper (arXiv:2605.26165)](https://arxiv.org/abs/2605.26165)
- [TSCG deterministic tool-schema (arXiv:2605.04107)](https://arxiv.org/abs/2605.04107)
- [TSCG GitHub (SKZL-AI)](https://github.com/SKZL-AI/tscg)
- [MTEB v2 announcement (HuggingFace)](https://huggingface.co/blog/isaacchung/mteb-v2)
- [Entity-Event KG RAG (arXiv:2506.05939)](https://arxiv.org/abs/2506.05939)
- [MemoryCD benchmark (arXiv:2603.25973)](https://arxiv.org/pdf/2603.25973)
- [RealMem benchmark (arXiv:2601.06966)](https://arxiv.org/pdf/2601.06966)
- [Atlan: What Is RAG in 2026](https://atlan.com/know/what-is-rag/)
- [GraphSearch agentic workflow (arXiv:2509.22009)](https://arxiv.org/pdf/2509.22009)
- [Respecting Temporal-Causal Consistency (EACL 2026)](https://aclanthology.org/2026.eacl-long.90/)
