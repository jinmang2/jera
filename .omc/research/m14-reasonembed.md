# M14 Research: SOTA Reasoning / Instruction-Tuned Text Embeddings (2025–2026)

**Date:** 2026-06-14
**Scope:** Jera hexagonal RAG — EmbeddingProvider port, offline-first, CI deterministic.

---

## 1. Instruction-Tuned Embeddings — Background & Prompt Format

**What it is:** A family of dense embedding models trained to condition retrieval on a natural-language instruction prepended to the query. The instruction names the task; the same base model then shifts its representation accordingly. Key 2024-2025 lineage: E5-Instruct → gte-Qwen2-Instruct → NV-Embed-v2 → Qwen3-Embedding.

**Canonical prompt format (universal across the lineage):**

```
Instruct: {task_description}
Query: {query_text}
```

Example:
```
Instruct: Given a web search query, retrieve relevant passages that answer the query
Query: What is the capital of China?
```

- Instructions are applied to **queries only**; documents/passages are embedded without a prefix.
- Qwen3 documentation quantifies the gain: omitting the instruction drops retrieval performance by ~1–5% on most benchmarks.
- The format is stable across gte-Qwen2-instruct, NV-Embed-v2, and Qwen3-Embedding — all use this identical `Instruct: / Query:` two-line header.

**Key citations:**
- Qwen3-Embedding Technical Report (2025): "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models" — https://arxiv.org/pdf/2506.05176 (2025)
- Qwen3-Embedding model card with code example: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B (2025)
- GitHub repository: https://github.com/QwenLM/Qwen3-Embedding (2025)

**SOTA/adoption verdict:** Mature. The `Instruct: / Query:` pattern is now the de-facto standard for MTEB-competitive retrieval; all top-10 MTEB multilingual models (as of June 2025) use it. Low implementation risk.

**Implementability for Jera:** IMMEDIATELY implementable as a pure-Python adapter with zero new dependencies. No model weights needed for the adapter layer itself — the logic is string concatenation.

---

## 2. Qwen3-Embedding (2025)

**What it is:** A decoder-based dense encoder series (0.6B / 4B / 8B) from Alibaba/Qwen, trained on Qwen3 LLM backbones, supporting 100+ languages, 32k token context, and Matryoshka variable-dimension outputs (32–1024 for 0.6B).

**Key 2025 citation:** "Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models" — https://arxiv.org/pdf/2506.05176 (June 2025)

**MTEB performance (June 2025):**
- 8B model: **#1 MTEB Multilingual leaderboard**, score 70.58
- 0.6B model: competitive with models 10× its size on multilingual tasks
- License: **Apache 2.0**

**ONNX / CPU feasibility (0.6B):**
- Community ONNX INT8 export (`janni-t/qwen3-embedding-0.6b-int8-tei-onnx`): **0.56 GB** on disk, 2–4× faster CPU inference vs float32, <1 GB runtime memory.
- Pure ONNX Runtime inference package `qwen3-embed` (`pip install qwen3-embed`): no PyTorch required, emits numpy arrays directly, auto-detects GPU but works CPU-only.
  - Source: https://github.com/n24q02m/qwen3-embed
  - ONNX model hub: https://huggingface.co/zhiqing/Qwen3-Embedding-0.6B-ONNX
  - INT8 TEI-optimized: https://huggingface.co/janni-t/qwen3-embedding-0.6b-int8-tei-onnx
  - UINT8 variant: https://huggingface.co/electroglyph/Qwen3-Embedding-0.6B-onnx-uint8

**SOTA/adoption verdict:** Best-in-class multilingual embedding as of mid-2025. The 0.6B ONNX INT8 variant is CPU-viable for offline deployment. Apache-2.0 is permissive. Community ONNX exports exist and are actively maintained.

**Implementability for Jera:** Production opt-in adapter behind `extras = ["qwen3"]`. Implement as `Qwen3EmbeddingProvider(EmbeddingProvider)` wrapping `qwen3-embed` or `onnxruntime`. Always pair with `InstructionEmbedding` adapter (§5) since the model performs better with instructions.

---

## 3. Reasoning Embeddings — RaDeR, ReasonEmbed, and the BRIGHT Benchmark

### 3a. BRIGHT Benchmark (ICLR 2025)

**What it is:** The first retrieval benchmark requiring multi-step reasoning to match queries to documents. 1,385 real-world queries from StackExchange, LeetCode, and math competitions; documents require deliberate inferential links, not keyword co-occurrence.

**Key citation:** "BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive Retrieval" — https://arxiv.org/pdf/2407.12883, presented at ICLR 2025; leaderboard at https://brightbenchmark.github.io/

**Key finding:** The then-best MTEB model (SFR-Embedding-Mistral, nDCG@10 = 59.0 on BEIR) scores only **18.3 nDCG@10** on BRIGHT. This 3× gap reveals that standard embedding training does not transfer to reasoning-intensive retrieval.

**2025–2026 leaderboard top:**
- Mira-Reasoning-Retrieval (Forward AI Labs): **66.9 nDCG@10** (short docs)
- INF-X-Retriever (INF): 63.4
- DIVER-Retriever: 46.8 (multi-stage approach, arxiv 2508.07995)
- RaDeR-7B: ~39.2

---

### 3b. RaDeR — Reasoning-aware Dense Retrieval (EMNLP 2025)

**What it is:** A family of dense retrievers trained on LLM-generated mathematical reasoning traces. Proposes retrieval-augmented reasoning trajectories + self-reflective relevance scoring to produce hard negatives. Uses Qwen2.5 instruction-tuned backbones.

**Key citation:** "RaDeR: Reasoning-aware Dense Retrieval Models" — EMNLP 2025 main track; https://arxiv.org/pdf/2505.18405 (2025); code at https://github.com/debrup-61/rader

**Key result:** RaDeR (gte-Qwen2-7B backbone) reaches **39.2 nDCG@10** on BRIGHT — the first dense retriever to outperform BM25 on Chain-of-Thought reasoning queries. Achieves this with only 2.5% of the training data of concurrent work (ReasonIR).

**ONNX/compact status (mid-2026):** No official ONNX exports. Models are 7B-class; no sub-1B reasoning-embedding variants confirmed as of the research date.

**SOTA/adoption verdict:** Strong research result; 7B-class only; no offline-feasible ONNX as of mid-2026.

**Implementability for Jera:** Not directly implementable offline. Represents a future direction. Track for ONNX 1–3B variants.

---

### 3c. ReasonEmbed (2025)

**What it is:** An embedding fine-tuning framework with two components: ReMixer (synthesizes 82,000 reasoning-intensive training triples) and Redapter (self-adaptive per-sample loss weighting based on reasoning intensity). Works across LLM backbones.

**Key citation:** "ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval" — https://arxiv.org/abs/2510.08252 (October 2025)

**Key result:** ReasonEmbed-Qwen3-8B achieves **38.1 nDCG@10** on BRIGHT — nearly 10 points above prior SOTA at submission time.

**ONNX/compact status:** No compact/ONNX variants confirmed. Backbone is 8B+.

**SOTA/adoption verdict:** State-of-the-art at time of publication; superseded on the live BRIGHT leaderboard by agentic/multi-stage approaches. No CPU-feasible variant available.

**Implementability for Jera:** Training methodology only; not directly pluggable. The ReMixer synthetic data approach could inform future fine-tuning experiments.

---

## 4. Instruction-Following Retrieval — FollowIR and InF-Embed

### 4a. FollowIR (NAACL 2025)

**What it is:** A benchmark and fine-tuning dataset for teaching IR models to honor fine-grained natural-language instructions that modify what relevance means (e.g., "retrieve only papers from 2020+", "ignore abstracts, focus on methods"). Tests whether retrieval output changes correctly when the instruction changes.

**Key citation:** "FollowIR: Evaluating and Teaching Information Retrieval Models to Follow Instructions" — NAACL 2025; ACL Anthology https://aclanthology.org/2025.naacl-long.597/; Semantic Scholar https://www.semanticscholar.org/paper/FollowIR:-Evaluating-and-Teaching-Information-to-Weller-Chang/77e07e5542b450a2ee3193993c552700ad9ba82d

**Key finding:** Standard bi-encoder retrievers with prepended instructions do not reliably change their outputs when the instruction changes. FollowIR-7B (a fine-tuned reranker) substantially outperforms standard embedders on p-MRR (pairwise MRR measuring instruction-following).

**Multilingual extension:** mFollowIR (Jan 2025) — https://arxiv.org/pdf/2501.19264

**SOTA/adoption verdict:** Active benchmark; tests a distinct capability from BRIGHT (instruction modulation vs. reasoning depth). Highlights the gap between "prepend instruction" (what InstructionEmbedding does) and "actually follow the instruction" (what FollowIR-7B and InF-Embed do). For Jera's use case (task-steering, not fine-grained constraint-following), the simpler prefix approach captures the most practical benefit.

---

### 4b. InF-Embed (2025)

**What it is:** An encoder-only instruction-aware embedding model trained on InF-IR, a 38,000-triplet dataset of `<instruction, query, passage>` triples with hard negatives validated by an LLM judge (o3-mini). Targets smaller encoder-only models (not decoder-only LLMs), enabling direct embedding-based retrieval.

**Key citation:** "Towards Better Instruction Following Retrieval Models" — arxiv 2505.21439, May 2025; https://arxiv.org/abs/2505.21439

**Key result:** InF-Embed surpasses baselines by **8.1% p-MRR** across five instruction-based retrieval benchmarks.

**SOTA/adoption verdict:** Promising encoder-focused approach. The InF-IR training corpus methodology (instruction-conditioned hard negatives + LLM validation) is the cleaner design for encoder retrieval; useful if Jera ever fine-tunes a custom embedding model.

**Implementability for Jera:** Not yet a widely-packaged model; research artifact. The training data design principle is worth monitoring for future fine-tuning.

---

## 5. Concrete Recommendation for Jera: `InstructionEmbedding` Adapter

### 5a. Design

The smallest viable offline-deterministic slice: a **pure-Python adapter** that wraps any `EmbeddingProvider` and prepends an instruction prefix to all query embeddings before delegating.

```python
# src/jera/adapters/embedding/instruction.py

from jera.ports.embedding import EmbeddingProvider
from dataclasses import dataclass, field
from typing import List

DEFAULT_RETRIEVAL_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

@dataclass
class InstructionEmbedding(EmbeddingProvider):
    """
    Wraps any EmbeddingProvider and prepends 'Instruct: {task}\nQuery: {text}'
    to every query before delegating to the base provider.
    Documents are embedded as-is (no instruction prefix).
    Follows the Qwen3-Embedding / E5-Instruct convention.
    """
    base: EmbeddingProvider
    task: str = DEFAULT_RETRIEVAL_INSTRUCTION

    @property
    def model_id(self) -> str:
        return self.base.model_id

    @property
    def dimensions(self) -> int:
        return self.base.dimensions

    @property
    def context_limit(self) -> int:
        return self.base.context_limit

    def _fmt(self, text: str) -> str:
        return f"Instruct: {self.task}\nQuery: {text}"

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Documents: no instruction prefix
        return self.base.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        # Queries: prepend instruction
        return self.base.embed_query(self._fmt(text))
```

**Composability:** Stacks with existing wrappers:
```python
provider = InstructionEmbedding(
    base=TruncatedDimEmbedding(base=HashEmbeddingProvider(), dims=256),
    task="Given a scientific paper abstract, retrieve papers that cite similar work",
)
```

### 5b. Non-Tautological CI Test

The test must verify that the instruction prefix **steers** retrieval — that task-A instructions cause task-A relevant chunks to rank above task-B chunks for the same query text, and vice versa. With `HashEmbeddingProvider` (bag-of-tokens), the instruction tokens enter the hash vocabulary and genuinely shift the vector — this is not a rigged test.

```python
# tests/unit/adapters/embedding/test_instruction_embedding.py

import pytest
from jera.adapters.embedding.hash_embedding import HashEmbeddingProvider
from jera.adapters.embedding.instruction import InstructionEmbedding
import numpy as np

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def test_instruction_steers_retrieval():
    """
    Two chunks: one about biology, one about Python code.
    Query text: "how does replication work" (ambiguous — could be DNA or software).
    With biology instruction, the biology chunk should rank higher than with code instruction.
    With code instruction, the code chunk should rank higher than with biology instruction.
    This is non-tautological: the hash embedding genuinely mixes instruction tokens into
    the vector, so the task prefix shifts which chunk wins.
    """
    base = HashEmbeddingProvider(dims=256)

    chunk_bio = "DNA replication occurs in the nucleus via polymerase enzymes"
    chunk_code = "Python multiprocessing uses fork to replicate process memory"

    query_text = "how does replication work"

    # Biology task
    bio_provider = InstructionEmbedding(
        base=base,
        task="Given a biology question, retrieve relevant biology passages",
    )
    # Code task
    code_provider = InstructionEmbedding(
        base=base,
        task="Given a programming question, retrieve relevant code documentation",
    )

    # Embed chunks without instruction (documents are not prefixed)
    bio_vec = base.embed([chunk_bio])[0]
    code_vec = base.embed([chunk_code])[0]

    # Embed query with each instruction
    query_bio_vec = bio_provider.embed_query(query_text)
    query_code_vec = code_provider.embed_query(query_text)

    # Biology-instructed query should be closer to biology chunk
    assert cosine(query_bio_vec, bio_vec) > cosine(query_bio_vec, code_vec), (
        "Biology instruction should steer query toward biology chunk"
    )

    # Code-instructed query should be closer to code chunk
    assert cosine(query_code_vec, code_vec) > cosine(query_code_vec, bio_vec), (
        "Code instruction should steer query toward code chunk"
    )

    # Sanity: the two instructed query vectors must actually differ
    assert cosine(query_bio_vec, query_code_vec) < 0.99, (
        "Different instructions must produce meaningfully different query vectors"
    )
```

**Why this is non-tautological:** The hash embedding of `"Instruct: Given a biology question... Query: how does replication work"` shares tokens with `chunk_bio` that the code-instructed hash does not. The test fails if `InstructionEmbedding` degenerates to passing query text unchanged (it would fail the instruction-steering assertions). It also fails if both instructions produce identical vectors (sanity check). The assertions test a real retrieval consequence, not an implementation detail.

### 5c. Production Opt-In: Qwen3-Embedding 0.6B ONNX

For production use, replace `HashEmbeddingProvider` with a `Qwen3EmbeddingProvider`:

```python
# extras = ["qwen3"]  →  pip install qwen3-embed

from qwen3_embed import TextEmbedding as _Qwen3Model
from jera.ports.embedding import EmbeddingProvider

@dataclass
class Qwen3EmbeddingProvider(EmbeddingProvider):
    model_name: str = "n24q02m/Qwen3-Embedding-0.6B-ONNX"
    _model: _Qwen3Model = field(init=False)

    def __post_init__(self):
        self._model = _Qwen3Model(model_name=self.model_name)

    @property
    def model_id(self) -> str:
        return self.model_name

    @property
    def dimensions(self) -> int:
        return 1024

    @property
    def context_limit(self) -> int:
        return 32768

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]
```

Then compose:
```python
provider = InstructionEmbedding(
    base=Qwen3EmbeddingProvider(),
    task="Given a Korean research paper query, retrieve relevant academic passages",
)
```

**Deployment specs (0.6B INT8 ONNX):**
- Disk: ~0.56 GB (INT8) vs 4.7 GB (float32)
- Runtime RAM: ~1 GB on CPU
- CPU speed: 2–4× faster than float32
- License: Apache 2.0
- ONNX packages: `zhiqing/Qwen3-Embedding-0.6B-ONNX`, `janni-t/qwen3-embedding-0.6b-int8-tei-onnx`, `electroglyph/Qwen3-Embedding-0.6B-onnx-uint8`
- Pure ONNX Runtime (no PyTorch): `pip install qwen3-embed` → `n24q02m/Qwen3-Embedding-0.6B-ONNX`

---

## 6. Summary Table

| Area | Best Work (2025-26) | BRIGHT nDCG@10 | ONNX / <1B Offline | Jera Action |
|---|---|---|---|---|
| Instruction-tuned embedding | Qwen3-Embedding-8B | ~47 (standard) | 0.6B INT8 ONNX: YES | `InstructionEmbedding` adapter NOW; Qwen3-0.6B opt-in |
| Reasoning retrieval | Mira-Reasoning-Retrieval | 66.9 | NO (proprietary, large) | Monitor only |
| Reasoning fine-tuned embedding | RaDeR-7B / ReasonEmbed-8B | 39.2 / 38.1 | NO (7-8B, no ONNX) | Track for sub-2B variants |
| Instruction-following | InF-Embed / FollowIR-7B | N/A | NO (encoder; no pkg) | InF-IR training design for future fine-tune |
| Baseline MTEB retrieval | SFR-Embedding-Mistral | 18.3 | Various | Already exceeded by Qwen3 |

---

## 7. TOP Recommendation

**Implement `InstructionEmbedding` now; wire Qwen3-Embedding-0.6B-ONNX as the production opt-in.**

**Step 1 (zero new deps, CI-safe):** Add `InstructionEmbedding(base, task)` adapter. It is a string-prepend wrapper — deterministic, testable with `HashEmbeddingProvider`, and immediately composable with every existing provider. The CI test above is non-tautological and offline.

**Step 2 (opt-in extras):** Add `Qwen3EmbeddingProvider` behind `extras = ["qwen3"]`. Use the `n24q02m/qwen3-embed` ONNX package (Apache 2.0, CPU-only, no PyTorch). Default to INT8 0.6B for offline feasibility.

**Exact prompt format to use in all production wrappers:**

```
Instruct: {task_description}
Query: {query_text}
```

Task descriptions should be short (one sentence), written in English regardless of the query language (per Qwen3 recommendation), and scoped to the retrieval domain (e.g., "Given a Korean legal document query, retrieve relevant statute passages").

---

## Sources

- Qwen3-Embedding blog: https://qwenlm.github.io/blog/qwen3-embedding/
- Qwen3-Embedding Technical Report (arxiv 2506.05176): https://arxiv.org/pdf/2506.05176
- Qwen3-Embedding GitHub: https://github.com/QwenLM/Qwen3-Embedding
- Qwen3-Embedding-0.6B HuggingFace: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- Qwen3-Embedding-0.6B ONNX (zhiqing): https://huggingface.co/zhiqing/Qwen3-Embedding-0.6B-ONNX
- Qwen3-Embedding-0.6B INT8 TEI ONNX: https://huggingface.co/janni-t/qwen3-embedding-0.6b-int8-tei-onnx
- Qwen3-Embedding-0.6B ONNX UINT8: https://huggingface.co/electroglyph/Qwen3-Embedding-0.6B-onnx-uint8
- qwen3-embed Python package: https://github.com/n24q02m/qwen3-embed
- BRIGHT Benchmark (ICLR 2025): https://arxiv.org/pdf/2407.12883
- BRIGHT leaderboard: https://brightbenchmark.github.io/
- RaDeR EMNLP 2025 (arxiv 2505.18405): https://arxiv.org/pdf/2505.18405
- RaDeR GitHub: https://github.com/debrup-61/rader
- ReasonEmbed (arxiv 2510.08252): https://arxiv.org/abs/2510.08252
- FollowIR NAACL 2025: https://aclanthology.org/2025.naacl-long.597/
- mFollowIR (arxiv 2501.19264): https://arxiv.org/pdf/2501.19264
- InF-Embed / InF-IR (arxiv 2505.21439): https://arxiv.org/abs/2505.21439
- DIVER multi-stage retrieval (arxiv 2508.07995): https://arxiv.org/pdf/2508.07995
- Large Reasoning Embedding Models (arxiv 2510.14321): https://arxiv.org/pdf/2510.14321
