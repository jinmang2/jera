# M14 Research: Visual Document RAG — ColPali Family & Late-Interaction over Images

**Compiled:** 2026-06-14  
**Purpose:** SOTA survey for implementing a `VisualMultiVectorEmbedding` adapter that plugs into
Jera's existing `MultiVectorEmbedding` port and `MaxSimVectorStore` (M10 infrastructure).

---

## Executive Summary

ColPali (2024) proved that a VLM can replace the entire OCR → text-chunk → embed pipeline for
document retrieval by treating each page as an image and producing one 128-d vector per image
patch, scored with the standard ColBERT MaxSim formula. Jera's M10 infrastructure (`embed_multi`
→ `MaxSimVectorStore.search_maxsim`) is architecturally identical to ColPali's retrieval
mechanism — it already operates over per-patch / per-token vectors with MaxSim. ColPali is
ColBERT over image patches instead of text tokens; the same ports accommodate both.

The critical practical constraint: **all current ColPali-family models require a GPU for
reasonable indexing throughput**. ColSmolVLM-256M is the only variant plausibly runnable
on consumer hardware with <4 GB VRAM; true CPU-only inference exists only through auxiliary
paths (EmbedAnything ONNX, ColModernVBERT, or the full-distillation NanoVDR approach). A
deterministic offline CI adapter can be built from scratch using hash-based patch vectors
derived from image bytes — no model needed — which is exactly the pattern Jera already uses
for `HashMultiVectorEmbedding`.

---

## 1. ColPali (Faysse et al., 2024)

**What it is:** VLM-based late-interaction document retrieval; replaces OCR + text embedding
with a single pass of PaliGemma over page images, producing per-patch ColBERT-style vectors.

**Citation:**  
Faysse, M., Sibille, H., Wu, T., Omrani, B., Viaud, G., Hudelot, C., & Colombo, P. (2024).  
*ColPali: Efficient Document Retrieval with Vision Language Models.*  
arXiv:2407.01449 (published ICLR 2025).  
<https://arxiv.org/abs/2407.01449>

**Architecture:**

```
PDF page
  └─► PIL/pdf2image/pymupdf → PIL.Image (448×448 px, square-padded)
        └─► SigLIP-So400m/14 vision encoder → 32×32 patch grid = 1024 patch tokens
              └─► Gemma-2B LM trunk (full-block prefix attention)
                    └─► Linear projection head D=128
                          └─► 1024 × 128-d patch vectors per page
```

Key numbers:
- Patches per page: **1024** (32×32 grid at 448 px input; ~6 "describe the image" text tokens also stored = 1030 total in some configs)
- Vector dimension: **128** (projected from PaliGemma's hidden size via LoRA-trained head)
- Storage: **~257 KB/page** at float16; ~85 KB/page with 3× token pooling (66 % reduction, 97.8 % NDCG@5 retention)
- At 1 M pages full: ~250 GB; pooled (3×): ~83 GB
- MaxSim formula: `score(q, d) = Σ_{qi ∈ q} max_{dj ∈ d} cosine(qi, dj)` — identical to Jera's `_maxsim_score`

**ViDoRe benchmark (nDCG@5):**  
ColPali achieves **81.3** average nDCG@5 across all ViDoRe tasks (AI: 96.2, Healthcare: 94.4,
TabFQuAD: 83.9), vs. the best text-only pipeline baseline at ~65. Particularly strong on
visually complex content (infographics, charts, tables, mixed-language PDFs). Published as
ICLR 2025.

Source: <https://huggingface.co/blog/manu/colpali>

**GPU requirement:** PaliGemma-3B (SigLIP + Gemma-2B). ColPali-v1.2 fits in **~8 GB VRAM**
(float16). Indexing: ~18 pages/sec on an A100 80 GB. No published ONNX or CPU path.

**SOTA/adoption verdict:** Foundational. The ColPali codebase (`illuin-tech/colpali`, now
`colpali-engine` on PyPI) is the canonical training/inference library for all ColVision
models. ViDoRe is the de-facto benchmark. Widely adopted in Qdrant, Vespa, Weaviate, and
Elasticsearch integrations (2024-2025).

Sources:  
- <https://qdrant.tech/blog/qdrant-colpali/>  
- <https://github.com/illuin-tech/colpali>  
- <https://huggingface.co/vidore/colpali>

---

## 2. ColQwen2 / ColQwen2.5 / ColQwen3.5 (2024–2026)

**What it is:** Drop-in ColPali successors built on Qwen2-VL and Qwen2.5-VL backbones
instead of PaliGemma, with dynamic-resolution patch counts and stronger base model quality.

**Citations:**

ColQwen2 model card (vidore/colqwen2-v1.0):  
<https://huggingface.co/docs/transformers/en/model_doc/colqwen2>

ColQwen2.5-v0.2 model card:  
<https://huggingface.co/vidore/colqwen2.5-v0.2>

ColQwen3.5 community fine-tune:  
<https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3>

**Architecture differences from ColPali:**

- Backbone: **Qwen2-VL-3B** (ColQwen2) or **Qwen2.5-VL-3B** (ColQwen2.5); ~3B params
- Dynamic resolution: image is not resized to 448×448 but processed at native aspect ratio
  up to **768 patches max** (vs. ColPali's fixed 1024). Reduces index size ~25 % on typical
  landscape pages; more patches retained for tall/complex pages.
- ColQwen2.5-v0.2 adds improved instruction tuning and multilingual training data.
- ColQwen3.5 (4.5B, 2026, community): builds on Qwen2.5-VL-7B with LoRA r=16/α=64,
  evolutionary model-soup merging; **90.9 nDCG@5** on ViDoRe (top-3 as of 2026-04).

**ViDoRe scores:**
| Model | nDCG@5 |
|---|---|
| ColQwen2-v1.0 | 89.3 |
| ColQwen2.5-v0.2 | 89.4 |
| Tomoro-ColQwen3-8B | 90.6 |
| ColQwen3.5-4.5B-v3 | 90.9 |
| Argus-9B (2026) | 86.0 on subset |

Source: <https://github.com/illuin-tech/colpali> (README leaderboard section)

**GPU requirement:**
- ColQwen2 / ColQwen2.5 (3B base): **~8–10 GB VRAM** float16; comfortably fits RTX 3080/4070.
- 4-bit quantized (bitsandbytes): ~4–5 GB, RTX 3060 class.
- ColQwen3.5 (7B base): **~16 GB VRAM** float16; 4-bit ~8 GB.
- Indexing throughput: ~12 pages/sec (ColQwen2.5, A100 80 GB).

Source: <https://www.spheron.network/blog/colpali-multimodal-document-rag-gpu-cloud/>

**ONNX/CPU:** No official ONNX export exists for ColQwen2/2.5 as of mid-2026. The dynamic
vision encoder (Qwen2-VL uses dynamic NTK-scaled RoPE for variable-length patches) complicates
static ONNX tracing. **EmbedAnything** (Rust/Candle + ONNX backend) does support ColPali
inference with ONNX but this has not been extended to ColQwen2 publicly:
<https://github.com/StarlightSearch/EmbedAnything>

**SOTA/adoption verdict:** ColQwen2.5 is the current practical sweet spot — ~10 NDCG points
above ColPali-v1 on ViDoRe, fits on consumer GPUs, supported in `colpali-engine >= 0.3.1`.
The ColQwen3.5 line is leading SOTA as of 2026 but requires 7B+ class hardware. The entire
family is GPU-mandatory for production indexing.

**Implementability for Jera:**  
Real adapter: wrap `colpali-engine`'s `ColQwen2_5` model + processor; call
`model(**inputs).last_hidden_state` → linear proj → list of patch vectors; return from
`embed_multi()`. Images passed in as `PIL.Image` objects, texts (queries) processed via the
same model's text path for `embed_query_multi()`. Needs GPU; lives behind `extras = ["visual"]`.

Sources:  
- <https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_reranker_and_vlms>  
- <https://medium.com/@juan.ovallevillamil/the-king-of-multi-modal-rag-colpali-3a03b0db476c>

---

## 3. ColSmolVLM (2025)

**What it is:** Lightweight ColPali-style model based on SmolVLM (256M and 500M parameter
VLMs from HuggingFace), designed for on-device and consumer-hardware visual retrieval.

**Citation:**  
Maraval, A. et al. (2025). *SmolVLM: Redefining small and efficient multimodal models.*  
arXiv:2504.05299.  
<https://arxiv.org/pdf/2504.05299>

Model cards:  
- <https://huggingface.co/vidore/ColSmolVLM-Instruct-500M-base>  
- <https://huggingface.co/vidore/ColSmolVLM-256M-Base>  
- Cookbook: <https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_smol_vlm>

**Architecture:**
- Backbone: SmolVLM-256M or SmolVLM-500M (SigLIP vision encoder + SmolLM text trunk)
- Same ColBERT-style projection head as ColPali (D=128)
- Patch count: similar to ColPali (~1024 patches at standard resolution)
- Training set: 127,460 query-page pairs (63 % academic VQA, 37 % synthetic via Claude-3 Sonnet)

**GPU requirement:** SmolVLM-256M inference uses **< 1 GB GPU memory**; 500M fits in ~2 GB.
The 256M variant has been demonstrated in-browser via WebGPU, making it the only ColPali-family
model with a credible CPU-light path. ColSmolVLM-256M inference on a CPU is slow (seconds
per page) but feasible in principle for small batches.

**ViDoRe scores:** ColSmolVLM claims performance "comparable to models 10× its size" — roughly
matches ColPali-v1 (~81 nDCG@5) despite being 10× smaller. Not in the top tier vs.
ColQwen2.5.

**SOTA/adoption verdict:** Niche but important for Jera's offline-first mandate. ColSmolVLM-256M
is the only ColPali-family model with a realistic CPU-only-ish inference path (slow but
possible without discrete GPU), making it the bridge between the deterministic CI adapter
and production GPU-based adapters.

**Implementability for Jera:**  
Opt-in real adapter wrapping `vidore/ColSmolVLM-Instruct-500M-base` from `transformers`.
Could run on CPU for integration tests with real model weights (slow, ~2–5 sec/page), gated
behind `extras = ["visual-smol"]`. For true offline CI, still use the deterministic hash
adapter below.

---

## 4. ColModernVBERT (Teiletche et al., 2025)

**What it is:** 250M-parameter encoder-only visual document retriever built on ModernBERT
(not a VLM); processes images directly to patch embeddings without a language model trunk,
enabling fast CPU inference.

**Citation:**  
Teiletche, P. et al. (2025). *ModernVBERT: Towards Smaller Visual Document Retrievers.*  
arXiv:2510.01149.  
<https://arxiv.org/pdf/2510.01149>

**Architecture:**
- Backbone: ModernBERT encoder (250M params, no LM decoder, rotary embeddings, Flash Attention 2)
- Image → patches → BERT-style encoder → projection head (D=128)
- No text generation step; purely encoder forward pass
- ONNX export supported (encoder-only architectures export cleanly via `optimum`)

**ViDoRe scores:** ~83 nDCG@5 (ColPali: 81, ColQwen2: 89). Competitive at 250M parameters.  
"Only 0.6 nDCG@5 points below ColPali despite 10× fewer parameters."

Sources:  
- <https://pub.towardsai.net/prod-scale-visual-document-retrieval-with-colmodernvbert-and-qdrant-for-vlms-4e98dfc75e99>  
- <https://www.emergentmind.com/topics/colmodernvbert>

**CPU inference:** The paper reports ~150 ms/page CPU latency (vs. multi-second for VLM-based
models), making ColModernVBERT the first viable CPU-production-class visual retriever.
ONNX runtime brings this further down.

**SOTA/adoption verdict:** Underrated. For Jera's offline-first CI, ColModernVBERT is the
most realistic real-model option that doesn't mandate a GPU — 250M encoder-only, ~150 ms/page
CPU, ONNX-exportable. Models released at `huggingface.co/ModernVBERT`.

**Implementability for Jera:**  
A `ColModernVBERTEmbedding` adapter implementing `MultiVectorEmbedding` can load via
`transformers`/`optimum` ONNX runtime with CPU provider. Images → ResizePad → patch tokens
→ encoder forward → 128-d projection. This is the recommended **opt-in real adapter** for
teams without GPUs. Gate behind `extras = ["visual-bert"]`.

---

## 5. NanoVDR (2026)

**What it is:** 70M-parameter text-only encoder distilled from a 2B VLM retriever; processes
rendered-text representations of documents (not images), trading visual fidelity for extreme
lightness and CPU friendliness.

**Citation:**  
*NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual
Document Retrieval.*  
arXiv:2603.12824 (2026).  
<https://arxiv.org/pdf/2603.12824>

**Architecture:** Knowledge distillation from a VLM teacher (ColPali-class) to a 70M
BERT-family text encoder. At inference, documents are represented as text extracted from
images (or rendered markdown), not raw images. Queries are standard text queries. No patch
vectors; single dense vector (or sparse) per chunk. **Not a MaxSim / multi-vector model.**

**CPU feasibility:** Excellent — 70M text encoder, ONNX exportable, runs in ~10 ms/chunk on
CPU.

**SOTA/adoption verdict:** Interesting research direction but architecturally incompatible
with Jera's MaxSim port (it produces single dense vectors). It would plug into `EmbeddingProvider`
not `MultiVectorEmbedding`. Lower retrieval quality than ColPali-family on visual-heavy documents
because it abandons the patch-vector representation.

**Implementability for Jera:** Not a fit for the `VisualMultiVectorEmbedding` goal. Could
be a fallback dense adapter for CPU environments with degraded visual fidelity.

---

## 6. Visual RAG Toolkit — Training-Free Pooling (2026)

**What it is:** Paper proposing training-free row/column pooling to shrink ColPali-style
patch vector indices by ~10–32× before MaxSim, enabling fast multi-stage search.

**Citation:**  
*Visual RAG Toolkit: Scaling Multi-Vector Visual Retrieval with Training-Free Pooling and
Multi-Stage Search.*  
arXiv:2602.12510 (2026).  
<https://arxiv.org/pdf/2602.12510>

**Key findings:**
- **Mean row pooling:** average all patch vectors in a row → reduces 1024 patches to 32 row
  vectors per page (32×). Use as first-pass filter.
- **Mean column pooling:** average column-wise → 32 column vectors.
- Stage 1: score candidates with pooled vectors (fast, approximate).
- Stage 2: re-rank top-K with full patch vectors (accurate).
- Result: ~10× throughput gain with <1 % nDCG@5 drop on ViDoRe.
- Storage: pooled index ~85 KB/page vs. 257 KB/page full.

**Implementability for Jera:**  
This pooling is a pure post-processing transform on the patch vector matrix returned by
`embed_multi`. Can be implemented as a wrapper `PooledVisualMultiVectorEmbedding` that calls
the real adapter and applies row/column mean pooling before returning. `MaxSimVectorStore`
is unchanged — it already handles variable-length document vectors. The pooling reduces CI
test complexity too (fewer vectors to compare).

---

## 7. Hierarchical Patch Compression (2025–2026)

**What it is:** Dynamic pruning + quantization of ColPali patch vectors to reduce index size
without retraining.

**Citations:**  
*Hierarchical Patch Compression for ColPali: Efficient Multi-Vector Document Retrieval with
Dynamic Pruning and Quantization.*  
arXiv:2506.21601 (2025).  
<https://arxiv.org/html/2506.21601v1>

*Hierarchical Patch Compression for ColPali.*  
Proceedings paper: <https://www.scitepress.org/Papers/2025/137325/137325.pdf>

**Key findings:**  
- Factor-3 token pooling: 1024 → 342 patches per page, 97.8 % NDCG@5 retention.
- Binary encoding of centroid indices enables CPU-only retrieval with HPC-ColPali path.
- int8 quantization of patch vectors: ~25 KB/page (ColQwen2.5 dynamic-res baseline).

---

## 8. CPU/ONNX Feasibility Summary

| Path | CPU viable? | Quality | Notes |
|---|---|---|---|
| ColPali (PaliGemma-3B) | No (seconds/page on CPU) | 81 nDCG@5 | GPU mandatory for production |
| ColQwen2.5 (Qwen2.5-VL-3B) | No | 89 nDCG@5 | GPU mandatory |
| ColSmolVLM-256M | Marginal (~2–5 s/page) | ~81 nDCG@5 | Possible for small batches / dev use |
| ColModernVBERT-250M | **Yes (~150 ms/page)** | 83 nDCG@5 | ONNX; encoder-only; recommended CPU path |
| EmbedAnything + ONNX (ColPali) | Partial | ColPali-class | ONNX projection layer only; still needs SigLIP |
| NanoVDR-70M | Yes (~10 ms) | Lower (text-only distill) | Single-vector, not MaxSim |
| Deterministic hash adapter | Yes (microseconds) | N/A (CI only) | `HashMultiVectorEmbedding` pattern |

**Honest assessment:** Production visual RAG (ColPali family, VLM backbone) is **GPU-only** for
any reasonable throughput. The only CPU paths with real model weights are ColModernVBERT (250M
encoder, ~150 ms/page, ONNX) and ColSmolVLM-256M (slow). For Jera's CI contract
(deterministic, offline, non-tautological), a **hash-based patch adapter** is the correct
solution — matching the existing `HashMultiVectorEmbedding` design pattern exactly.

---

## 9. MediaType / Parser Implications

ColPali operates at the **page-image** level. The current Jera parser stack (M5a, M3) produces
text chunks from PDFs. To feed ColPali, a new parser adapter is needed:

```
PDF bytes
  └─► pdf2image / pymupdf / pdfplumber → PIL.Image per page
        └─► VisualMultiVectorEmbedding.embed_multi([img_bytes, ...])
              └─► MaxSimVectorStore
```

Key implications:
1. **MediaType.IMAGE_PAGE** (new domain concept): a chunk whose "text" is the page image
   bytes (or a path reference), not extracted text. The chunk content is opaque bytes.
2. The parser role reverses: instead of extracting text, it extracts images. A
   `PdfToImageParser` produces `Chunk(content=<png_bytes>, media_type=IMAGE_PAGE, page=n)`.
3. The `VisualMultiVectorEmbedding.embed_multi()` signature changes from `texts: Sequence[str]`
   to `images: Sequence[bytes]` — this is a **new port**, not the existing
   `MultiVectorEmbedding`. The existing port is text-in; the visual one is bytes-in.
4. For CI, the deterministic adapter ignores actual image content — it derives patch vectors
   from a hash of the image bytes (e.g., SHA-1 of each fixed pixel region), which is
   non-tautological as long as the test corpus has images that genuinely differ in content.

**PDF → image libraries:**
- `pdf2image` (poppler dependency, pure Python): simplest
- `pymupdf` (fitz, BSD license, no poppler): fastest, no system deps — preferred for Jera
- `pdfplumber`: better for text; inferior for image export

Source: <https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/>

---

## 10. Implementability for Jera — Detailed Design

### Port extension needed

The existing `MultiVectorEmbedding` protocol takes `texts: Sequence[str]`. Visual documents
require images. Define a **new protocol** alongside the existing one:

```python
# jera/ports/visual_multi_vector_embedding.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class VisualMultiVectorEmbedding(Protocol):
    model_id: str
    dimensions: int
    patch_count: int  # expected patches per image (may be dynamic)

    def embed_images(self, images: Sequence[bytes]) -> list[list[list[float]]]:
        """Return patch-vector matrices for each page image.
        Shape: [len(images), n_patches_i, dimensions]
        """
        ...

    def embed_query_multi(self, text: str) -> list[list[float]]:
        """Text query → per-token query vectors (same as MultiVectorEmbedding)."""
        ...
```

`MaxSimVectorStore` is **unchanged** — it accepts `list[list[float]]` doc vectors regardless
of whether they came from text tokens or image patches.

### Adapter A: Deterministic CI adapter (no model, offline)

```python
class HashVisualMultiVectorEmbedding:
    """Deterministic patch vectors from image bytes — CI offline stand-in for ColPali."""

    model_id = "hash-visual-multivec-v1-128"
    dimensions = 128
    patch_count = 32  # 4×8 grid for speed in CI

    def embed_images(self, images: Sequence[bytes]) -> list[list[list[float]]]:
        results = []
        for img_bytes in images:
            # Divide the image into a fixed H×W patch grid deterministically.
            # Each patch's vector = _token_vector(sha1_hex(patch_region_bytes), 128)
            # For CI, approximate: hash 32 equal-sized byte slices of the whole image.
            chunk_size = max(1, len(img_bytes) // self.patch_count)
            patch_vecs = []
            for i in range(self.patch_count):
                region = img_bytes[i * chunk_size : (i + 1) * chunk_size]
                digest = hashlib.sha1(region).hexdigest()
                patch_vecs.append(_token_vector(digest, self.dimensions))
            results.append(patch_vecs)
        return results

    def embed_query_multi(self, text: str) -> list[list[float]]:
        # Reuse text token hashing — queries are always text
        return HashMultiVectorEmbedding(self.dimensions).embed_query_multi(text)
```

### Adapter B: Real opt-in adapter (GPU, ColQwen2.5)

```python
class ColQwen2Embedding:
    """ColQwen2.5-v0.2 visual multi-vector embedding — requires CUDA GPU.

    Install: pip install jera[visual]  (pulls colpali-engine, transformers>=4.45)
    """
    model_id = "vidore/colqwen2.5-v0.2"
    dimensions = 128

    def __init__(self, device: str = "cuda:0") -> None:
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
        self._model = ColQwen2_5.from_pretrained(self.model_id, ...).to(device)
        self._proc  = ColQwen2_5_Processor.from_pretrained(self.model_id)

    def embed_images(self, images: Sequence[bytes]) -> list[list[list[float]]]:
        pil_images = [PIL.Image.open(io.BytesIO(b)) for b in images]
        inputs = self._proc.process_images(pil_images)
        with torch.no_grad():
            out = self._model(**inputs)
        # out.last_hidden_state shape: [B, n_patches, 128]
        return out.tolist()

    def embed_query_multi(self, text: str) -> list[list[float]]:
        inputs = self._proc.process_queries([text])
        with torch.no_grad():
            out = self._model(**inputs)
        return out[0].tolist()
```

### Adapter C: CPU-viable opt-in adapter (ColModernVBERT, ONNX)

```python
class ColModernVBERTEmbedding:
    """ColModernVBERT-250M via optimum ONNX runtime — CPU viable (~150 ms/page).

    Install: pip install jera[visual-bert]  (pulls optimum[onnxruntime])
    """
    model_id = "ModernVBERT/colmodernvbert"
    dimensions = 128
    # ...encoder-only, no LM trunk, clean ONNX export
```

### Non-tautological deterministic test design

The key requirement is that the test proves **correct ranking via MaxSim over patch vectors**,
not merely that the same query returns itself (tautology).

```
Corpus:
  doc_A = image_bytes_A  (synthesized: a solid blue 32×32 PNG → deterministic byte pattern)
  doc_B = image_bytes_B  (synthesized: a solid red 32×32 PNG → different byte pattern)

Query = "blue page"

HashVisualMultiVectorEmbedding:
  - embed_images([A, B]) → patch_vecs_A, patch_vecs_B
    (patch_vecs_A are hashed from blue-pixel bytes; patch_vecs_B from red-pixel bytes)
  - embed_query_multi("blue") → query_token_vecs
    (hashed from the token text "blue")

PROBLEM: The hash adapter hashes image bytes, not semantic content — "blue" query token
has no inherent affinity for blue-pixel bytes. The test cannot rely on semantic matching.

CORRECT DESIGN: Make doc_A and doc_B differ by a known controlled property:
  - doc_A = image_bytes constructed so that one of its 32 byte-slices is the SHA1 of "blue"
    (i.e., artificially embed a token-like signal into one image's bytes)
  - doc_B = image constructed from unrelated bytes

OR: Use a different non-tautological approach — embed two images where A shares a byte-region
hash with the query token, B does not. The test then verifies MaxSim scores A > B.

SIMPLER CORRECT DESIGN (matches HashMultiVectorEmbedding test pattern exactly):

  query_vecs = embed_query_multi("query_token_A")
    → hash_vec("query_token_a")  [one vector, deterministic]

  doc_A_patches = embed_images([image_A])
    where image_A is crafted so that its first byte-slice hashes to a vector very close
    to hash_vec("query_token_a"):
    → image_A_bytes[0:chunk] = sha1("query_token_a").digest() * (chunk // 20 + 1)  [first chunk]
    → remaining chunks = random/zero bytes

  doc_B_patches = embed_images([image_B])
    where image_B has no byte-slice close to hash_vec("query_token_a")

  store.add([("A", doc_A_patches[0]), ("B", doc_B_patches[0])])
  results = store.search_maxsim(query_vecs, top_k=2)

  assert results[0].chunk_id == "A"  # non-tautological: A wins because its first patch
                                      # vector is identical to the query token vector
                                      # (same SHA1 input → same hash vector → cosine = 1.0)
```

This design is **non-tautological** because:
1. The images are not the query itself — they are synthetic byte arrays
2. The ranking is determined by the MaxSim path through `MaxSimVectorStore`
3. The test proves that patch-vector identity correctly propagates through the pipeline
4. It matches the exact pattern of `test_target_ranks_first_by_maxsim` in M10

---

## 11. Top Recommendation (Decisive)

**Implement in this order:**

### Phase 1 (M14a) — Deterministic CI slice, zero GPU, no new deps

1. Define `VisualMultiVectorEmbedding` protocol in `jera/ports/visual_multi_vector_embedding.py`
   with `embed_images(images: Sequence[bytes]) -> list[list[list[float]]]` and
   `embed_query_multi(text: str) -> list[list[float]]`.
2. Implement `HashVisualMultiVectorEmbedding` in
   `jera/adapters/embedding/hash_visual_multivector.py` using the controlled-byte-slice
   strategy above. Reuses `_token_vector` from the existing `hash_multivector.py`.
3. Add `PdfToImageParser` stub (uses `pymupdf` if available, falls back to raw bytes for
   CI — no real PDF needed in tests; tests use synthetic PNG bytes).
4. Write the non-tautological test in `tests/unit/test_visual_maxsim_retrieval.py`:
   - Craft two synthetic images where image_A's first byte-chunk is seeded from a known
     token string; image_B has different bytes.
   - Query = that token string.
   - Assert image_A ranks #1 via `MaxSimVectorStore.search_maxsim` (unchanged).
   - **No model download, no GPU, runs in CI.**
5. `MaxSimVectorStore` requires **zero changes** — it already handles patch vectors.

### Phase 2 (M14b) — Real CPU adapter (opt-in)

6. Implement `ColModernVBERTEmbedding` with `transformers`/`optimum` ONNX runtime.
   Gate behind `pip install jera[visual-bert]`.
   ~150 ms/page CPU, 250M params, ONNX-exportable. Best choice for CPU-only environments.

### Phase 3 (M14c) — Production GPU adapter (opt-in)

7. Implement `ColQwen2Embedding` wrapping `colpali-engine`.
   Gate behind `pip install jera[visual]` + CUDA.
   ColQwen2.5-v0.2 or ColQwen3.5 depending on VRAM available.
   Use pooled variant (3× row pooling) for index-size reduction via training-free wrapper.

### What NOT to do

- Do not attempt GPU-free ColPali/ColQwen2 inference in CI — it is impractically slow (>30 s/page)
- Do not build a NanoVDR adapter for the `VisualMultiVectorEmbedding` port — it is single-vector
- Do not modify `MaxSimVectorStore` — it is already correct for patch vectors

---

## Sources Index

| Title | URL | Year |
|---|---|---|
| ColPali: Efficient Document Retrieval with Vision Language Models | <https://arxiv.org/abs/2407.01449> | 2024 |
| ColPali HuggingFace Blog | <https://huggingface.co/blog/manu/colpali> | 2024 |
| colpali-engine GitHub | <https://github.com/illuin-tech/colpali> | 2024–2026 |
| vidore/colpali model card | <https://huggingface.co/vidore/colpali> | 2024 |
| ColQwen2 transformers docs | <https://huggingface.co/docs/transformers/en/model_doc/colqwen2> | 2024 |
| vidore/colqwen2.5-v0.2 model card | <https://huggingface.co/vidore/colqwen2.5-v0.2> | 2025 |
| ColQwen3.5-4.5B-v3 model card | <https://huggingface.co/athrael-soju/colqwen3.5-4.5B-v3> | 2026 |
| ColSmolVLM-Instruct-500M-base | <https://huggingface.co/vidore/ColSmolVLM-Instruct-500M-base> | 2025 |
| ColSmolVLM-256M-Base | <https://huggingface.co/vidore/ColSmolVLM-256M-Base> | 2025 |
| SmolVLM: Redefining small and efficient multimodal models | <https://arxiv.org/pdf/2504.05299> | 2025 |
| Smol Multimodal RAG cookbook | <https://huggingface.co/learn/cookbook/en/multimodal_rag_using_document_retrieval_and_smol_vlm> | 2025 |
| ModernVBERT: Towards Smaller Visual Document Retrievers | <https://arxiv.org/pdf/2510.01149> | 2025 |
| ColModernVBERT + Qdrant blog | <https://pub.towardsai.net/prod-scale-visual-document-retrieval-with-colmodernvbert-and-qdrant-for-vlms-4e98dfc75e99> | 2025 |
| NanoVDR: Distilling a 2B VLR into 70M Text Encoder | <https://arxiv.org/pdf/2603.12824> | 2026 |
| Visual RAG Toolkit: Training-Free Pooling | <https://arxiv.org/pdf/2602.12510> | 2026 |
| Hierarchical Patch Compression for ColPali | <https://arxiv.org/html/2506.21601v1> | 2025–2026 |
| Qdrant ColPali integration blog | <https://qdrant.tech/blog/qdrant-colpali/> | 2024 |
| ColPali + GPU Cloud deployment (Spheron) | <https://www.spheron.network/blog/colpali-multimodal-document-rag-gpu-cloud/> | 2026 |
| Vespa PDF retrieval with VLMs | <https://blog.vespa.ai/retrieval-with-vision-language-models-colpali/> | 2024 |
| Multimodal RAG cookbook (ColQwen2 + reranker) | <https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_reranker_and_vlms> | 2024 |
| EmbedAnything (ONNX/Candle ColPali) | <https://github.com/StarlightSearch/EmbedAnything> | 2024–2025 |
| Reproducibility of Late-Interaction Visual Retrieval | <https://arxiv.org/pdf/2505.07730> | 2025 |
| ViDoRe Leaderboard | <https://huggingface.co/spaces/vidore/vidore-leaderboard> | live |
