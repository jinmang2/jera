"""Visual late-interaction (ColPali-style) multi-vector embedding — deterministic CI core.

ColPali (Faysse et al., ICLR 2025, arXiv:2407.01449) retrieves over *page images*: a VLM turns
each page into one vector per image patch and scores with the standard ColBERT MaxSim formula —
``score(q, d) = Σ_qi max_dj cosine(qi, dj)``. That is architecturally identical to Jera's M10
`MaxSimVectorStore`; ColPali is just ColBERT over image patches instead of text tokens, so it
reuses the same `MaxSimVectorStore` for scoring/storage.

This adapter is the deterministic, offline CI core: an image's bytes are split into a fixed patch
grid and each patch is hashed to an L2-normalised vector (reusing the token-vector scheme, so
patches and query tokens live in one space). No model, no GPU. The opt-in production adapter is
`ColModernVBERTEmbedding` (encoder-only, ~250M params, CPU/ONNX via optimum — arXiv:2510.01149) or
a ColPali/ColQwen wrapper (GPU) — both behind a `visual` extra; this adapter exists so the visual
MaxSim mechanism is testable without any download.

Honest limit: cross-modal *text-query → image* alignment is what the real VLM learns; hash patches
and hash tokens do not share that learned alignment, so the deterministic non-tautological property
proven here is **image↔image** visual MaxSim retrieval (find the page that shares visual patches).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from jera.adapters.embedding.hash_multivector import HashMultiVectorEmbedding, _token_vector


class VisualMultiVectorEmbedding:
    """Deterministic ColPali-style patch embedding for image bytes + a text-query path.

    Args:
        dimensions: per-patch / per-token vector size (default 64, matching HashMultiVector).
        n_patches: number of patches an image is split into (a fixed grid; default 16).
    """

    def __init__(self, dimensions: int = 64, n_patches: int = 16) -> None:
        if n_patches <= 0:
            raise ValueError(f"n_patches must be > 0, got {n_patches}")
        self.model_id = f"hash-visual-multivec-v1-{dimensions}"
        self.dimensions = dimensions
        self._n_patches = n_patches
        self._text = HashMultiVectorEmbedding(dimensions=dimensions)

    def _patches(self, image: bytes) -> list[bytes]:
        """Split image bytes into ``n_patches`` contiguous, equal-ish patches (pad if short)."""
        data = image or b"\x00"
        size = max(1, len(data) // self._n_patches)
        patches = [data[i * size : (i + 1) * size] for i in range(self._n_patches - 1)]
        patches.append(data[(self._n_patches - 1) * size :])  # remainder into the last patch
        return [p if p else b"\x00" for p in patches]

    def embed_image_patches(self, image: bytes) -> list[list[float]]:
        """Image bytes → one L2-normalised vector per patch (the document side of ColPali)."""
        return [
            _token_vector(hashlib.sha1(p).hexdigest(), self.dimensions)
            for p in self._patches(image)
        ]

    def embed_images(self, images: Sequence[bytes]) -> list[list[list[float]]]:
        """Batch ``embed_image_patches`` — ready for ``MaxSimVectorStore.add``."""
        return [self.embed_image_patches(img) for img in images]

    def embed_query_image(self, image: bytes) -> list[list[float]]:
        """An image query → patch vectors (image↔image visual retrieval)."""
        return self.embed_image_patches(image)

    def embed_query_multi(self, text: str) -> list[list[float]]:
        """A text query → per-token vectors. (Cross-modal alignment is the real VLM's job; with
        hash vectors this path is mechanism-only — see the module docstring.)"""
        return self._text.embed_query_multi(text)
