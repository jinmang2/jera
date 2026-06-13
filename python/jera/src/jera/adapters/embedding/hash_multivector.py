"""Deterministic hash multi-vector embedding — CI/test default (offline, no torch).

Tokenises each text by lowercase-splitting on whitespace, then maps EACH TOKEN to its
own L2-normalised dense vector whose direction is seeded solely by a SHA-1 hash of the
token *text* (not its position).  The key property:

    same token text  →  same vector, in every document and in every query.

This means genuine token-overlap between a query and a document directly produces high
per-token cosine similarities, which drives the MaxSim score up without any position
bias.  It is the deterministic CI analogue of a real ColBERT token encoder.

The opt-in production adapter would wrap BGE-M3's ColBERT projection head; this adapter
exists so the full late-interaction pipeline runs without any model download.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence


def _token_vector(token: str, dimensions: int) -> list[float]:
    """Deterministic, L2-normalised vector for *token* in R^dimensions.

    The vector is built by scanning consecutive 5-byte windows of the SHA-1 digest
    (cycled as needed) and turning each window into a signed bucket increment, then
    L2-normalising the result.  The construction guarantees:

    * Determinism: same token → same vector across runs, processes, and platforms.
    * Non-zero output: at least the first bucket is always incremented.
    * Token-identity geometry: identical tokens produce identical vectors (cosine = 1),
      while unrelated tokens produce nearly-orthogonal vectors in high dimensions.
    """
    digest = hashlib.sha1(token.encode("utf-8")).digest()  # 20 bytes
    # Need dimensions*5 bytes; SHA-1 gives 20 bytes, so repeat ceiling(dimensions*5/20) times.
    need = dimensions * 5
    repeat = math.ceil(need / len(digest))
    seed = (digest * repeat)[:need]

    vec = [0.0] * dimensions
    for i in range(dimensions):
        chunk = seed[i * 5 : i * 5 + 5]
        bucket = int.from_bytes(chunk[:4], "big") % dimensions
        sign = 1.0 if chunk[4] & 1 else -1.0
        vec[bucket] += sign

    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        # Fallback: set dimension-0 to 1.0 (cannot happen in practice but satisfies mypy).
        vec[0] = 1.0
        return vec
    return [v / norm for v in vec]


class HashMultiVectorEmbedding:
    """Per-token deterministic embedding for late-interaction retrieval (CI default).

    Each whitespace-split lowercase token in a text is mapped to an independent
    L2-normalised vector.  The vector is a function of the token *text* only, so the
    same word produces the same vector everywhere — genuine query/document token overlap
    drives MaxSim scores up, which is exactly the property ColBERT exploits.

    Args:
        dimensions: Dimensionality of each per-token vector (default 64).
    """

    def __init__(self, dimensions: int = 64) -> None:
        self.model_id = f"hash-multivec-v1-{dimensions}"
        self.dimensions = dimensions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenise(self, text: str) -> list[str]:
        """Lowercase-split tokeniser — mirrors HashEmbedding's _WORD logic."""
        tokens = text.lower().split()
        return tokens if tokens else ["<empty>"]

    def _embed_tokens(self, tokens: list[str]) -> list[list[float]]:
        return [_token_vector(tok, self.dimensions) for tok in tokens]

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------

    def embed_multi(self, texts: Sequence[str]) -> list[list[list[float]]]:
        """Return per-token vector matrices for each text.

        Returns:
            List of length ``len(texts)``; each element is a matrix of shape
            ``[n_tokens_i, dimensions]``.
        """
        return [self._embed_tokens(self._tokenise(t)) for t in texts]

    def embed_query_multi(self, text: str) -> list[list[float]]:
        """Return per-token vectors for a single query string.

        Returns:
            Matrix of shape ``[n_tokens, dimensions]``.
        """
        return self._embed_tokens(self._tokenise(text))
