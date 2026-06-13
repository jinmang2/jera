"""MultiVectorEmbedding port — late-interaction (ColBERT-style) token-level embeddings.

A real adapter would wrap BGE-M3's ColBERT head or ColBERTv2/PLAID; this port defines
the interface any such adapter must satisfy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class MultiVectorEmbedding(Protocol):
    """Produces one dense vector *per token* for each input text.

    Attributes:
        model_id:   Unique identifier recorded in index metadata; changing it
                    requires full re-indexing (same as ``EmbeddingProvider``).
        dimensions: Dimensionality of each per-token vector.

    The opt-in production adapter would wrap BGE-M3's ColBERT head and return its
    token-level projection.  This port is intentionally model-agnostic.
    """

    model_id: str
    dimensions: int

    def embed_multi(self, texts: Sequence[str]) -> list[list[list[float]]]:
        """Return per-token vectors for each text.

        Shape: ``[len(texts), n_tokens_i, dimensions]`` where ``n_tokens_i`` may
        differ per text.
        """
        ...

    def embed_query_multi(self, text: str) -> list[list[float]]:
        """Return per-token vectors for a single query string.

        Shape: ``[n_tokens, dimensions]``.
        """
        ...
