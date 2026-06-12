"""Deterministic hashing embedding — TEST/CI default (offline, no torch, reproducible).

Each token is hashed into a bucket; the vector is the L2-normalized bag of hashed tokens.
This gives lexical-overlap-sensitive geometry (exact-term queries behave sensibly) but NO
semantic/paraphrase geometry. We never claim it demonstrates semantic superiority — that
would be the fake path Gate 7 forbids. It exists so the full pipeline runs deterministically
without any model download.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from jera.domain.vectors import DenseVector

_WORD = re.compile(r"\w+", re.UNICODE)


class HashEmbedding:
    def __init__(self, dimensions: int = 256) -> None:
        self.model_id = f"hash-emb-v1-{dimensions}"
        self.dimensions = dimensions
        self.context_limit: int | None = None

    def embed(self, texts: Sequence[str]) -> list[DenseVector]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> DenseVector:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> DenseVector:
        vec = [0.0] * self.dimensions
        for token in _WORD.findall(text.lower()):
            h = hashlib.sha1(token.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:4], "big") % self.dimensions
            sign = 1.0 if h[4] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]
