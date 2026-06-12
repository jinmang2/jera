"""Deterministic local BM25 sparse provider (Fittable) — default sparse path.

Uses a hashing vocabulary (no global token table needed) so it is self-contained. The
BM25 score is reconstructed as a sparse dot product:

  doc value[t]   = idf[t] * tf*(k1+1) / (tf + k1*(1 - b + b*dl/avgdl))
  query value[t] = query term frequency

so ``doc.dot(query) == BM25(query, doc)``. ``fit(corpus)`` computes idf and avgdl; the ingest
pipeline calls it on the batch being indexed (incremental re-fit per batch — an accepted M1
limitation documented in the ADR).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence

from jera.domain.vectors import SparseVector

_WORD = re.compile(r"\w+", re.UNICODE)
_DIM = 1 << 20  # hashing space


def _index(token: str) -> int:
    return int.from_bytes(hashlib.sha1(token.encode("utf-8")).digest()[:4], "big") % _DIM


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25Local:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.model_id = "bm25-local-v1"
        self.k1 = k1
        self.b = b
        self._idf: dict[int, float] = {}
        self._avgdl: float = 0.0
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, corpus: Sequence[str]) -> None:
        n = len(corpus)
        if n == 0:
            self._idf, self._avgdl, self._fitted = {}, 0.0, True
            return
        df: Counter[int] = Counter()
        total_len = 0
        for text in corpus:
            tokens = _tokenize(text)
            total_len += len(tokens)
            for idx in {_index(t) for t in tokens}:
                df[idx] += 1
        self._avgdl = total_len / n
        self._idf = {idx: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for idx, freq in df.items()}
        self._fitted = True

    def encode(self, texts: Sequence[str]) -> list[SparseVector]:
        if not self._fitted:
            raise RuntimeError("BM25Local.encode requires fit(corpus) first.")
        return [self._encode_doc(t) for t in texts]

    def encode_query(self, text: str) -> SparseVector:
        tf = Counter(_index(t) for t in _tokenize(text))
        if not tf:
            return SparseVector(indices=[], values=[])
        indices = list(tf.keys())
        values = [float(tf[i]) for i in indices]
        return SparseVector(indices=indices, values=values)

    def _encode_doc(self, text: str) -> SparseVector:
        tokens = _tokenize(text)
        dl = len(tokens)
        tf = Counter(_index(t) for t in tokens)
        indices: list[int] = []
        values: list[float] = []
        denom_norm = self.k1 * (1 - self.b + self.b * dl / self._avgdl) if self._avgdl else self.k1
        for idx, freq in tf.items():
            idf = self._idf.get(idx)
            if idf is None:
                continue
            weight = idf * (freq * (self.k1 + 1)) / (freq + denom_norm)
            indices.append(idx)
            values.append(weight)
        return SparseVector(indices=indices, values=values)
