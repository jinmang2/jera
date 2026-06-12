"""Semantic chunker (candidate strategy).

Merges adjacent elements while their embedding cosine similarity exceeds a threshold, then
falls back to token windowing. It depends on an EmbeddingProvider, which is the reason it is
NOT the M1 default (the baseline must run with zero model dependencies). With deterministic
hash embeddings the *boundaries* are lexical-overlap-driven, not truly semantic — its real
value appears under the `local` profile (fastembed). The chunking gate compares its output
SHAPE against the heading-aware baseline on the same fixture; semantic superiority is only
asserted under `local`.
"""

from __future__ import annotations

import math

from jera.adapters.chunking.heading_aware import HeadingAwareChunker
from jera.domain.chunk import Chunk
from jera.domain.document import ParsedDocument
from jera.domain.ids import stable_id
from jera.ports.embedding import EmbeddingProvider


class SemanticChunker:
    strategy = "semantic"
    version = "0.1.0"

    def __init__(
        self,
        embedding: EmbeddingProvider,
        threshold: float = 0.55,
        max_tokens: int = 180,
    ) -> None:
        self._embedding = embedding
        self._threshold = threshold
        self._baseline = HeadingAwareChunker(max_tokens=max_tokens, overlap_tokens=0)

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        # Start from heading-aware blocks, then re-key chunk ids/strategy for this strategy.
        base = self._baseline.chunk(document)
        if len(base) <= 1:
            return [self._retag(c, idx) for idx, c in enumerate(base)]

        vectors = self._embedding.embed([c.text for c in base])
        merged: list[Chunk] = []
        buffer = base[0]
        buf_vec = vectors[0]
        for chunk, vec in zip(base[1:], vectors[1:], strict=True):
            if (
                chunk.section_path == buffer.section_path
                and _cosine(buf_vec, vec) >= self._threshold
            ):
                buffer = _join(buffer, chunk)
                buf_vec = _mean(buf_vec, vec)
            else:
                merged.append(buffer)
                buffer = chunk
                buf_vec = vec
        merged.append(buffer)
        return [self._retag(c, idx) for idx, c in enumerate(merged)]

    def _retag(self, chunk: Chunk, idx: int) -> Chunk:
        new_id = stable_id(
            chunk.document_id,
            self.strategy,
            self.version,
            "/".join(chunk.section_path),
            str(idx),
        )
        return chunk.model_copy(
            update={
                "chunk_id": new_id,
                "chunk_strategy": self.strategy,
                "chunk_version": self.version,
            }
        )


def _join(a: Chunk, b: Chunk) -> Chunk:
    return a.model_copy(
        update={
            "text": f"{a.text}\n\n{b.text}",
            "element_ids": a.element_ids + b.element_ids,
            "char_span": (a.char_span[0], b.char_span[1]),
            "page_span": a.page_span.merge(b.page_span),
            "token_count": a.token_count + b.token_count,
        }
    )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _mean(a: list[float], b: list[float]) -> list[float]:
    return [(x + y) / 2 for x, y in zip(a, b, strict=True)]
