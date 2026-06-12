"""Semantic chunker: embedding-breakpoint splitting within heading sections.

For each section, sentences are embedded and split at points where the cosine distance between
consecutive sentences exceeds a percentile threshold (the LlamaIndex semantic-splitter idea),
with a hard token cap. Deterministic given the embedding provider, so it is reproducible in
tests (hash embeddings → lexical boundaries) and genuinely semantic under `local` (fastembed).
"""

from __future__ import annotations

import math

from jera.adapters.chunking.sections import Section, group_sections
from jera.adapters.chunking.sentences import split_sentences_with_offsets
from jera.adapters.chunking.tokenizer import count_tokens
from jera.domain.chunk import Chunk
from jera.domain.document import ParsedDocument
from jera.domain.ids import stable_id
from jera.domain.vectors import DenseVector
from jera.ports.embedding import EmbeddingProvider


class SemanticChunker:
    strategy = "semantic"
    version = "2.0.0"

    def __init__(
        self,
        embedding: EmbeddingProvider,
        *,
        breakpoint_percentile: float = 90.0,
        max_tokens: int = 180,
    ) -> None:
        if not 0 < breakpoint_percentile < 100:
            raise ValueError("breakpoint_percentile must be in (0, 100)")
        self._embedding = embedding
        self._percentile = breakpoint_percentile
        self._max_tokens = max_tokens

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in group_sections(document.elements):
            chunks.extend(self._chunk_section(document, section))
        return chunks

    def _chunk_section(self, document: ParsedDocument, section: Section) -> list[Chunk]:
        sentences = split_sentences_with_offsets(section.text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [self._emit(document, section, sentences, 0)]

        vectors = self._embedding.embed([s for s, _, _ in sentences])
        distances = [1.0 - _cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
        threshold = _percentile(distances, self._percentile)

        chunks: list[Chunk] = []
        start_idx = 0
        running_tokens = count_tokens(sentences[0][0])
        for i in range(len(sentences) - 1):
            next_tokens = count_tokens(sentences[i + 1][0])
            over_budget = running_tokens + next_tokens > self._max_tokens
            semantic_break = distances[i] > threshold
            if over_budget or semantic_break:
                chunks.append(
                    self._emit(document, section, sentences[start_idx : i + 1], len(chunks))
                )
                start_idx = i + 1
                running_tokens = next_tokens
            else:
                running_tokens += next_tokens
        chunks.append(self._emit(document, section, sentences[start_idx:], len(chunks)))
        return chunks

    def _emit(
        self,
        document: ParsedDocument,
        section: Section,
        sentences: list[tuple[str, int, int]],
        index: int,
    ) -> Chunk:
        char_start = sentences[0][1]
        char_end = sentences[-1][2]
        text = section.text[char_start:char_end]
        element_ids, page_span = section.attribute(char_start, char_end)
        chunk_id = stable_id(
            document.document_id,
            self.strategy,
            self.version,
            "/".join(section.section_path),
            str(index),
            str(char_start),
        )
        return Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            source_id=document.source_id,
            text=text,
            page_span=page_span,
            section_path=section.section_path,
            element_ids=element_ids,
            char_span=(char_start, char_end),
            token_count=count_tokens(text),
            chunk_strategy=self.strategy,
            chunk_version=self.version,
            parent_chunk_id=None,
        )


def _cosine(a: DenseVector, b: DenseVector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (deterministic)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)
