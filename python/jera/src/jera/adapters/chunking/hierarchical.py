"""Hierarchical (RAPTOR-lite) chunker: cluster leaf chunks, synthesize parent summaries.

Builds a 2-level tree: leaf chunks (from a base chunker) are clustered by embedding
similarity; each cluster gets a parent chunk whose text is an extractive summary of its
children, with children linked via ``parent_chunk_id``. This lets retrieval match both
fine-grained passages and higher-level abstractions (the RAPTOR motivation), while staying
deterministic and offline — summaries are extractive (first sentence per child), not LLM calls.
"""

from __future__ import annotations

import math

from jera.adapters.chunking.heading_aware import HeadingAwareChunker
from jera.adapters.chunking.sentences import split_sentences_with_offsets
from jera.adapters.chunking.tokenizer import count_tokens
from jera.domain.chunk import Chunk
from jera.domain.document import ParsedDocument
from jera.domain.ids import stable_id
from jera.domain.vectors import DenseVector
from jera.ports.chunker import Chunker
from jera.ports.embedding import EmbeddingProvider


class HierarchicalChunker:
    strategy = "hierarchical"
    version = "1.0.0"

    def __init__(
        self,
        embedding: EmbeddingProvider,
        *,
        base: Chunker | None = None,
        cluster_threshold: float = 0.5,
        summary_max_tokens: int = 120,
    ) -> None:
        self._embedding = embedding
        self._base = base or HeadingAwareChunker()
        self._threshold = cluster_threshold
        self._summary_max_tokens = summary_max_tokens

    def chunk(self, document: ParsedDocument) -> list[Chunk]:
        leaves = self._base.chunk(document)
        if not leaves:
            return []
        if len(leaves) == 1:
            return [self._as_leaf(leaves[0], parent_chunk_id=None)]

        vectors = self._embedding.embed([leaf.text for leaf in leaves])
        clusters = _cluster(vectors, self._threshold)

        out: list[Chunk] = []
        for cluster_idx, member_indices in enumerate(clusters):
            members = [leaves[i] for i in member_indices]
            parent = self._build_parent(document, members, cluster_idx)
            out.append(parent)
            out.extend(self._as_leaf(leaf, parent_chunk_id=parent.chunk_id) for leaf in members)
        return out

    def _build_parent(
        self, document: ParsedDocument, members: list[Chunk], cluster_idx: int
    ) -> Chunk:
        summary = self._summarize([m.text for m in members])
        section_path = _common_prefix([m.section_path for m in members]) or members[0].section_path
        page_span = members[0].page_span
        element_ids: list[str] = []
        for m in members:
            page_span = page_span.merge(m.page_span)
            element_ids.extend(m.element_ids)
        chunk_id = stable_id(
            document.document_id, self.strategy, self.version, "parent", str(cluster_idx)
        )
        return Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            source_id=document.source_id,
            text=summary,
            page_span=page_span,
            section_path=section_path,
            element_ids=tuple(dict.fromkeys(element_ids)),  # de-dup, keep order
            char_span=(0, len(summary)),  # synthesized text → span within itself
            token_count=count_tokens(summary),
            chunk_strategy=self.strategy,
            chunk_version=self.version,
            parent_chunk_id=None,
        )

    def _as_leaf(self, leaf: Chunk, *, parent_chunk_id: str | None) -> Chunk:
        new_id = stable_id(leaf.chunk_id, self.strategy, "leaf")
        return leaf.model_copy(
            update={
                "chunk_id": new_id,
                "chunk_strategy": self.strategy,
                "chunk_version": self.version,
                "parent_chunk_id": parent_chunk_id,
            }
        )

    def _summarize(self, texts: list[str]) -> str:
        firsts: list[str] = []
        for text in texts:
            sentences = split_sentences_with_offsets(text)
            firsts.append(sentences[0][0] if sentences else text.strip())
        summary = " ".join(firsts)
        words = summary.split()
        if len(words) > self._summary_max_tokens:
            summary = " ".join(words[: self._summary_max_tokens])
        return summary


def _cluster(vectors: list[DenseVector], threshold: float) -> list[list[int]]:
    """Greedy single-pass agglomeration: assign each vector to the nearest cluster centroid
    above ``threshold``, else start a new cluster. Deterministic in input order."""
    clusters: list[list[int]] = []
    centroids: list[DenseVector] = []
    for idx, vec in enumerate(vectors):
        best_cluster = -1
        best_sim = threshold
        for c_idx, centroid in enumerate(centroids):
            sim = _cosine(vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_cluster = c_idx
        if best_cluster == -1:
            clusters.append([idx])
            centroids.append(list(vec))
        else:
            members = clusters[best_cluster]
            members.append(idx)
            centroids[best_cluster] = _running_mean(centroids[best_cluster], vec, len(members))
    return clusters


def _running_mean(centroid: DenseVector, vec: DenseVector, n: int) -> DenseVector:
    return [c + (v - c) / n for c, v in zip(centroid, vec, strict=True)]


def _common_prefix(paths: list[tuple[str, ...]]) -> tuple[str, ...]:
    if not paths:
        return ()
    prefix = list(paths[0])
    for path in paths[1:]:
        i = 0
        while i < len(prefix) and i < len(path) and prefix[i] == path[i]:
            i += 1
        prefix = prefix[:i]
    return tuple(prefix)


def _cosine(a: DenseVector, b: DenseVector) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
