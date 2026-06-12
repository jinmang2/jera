"""QueryPipeline: analyze → dense/sparse → fuse → attach chunks → rerank → cite → generate.

Exposes the stages separately (``retrieve`` for dense/sparse/hybrid, ``rerank``) so the
retrieval gate can exercise each, plus ``answer`` for the full E2E path.
"""

from __future__ import annotations

from jera.domain.answer import Answer
from jera.domain.chunk import Chunk
from jera.domain.retrieval import (
    FusionMethod,
    Query,
    RetrievalMode,
    RetrievalResult,
    ScoredChunk,
)
from jera.ports.embedding import EmbeddingProvider
from jera.ports.generator import GeneratorLLM
from jera.ports.metadata_store import MetadataStore
from jera.ports.reranker import Reranker
from jera.ports.sparse import SparseVectorProvider
from jera.ports.vector_store import VectorStore


class QueryPipeline:
    def __init__(
        self,
        *,
        embedding: EmbeddingProvider,
        sparse: SparseVectorProvider,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        reranker: Reranker,
        generator: GeneratorLLM,
        collection: str,
    ) -> None:
        self._embedding = embedding
        self._sparse = sparse
        self._vectors = vector_store
        self._meta = metadata_store
        self._reranker = reranker
        self._generator = generator
        self._collection = collection

    @staticmethod
    def analyze(text: str) -> str:
        """Pure normalization step (no port/adapter): collapse whitespace, strip."""
        return " ".join(text.split())

    def retrieve(self, query: Query) -> RetrievalResult:
        text = self.analyze(query.text)
        dense = (
            self._embedding.embed_query(text) if query.mode is not RetrievalMode.SPARSE else None
        )
        sparse = self._sparse.encode_query(text) if query.mode is not RetrievalMode.DENSE else None
        scored = self._vectors.search(
            self._collection,
            dense=dense,
            sparse=sparse,
            top_k=query.top_k,
            fusion=query.fusion,
        )
        scored = self._attach_chunks(scored)
        return RetrievalResult(query=query, stage=query.mode.value, results=scored)

    def rerank(self, query_text: str, scored: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return self._reranker.rerank(self.analyze(query_text), scored, top_k)

    def answer(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        fusion: FusionMethod = FusionMethod.RRF,
        rerank_top_k: int | None = None,
    ) -> Answer:
        query = Query(text=query_text, top_k=top_k, mode=mode, fusion=fusion)
        retrieved = self.retrieve(query)
        if not retrieved.results:
            # Defined empty-result behavior: empty answer, no citations, no error.
            return Answer(query=self.analyze(query_text), text="", citations=[])

        retrieved_ids = {c.chunk_id for c in retrieved.results}
        reranked = self.rerank(query_text, retrieved.results, rerank_top_k or top_k)

        # Citation-correctness invariant: only cite chunks that were actually retrieved.
        contexts: list[Chunk] = []
        for sc in reranked:
            assert sc.chunk_id in retrieved_ids, "rerank introduced a non-retrieved chunk"
            if sc.chunk is not None:
                contexts.append(sc.chunk)

        answer = self._generator.generate(self.analyze(query_text), contexts)
        valid_ids = {c.chunk_id for c in contexts}
        for citation in answer.citations:
            assert citation.chunk_id in valid_ids, "citation does not resolve to a context chunk"
        return answer

    def _attach_chunks(self, scored: list[ScoredChunk]) -> list[ScoredChunk]:
        chunks = {c.chunk_id: c for c in self._meta.get_chunks([s.chunk_id for s in scored])}
        out: list[ScoredChunk] = []
        for s in scored:
            chunk = chunks.get(s.chunk_id)
            out.append(s.with_chunk(chunk) if chunk else s)
        return out
