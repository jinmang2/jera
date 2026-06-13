"""QueryPipeline: analyze → dense/sparse → fuse → attach chunks → rerank → cite → generate.

Exposes the stages separately (``retrieve`` for dense/sparse/hybrid, ``rerank``) so the
retrieval gate can exercise each, plus ``answer`` for the full E2E path.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from jera.adapters.vector_store.fusion import reciprocal_rank_fusion
from jera.config.pricing import ModelPricing, pricing_for
from jera.domain.answer import Answer
from jera.domain.chunk import Chunk
from jera.domain.retrieval import (
    FusionMethod,
    Query,
    RetrievalMode,
    RetrievalResult,
    ScoredChunk,
)
from jera.ports.context_processor import ContextProcessor
from jera.ports.embedding import EmbeddingProvider
from jera.ports.generator import GeneratorLLM
from jera.ports.metadata_store import MetadataStore
from jera.ports.query_transformer import QueryTransformer
from jera.ports.reranker import Reranker
from jera.ports.sparse import SparseVectorProvider
from jera.ports.vector_store import VectorStore


class QueryStats(BaseModel, frozen=True):
    """Per-call observability: stage timings, model ids, and a rough cost estimate.

    ``timings_ms`` keys: ``"retrieve"``, ``"rerank"``, ``"generate"``, ``"total"`` (wall-clock ms).
    ``estimated_cost_usd`` is a documented estimate based on character-count token approximation
    (chars / 4) and published list prices from ``jera.config.pricing``; treat it as a rough guide,
    not a billing figure.
    ``model_ids`` maps ``"embedding"``, ``"reranker"``, ``"generator"`` to the adapter's model id.
    """

    timings_ms: dict[str, float]
    estimated_cost_usd: float
    model_ids: dict[str, str]


def estimate_query_cost(
    *,
    query_text: str,
    context_texts: list[str],
    answer_text: str,
    embedding_pricing: ModelPricing | None,
    generator_pricing: ModelPricing | None,
) -> float:
    """Estimate the USD cost of one query call using a chars/4 token approximation.

    Token approximation: 1 token ≈ 4 characters (standard heuristic).
    - Embedding: query tokens * input_per_mtok / 1e6
    - Generator: (query + context) tokens * input_per_mtok / 1e6
                 + answer tokens * output_per_mtok / 1e6
    Free-local or unpriced models contribute 0.0.

    Returns the total estimated cost in USD as a float.
    """
    cost = 0.0

    # Embedding cost: encode the query text
    if embedding_pricing is not None and not embedding_pricing.free_local:
        query_tokens = len(query_text) / 4.0
        cost += query_tokens / 1e6 * embedding_pricing.input_per_mtok

    # Generator cost: prompt = query + all context texts; completion = answer
    if generator_pricing is not None and not generator_pricing.free_local:
        prompt_chars = len(query_text) + sum(len(t) for t in context_texts)
        prompt_tokens = prompt_chars / 4.0
        completion_tokens = len(answer_text) / 4.0
        cost += prompt_tokens / 1e6 * generator_pricing.input_per_mtok
        cost += completion_tokens / 1e6 * generator_pricing.output_per_mtok

    return cost


@dataclass(frozen=True)
class AnsweredQuery:
    """The full result of the answer path: the answer plus the evidence it was built from.

    ``retrieved_ids`` is the retrieval ranking (pre-rerank, used for context_precision);
    ``contexts`` are the reranked chunks actually handed to the generator (used for faithfulness).
    ``stats`` carries per-stage timings, model ids, and a rough cost estimate (None on old paths).
    """

    answer: Answer
    contexts: list[Chunk]
    retrieved_ids: list[str]
    stats: QueryStats | None = field(default=None)


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
        query_transformer: QueryTransformer | None = None,
        context_processors: Sequence[ContextProcessor] | None = None,
    ) -> None:
        self._embedding = embedding
        self._sparse = sparse
        self._vectors = vector_store
        self._meta = metadata_store
        self._reranker = reranker
        self._generator = generator
        self._collection = collection
        # Multi-query retrieval (off unless set): expand the query into variants, retrieve each,
        # and RRF-fuse the rankings. The answer path uses it automatically when present.
        self._query_transformer = query_transformer
        # Context engineering (off unless set): processors applied in order to the reranked
        # contexts before generation (curate → compress → reorder).
        self._context_processors: tuple[ContextProcessor, ...] = tuple(context_processors or ())

    @property
    def embedding(self) -> EmbeddingProvider:
        """Read-only access to the embedding provider (e.g. for answer-relevance scoring)."""
        return self._embedding

    @staticmethod
    def analyze(text: str) -> str:
        """Pure normalization step (no port/adapter): collapse whitespace, strip."""
        return " ".join(text.split())

    def retrieve(self, query: Query) -> RetrievalResult:
        scored = self._attach_chunks(self._search_text(self.analyze(query.text), query))
        return RetrievalResult(query=query, stage=query.mode.value, results=scored)

    def retrieve_multi(self, query: Query) -> RetrievalResult:
        """Multi-query retrieval: expand into variants, retrieve each, RRF-fuse the rankings.

        Falls back to single-query ``retrieve`` when no transformer is set or only the original
        variant survives. Each variant is retrieved at ``top_k`` and the fused result truncated
        to ``top_k`` — a chunk buried by the original phrasing can win via a variant ranking.
        """
        if self._query_transformer is None:
            return self.retrieve(query)
        variants = self._query_transformer.transform(self.analyze(query.text))
        if len(variants) <= 1:
            return self.retrieve(query)

        rankings: dict[str, list[str]] = {}
        for i, variant in enumerate(variants):
            scored = self._search_text(self.analyze(variant), query)
            rankings[f"q{i}"] = [s.chunk_id for s in scored]
        fused = reciprocal_rank_fusion(rankings)[: query.top_k]
        results = self._attach_chunks(
            [ScoredChunk(chunk_id=cid, score=score) for cid, score in fused]
        )
        return RetrievalResult(query=query, stage="multi_query", results=results)

    def _search_text(self, text: str, query: Query) -> list[ScoredChunk]:
        """Single-variant dense/sparse/hybrid search (no chunk attachment)."""
        dense = (
            self._embedding.embed_query(text) if query.mode is not RetrievalMode.SPARSE else None
        )
        sparse = self._sparse.encode_query(text) if query.mode is not RetrievalMode.DENSE else None
        return self._vectors.search(
            self._collection,
            dense=dense,
            sparse=sparse,
            top_k=query.top_k,
            fusion=query.fusion,
        )

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
        return self.answer_with_contexts(
            query_text, top_k=top_k, mode=mode, fusion=fusion, rerank_top_k=rerank_top_k
        ).answer

    def answer_with_contexts(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        fusion: FusionMethod = FusionMethod.RRF,
        rerank_top_k: int | None = None,
    ) -> AnsweredQuery:
        """The full answer path, also returning the evidence (for generation evaluation).

        Populates ``AnsweredQuery.stats`` with per-stage wall-clock timings (ms), the model ids
        of the embedding/reranker/generator adapters, and a rough USD cost estimate.
        """
        t_total_start = time.perf_counter()

        query = Query(text=query_text, top_k=top_k, mode=mode, fusion=fusion)

        t0 = time.perf_counter()
        retrieved = self.retrieve_multi(query)
        t_retrieve_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_ids = [c.chunk_id for c in retrieved.results]

        model_ids = {
            "embedding": self._embedding.model_id,
            "reranker": self._reranker.model_id,
            "generator": self._generator.model_id,
        }

        if not retrieved.results:
            # Defined empty-result behavior: empty answer, no citations, no error.
            empty = Answer(query=self.analyze(query_text), text="", citations=[])
            t_total_ms = (time.perf_counter() - t_total_start) * 1000.0
            stats = QueryStats(
                timings_ms={
                    "retrieve": t_retrieve_ms,
                    "rerank": 0.0,
                    "generate": 0.0,
                    "total": t_total_ms,
                },
                estimated_cost_usd=0.0,
                model_ids=model_ids,
            )
            return AnsweredQuery(
                answer=empty, contexts=[], retrieved_ids=retrieved_ids, stats=stats
            )

        retrieved_id_set = set(retrieved_ids)

        t0 = time.perf_counter()
        reranked = self.rerank(query_text, retrieved.results, rerank_top_k or top_k)
        t_rerank_ms = (time.perf_counter() - t0) * 1000.0

        # Citation-correctness invariant: only cite chunks that were actually retrieved.
        contexts: list[Chunk] = []
        for sc in reranked:
            assert sc.chunk_id in retrieved_id_set, "rerank introduced a non-retrieved chunk"
            if sc.chunk is not None:
                contexts.append(sc.chunk)

        # Generation tail (context processors → generate → citation invariant → cost → stats) is
        # shared with the advanced pipelines via generate_from_contexts; pass the measured
        # retrieve/rerank timings so the stats stay end-to-end honest.
        return self.generate_from_contexts(
            query_text,
            contexts,
            retrieved_ids=retrieved_ids,
            upstream_timings_ms={"retrieve": t_retrieve_ms, "rerank": t_rerank_ms},
        )

    def generate_from_contexts(
        self,
        query_text: str,
        contexts: Sequence[Chunk],
        *,
        retrieved_ids: list[str] | None = None,
        upstream_timings_ms: dict[str, float] | None = None,
        generator: GeneratorLLM | None = None,
    ) -> AnsweredQuery:
        """Generate an answer from an explicit, already-retrieved+reranked context list.

        This is the shared generation tail used by ``answer_with_contexts`` *and* the advanced
        retrieval pipelines (CRAG, decomposition, iterative) that build their OWN corrected/
        accumulated context and must feed *that* to the generator — not silently re-run vanilla
        retrieval. It applies the configured context processors, enforces the citation-resolves-
        to-context invariant, and records stats. It performs **no** retrieval or reranking of its
        own (``retrieve``/``rerank`` timings default to 0 or come from ``upstream_timings_ms``).

        Args:
            query_text:   The (original) user query, used for processing + generation + cost.
            contexts:     The chunks to answer from, already ranked best-first by the caller.
            retrieved_ids: The caller's retrieval ranking ids (for context_precision). Defaults
                          to the context chunk ids when not supplied.
            upstream_timings_ms: Optional ``{"retrieve":…, "rerank":…}`` measured by the caller,
                          folded into the returned stats so end-to-end timing stays honest.
            generator:    Optional generator override. Composable wrappers (decomposition,
                          iterative) inject their own generator here so they still reuse this
                          shared tail (context processing + citation invariant + stats) instead
                          of bypassing it. Defaults to the pipeline's own generator.
        """
        gen = generator if generator is not None else self._generator
        t_total_start = time.perf_counter()
        normalized = self.analyze(query_text)

        # Context engineering (off unless configured): curate → compress → reorder. Each
        # processor preserves chunk_id (compression only trims text), so citations still resolve.
        processed: list[Chunk] = list(contexts)
        for processor in self._context_processors:
            processed = processor.process(normalized, processed)

        t0 = time.perf_counter()
        answer = gen.generate(normalized, processed)
        t_generate_ms = (time.perf_counter() - t0) * 1000.0

        valid_ids = {c.chunk_id for c in processed}
        for citation in answer.citations:
            assert citation.chunk_id in valid_ids, "citation does not resolve to a context chunk"

        t_local_ms = (time.perf_counter() - t_total_start) * 1000.0
        timings: dict[str, float] = dict(upstream_timings_ms or {})
        timings.setdefault("retrieve", 0.0)
        timings.setdefault("rerank", 0.0)
        timings["generate"] = t_generate_ms
        # total = caller's upstream retrieve/rerank + this method's wall-clock (gen + processing).
        timings["total"] = timings["retrieve"] + timings["rerank"] + t_local_ms

        estimated_cost = estimate_query_cost(
            query_text=query_text,
            context_texts=[c.text for c in processed],
            answer_text=answer.text,
            embedding_pricing=pricing_for(self._embedding.model_id),
            generator_pricing=pricing_for(gen.model_id),
        )
        stats = QueryStats(
            timings_ms=timings,
            estimated_cost_usd=estimated_cost,
            model_ids={
                "embedding": self._embedding.model_id,
                "reranker": self._reranker.model_id,
                "generator": gen.model_id,
            },
        )
        return AnsweredQuery(
            answer=answer,
            contexts=processed,
            retrieved_ids=(
                retrieved_ids if retrieved_ids is not None else [c.chunk_id for c in processed]
            ),
            stats=stats,
        )

    def _attach_chunks(self, scored: list[ScoredChunk]) -> list[ScoredChunk]:
        chunks = {c.chunk_id: c for c in self._meta.get_chunks([s.chunk_id for s in scored])}
        out: list[ScoredChunk] = []
        for s in scored:
            chunk = chunks.get(s.chunk_id)
            out.append(s.with_chunk(chunk) if chunk else s)
        return out
