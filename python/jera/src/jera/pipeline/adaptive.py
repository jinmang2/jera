"""AdaptiveQueryPipeline — Adaptive-RAG routing wrapper around QueryPipeline.

Routes each query to one of three execution paths based on its complexity:

* NO_RETRIEVAL  — the generator is called with an *empty* context list; no vector-store
                  search is performed.  This is the real compute saving: zero embedding
                  calls, zero BM25 scoring, zero vector-store round-trips.

* SINGLE_STEP   — delegates to ``QueryPipeline.answer_with_contexts`` (standard path).

* MULTI_STEP    — also delegates to ``answer_with_contexts``, which internally calls
                  ``retrieve_multi``.  When a ``QueryTransformer`` is attached to the
                  wrapped pipeline, ``retrieve_multi`` expands the query into variants
                  and RRF-fuses their rankings, giving genuine multi-hop coverage.  If no
                  transformer is set, ``retrieve_multi`` falls back to ``retrieve``
                  (single-variant) — this is the documented graceful-degradation behaviour
                  of the underlying pipeline; the adapter documents it here but does not
                  try to override it.

Reference: Jeong et al., "Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language
Models through Question Complexity" (NAACL 2024).
"""

from __future__ import annotations

from dataclasses import dataclass

from jera.domain.answer import Answer
from jera.domain.retrieval import FusionMethod, RetrievalMode
from jera.pipeline.query import AnsweredQuery, QueryPipeline
from jera.ports.query_router import QueryComplexity, QueryRouter


@dataclass(frozen=True)
class AdaptiveAnsweredQuery:
    """Result of an adaptive answer call: the answer plus the routing decision.

    ``complexity`` records which tier the router selected so callers can log
    routing statistics, run A/B evaluations, or display provenance in a UI.
    """

    answer: Answer
    answered_query: AnsweredQuery
    complexity: QueryComplexity


class AdaptiveQueryPipeline:
    """Wraps a ``QueryPipeline`` with an Adaptive-RAG complexity router.

    Parameters
    ----------
    pipeline:
        A fully configured ``QueryPipeline``.  The adaptive wrapper never mutates
        it — all state (embedding, vector store, transformer, …) lives in the
        wrapped pipeline.
    router:
        Any ``QueryRouter`` implementation.  The ``HeuristicQueryRouter`` is the
        default offline/CI-real choice; a trained TF-IDF+SVM router can be
        swapped in without touching this class.
    """

    def __init__(self, pipeline: QueryPipeline, router: QueryRouter) -> None:
        self._pipeline = pipeline
        self._router = router

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        fusion: FusionMethod = FusionMethod.RRF,
        rerank_top_k: int | None = None,
    ) -> AdaptiveAnsweredQuery:
        """Route *query_text*, execute the appropriate retrieval path, and return
        an ``AdaptiveAnsweredQuery`` that exposes ``.answer`` and ``.complexity``.

        Routing contract
        ----------------
        NO_RETRIEVAL  → generator receives an empty context list; ``VectorStore.search``
                        is never called (verifiable via a counting proxy — see tests).
        SINGLE_STEP   → ``QueryPipeline.answer_with_contexts`` (single retrieve pass).
        MULTI_STEP    → ``QueryPipeline.answer_with_contexts`` (retrieve_multi path);
                        if no ``QueryTransformer`` is configured on the wrapped pipeline,
                        this degrades to a single retrieve pass (documented upstream).
        """
        complexity = self._router.route(query_text)

        if complexity is QueryComplexity.NO_RETRIEVAL:
            answered = self._answer_no_retrieval(query_text)
        else:
            answered = self._pipeline.answer_with_contexts(
                query_text,
                top_k=top_k,
                mode=mode,
                fusion=fusion,
                rerank_top_k=rerank_top_k,
            )

        return AdaptiveAnsweredQuery(
            answer=answered.answer,
            answered_query=answered,
            complexity=complexity,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _answer_no_retrieval(self, query_text: str) -> AnsweredQuery:
        """Generate an answer with zero retrieval — no vector-store search performed.

        Calls the generator directly with an empty context list, exactly mirroring
        what ``QueryPipeline.answer_with_contexts`` does when retrieval returns no
        results.  The ``AnsweredQuery`` is constructed manually so the stats bucket
        records zero retrieve/rerank time and no cost for embedding/reranker.
        """
        import time

        from jera.pipeline.query import QueryStats

        normalized = self._pipeline.analyze(query_text)

        t_start = time.perf_counter()
        answer: Answer = self._pipeline._generator.generate(normalized, [])  # noqa: SLF001
        t_generate_ms = (time.perf_counter() - t_start) * 1000.0

        stats = QueryStats(
            timings_ms={
                "retrieve": 0.0,
                "rerank": 0.0,
                "generate": t_generate_ms,
                "total": t_generate_ms,
            },
            estimated_cost_usd=0.0,
            model_ids={
                "embedding": self._pipeline.embedding.model_id,
                "reranker": self._pipeline._reranker.model_id,  # noqa: SLF001
                "generator": self._pipeline._generator.model_id,  # noqa: SLF001
            },
        )
        return AnsweredQuery(answer=answer, contexts=[], retrieved_ids=[], stats=stats)
