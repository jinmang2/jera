"""CorrectiveQueryPipeline — Corrective RAG (CRAG) loop.

Implements the corrective retrieval-augmented generation strategy from:

    Yan et al., "Corrective Retrieval Augmented Generation",
    ICLR 2025, arXiv:2401.15884.

The core idea: grade the initial retrieval; if it is not ``CORRECT``, expand the query
via a ``QueryTransformer``, retrieve for each variant, and fuse all rankings with RRF
before handing the corrected context to the generator.

Architecture
------------
* Delegates entirely to ``QueryPipeline``'s **public** API —
  ``retrieve``, ``rerank``, ``answer_with_contexts``.  No private attributes are
  accessed.
* Uses ``RetrievalEvaluator`` (injected) to grade the initial result.
* Uses ``QueryTransformer`` (injected) to generate corrective query variants.
  The ``QueryTransformer`` protocol specifies that its output is "original first",
  so ``transform`` already includes the original query at index 0.  We retrieve for
  each returned variant (including the original) — no separate "original" slot is
  added, which would double-count it in the RRF scores.
* Fuses all per-variant rankings with ``reciprocal_rank_fusion`` (the same function
  used by ``QueryPipeline.retrieve_multi``).
* Returns a ``CorrectiveResult`` exposing ``.answer``, ``.grade``, and ``.corrected``
  for downstream observability and testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jera.adapters.vector_store.fusion import reciprocal_rank_fusion
from jera.domain.answer import Answer
from jera.domain.chunk import Chunk
from jera.domain.retrieval import (
    FusionMethod,
    Query,
    RetrievalMode,
    ScoredChunk,
)
from jera.pipeline.query import AnsweredQuery, QueryPipeline
from jera.ports.query_transformer import QueryTransformer
from jera.ports.retrieval_evaluator import RetrievalEvaluator, RetrievalGrade


@dataclass(frozen=True)
class CorrectiveResult:
    """The full result of one ``CorrectiveQueryPipeline.answer`` call.

    Attributes
    ----------
    answer:
        The ``Answer`` domain object (text + citations) from the generator.
    contexts:
        Reranked ``Chunk`` objects handed to the generator (for faithfulness eval).
    retrieved_ids:
        Chunk ids in the final retrieval ranking (pre-rerank, for context_precision).
    grade:
        The ``RetrievalGrade`` assigned to the *initial* retrieval.
    corrected:
        ``True`` when the corrective loop ran (grade was not ``CORRECT``); ``False``
        when the initial retrieval was accepted as-is.
    stats:
        Per-stage timing / cost stats from the underlying ``QueryPipeline`` call
        (``None`` when unavailable).
    """

    answer: Answer
    contexts: list[Chunk]
    retrieved_ids: list[str]
    grade: RetrievalGrade
    corrected: bool
    stats: object = field(default=None)  # QueryStats | None — kept as object to avoid re-export


class CorrectiveQueryPipeline:
    """Wraps a ``QueryPipeline`` with a retrieval-grader + corrective re-query loop.

    Parameters
    ----------
    pipeline:
        The underlying ``QueryPipeline`` providing retrieve / rerank / generate.
    evaluator:
        A ``RetrievalEvaluator`` that grades the initial retrieval.
    transformer:
        A ``QueryTransformer`` used when the grade is not ``CORRECT`` to produce
        corrective query variants.  Per the ``QueryTransformer`` protocol the returned
        list has the original query at index 0 — we retrieve for every returned variant
        without adding a separate "original" ranking so the original is never
        double-counted in the RRF scores.
    """

    def __init__(
        self,
        pipeline: QueryPipeline,
        evaluator: RetrievalEvaluator,
        transformer: QueryTransformer,
    ) -> None:
        self._pipeline = pipeline
        self._evaluator = evaluator
        self._transformer = transformer

    def answer(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        fusion: FusionMethod = FusionMethod.RRF,
        rerank_top_k: int | None = None,
    ) -> CorrectiveResult:
        """Run the CRAG loop and return a ``CorrectiveResult``.

        Steps
        -----
        (a) Retrieve with the original query via ``pipeline.retrieve``.
        (b) Grade the result with the injected ``evaluator``.
        (c) If grade is ``CORRECT``: hand the initial retrieval directly to (d).
            If grade is not ``CORRECT``: run the corrective re-query loop —
            call ``transformer.transform`` to get all variants (original first),
            retrieve for each variant, RRF-merge all per-variant rankings, take
            top_k, re-attach chunks from the initial retrieval cache.
        (d) Rerank the final candidate set.
        (e) Drive the generator via ``pipeline.answer_with_contexts`` (which owns
            the timing/cost/stats instrumentation).  The ``CorrectiveResult`` exposes
            the corrected contexts and retrieved_ids alongside the pipeline's answer.
        """
        query = Query(text=query_text, top_k=top_k, mode=mode, fusion=fusion)

        # (a) Initial retrieval.
        initial = self._pipeline.retrieve(query)

        # (b) Grade.
        grade: RetrievalGrade = self._evaluator.grade(query_text, initial)

        if grade is RetrievalGrade.CORRECT:
            # Fast path: initial retrieval is good enough — skip correction, but the answer
            # must still be generated from THESE (reranked initial) contexts, not a fresh
            # re-retrieval, so the answer and the reported contexts stay coherent.
            reranked = self._pipeline.rerank(query_text, initial.results, rerank_top_k or top_k)
            contexts = [sc.chunk for sc in reranked if sc.chunk is not None]
            answered: AnsweredQuery = self._pipeline.generate_from_contexts(
                query_text, contexts, retrieved_ids=[sc.chunk_id for sc in initial.results]
            )
            return CorrectiveResult(
                answer=answered.answer,
                contexts=answered.contexts,
                retrieved_ids=[sc.chunk_id for sc in reranked],
                grade=grade,
                corrected=False,
                stats=answered.stats,
            )

        # (c) Corrective re-query loop.
        #
        # transformer.transform returns variants with the original at index 0 (per protocol).
        # We retrieve for every variant — the original is included naturally so no separate
        # "original" slot is needed (which would double-count it in RRF).
        variants = self._transformer.transform(QueryPipeline.analyze(query_text))

        rankings: dict[str, list[str]] = {}
        # Keep a cache of all ScoredChunks seen so we can re-attach Chunk objects.
        chunk_cache: dict[str, ScoredChunk] = {sc.chunk_id: sc for sc in initial.results}

        for i, variant in enumerate(variants):
            variant_query = Query(text=variant, top_k=top_k, mode=mode, fusion=fusion)
            variant_result = self._pipeline.retrieve(variant_query)
            rankings[f"q{i}"] = [sc.chunk_id for sc in variant_result.results]
            for sc in variant_result.results:
                if sc.chunk_id not in chunk_cache or chunk_cache[sc.chunk_id].chunk is None:
                    chunk_cache[sc.chunk_id] = sc

        fused = reciprocal_rank_fusion(rankings)[:top_k]

        # Build the corrected ScoredChunk list, re-attaching chunks from cache.
        corrected_scored: list[ScoredChunk] = []
        for cid, score in fused:
            cached = chunk_cache.get(cid)
            corrected_scored.append(
                ScoredChunk(
                    chunk_id=cid,
                    score=score,
                    chunk=cached.chunk if cached is not None else None,
                )
            )

        retrieved_ids = [sc.chunk_id for sc in corrected_scored]

        # (d) Rerank the corrected candidate set.
        reranked = self._pipeline.rerank(query_text, corrected_scored, rerank_top_k or top_k)
        contexts: list[Chunk] = [sc.chunk for sc in reranked if sc.chunk is not None]

        # (e) Generate from the CORRECTED contexts. This is the whole point of CRAG: the
        # generator must answer from the corrected evidence, not a fresh vanilla retrieval.
        # (The previous implementation called answer_with_contexts, which re-ran retrieval
        # internally and silently discarded the correction — the answer ignored the lift.)
        answered = self._pipeline.generate_from_contexts(
            query_text, contexts, retrieved_ids=retrieved_ids
        )

        return CorrectiveResult(
            answer=answered.answer,
            contexts=answered.contexts,
            retrieved_ids=retrieved_ids,
            grade=grade,
            corrected=True,
            stats=answered.stats,
        )
