"""DecompositionalQueryPipeline — sequential sub-question retrieval for multi-hop RAG.

Implements the query decomposition approach from Pereira et al., "Question Decomposition for
RAG", ACL-SRW 2025 (arXiv:2507.00355), which reports +36.7% MRR@10 on multi-hop benchmarks.

Algorithm
---------
1. **Decompose** — split the original query into an ordered list of sub-questions via a
   :class:`~jera.ports.query_decomposer.QueryDecomposer`.  A non-compound query returns
   ``[query]`` and the pipeline degrades gracefully to single-step retrieval.
2. **Retrieve per hop** — for each sub-question, call
   ``QueryPipeline.retrieve(Query(sub_q, ...))`` and collect its top-``top_k`` chunks.
3. **Accumulate** — merge all per-hop chunks into a single ordered list, deduplicating by
   ``chunk_id`` (first occurrence wins, preserving hop order).  The total is capped at
   ``top_k × n_sub_questions`` to keep the generator prompt bounded.
4. **Generate once** — call ``GeneratorLLM.generate(original_query, accumulated_chunks)``
   exactly once so the answer has access to all bridge entities.

This is distinct from multi-query RRF (parallel variants fused by rank): decomposition is
*sequential* and accumulates, not fuses — every retrieved chunk reaches the generator,
regardless of per-hop rank position.
"""

from __future__ import annotations

from dataclasses import dataclass

from jera.domain.answer import Answer
from jera.domain.chunk import Chunk
from jera.domain.retrieval import FusionMethod, Query, RetrievalMode
from jera.pipeline.query import QueryPipeline
from jera.ports.generator import GeneratorLLM
from jera.ports.query_decomposer import QueryDecomposer


@dataclass(frozen=True)
class DecompositionalResult:
    """Result of a decompositional answer call.

    Attributes
    ----------
    answer:
        The generated answer for the *original* query, built from all accumulated contexts.
    sub_questions:
        The ordered sub-questions produced by the decomposer (``[original]`` for non-compound
        queries).
    contexts:
        Accumulated unique chunks from all retrieval hops, in hop order (first-occurrence wins
        on dedup).  These are the exact chunks handed to the generator.
    """

    answer: Answer
    sub_questions: list[str]
    contexts: list[Chunk]


class DecompositionalQueryPipeline:
    """Wraps a :class:`~jera.pipeline.query.QueryPipeline` with sequential sub-question retrieval.

    Parameters
    ----------
    pipeline:
        The base retrieval pipeline (provides ``retrieve`` and ``rerank``).
    decomposer:
        Splits the user query into ordered sub-questions.
    generator:
        Produces the final answer from the original query + accumulated contexts.
    """

    def __init__(
        self,
        pipeline: QueryPipeline,
        decomposer: QueryDecomposer,
        generator: GeneratorLLM,
    ) -> None:
        self._pipeline = pipeline
        self._decomposer = decomposer
        self._generator = generator

    def answer(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        fusion: FusionMethod = FusionMethod.RRF,
    ) -> DecompositionalResult:
        """Decompose → retrieve per sub-question → accumulate → generate once.

        Parameters
        ----------
        query_text:
            The original user question (may be multi-hop).
        top_k:
            Number of chunks to retrieve per sub-question.
        mode:
            Retrieval mode (dense / sparse / hybrid) forwarded to each sub-question query.
        fusion:
            Fusion method forwarded to each sub-question query.

        Returns
        -------
        DecompositionalResult
            Carries the generated answer, the sub-questions, and the accumulated contexts.
        """
        normalized = QueryPipeline.analyze(query_text)
        sub_questions = self._decomposer.decompose(normalized)

        # Accumulate unique chunks across hops; dedup by chunk_id, first-occurrence wins.
        seen: set[str] = set()
        accumulated: list[Chunk] = []

        for sub_q in sub_questions:
            query = Query(text=sub_q, top_k=top_k, mode=mode, fusion=fusion)
            result = self._pipeline.retrieve(query)
            for scored in result.results:
                if scored.chunk is not None and scored.chunk_id not in seen:
                    seen.add(scored.chunk_id)
                    accumulated.append(scored.chunk)

        # Cap total contexts: top_k * n_sub_questions (already bounded by accumulation above,
        # but apply an explicit ceiling in case top_k is large and sub_questions is long).
        max_contexts = top_k * len(sub_questions)
        contexts: list[Chunk] = accumulated[:max_contexts]

        answer = self._generator.generate(normalized, contexts)
        return DecompositionalResult(
            answer=answer,
            sub_questions=sub_questions,
            contexts=contexts,
        )
