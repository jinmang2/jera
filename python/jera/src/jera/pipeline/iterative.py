"""IterativeRetrievalPipeline — sequential multi-hop retrieval (IRCoT / Search-R1 pattern).

Implements an iterative think→search→stop loop where each retrieval round's results
condition the next query.  This is distinct from:

* **CRAG** (``corrective.py``) — single corrective *re-query* triggered by a quality
  grade; retrieval is not sequential and does not accumulate across rounds.
* **Decomposition** (``decompositional.py``) — *parallel* sub-questions derived
  up-front from the original query; all sub-questions are known before any retrieval.

Here retrieval is *sequential*: the output of round N is inspected by the
``FollowupController`` before round N+1's query is formed.  The loop terminates when
the controller returns ``None`` or ``max_hops`` rounds have completed.

References
----------
* Search-R1 (Jin et al., 2025): arXiv:2503.09516
* A-RAG hierarchical retrieval (Li et al., 2026): arXiv:2602.03442
* IRCoT (Trivedi et al., 2022): https://arxiv.org/abs/2212.10509
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jera.domain.answer import Answer
from jera.domain.chunk import Chunk
from jera.domain.retrieval import FusionMethod, Query, RetrievalMode
from jera.pipeline.query import QueryPipeline
from jera.ports.followup_controller import FollowupController
from jera.ports.generator import GeneratorLLM


@dataclass(frozen=True)
class IterativeResult:
    """Result of one ``IterativeRetrievalPipeline.answer`` call.

    Attributes
    ----------
    answer:
        The ``Answer`` domain object produced by the generator from *all*
        accumulated contexts.
    queries:
        Ordered list of retrieval queries issued, starting with the original query
        and followed by each follow-up query returned by the controller.  Length
        equals ``rounds``.
    contexts:
        Accumulated unique ``Chunk`` objects from all retrieval rounds, in
        accumulation order (first-occurrence wins on dedup by ``chunk_id``).
        These are the exact chunks handed to the generator.
    rounds:
        Number of retrieval rounds completed (>= 1).
    """

    answer: Answer
    queries: list[str]
    contexts: list[Chunk]
    rounds: int = field(default=1)


class IterativeRetrievalPipeline:
    """Wraps a :class:`~jera.pipeline.query.QueryPipeline` with a sequential retrieval loop.

    The loop runs as follows::

        current_query = original_query
        for round_index in range(max_hops):
            result = pipeline.retrieve(Query(current_query, ...))
            accumulate unique chunks (dedup by chunk_id, first-occurrence wins)
            next_q = controller.next_query(original_query, accumulated, round_index)
            if next_q is None:
                break          # controller says STOP
            current_query = next_q

        answer = generator.generate(original_query, accumulated_contexts)

    The generator is called *once* at the end with the full accumulated context so
    it can synthesise across bridge entities found in different rounds.

    Parameters
    ----------
    pipeline:
        The base ``QueryPipeline`` used for all retrieval calls.
    controller:
        A ``FollowupController`` that decides the next query (or None to stop).
    generator:
        A ``GeneratorLLM`` that produces the final answer.
    max_hops:
        Hard cap on the number of retrieval rounds.  The controller's own stop
        condition is the *primary* termination signal; ``max_hops`` is a safety
        net that prevents runaway loops regardless of controller behaviour.
        Must be >= 1.  Default 3.
    """

    def __init__(
        self,
        pipeline: QueryPipeline,
        controller: FollowupController,
        generator: GeneratorLLM,
        *,
        max_hops: int = 3,
    ) -> None:
        if max_hops < 1:
            raise ValueError(f"max_hops must be >= 1, got {max_hops}")
        self._pipeline = pipeline
        self._controller = controller
        self._generator = generator
        self._max_hops = max_hops

    def answer(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        fusion: FusionMethod = FusionMethod.RRF,
    ) -> IterativeResult:
        """Run the iterative retrieval loop and generate an answer.

        Parameters
        ----------
        query_text:
            The user's original question.  Normalised via
            ``QueryPipeline.analyze`` before the first retrieval call.
        top_k:
            Number of chunks to retrieve per round.
        mode:
            Retrieval mode (dense / sparse / hybrid) used for every round.
        fusion:
            Fusion method forwarded to each round's ``retrieve`` call.

        Returns
        -------
        IterativeResult
            Carries the generated answer, all issued queries, accumulated
            contexts, and the number of rounds completed.
        """
        original_query = QueryPipeline.analyze(query_text)

        seen_ids: set[str] = set()
        accumulated: list[Chunk] = []
        queries_issued: list[str] = []
        current_query = original_query

        for round_index in range(self._max_hops):
            queries_issued.append(current_query)

            q = Query(text=current_query, top_k=top_k, mode=mode, fusion=fusion)
            result = self._pipeline.retrieve(q)

            # Accumulate unique chunks; dedup by chunk_id, first-occurrence wins.
            # Skip zero-score results: a BM25/dense score of 0.0 means the chunk
            # has no lexical or semantic overlap with this round's query — including
            # it would only add noise to the controller's bridge-term analysis.
            for scored in result.results:
                if (
                    scored.chunk is not None
                    and scored.chunk_id not in seen_ids
                    and scored.score > 0.0
                ):
                    seen_ids.add(scored.chunk_id)
                    accumulated.append(scored.chunk)

            # Ask the controller whether to continue.
            next_q = self._controller.next_query(original_query, accumulated, round_index)
            if next_q is None:
                # STOP signal from the controller.
                break
            # Safety: if we are about to exceed max_hops, stop before issuing another query.
            if round_index + 1 >= self._max_hops:
                break
            current_query = next_q

        rounds_completed = len(queries_issued)
        # Route generation through the pipeline's shared tail so configured context processors
        # and the citation invariant apply here too (not only on the standard path). The
        # injected generator is reused via the `generator` override.
        answered = self._pipeline.generate_from_contexts(
            original_query, accumulated, generator=self._generator
        )
        return IterativeResult(
            answer=answered.answer,
            queries=queries_issued,
            contexts=answered.contexts,
            rounds=rounds_completed,
        )
