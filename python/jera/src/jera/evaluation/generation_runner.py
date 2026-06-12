"""GenerationEvalRunner: scores the answer-path with RAGAS-lite generation metrics.

Complements `EvalRunner` (which grades retrieval). For each case it runs the full
``answer_with_contexts`` path and computes:

- ``faithfulness``      — answer grounded in the retrieved context chunks (always)
- ``context_precision`` — relevant chunks ranked near the top (always; uses ``gold``)
- ``answer_correctness``— token-F1 vs ``case.reference_answer`` (only if present)
- ``answer_relevance``  — cosine(answer, query) over the pipeline's embedding (only if a
                          reference answer is present, i.e. the case opted into generation grading)

Deterministic under the test profile (extractive generator + hash embedding); the same runner
measures real LLM answers under local/prod without code changes.
"""

from __future__ import annotations

from jera.domain.retrieval import FusionMethod, RetrievalMode
from jera.evaluation.report import GenerationCaseResult, GenerationReport
from jera.evaluation_contracts.dataset import EvalCase, EvalDataset
from jera.evaluation_contracts.generation_metrics import (
    answer_correctness,
    answer_relevance,
    context_precision,
    faithfulness,
)
from jera.pipeline.query import QueryPipeline


class GenerationEvalRunner:
    def __init__(self, query_pipeline: QueryPipeline) -> None:
        self._query = query_pipeline

    def run(
        self,
        dataset: EvalDataset,
        *,
        k: int = 5,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        fusion: FusionMethod = FusionMethod.RRF,
    ) -> GenerationReport:
        cases = [self._score_case(case, k, mode, fusion) for case in dataset.cases]
        return GenerationReport.from_cases(dataset.name, mode.value, k, cases)

    def _score_case(
        self, case: EvalCase, k: int, mode: RetrievalMode, fusion: FusionMethod
    ) -> GenerationCaseResult:
        bundle = self._query.answer_with_contexts(case.query, top_k=k, mode=mode, fusion=fusion)
        context_texts = [c.text for c in bundle.contexts]

        correctness: float | None = None
        relevance: float | None = None
        if case.reference_answer is not None:
            correctness = answer_correctness(bundle.answer.text, case.reference_answer)
            answer_vec = self._query.embedding.embed_query(bundle.answer.text)
            query_vec = self._query.embedding.embed_query(case.query)
            relevance = answer_relevance(answer_vec, query_vec)

        return GenerationCaseResult(
            case_id=case.case_id,
            faithfulness=faithfulness(bundle.answer.text, context_texts),
            context_precision=context_precision(bundle.retrieved_ids, case.gold, k),
            answer_correctness=correctness,
            answer_relevance=relevance,
        )
