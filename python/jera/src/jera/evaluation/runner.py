"""EvalRunner: runs an EvalDataset through a QueryPipeline and scores each retrieval mode.

Mechanism-level by design: with deterministic local providers this measures retrieval/fusion
behavior, not semantic model quality. Under the `local`/`prod` profiles the same harness
measures real model quality without code changes.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.retrieval import FusionMethod, Query, RetrievalMode
from jera.evaluation.report import CaseResult, EvalReport, ModeReport
from jera.evaluation_contracts.dataset import EvalCase, EvalDataset
from jera.evaluation_contracts.metrics import mrr, ndcg_at_k, recall_at_k
from jera.pipeline.query import QueryPipeline


class EvalRunner:
    def __init__(self, query_pipeline: QueryPipeline) -> None:
        self._query = query_pipeline

    def run(
        self,
        dataset: EvalDataset,
        *,
        k: int = 5,
        modes: Sequence[RetrievalMode] = (
            RetrievalMode.DENSE,
            RetrievalMode.SPARSE,
            RetrievalMode.HYBRID,
        ),
        fusion: FusionMethod = FusionMethod.RRF,
    ) -> EvalReport:
        reports: dict[str, ModeReport] = {}
        for mode in modes:
            cases = [self._score_case(case, mode, k, fusion) for case in dataset.cases]
            reports[mode.value] = ModeReport.from_cases(mode.value, k, cases)
        return EvalReport(dataset=dataset.name, k=k, modes=reports)

    def _score_case(
        self, case: EvalCase, mode: RetrievalMode, k: int, fusion: FusionMethod
    ) -> CaseResult:
        result = self._query.retrieve(Query(text=case.query, top_k=k, mode=mode, fusion=fusion))
        ranked = [r.chunk_id for r in result.results]
        return CaseResult(
            case_id=case.case_id,
            ranked_ids=ranked,
            recall_at_k=recall_at_k(ranked, case.gold, k),
            mrr=mrr(ranked, case.gold),
            ndcg_at_k=ndcg_at_k(ranked, case.gold, k),
        )
