"""Evaluation report models: per-case results aggregated per retrieval mode."""

from __future__ import annotations

from statistics import fmean

from pydantic import BaseModel


class CaseResult(BaseModel):
    """Metrics for a single eval case under one retrieval mode."""

    model_config = {"frozen": True}

    case_id: str
    ranked_ids: list[str]
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


class ModeReport(BaseModel):
    """Aggregate metrics for one retrieval mode across all cases."""

    model_config = {"frozen": True}

    mode: str
    k: int
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    cases: list[CaseResult]

    @classmethod
    def from_cases(cls, mode: str, k: int, cases: list[CaseResult]) -> ModeReport:
        if not cases:
            return cls(
                mode=mode, k=k, mean_recall_at_k=0.0, mean_mrr=0.0, mean_ndcg_at_k=0.0, cases=[]
            )
        return cls(
            mode=mode,
            k=k,
            mean_recall_at_k=fmean(c.recall_at_k for c in cases),
            mean_mrr=fmean(c.mrr for c in cases),
            mean_ndcg_at_k=fmean(c.ndcg_at_k for c in cases),
            cases=cases,
        )


def _mean_opt(values: list[float]) -> float | None:
    return fmean(values) if values else None


class GenerationCaseResult(BaseModel):
    """RAGAS-lite generation metrics for one eval case (answer-path)."""

    model_config = {"frozen": True}

    case_id: str
    faithfulness: float
    context_precision: float
    # only present when the case carries a reference_answer
    answer_correctness: float | None = None
    answer_relevance: float | None = None


class GenerationReport(BaseModel):
    """Aggregate generation-quality metrics across all cases for one retrieval mode."""

    model_config = {"frozen": True}

    dataset: str
    mode: str
    k: int
    mean_faithfulness: float
    mean_context_precision: float
    mean_answer_correctness: float | None
    mean_answer_relevance: float | None
    cases: list[GenerationCaseResult]

    @classmethod
    def from_cases(
        cls, dataset: str, mode: str, k: int, cases: list[GenerationCaseResult]
    ) -> GenerationReport:
        return cls(
            dataset=dataset,
            mode=mode,
            k=k,
            mean_faithfulness=fmean(c.faithfulness for c in cases) if cases else 0.0,
            mean_context_precision=fmean(c.context_precision for c in cases) if cases else 0.0,
            mean_answer_correctness=_mean_opt(
                [c.answer_correctness for c in cases if c.answer_correctness is not None]
            ),
            mean_answer_relevance=_mean_opt(
                [c.answer_relevance for c in cases if c.answer_relevance is not None]
            ),
            cases=cases,
        )

    def summary_table(self) -> str:
        def fmt(x: float | None) -> str:
            return f"{x:>7.3f}" if x is not None else "    n/a"

        return "\n".join(
            [
                "metric             value",
                f"faithfulness     {fmt(self.mean_faithfulness)}",
                f"context_prec     {fmt(self.mean_context_precision)}",
                f"answer_correct   {fmt(self.mean_answer_correctness)}",
                f"answer_relevance {fmt(self.mean_answer_relevance)}",
            ]
        )


class EvalReport(BaseModel):
    """A full evaluation run: one ModeReport per retrieval mode evaluated."""

    dataset: str
    k: int
    modes: dict[str, ModeReport]

    def best_mode_by_recall(self) -> str:
        return max(self.modes.values(), key=lambda m: m.mean_recall_at_k).mode

    def summary_table(self) -> str:
        rows = ["mode      recall@k   mrr      ndcg@k"]
        for name in sorted(self.modes):
            m = self.modes[name]
            rows.append(
                f"{m.mode:<8}  {m.mean_recall_at_k:>7.3f}  "
                f"{m.mean_mrr:>7.3f}  {m.mean_ndcg_at_k:>7.3f}"
            )
        return "\n".join(rows)
