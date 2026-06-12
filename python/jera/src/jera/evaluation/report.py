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
