"""AblationRunner — compare arbitrary named RAG configurations on one corpus + question set.

The matrix (`run_matrix`) compares chunk-strategy × retrieval-mode. This is the general form:
each configuration is a *named* ``Settings`` (baseline, contextual retrieval, context processing,
proposition chunking, multi-query, listwise rerank, …). Every config is built fresh, the corpus is
re-ingested, the gold dataset is **rebuilt per config by substring labeling** (chunk ids differ
across chunk strategies), and each config is scored on retrieval (recall/MRR/nDCG), RAGAS-lite
generation, and claim-level (RAGChecker) metrics. The result is a single comparison table so the
question "which technique actually helps, and on what?" is answerable, not assumed.

Fully offline-deterministic under the test profile; the same harness measures real models without
code changes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import fmean

from pydantic import BaseModel

from jera.config.registry import build_system
from jera.config.settings import Settings
from jera.domain.document import MediaType
from jera.domain.retrieval import FusionMethod, RetrievalMode
from jera.evaluation.dataset_builder import CaseSpec, build_gold_dataset
from jera.evaluation.generation_runner import GenerationEvalRunner
from jera.evaluation.runner import EvalRunner
from jera.evaluation_contracts.dataset import EvalDataset
from jera.evaluation_contracts.ragchecker_metrics import (
    abstention_score,
    claim_precision,
    noise_sensitivity,
)


@dataclass(frozen=True)
class AblationConfig:
    """A named configuration to evaluate."""

    name: str
    settings: Settings


@dataclass(frozen=True)
class AblationCase:
    """A question + its gold substring (for labeling) + an optional reference answer."""

    spec: CaseSpec
    reference_answer: str | None = None


class AblationEntry(BaseModel):
    """Aggregate metrics for one configuration."""

    model_config = {"frozen": True}

    name: str
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    mean_faithfulness: float
    mean_context_precision: float
    mean_claim_precision: float
    mean_noise_sensitivity: float
    mean_abstention: float


class AblationReport(BaseModel):
    """A full ablation run: one entry per configuration."""

    model_config = {"frozen": True}

    k: int
    mode: str
    entries: list[AblationEntry]

    def best_by(self, metric: str) -> str:
        """Name of the config with the highest value of ``metric`` (ties → first)."""
        return max(self.entries, key=lambda e: getattr(e, metric)).name

    def comparison_table(self) -> str:
        header = f"{'config':<22} recall@k    mrr   ndcg  faithf  ctx_prec  claim_p  noise  abstain"
        rows = [header]
        for e in self.entries:
            rows.append(
                f"{e.name:<22} {e.mean_recall_at_k:>7.3f}  {e.mean_mrr:>5.3f} "
                f"{e.mean_ndcg_at_k:>5.3f} {e.mean_faithfulness:>6.3f} "
                f"{e.mean_context_precision:>8.3f} {e.mean_claim_precision:>7.3f} "
                f"{e.mean_noise_sensitivity:>5.3f} {e.mean_abstention:>7.3f}"
            )
        return "\n".join(rows)


@dataclass
class AblationRunner:
    corpus: dict[str, tuple[MediaType, str]]
    cases: Sequence[AblationCase]
    k: int = 5
    mode: RetrievalMode = RetrievalMode.HYBRID
    fusion: FusionMethod = FusionMethod.RRF
    _refs: dict[str, str | None] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self._refs = {c.spec.case_id: c.reference_answer for c in self.cases}

    def run(self, configs: Sequence[AblationConfig]) -> AblationReport:
        entries = [self._score_config(cfg) for cfg in configs]
        return AblationReport(k=self.k, mode=self.mode.value, entries=entries)

    def _score_config(self, config: AblationConfig) -> AblationEntry:
        system = build_system(config.settings)
        labeled = build_gold_dataset(
            system,
            name=config.name,
            documents=self.corpus,
            cases=[c.spec for c in self.cases],
        )
        with_refs = EvalDataset(
            name=config.name,
            cases=[
                case.model_copy(update={"reference_answer": self._refs.get(case.case_id)})
                for case in labeled.cases
            ],
        )

        retrieval = EvalRunner(system.query).run(labeled, k=self.k, modes=[self.mode])
        mode_report = retrieval.modes[self.mode.value]
        generation = GenerationEvalRunner(system.query).run(
            with_refs, k=self.k, mode=self.mode, fusion=self.fusion
        )
        claim_p, noise, abstain = self._claim_metrics(system, with_refs)

        return AblationEntry(
            name=config.name,
            mean_recall_at_k=mode_report.mean_recall_at_k,
            mean_mrr=mode_report.mean_mrr,
            mean_ndcg_at_k=mode_report.mean_ndcg_at_k,
            mean_faithfulness=generation.mean_faithfulness,
            mean_context_precision=generation.mean_context_precision,
            mean_claim_precision=claim_p,
            mean_noise_sensitivity=noise,
            mean_abstention=abstain,
        )

    def _claim_metrics(self, system: object, dataset: EvalDataset) -> tuple[float, float, float]:
        from jera.config.registry import RagSystem

        assert isinstance(system, RagSystem)
        precisions: list[float] = []
        noises: list[float] = []
        abstentions: list[float] = []
        for case in dataset.cases:
            bundle = system.query.answer_with_contexts(case.query, top_k=self.k, mode=self.mode)
            ctx_texts = [c.text for c in bundle.contexts]
            precisions.append(claim_precision(bundle.answer.text, ctx_texts))
            abstentions.append(abstention_score(bundle.answer.text))
            if case.reference_answer is not None:
                noises.append(
                    noise_sensitivity(bundle.answer.text, ctx_texts, case.reference_answer)
                )
        return (
            fmean(precisions) if precisions else 0.0,
            fmean(noises) if noises else 0.0,
            fmean(abstentions) if abstentions else 0.0,
        )
