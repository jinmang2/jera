"""GenerationEvalRunner: RAGAS-lite metrics computed end-to-end over the answer path."""

from __future__ import annotations

from jera.config.registry import RagSystem
from jera.domain.document import MediaType
from jera.domain.retrieval import RetrievalMode
from jera.evaluation import CaseSpec, GenerationEvalRunner, build_gold_dataset
from jera.evaluation.report import GenerationCaseResult, GenerationReport
from jera.evaluation_contracts.dataset import EvalDataset

CORPUS = {
    "doc": (
        MediaType.MARKDOWN,
        """# Notes

## Fusion
Hybrid retrieval uses reciprocal rank fusion to merge rankings.

## Identifier
The ranking module identifier is ZX9000 here.
""",
    )
}


def _dataset(system: RagSystem, *, with_reference: bool) -> EvalDataset:
    ds = build_gold_dataset(
        system, name="gen", documents=CORPUS, cases=[CaseSpec("id", "ZX9000", "ZX9000")]
    )
    if not with_reference:
        return ds
    case = ds.cases[0].model_copy(
        update={"reference_answer": "The ranking module identifier is ZX9000 here."}
    )
    return EvalDataset(name="gen", cases=[case])


def test_generation_runner_grades_grounded_extractive_answer(system: RagSystem) -> None:
    ds = _dataset(system, with_reference=True)
    report = GenerationEvalRunner(system.query).run(ds, k=5, mode=RetrievalMode.SPARSE)

    # End-to-end: the extractive answer stitches retrieved context (grounded) but adds a boilerplate
    # preamble (ungrounded), so faithfulness lands strictly inside (0, 1) — neither hallucinated
    # nor trivially perfect. (Discrimination itself is proven in test_generation_metrics.py.)
    assert 0.0 < report.mean_faithfulness < 1.0
    # The gold chunk is retrieved at rank 1 for the exact identifier → perfect precision.
    assert report.mean_context_precision == 1.0


def test_generation_runner_reports_correctness_and_relevance_with_reference(
    system: RagSystem,
) -> None:
    ds = _dataset(system, with_reference=True)
    report = GenerationEvalRunner(system.query).run(ds, k=5, mode=RetrievalMode.SPARSE)
    assert report.mean_answer_correctness is not None
    assert 0.0 < report.mean_answer_correctness <= 1.0
    assert report.mean_answer_relevance is not None
    assert 0.0 <= report.mean_answer_relevance <= 1.0
    assert "faithfulness" in report.summary_table()


def test_generation_runner_omits_reference_metrics_when_absent(system: RagSystem) -> None:
    ds = _dataset(system, with_reference=False)
    report = GenerationEvalRunner(system.query).run(ds, k=5, mode=RetrievalMode.SPARSE)
    # No reference answer → correctness/relevance are not computed (n/a), not faked as 0.
    assert report.mean_answer_correctness is None
    assert report.mean_answer_relevance is None
    assert "n/a" in report.summary_table()
    # but the always-on metrics are still present
    assert report.mean_faithfulness > 0.0


def test_generation_report_aggregation_handles_optional_means() -> None:
    cases = [
        GenerationCaseResult(
            case_id="a",
            faithfulness=1.0,
            context_precision=1.0,
            answer_correctness=0.8,
            answer_relevance=0.6,
        ),
        GenerationCaseResult(
            case_id="b",
            faithfulness=0.0,
            context_precision=0.0,
            answer_correctness=None,  # this case had no reference answer
            answer_relevance=None,
        ),
    ]
    report = GenerationReport.from_cases("ds", "sparse", 5, cases)
    assert report.mean_faithfulness == 0.5
    assert report.mean_context_precision == 0.5
    # optional means average only the present values (a single 0.8 / 0.6 here)
    assert report.mean_answer_correctness == 0.8
    assert report.mean_answer_relevance == 0.6


def test_generation_report_empty_cases_is_safe() -> None:
    report = GenerationReport.from_cases("ds", "sparse", 5, [])
    assert report.mean_faithfulness == 0.0
    assert report.mean_answer_correctness is None
    assert report.cases == []
