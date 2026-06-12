"""Evaluation harness tests: gold labeling, metric computation, aggregation, failure modes."""

from __future__ import annotations

import pytest

from jera.config.registry import RagSystem
from jera.domain.document import MediaType
from jera.domain.retrieval import RetrievalMode
from jera.evaluation import CaseSpec, EvalRunner, build_gold_dataset
from jera.evaluation.report import CaseResult, ModeReport

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


def test_build_gold_dataset_labels_by_substring(system: RagSystem) -> None:
    ds = build_gold_dataset(
        system,
        name="t",
        documents=CORPUS,
        cases=[CaseSpec("id", "ZX9000", "ZX9000")],
    )
    assert ds.cases[0].gold, "expected at least one gold chunk"
    # every gold chunk really contains the marker
    for g in ds.cases[0].gold:
        chunk = system.metadata_store.get_chunk(g.chunk_id)
        assert chunk is not None and "ZX9000" in chunk.text


def test_build_gold_dataset_raises_on_unsatisfiable_label(system: RagSystem) -> None:
    with pytest.raises(ValueError, match="unsatisfiable"):
        build_gold_dataset(
            system,
            name="t",
            documents=CORPUS,
            cases=[CaseSpec("missing", "anything", "NO_SUCH_STRING_42")],
        )


def test_runner_scores_perfect_for_exact_identifier(system: RagSystem) -> None:
    ds = build_gold_dataset(
        system, name="t", documents=CORPUS, cases=[CaseSpec("id", "ZX9000", "ZX9000")]
    )
    report = EvalRunner(system.query).run(ds, k=5, modes=[RetrievalMode.SPARSE])
    sparse = report.modes["sparse"]
    assert sparse.mean_recall_at_k == 1.0
    assert sparse.mean_mrr == 1.0  # the gold chunk is retrieved at rank 1


def test_runner_reports_one_mode_report_per_mode(system: RagSystem) -> None:
    ds = build_gold_dataset(
        system, name="t", documents=CORPUS, cases=[CaseSpec("id", "ZX9000", "ZX9000")]
    )
    report = EvalRunner(system.query).run(
        ds, k=3, modes=[RetrievalMode.DENSE, RetrievalMode.SPARSE, RetrievalMode.HYBRID]
    )
    assert set(report.modes) == {"dense", "sparse", "hybrid"}
    assert "recall@k" in report.summary_table()


def test_mode_report_aggregation_is_mean() -> None:
    cases = [
        CaseResult(case_id="a", ranked_ids=[], recall_at_k=1.0, mrr=1.0, ndcg_at_k=1.0),
        CaseResult(case_id="b", ranked_ids=[], recall_at_k=0.0, mrr=0.0, ndcg_at_k=0.0),
    ]
    report = ModeReport.from_cases("dense", 5, cases)
    assert report.mean_recall_at_k == 0.5
    assert report.mean_mrr == 0.5
    assert report.mean_ndcg_at_k == 0.5


def test_empty_cases_yield_zero_report() -> None:
    report = ModeReport.from_cases("dense", 5, [])
    assert report.mean_recall_at_k == 0.0
    assert report.cases == []
