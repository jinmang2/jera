"""Unit tests for evaluation/computation.py — zero API calls.

All tests use FakeToolUseLLM (deterministic) and a mock QueryPipeline so no
retrieval infrastructure is needed.  ComputationEval.run() is exercised
end-to-end: retrieve → gen.run() → numeric_accuracy + chunk_recall.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jera.adapters.generator.tool_augmented_generator import ToolAugmentedGenerator
from jera.domain.chunk import Chunk
from jera.domain.document import PageSpan
from jera.domain.retrieval import ScoredChunk
from jera.evaluation.computation import (
    ComputationCaseResult,
    ComputationEval,
    ComputationReport,
)
from jera.evaluation_contracts.dataset import CaseKind, EvalCase, EvalDataset, GoldChunk
from jera.tooluse import CalculatorTool, FakeToolUseLLM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, text: str = "Some text.") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc1",
        source_id="src1",
        text=text,
        page_span=PageSpan(start_page=1, end_page=1),
        section_path=("Section 1",),
        element_ids=("e1",),
        char_span=(0, len(text)),
        token_count=len(text.split()),
        chunk_strategy="heading_aware",
        chunk_version="v1",
    )


def _make_scored(chunk_id: str, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        score=score,
        chunk=_make_chunk(chunk_id),
    )


def _fake_pipeline(chunk_ids: list[str]) -> MagicMock:
    """Build a mock QueryPipeline that returns the given chunk_ids."""
    pipeline = MagicMock()
    query_mock = MagicMock()
    query_mock.results = [_make_scored(cid) for cid in chunk_ids]
    pipeline.retrieve.return_value = query_mock
    return pipeline


def _make_gen(mode: str = "single") -> ToolAugmentedGenerator:
    return ToolAugmentedGenerator(
        llm=FakeToolUseLLM(mode=mode),  # type: ignore[arg-type]
        tools=[CalculatorTool()],
    )


def _computation_case(
    case_id: str = "c1",
    query: str = "What is 1 + 1?",
    expected_value: float = 2.0,
    gold_ids: list[str] | None = None,
) -> EvalCase:
    gold = [GoldChunk(chunk_id=gid) for gid in (gold_ids or ["chunk-c1"])]
    return EvalCase(
        case_id=case_id,
        query=query,
        gold=gold,
        kind=CaseKind.COMPUTATION,
        expected_value=expected_value,
        tolerance=0.001,
    )


# ---------------------------------------------------------------------------
# ComputationCaseResult / ComputationReport
# ---------------------------------------------------------------------------


class TestComputationReport:
    def test_empty_report_means_zero(self) -> None:
        report = ComputationReport(dataset="test", k=5)
        assert report.mean_numeric_accuracy == 0.0
        assert report.mean_chunk_recall == 0.0

    def test_mean_numeric_accuracy(self) -> None:
        report = ComputationReport(
            dataset="test",
            k=5,
            cases=[
                ComputationCaseResult(
                    case_id="c1",
                    expected_value=2.0,
                    predicted_value=2.0,
                    numeric_acc=1.0,
                    chunk_recall_at_k=1.0,
                    answer_text="2.0",
                ),
                ComputationCaseResult(
                    case_id="c2",
                    expected_value=5.0,
                    predicted_value=99.0,
                    numeric_acc=0.0,
                    chunk_recall_at_k=0.5,
                    answer_text="99.0",
                ),
            ],
        )
        assert report.mean_numeric_accuracy == pytest.approx(0.5)
        assert report.mean_chunk_recall == pytest.approx(0.75)

    def test_to_markdown_contains_header(self) -> None:
        report = ComputationReport(dataset="mydata", k=5)
        md = report.to_markdown()
        assert "mydata" in md
        assert "numeric accuracy" in md.lower()


# ---------------------------------------------------------------------------
# ComputationEval.run() — full end-to-end (offline)
# ---------------------------------------------------------------------------


class TestComputationEvalRun:
    def setup_method(self) -> None:
        # FakeToolUseLLM single-mode: tool call "1 + 1" → 2.0
        self.gen = _make_gen(mode="single")
        # Retrieved chunk with id "chunk-c1" → gold hit
        self.pipeline = _fake_pipeline(["chunk-c1", "chunk-c2"])
        self.eval = ComputationEval(
            generator=self.gen,
            query_pipeline=self.pipeline,
            k=2,
        )

    def test_run_returns_computation_report(self) -> None:
        dataset = EvalDataset(
            name="test",
            cases=[_computation_case(expected_value=2.0, gold_ids=["chunk-c1"])],
        )
        report = self.eval.run(dataset)
        assert isinstance(report, ComputationReport)

    def test_run_scores_one_computation_case(self) -> None:
        dataset = EvalDataset(
            name="test",
            cases=[_computation_case(expected_value=2.0, gold_ids=["chunk-c1"])],
        )
        report = self.eval.run(dataset)
        assert len(report.cases) == 1

    def test_numeric_accuracy_hit(self) -> None:
        # FakeToolUseLLM "single" returns final_value=2.0; expected_value=2.0
        dataset = EvalDataset(
            name="test",
            cases=[_computation_case(expected_value=2.0, gold_ids=["chunk-c1"])],
        )
        report = self.eval.run(dataset)
        assert report.cases[0].numeric_acc == pytest.approx(1.0)
        assert report.cases[0].predicted_value == pytest.approx(2.0)

    def test_numeric_accuracy_miss(self) -> None:
        # expected_value is 999.0 but FakeToolUseLLM returns 2.0
        dataset = EvalDataset(
            name="test",
            cases=[_computation_case(expected_value=999.0, gold_ids=["chunk-c1"])],
        )
        report = self.eval.run(dataset)
        assert report.cases[0].numeric_acc == pytest.approx(0.0)

    def test_chunk_recall_hit(self) -> None:
        # Pipeline returns ["chunk-c1", "chunk-c2"]; gold = ["chunk-c1"] → recall=1.0
        dataset = EvalDataset(
            name="test",
            cases=[_computation_case(expected_value=2.0, gold_ids=["chunk-c1"])],
        )
        report = self.eval.run(dataset)
        assert report.cases[0].chunk_recall_at_k == pytest.approx(1.0)

    def test_chunk_recall_miss(self) -> None:
        # Gold chunk not in retrieved set
        dataset = EvalDataset(
            name="test",
            cases=[_computation_case(expected_value=2.0, gold_ids=["chunk-missing"])],
        )
        report = self.eval.run(dataset)
        assert report.cases[0].chunk_recall_at_k == pytest.approx(0.0)

    def test_non_computation_cases_skipped(self) -> None:
        retrieval_case = EvalCase(
            case_id="r1",
            query="Who wrote this?",
            gold=[GoldChunk(chunk_id="chunk-r1")],
            kind=CaseKind.RETRIEVAL,
        )
        computation_case = _computation_case()
        dataset = EvalDataset(name="test", cases=[retrieval_case, computation_case])
        report = self.eval.run(dataset)
        # Only the computation case is scored
        assert len(report.cases) == 1
        assert report.cases[0].case_id == "c1"

    def test_no_expected_value_gives_zero_accuracy(self) -> None:
        case = EvalCase(
            case_id="c-none",
            query="Unknown numeric?",
            gold=[GoldChunk(chunk_id="chunk-c1")],
            kind=CaseKind.COMPUTATION,
            expected_value=None,
        )
        dataset = EvalDataset(name="test", cases=[case])
        report = self.eval.run(dataset)
        assert report.cases[0].numeric_acc == pytest.approx(0.0)

    def test_pipeline_retrieve_called_per_case(self) -> None:
        dataset = EvalDataset(
            name="test",
            cases=[
                _computation_case(case_id="c1"),
                _computation_case(case_id="c2"),
            ],
        )
        self.eval.run(dataset)
        assert self.pipeline.retrieve.call_count == 2


# ---------------------------------------------------------------------------
# Multi-block FakeToolUseLLM path (exercises the full loop)
# ---------------------------------------------------------------------------


class TestComputationEvalMultiBlock:
    def test_multi_block_final_value_scored(self) -> None:
        # FakeToolUseLLM "multi" last tool call: 7.0 + 20.0 = 27.0
        gen = _make_gen(mode="multi")
        pipeline = _fake_pipeline(["chunk-c1"])
        eval_ = ComputationEval(generator=gen, query_pipeline=pipeline, k=1)
        dataset = EvalDataset(
            name="test",
            cases=[_computation_case(expected_value=27.0, gold_ids=["chunk-c1"])],
        )
        report = eval_.run(dataset)
        assert report.cases[0].numeric_acc == pytest.approx(1.0)
        assert report.cases[0].predicted_value == pytest.approx(27.0)
