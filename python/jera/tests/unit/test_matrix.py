"""Unit tests for evaluation matrix (dense/sparse/hybrid × chunking strategies).

All tests run fully offline under the TEST profile (hash embeddings, BM25,
in-memory store) — no fastembed model weights are downloaded, no paid API calls.
"""

from __future__ import annotations

from jera.config.settings import Settings
from jera.domain.document import MediaType
from jera.domain.retrieval import RetrievalMode
from jera.evaluation.matrix import MatrixReport, StrategyEntry, run_matrix
from jera.evaluation_contracts.dataset import EvalCase, EvalDataset, GoldChunk

# ---------------------------------------------------------------------------
# Minimal corpus and dataset for offline tests
# ---------------------------------------------------------------------------

_CORPUS: dict[str, tuple[MediaType, str]] = {
    "doc-a": (
        MediaType.MARKDOWN,
        """# 경제 보고서

## 성장률
2023년 한국 경제성장률은 1.4%로 전망됩니다.

## 물가
소비자물가 상승률은 3.6%를 기록하였습니다.
""",
    ),
}

# A retrieval case whose gold chunk_id will be assigned by the pipeline.
# We use a substring label so build_gold_dataset-style matching can be
# replicated — but for the matrix test we only need the eval to complete
# and produce entries, not achieve high recall.
_DATASET = EvalDataset(
    name="test_matrix",
    cases=[
        EvalCase(
            case_id="t-001",
            query="한국 경제성장률 전망",
            gold=[GoldChunk(chunk_id="doc-a-0")],
        ),
        EvalCase(
            case_id="t-002",
            query="소비자물가 상승률",
            gold=[GoldChunk(chunk_id="doc-a-1")],
        ),
    ],
)

# Dataset variant with reference answers to exercise answer_correctness /
# answer_relevance metrics.
_DATASET_WITH_REF = EvalDataset(
    name="test_matrix_ref",
    cases=[
        EvalCase(
            case_id="t-001",
            query="한국 경제성장률 전망",
            gold=[GoldChunk(chunk_id="doc-a-0")],
            reference_answer="2023년 한국 경제성장률은 1.4%로 전망됩니다.",
        ),
        EvalCase(
            case_id="t-002",
            query="소비자물가 상승률",
            gold=[GoldChunk(chunk_id="doc-a-1")],
            reference_answer="소비자물가 상승률은 3.6%를 기록하였습니다.",
        ),
    ],
)

_TEST_SETTINGS = Settings(profile="test")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# MatrixReport structure
# ---------------------------------------------------------------------------


class TestMatrixReportStructure:
    def test_has_entry_per_strategy_and_mode(self) -> None:
        report = run_matrix(
            _DATASET,
            corpus=_CORPUS,
            strategies=["heading_aware", "semantic", "hierarchical"],
            modes=[RetrievalMode.DENSE, RetrievalMode.SPARSE, RetrievalMode.HYBRID],
            settings_base=_TEST_SETTINGS,
        )
        assert len(report.entries) == 9  # 3 strategies × 3 modes

    def test_strategies_and_modes_recorded(self) -> None:
        report = run_matrix(
            _DATASET,
            corpus=_CORPUS,
            strategies=["heading_aware", "semantic"],
            modes=[RetrievalMode.DENSE, RetrievalMode.SPARSE],
            settings_base=_TEST_SETTINGS,
        )
        assert report.strategies == ["heading_aware", "semantic"]
        assert set(report.modes) == {"dense", "sparse"}

    def test_dataset_name_preserved(self) -> None:
        report = run_matrix(
            _DATASET,
            corpus=_CORPUS,
            strategies=["heading_aware"],
            settings_base=_TEST_SETTINGS,
        )
        assert report.dataset == "test_matrix"

    def test_get_returns_entry(self) -> None:
        report = run_matrix(
            _DATASET,
            corpus=_CORPUS,
            strategies=["heading_aware"],
            modes=[RetrievalMode.DENSE],
            settings_base=_TEST_SETTINGS,
        )
        entry = report.get("heading_aware", "dense")
        assert entry is not None
        assert isinstance(entry, StrategyEntry)
        assert entry.strategy == "heading_aware"
        assert entry.mode == "dense"

    def test_get_returns_none_for_missing(self) -> None:
        report = MatrixReport(dataset="x", k=5, strategies=[], modes=[], entries=[])
        assert report.get("heading_aware", "dense") is None

    def test_metrics_are_floats_in_unit_interval(self) -> None:
        report = run_matrix(
            _DATASET,
            corpus=_CORPUS,
            strategies=["heading_aware"],
            modes=[RetrievalMode.DENSE],
            settings_base=_TEST_SETTINGS,
        )
        entry = report.get("heading_aware", "dense")
        assert entry is not None
        for metric in (entry.mean_recall_at_k, entry.mean_mrr, entry.mean_ndcg_at_k):
            assert 0.0 <= metric <= 1.0


# ---------------------------------------------------------------------------
# MatrixReport.to_markdown()
# ---------------------------------------------------------------------------


class TestToMarkdown:
    def _make_report(self) -> MatrixReport:
        return run_matrix(
            _DATASET,
            corpus=_CORPUS,
            strategies=["heading_aware", "semantic"],
            modes=[RetrievalMode.DENSE, RetrievalMode.SPARSE],
            settings_base=_TEST_SETTINGS,
        )

    def test_contains_dataset_header(self) -> None:
        md = self._make_report().to_markdown()
        assert "test_matrix" in md

    def test_contains_strategy_headings(self) -> None:
        md = self._make_report().to_markdown()
        assert "## heading_aware" in md
        assert "## semantic" in md

    def test_contains_mode_rows(self) -> None:
        md = self._make_report().to_markdown()
        assert "dense" in md
        assert "sparse" in md

    def test_markdown_has_table_separator(self) -> None:
        md = self._make_report().to_markdown()
        # Markdown tables use | and - separators
        assert "|" in md
        assert "---" in md

    def test_markdown_mentions_faithfulness(self) -> None:
        md = self._make_report().to_markdown()
        assert "faithful" in md


# ---------------------------------------------------------------------------
# run_matrix — edge cases
# ---------------------------------------------------------------------------


class TestRunMatrixEdgeCases:
    def test_invalid_strategy_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Unknown chunk strategies"):
            run_matrix(
                _DATASET,
                corpus=_CORPUS,
                strategies=["invalid_strategy"],
                settings_base=_TEST_SETTINGS,
            )

    def test_single_strategy_single_mode(self) -> None:
        report = run_matrix(
            _DATASET,
            corpus=_CORPUS,
            strategies=["hierarchical"],
            modes=[RetrievalMode.HYBRID],
            settings_base=_TEST_SETTINGS,
        )
        assert len(report.entries) == 1
        assert report.entries[0].strategy == "hierarchical"
        assert report.entries[0].mode == "hybrid"

    def test_empty_corpus_produces_zero_metrics(self) -> None:
        report = run_matrix(
            _DATASET,
            corpus={},
            strategies=["heading_aware"],
            modes=[RetrievalMode.DENSE],
            settings_base=_TEST_SETTINGS,
        )
        entry = report.get("heading_aware", "dense")
        assert entry is not None
        assert entry.mean_recall_at_k == 0.0
        assert entry.mean_mrr == 0.0


# ---------------------------------------------------------------------------
# Generation metrics on StrategyEntry
# ---------------------------------------------------------------------------


class TestGenerationMetrics:
    """RAGAS-lite generation metrics are populated for each strategy entry."""

    def test_generation_fields_present_on_entry(self) -> None:
        """Every StrategyEntry exposes the four generation-metric fields."""
        report = run_matrix(
            _DATASET,
            corpus=_CORPUS,
            strategies=["heading_aware", "semantic"],
            modes=[RetrievalMode.DENSE, RetrievalMode.HYBRID],
            settings_base=_TEST_SETTINGS,
        )
        assert len(report.entries) >= 2
        for entry in report.entries:
            assert hasattr(entry, "mean_faithfulness")
            assert hasattr(entry, "mean_context_precision")
            assert hasattr(entry, "mean_answer_correctness")
            assert hasattr(entry, "mean_answer_relevance")

    def test_faithfulness_and_context_precision_in_unit_interval(self) -> None:
        """faithfulness and context_precision are floats in [0, 1]."""
        report = run_matrix(
            _DATASET,
            corpus=_CORPUS,
            strategies=["heading_aware", "semantic"],
            modes=[RetrievalMode.DENSE, RetrievalMode.HYBRID],
            settings_base=_TEST_SETTINGS,
        )
        for entry in report.entries:
            assert 0.0 <= entry.mean_faithfulness <= 1.0
            assert 0.0 <= entry.mean_context_precision <= 1.0

    def test_no_reference_answer_yields_none_optional_metrics(self) -> None:
        """Without reference_answer on cases, correctness and relevance are None."""
        report = run_matrix(
            _DATASET,  # no reference_answer on any case
            corpus=_CORPUS,
            strategies=["heading_aware"],
            modes=[RetrievalMode.HYBRID],
            settings_base=_TEST_SETTINGS,
        )
        entry = report.get("heading_aware", "hybrid")
        assert entry is not None
        assert entry.mean_answer_correctness is None
        assert entry.mean_answer_relevance is None

    def test_with_reference_answer_optional_metrics_in_unit_interval(self) -> None:
        """With reference_answer, correctness and relevance are floats in [0, 1]."""
        report = run_matrix(
            _DATASET_WITH_REF,
            corpus=_CORPUS,
            strategies=["heading_aware"],
            modes=[RetrievalMode.HYBRID],
            settings_base=_TEST_SETTINGS,
        )
        entry = report.get("heading_aware", "hybrid")
        assert entry is not None
        assert entry.mean_answer_correctness is not None
        assert entry.mean_answer_relevance is not None
        assert 0.0 <= entry.mean_answer_correctness <= 1.0
        assert 0.0 <= entry.mean_answer_relevance <= 1.0

    def test_generation_metrics_shared_across_modes_within_strategy(self) -> None:
        """Generation metrics are per-strategy (same value across modes)."""
        report = run_matrix(
            _DATASET,
            corpus=_CORPUS,
            strategies=["heading_aware"],
            modes=[RetrievalMode.DENSE, RetrievalMode.SPARSE, RetrievalMode.HYBRID],
            settings_base=_TEST_SETTINGS,
        )
        dense = report.get("heading_aware", "dense")
        sparse = report.get("heading_aware", "sparse")
        hybrid = report.get("heading_aware", "hybrid")
        assert dense is not None and sparse is not None and hybrid is not None
        assert dense.mean_faithfulness == sparse.mean_faithfulness == hybrid.mean_faithfulness
        assert (
            dense.mean_context_precision
            == sparse.mean_context_precision
            == hybrid.mean_context_precision
        )
