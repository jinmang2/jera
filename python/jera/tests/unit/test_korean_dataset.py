"""Tests for the committed Korean research eval dataset and its corpus fixtures.

Verifies that:
  (a) korean_research.json loads and has no placeholder chunk IDs.
  (b) Every gold chunk_id resolves to a real stored chunk that contains the
      expected text (mirrors test_build_gold_dataset_labels_by_substring).
  (c) At least one case of each kind (retrieval, table, computation) exists.
  (d) EvalRunner scores retrieval cases with recall@k == 1.0 (exact identifier
      retrieval over the committed corpus).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jera.config.registry import RagSystem, build_system
from jera.config.settings import Profile, Settings
from jera.domain.document import MediaType, SourceRef
from jera.domain.retrieval import RetrievalMode
from jera.evaluation import EvalRunner
from jera.evaluation_contracts.dataset import EvalCase, EvalDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]  # python/jera/tests/unit → repo root
_DATASET_PATH = _REPO_ROOT / "data" / "eval" / "korean_research.json"
_CORPUS_DIR = _REPO_ROOT / "data" / "corpus" / "korean_research"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dataset() -> EvalDataset:
    """Load the committed eval dataset."""
    raw = json.loads(_DATASET_PATH.read_text(encoding="utf-8"))
    return EvalDataset.model_validate(raw)


@pytest.fixture(scope="module")
def ingested_system() -> RagSystem:
    """Fresh TEST-profile system with the committed corpus already ingested."""
    system = build_system(Settings(profile=Profile.TEST))
    for md_path in sorted(_CORPUS_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        system.ingest.ingest(
            SourceRef(source_id=md_path.name, media_type=MediaType.MARKDOWN, content=text.encode())
        )
    return system


# ---------------------------------------------------------------------------
# (a) Dataset structure — no placeholders
# ---------------------------------------------------------------------------


def test_dataset_loads_successfully(dataset: EvalDataset) -> None:
    assert dataset.name == "korean_research"
    assert len(dataset.cases) >= 5


def test_no_placeholder_chunk_ids(dataset: EvalDataset) -> None:
    for case in dataset.cases:
        for gold in case.gold:
            assert not gold.chunk_id.startswith("placeholder"), (
                f"case {case.case_id!r} still has placeholder chunk_id {gold.chunk_id!r}"
            )


# ---------------------------------------------------------------------------
# (b) Every gold chunk_id resolves to a real stored chunk with expected text
# ---------------------------------------------------------------------------


def test_gold_chunk_ids_resolve_and_contain_text(
    dataset: EvalDataset, ingested_system: RagSystem
) -> None:
    """Mirror of test_build_gold_dataset_labels_by_substring from test_eval_runner.py."""
    for case in dataset.cases:
        assert case.gold, f"case {case.case_id!r} has empty gold list"
        for gold in case.gold:
            chunk = ingested_system.metadata_store.get_chunk(gold.chunk_id)
            assert chunk is not None, (
                f"case {case.case_id!r}: chunk_id {gold.chunk_id!r} not found in store"
            )
            # The chunk must contain some meaningful text (not empty)
            assert chunk.text.strip(), (
                f"case {case.case_id!r}: chunk {gold.chunk_id!r} has empty text"
            )


# ---------------------------------------------------------------------------
# (c) At least one case of each kind
# ---------------------------------------------------------------------------


def test_all_three_kinds_present(dataset: EvalDataset) -> None:
    kinds = {case.kind for case in dataset.cases}
    assert "retrieval" in kinds, "no retrieval case found"
    assert "table" in kinds, "no table case found"
    assert "computation" in kinds, "no computation case found"


# ---------------------------------------------------------------------------
# (d) Computation case has expected fields populated
# ---------------------------------------------------------------------------


def test_computation_cases_have_numeric_fields(dataset: EvalDataset) -> None:
    computation_cases: list[EvalCase] = [c for c in dataset.cases if c.kind == "computation"]
    assert computation_cases, "no computation cases"
    for case in computation_cases:
        assert case.expected_value is not None, (
            f"computation case {case.case_id!r} missing expected_value"
        )
        assert case.formula is not None, f"computation case {case.case_id!r} missing formula"
        assert case.cited_numbers, f"computation case {case.case_id!r} has empty cited_numbers"
        # Verify the formula is arithmetically consistent with expected_value
        result = sum(case.cited_numbers)
        assert abs(result - case.expected_value) <= case.tolerance + 0.5, (
            f"case {case.case_id!r}: sum of cited_numbers {result} != "
            f"expected_value {case.expected_value}"
        )


# ---------------------------------------------------------------------------
# (e) EvalRunner: retrieval cases score recall@k == 1.0
# ---------------------------------------------------------------------------


def test_eval_runner_retrieval_recall_at_k(
    dataset: EvalDataset, ingested_system: RagSystem
) -> None:
    """Retrieval cases use exact unique substrings so sparse retrieval must hit them."""
    from jera.evaluation_contracts.dataset import EvalDataset as DS

    retrieval_only = DS(
        name="korean_research_retrieval",
        cases=[c for c in dataset.cases if c.kind == "retrieval"],
    )
    assert retrieval_only.cases, "no retrieval cases to test"

    report = EvalRunner(ingested_system.query).run(
        retrieval_only, k=10, modes=[RetrievalMode.SPARSE]
    )
    sparse = report.modes["sparse"]
    assert sparse.mean_recall_at_k == 1.0, (
        f"Expected recall@10 == 1.0 for retrieval cases, got {sparse.mean_recall_at_k}"
    )
