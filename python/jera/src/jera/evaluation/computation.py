"""Computation eval: numeric-accuracy scoring for ComputationQ eval cases.

Design notes (S7)
-----------------
The computation scorer holds a **concrete** ``ToolAugmentedGenerator`` directly —
not a ``GeneratorLLM``-typed reference — because ``run()`` is NOT on the
``GeneratorLLM`` port (that port only exposes ``generate()``).  Calling through
the port would lose the typed ``RunResult.final_value`` and force us to
regex-extract a number from prose, violating Principle 3.

The two metrics produced per case:
  - ``numeric_accuracy`` — FinQA-style relative tolerance comparing
    ``run_result.final_value`` to ``case.expected_value``.
  - ``chunk_recall_at_k`` — whether the supporting gold chunks appear in the
    top-k retrieved results (using the same recall@k formula as ``EvalRunner``).

The computation eval is intentionally **separate** from ``EvalRunner.run``; it
does not mutate the retrieval runner or its reports.

Offline use (CI)
----------------
Pass a ``ToolAugmentedGenerator`` built with ``FakeToolUseLLM`` for zero-API
testing.  The ``FakeToolUseLLM`` yields deterministic ``final_value`` values so
``numeric_accuracy`` scores can be asserted exactly.  Pass one built with
``ClaudeToolUseGenerator(enabled=True, ...)`` for real paid runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jera.adapters.generator.tool_augmented_generator import ToolAugmentedGenerator
from jera.domain.retrieval import FusionMethod, Query, RetrievalMode
from jera.evaluation_contracts.dataset import CaseKind, EvalCase, EvalDataset
from jera.evaluation_contracts.metrics import numeric_accuracy, recall_at_k
from jera.pipeline.query import QueryPipeline

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputationCaseResult:
    """Metrics for one ComputationQ case."""

    case_id: str
    expected_value: float | None
    predicted_value: float | None
    numeric_acc: float  # 1.0 = within tolerance, 0.0 = wrong / None
    chunk_recall_at_k: float  # supporting-chunk recall from retrieval
    answer_text: str


@dataclass
class ComputationReport:
    """Aggregate computation-eval results across all ComputationQ cases."""

    dataset: str
    k: int
    cases: list[ComputationCaseResult] = field(default_factory=list)

    @property
    def mean_numeric_accuracy(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.numeric_acc for c in self.cases) / len(self.cases)

    @property
    def mean_chunk_recall(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.chunk_recall_at_k for c in self.cases) / len(self.cases)

    def to_markdown(self) -> str:
        """Render a compact markdown table of per-case results."""
        lines = [
            f"## Computation Eval — {self.dataset}",
            "",
            f"- Cases: {len(self.cases)}",
            f"- Mean numeric accuracy: {self.mean_numeric_accuracy:.3f}",
            f"- Mean chunk recall@{self.k}: {self.mean_chunk_recall:.3f}",
            "",
            f"| case_id | expected | predicted | num_acc | recall@{self.k} |",
            "|---------|----------|-----------|---------|---------|",
        ]
        for c in self.cases:
            exp = f"{c.expected_value:.4g}" if c.expected_value is not None else "—"
            pred = f"{c.predicted_value:.4g}" if c.predicted_value is not None else "—"
            acc = f"{c.numeric_acc:.2f}"
            rec = f"{c.chunk_recall_at_k:.2f}"
            lines.append(f"| {c.case_id} | {exp} | {pred} | {acc} | {rec} |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class ComputationEval:
    """Scores ComputationQ cases via a ToolAugmentedGenerator.

    Parameters
    ----------
    generator:
        A **concrete** ``ToolAugmentedGenerator`` (not the port-typed
        ``GeneratorLLM``).  Use ``FakeToolUseLLM`` for offline CI;
        ``ClaudeToolUseGenerator`` for real paid runs.
    query_pipeline:
        The retrieval pipeline used to fetch supporting chunks before calling
        the generator.  The same pipeline used for retrieval eval is fine.
    k:
        Number of chunks to retrieve per case (used for both generation
        context and chunk-recall scoring).
    mode:
        Retrieval mode passed to the query pipeline.
    fusion:
        Fusion method for hybrid retrieval.
    """

    def __init__(
        self,
        generator: ToolAugmentedGenerator,
        query_pipeline: QueryPipeline,
        k: int = 5,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        fusion: FusionMethod = FusionMethod.RRF,
    ) -> None:
        self._gen = generator
        self._pipeline = query_pipeline
        self._k = k
        self._mode = mode
        self._fusion = fusion

    def run(self, dataset: EvalDataset) -> ComputationReport:
        """Score all ``computation`` cases in *dataset*.

        Non-computation cases are silently skipped so the same dataset can be
        passed without filtering.
        """
        report = ComputationReport(dataset=dataset.name, k=self._k)
        for case in dataset.cases:
            if case.kind != CaseKind.COMPUTATION:
                continue
            report.cases.append(self._score_case(case))
        return report

    # ------------------------------------------------------------------
    # Per-case scoring
    # ------------------------------------------------------------------

    def _score_case(self, case: EvalCase) -> ComputationCaseResult:
        # --- Step 1: retrieve supporting chunks ---
        retrieval = self._pipeline.retrieve(
            Query(
                text=case.query,
                top_k=self._k,
                mode=self._mode,
                fusion=self._fusion,
            )
        )
        contexts = [r.chunk for r in retrieval.results if r.chunk is not None]
        ranked_ids = [r.chunk_id for r in retrieval.results]

        # --- Step 2: call the typed run() path (not generate()) ---
        # run() returns RunResult{answer_text, tool_calls, final_value: float|None}
        # final_value is typed — no regex extraction needed (Principle 3).
        run_result = self._gen.run(case.query, contexts)

        # --- Step 3: numeric accuracy ---
        num_acc: float
        if case.expected_value is None:
            # No ground truth — treat as unscored (0.0 so it doesn't inflate mean)
            num_acc = 0.0
        else:
            num_acc = numeric_accuracy(
                run_result.final_value,
                case.expected_value,
                case.tolerance,
            )

        # --- Step 4: supporting-chunk recall@k ---
        chunk_recall = recall_at_k(ranked_ids, case.gold, self._k)

        return ComputationCaseResult(
            case_id=case.case_id,
            expected_value=case.expected_value,
            predicted_value=run_result.final_value,
            numeric_acc=num_acc,
            chunk_recall_at_k=chunk_recall,
            answer_text=run_result.answer_text,
        )
