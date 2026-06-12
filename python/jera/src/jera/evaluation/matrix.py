"""Matrix evaluation: run EvalRunner across chunk strategies × retrieval modes.

Produces a ``MatrixReport`` with one ``EvalReport`` per strategy, which
``to_markdown()`` renders as a strategy × mode table of recall@k / MRR / nDCG /
faithfulness / context_precision.

Design notes
------------
- Each strategy gets a *fresh* ``RagSystem`` built via ``build_system()``.
  Chunk text differs per strategy, so embeddings cannot be reused across
  strategies — no cross-strategy embedding cache is introduced.
- Modes (dense / sparse / hybrid) are scored inside a single ``EvalRunner.run``
  call per strategy, reusing the same ingest for all modes within a strategy.
- Generation metrics (faithfulness, context_precision, answer_correctness,
  answer_relevance) are computed once per strategy via ``GenerationEvalRunner``
  using ``gen_mode`` (default ``RetrievalMode.HYBRID``) and stored on every
  ``StrategyEntry`` for that strategy.  They are the same across modes within a
  strategy because the generator is mode-independent per run.
- This module is offline-safe: under the TEST profile (hash embeddings, BM25,
  in-memory store) no model weights are downloaded.  The local profile requires
  ``uv sync --extra local`` and downloads fastembed models on first use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from jera.config.registry import RagSystem, build_system
from jera.config.settings import Settings
from jera.domain.document import MediaType
from jera.domain.retrieval import FusionMethod, RetrievalMode
from jera.evaluation.generation_runner import GenerationEvalRunner
from jera.evaluation.runner import EvalRunner
from jera.evaluation_contracts.dataset import EvalDataset

# Strategies accepted by ``Settings.chunk_strategy``.
_VALID_STRATEGIES = frozenset({"heading_aware", "semantic", "hierarchical"})

# Default mode sequence — mirrors EvalRunner default.
_DEFAULT_MODES: tuple[RetrievalMode, ...] = (
    RetrievalMode.DENSE,
    RetrievalMode.SPARSE,
    RetrievalMode.HYBRID,
)


@dataclass
class StrategyEntry:
    """Metrics for one (strategy, mode) cell in the matrix.

    Retrieval metrics (recall@k, mrr, ndcg@k) vary per mode.  Generation
    metrics (faithfulness, context_precision, answer_correctness,
    answer_relevance) are computed once per strategy and shared across modes.
    """

    strategy: str
    mode: str
    mean_recall_at_k: float
    mean_mrr: float
    mean_ndcg_at_k: float
    # RAGAS-lite generation metrics — shared across modes for a given strategy.
    mean_faithfulness: float = 0.0
    mean_context_precision: float = 0.0
    mean_answer_correctness: float | None = None
    mean_answer_relevance: float | None = None


@dataclass
class MatrixReport:
    """Matrix of retrieval + generation metrics: strategies (rows) × modes (columns).

    ``entries`` is ordered: outer loop strategies, inner loop modes —
    matching the iteration order of ``run_matrix``.
    """

    dataset: str
    k: int
    strategies: list[str]
    modes: list[str]
    entries: list[StrategyEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get(self, strategy: str, mode: str) -> StrategyEntry | None:
        """Return the entry for a given (strategy, mode) pair, or None."""
        for e in self.entries:
            if e.strategy == strategy and e.mode == mode:
                return e
        return None

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    def to_markdown(self) -> str:
        """Render the matrix as a GitHub-flavoured Markdown table.

        Produces one section per strategy; within each section the three
        retrieval modes are columns.  Each row also shows the per-strategy
        generation metrics (faithfulness and context_precision).  Example::

            ## heading_aware

            | mode    | recall@5 |    mrr | ndcg@5 | faithful | ctx_prec |
            |---------|----------|--------|--------|----------|----------|
            | dense   |    0.800 |  0.750 |  0.770 |    0.900 |    0.850 |
            ...
        """
        lines: list[str] = [f"# Matrix Eval — {self.dataset}\n"]
        k_str = str(self.k)
        header = (
            f"| {'mode':<8} | {'recall@' + k_str:<8} | {'mrr':>6}"
            f" | {'ndcg@' + k_str:>6} | {'faithful':>8} | {'ctx_prec':>8} |"
        )
        sep = f"|{'-' * 10}|{'-' * 10}|{'-' * 8}|{'-' * 8}|{'-' * 10}|{'-' * 10}|"

        for strategy in self.strategies:
            lines.append(f"## {strategy}\n")
            lines.append(header)
            lines.append(sep)
            for mode in self.modes:
                entry = self.get(strategy, mode)
                if entry is None:
                    lines.append(
                        f"| {mode:<8} | {'n/a':>8} | {'n/a':>6}"
                        f" | {'n/a':>6} | {'n/a':>8} | {'n/a':>8} |"
                    )
                else:
                    lines.append(
                        f"| {mode:<8} | {entry.mean_recall_at_k:>8.3f}"
                        f" | {entry.mean_mrr:>6.3f}"
                        f" | {entry.mean_ndcg_at_k:>6.3f}"
                        f" | {entry.mean_faithfulness:>8.3f}"
                        f" | {entry.mean_context_precision:>8.3f} |"
                    )
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Corpus ingestion helper
# ---------------------------------------------------------------------------


def _ingest_corpus(
    system: RagSystem,
    corpus: dict[str, tuple[MediaType, str | bytes]],
) -> None:
    """Ingest all documents in *corpus* into *system*.

    ``corpus`` maps a source id to a ``(MediaType, content)`` pair, where
    content is a UTF-8 string for text types or raw bytes for binary types
    (e.g. PDF).
    """
    from jera.domain.document import SourceRef

    for source_id, (media_type, content) in corpus.items():
        raw: bytes = content if isinstance(content, bytes) else content.encode()
        ref = SourceRef(source_id=source_id, media_type=media_type, content=raw)
        system.ingest.ingest(ref)


# ---------------------------------------------------------------------------
# run_matrix
# ---------------------------------------------------------------------------


def run_matrix(
    dataset: EvalDataset,
    *,
    corpus: dict[str, tuple[MediaType, str | bytes]],
    strategies: Sequence[str] = ("heading_aware", "semantic", "hierarchical"),
    modes: Sequence[RetrievalMode] = _DEFAULT_MODES,
    settings_base: Settings | None = None,
    k: int = 5,
    gen_mode: RetrievalMode = RetrievalMode.HYBRID,
    gen_fusion: FusionMethod = FusionMethod.RRF,
) -> MatrixReport:
    """Run EvalRunner and GenerationEvalRunner across all *strategies* × *modes*.

    Returns a ``MatrixReport`` with retrieval metrics per (strategy, mode) cell
    and RAGAS-lite generation metrics per strategy (shared across all modes for
    that strategy).

    Parameters
    ----------
    dataset:
        The evaluation dataset to score against.
    corpus:
        Documents to ingest, keyed by source id.  Each value is a
        ``(MediaType, content)`` pair.  The corpus is re-ingested per
        strategy so chunk text reflects the chosen strategy.
    strategies:
        Chunking strategies to evaluate.  Each must be one of
        ``"heading_aware"``, ``"semantic"``, or ``"hierarchical"``.
    modes:
        Retrieval modes to evaluate per strategy.
    settings_base:
        Base ``Settings`` object.  ``chunk_strategy`` is overridden per
        iteration; all other fields are preserved.  Defaults to
        ``Settings()`` (TEST profile).
    k:
        Top-k for recall / nDCG and for generation context window.
    gen_mode:
        Retrieval mode used by ``GenerationEvalRunner`` for generation
        metrics.  Defaults to ``RetrievalMode.HYBRID``.
    gen_fusion:
        Fusion method used by ``GenerationEvalRunner``.  Defaults to
        ``FusionMethod.RRF``.

    Raises
    ------
    ValueError
        If any strategy name is not recognised.
    """
    base = settings_base or Settings()

    unknown = set(strategies) - _VALID_STRATEGIES
    if unknown:
        raise ValueError(f"Unknown chunk strategies: {sorted(unknown)}")

    mode_names = [m.value for m in modes]
    report = MatrixReport(
        dataset=dataset.name,
        k=k,
        strategies=list(strategies),
        modes=mode_names,
    )

    for strategy in strategies:
        # Build a fresh system per strategy; chunk text differs across
        # strategies so embeddings cannot be reused.
        settings = Settings(
            **{
                **base.model_dump(),
                "chunk_strategy": strategy,
            }
        )
        system = build_system(settings)
        _ingest_corpus(system, corpus)

        # --- Retrieval metrics ---
        runner = EvalRunner(system.query)
        eval_report = runner.run(dataset, k=k, modes=modes)

        # --- Generation metrics (once per strategy, shared across modes) ---
        gen_runner = GenerationEvalRunner(system.query)
        gen_report = gen_runner.run(dataset, k=k, mode=gen_mode, fusion=gen_fusion)

        for mode in modes:
            mode_report = eval_report.modes.get(mode.value)
            if mode_report is None:
                continue
            report.entries.append(
                StrategyEntry(
                    strategy=strategy,
                    mode=mode.value,
                    mean_recall_at_k=mode_report.mean_recall_at_k,
                    mean_mrr=mode_report.mean_mrr,
                    mean_ndcg_at_k=mode_report.mean_ndcg_at_k,
                    mean_faithfulness=gen_report.mean_faithfulness,
                    mean_context_precision=gen_report.mean_context_precision,
                    mean_answer_correctness=gen_report.mean_answer_correctness,
                    mean_answer_relevance=gen_report.mean_answer_relevance,
                )
            )

    return report
