"""Run the dense/sparse/hybrid × chunking strategy matrix on the Korean eval dataset.

This script is NOT run by CI.  It requires:
  - ``uv sync --extra local``          (fastembed multilingual models)
  - ``JERA_PROFILE=local``             (set via env or --profile flag)
  - PDF files present under ``data/corpus/`` (verify with ``scripts/fetch_corpus.py``)

Outputs ``docs/eval/korean_research_results.md`` with a strategy × mode metric
table plus at least one narrative observation derived from the numbers.

Usage::

    JERA_PROFILE=local uv run python scripts/eval_local_matrix.py \\
        [--dataset data/eval/korean_research.json] \\
        [--manifest data/corpus/manifest.json] \\
        [--output docs/eval/korean_research_results.md] \\
        [--k 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jera.evaluation.matrix import MatrixReport


def _load_corpus(
    manifest_path: Path,
) -> dict[str, tuple]:
    """Load PDF files listed in the manifest into a corpus dict.

    Returns a dict mapping source_id → (MediaType.PDF, bytes).
    Skips entries whose PDF is not present locally.
    """
    from jera.domain.document import MediaType

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    entries: list[dict[str, str]] = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_dir = manifest_path.parent
    corpus: dict[str, tuple] = {}

    for entry in entries:
        filename = entry.get("filename", "")
        if not filename:
            continue
        pdf_path = corpus_dir / filename
        if not pdf_path.exists():
            print(f"  [SKIP] PDF not present, skipping: {filename}")
            continue
        corpus[filename] = (MediaType.PDF, pdf_path.read_bytes())
        print(f"  [LOADED] {filename}")

    return corpus


def _narrative_observation(report: MatrixReport) -> str:
    """Derive at least one observation from the matrix numbers."""
    lines: list[str] = []

    # Find the best (strategy, mode) cell by recall@k.
    best_entry = max(report.entries, key=lambda e: e.mean_recall_at_k, default=None)
    if best_entry is not None:
        lines.append(
            f"- Best recall@{report.k}: **{best_entry.strategy}** strategy with "
            f"**{best_entry.mode}** mode "
            f"(recall={best_entry.mean_recall_at_k:.3f}, "
            f"MRR={best_entry.mean_mrr:.3f}, "
            f"nDCG={best_entry.mean_ndcg_at_k:.3f})."
        )

    # Compare hybrid vs dense across strategies.
    for strategy in report.strategies:
        hybrid = report.get(strategy, "hybrid")
        dense = report.get(strategy, "dense")
        if hybrid and dense:
            diff = hybrid.mean_recall_at_k - dense.mean_recall_at_k
            if abs(diff) >= 0.01:
                direction = "higher" if diff > 0 else "lower"
                lines.append(
                    f"- Under **{strategy}**, hybrid recall is {abs(diff):.3f} "
                    f"{direction} than dense."
                )

    # Note which chunking strategy is most consistent (lowest spread across modes).
    spreads: dict[str, float] = {}
    for strategy in report.strategies:
        recalls = [e.mean_recall_at_k for e in report.entries if e.strategy == strategy]
        if len(recalls) > 1:
            spreads[strategy] = max(recalls) - min(recalls)
    if spreads:
        most_consistent = min(spreads, key=lambda s: spreads[s])
        lines.append(
            f"- **{most_consistent}** shows the most consistent recall across "
            f"retrieval modes (spread={spreads[most_consistent]:.3f})."
        )

    return "\n".join(lines) if lines else "- No significant differences observed across strategies."


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run dense/sparse/hybrid × chunking matrix eval on Korean dataset (local profile)."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/korean_research.json"),
        help="Path to the eval dataset JSON (default: data/eval/korean_research.json).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/corpus/manifest.json"),
        help="Path to corpus manifest.json (default: data/corpus/manifest.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/eval/korean_research_results.md"),
        help="Output markdown path (default: docs/eval/korean_research_results.md).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Top-k for recall / nDCG (default: 5).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Override JERA_PROFILE (default: reads env JERA_PROFILE, falls back to 'local').",
    )
    args = parser.parse_args()

    # Resolve profile — env var takes precedence over flag for consistency with
    # the Settings env_prefix="JERA_" convention; flag is a convenience override.
    import os

    profile_str = args.profile or os.environ.get("JERA_PROFILE", "local")
    if profile_str not in ("local", "test"):
        print(
            f"WARNING: profile={profile_str!r} — matrix eval is designed for 'local' "
            "(fastembed multilingual). Running anyway.",
            file=sys.stderr,
        )

    # Import jera components here so import errors surface with a clear message.
    try:
        from jera.config.settings import Settings
        from jera.domain.retrieval import RetrievalMode
        from jera.evaluation.matrix import run_matrix
        from jera.evaluation_contracts.dataset import EvalDataset
    except ImportError as exc:
        print(f"ERROR: {exc}\nMake sure you ran: uv sync --extra local", file=sys.stderr)
        return 1

    # Load dataset.
    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 1

    raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    dataset = EvalDataset.model_validate(raw)
    print(f"Loaded {len(dataset.cases)} eval cases from {args.dataset}")

    # Load corpus PDFs.
    print(f"\nLoading corpus from {args.manifest} ...")
    corpus = _load_corpus(args.manifest)
    if not corpus:
        print("ERROR: no PDFs loaded — run scripts/fetch_corpus.py first.", file=sys.stderr)
        return 1
    print(f"Corpus: {len(corpus)} document(s)\n")

    # Build settings for the chosen profile.
    settings_base = Settings(profile=profile_str)  # type: ignore[arg-type]

    strategies = ["heading_aware", "semantic", "hierarchical"]
    modes = [RetrievalMode.DENSE, RetrievalMode.SPARSE, RetrievalMode.HYBRID]

    print(f"Running matrix: {strategies} × {[m.value for m in modes]} ...")
    for strategy in strategies:
        print(f"  strategy={strategy} ...")

    report = run_matrix(
        dataset,
        corpus=corpus,
        strategies=strategies,
        modes=modes,
        settings_base=settings_base,
        k=args.k,
    )

    # Compose output document.
    matrix_md = report.to_markdown()
    observations = _narrative_observation(report)

    output_md = f"""{matrix_md}
## Observations

{observations}

---
*Generated by `scripts/eval_local_matrix.py` — profile: `{profile_str}`, k={args.k}*
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_md, encoding="utf-8")
    print(f"\nWrote results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
