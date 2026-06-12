"""Runnable demo evaluation on a small fixture corpus.

    uv run python scripts/eval.py

Builds a deterministic gold dataset, runs dense vs sparse vs hybrid retrieval through the
configured profile (default `test`), and prints a metric comparison table.
"""

from __future__ import annotations

from jera.rag import CaseSpec, EvalRunner, MediaType, Settings, build_gold_dataset, build_system

CORPUS = {
    "handbook": (
        MediaType.MARKDOWN,
        """# Jera Handbook

## Retrieval
Hybrid retrieval uses reciprocal rank fusion to merge dense and sparse rankings.

## Ranking
The reranking module identifier is ZX9000 and it runs after first-stage recall.

## Storage
Postgres owns documents and chunks. Qdrant owns dense and sparse named vectors.
""",
    ),
}

CASES = [
    CaseSpec("fusion", "how are dense and sparse rankings merged?", "reciprocal rank fusion"),
    CaseSpec("identifier", "ZX9000", "ZX9000"),
    CaseSpec("storage", "where are vectors stored?", "Qdrant owns dense and sparse"),
]


def main() -> None:
    system = build_system(Settings())
    dataset = build_gold_dataset(system, name="demo", documents=CORPUS, cases=CASES)
    report = EvalRunner(system.query).run(dataset, k=5)
    print(f"dataset={report.dataset!r}  k={report.k}  cases={len(dataset.cases)}\n")
    print(report.summary_table())
    print(f"\nbest mode by recall@{report.k}: {report.best_mode_by_recall()}")


if __name__ == "__main__":
    main()
