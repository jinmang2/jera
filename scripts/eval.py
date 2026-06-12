"""Runnable demo evaluation on a small fixture corpus.

    uv run python scripts/eval.py

Builds a deterministic gold dataset, runs dense vs sparse vs hybrid retrieval through the
configured profile (default `test`), prints a retrieval metric comparison table, then grades
the generated-answer path with RAGAS-lite generation metrics.
"""

from __future__ import annotations

from jera.evaluation_contracts.dataset import EvalDataset
from jera.rag import (
    CaseSpec,
    EvalRunner,
    GenerationEvalRunner,
    MediaType,
    Settings,
    build_gold_dataset,
    build_system,
)

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

# Reference answers (by case_id) opt cases into answer_correctness + answer_relevance grading.
REFERENCES = {
    "fusion": "Hybrid retrieval uses reciprocal rank fusion to merge dense and sparse rankings.",
    "identifier": "The reranking module identifier is ZX9000.",
    "storage": "Qdrant owns the dense and sparse named vectors.",
}


def main() -> None:
    system = build_system(Settings())
    dataset = build_gold_dataset(system, name="demo", documents=CORPUS, cases=CASES)

    report = EvalRunner(system.query).run(dataset, k=5)
    print(f"dataset={report.dataset!r}  k={report.k}  cases={len(dataset.cases)}\n")
    print("== retrieval ==")
    print(report.summary_table())
    print(f"\nbest mode by recall@{report.k}: {report.best_mode_by_recall()}")

    # Attach reference answers, then grade the generated-answer path (RAGAS-lite).
    graded = EvalDataset(
        name=dataset.name,
        cases=[
            c.model_copy(update={"reference_answer": REFERENCES.get(c.case_id)})
            for c in dataset.cases
        ],
    )
    gen = GenerationEvalRunner(system.query).run(graded, k=5)
    print("\n== generation (RAGAS-lite) ==")
    print(gen.summary_table())


if __name__ == "__main__":
    main()
