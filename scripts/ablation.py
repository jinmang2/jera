"""Runnable technique-ablation: which RAG configuration actually wins on this corpus?

    uv run python scripts/ablation.py

Builds a small corpus + question set, then scores several named configurations (baseline,
contextual retrieval, proposition chunking, multi-query, listwise rerank, context processing)
through the eval harness — retrieval (recall/MRR/nDCG), RAGAS-lite generation, and claim-level
(RAGChecker) metrics — and prints a comparison table plus the winner per metric. Fully offline
(test profile); the same harness measures real models without code changes.
"""

from __future__ import annotations

from jera.config.settings import Profile, Settings
from jera.domain.document import MediaType
from jera.domain.retrieval import RetrievalMode
from jera.evaluation.ablation import AblationCase, AblationConfig, AblationRunner
from jera.evaluation.dataset_builder import CaseSpec

CORPUS: dict[str, tuple[MediaType, str]] = {
    "handbook": (
        MediaType.MARKDOWN,
        "# Acme Handbook\n\n## Retrieval\n"
        "The outlook for hybrid retrieval is strong; it merges dense and sparse rankings via "
        "reciprocal rank fusion.\n\n## Ranking\n"
        "The reranking module identifier is ZX9000 and it runs after first-stage recall.\n\n"
        "## Storage\nPostgres owns documents and chunks. Qdrant owns dense and sparse vectors.\n",
    ),
}
CASES = [
    AblationCase(
        CaseSpec("fusion", "how are dense and sparse rankings merged?", "reciprocal rank fusion"),
        reference_answer="Hybrid retrieval merges dense and sparse rankings via RRF.",
    ),
    AblationCase(
        CaseSpec("identifier", "ZX9000", "ZX9000"),
        reference_answer="The reranking module identifier is ZX9000.",
    ),
    AblationCase(
        CaseSpec("storage", "where are vectors stored?", "Qdrant owns dense and sparse"),
        reference_answer="Qdrant owns the dense and sparse vectors.",
    ),
]

CONFIGS = [
    AblationConfig("baseline", Settings(profile=Profile.TEST)),
    AblationConfig("contextual", Settings(profile=Profile.TEST, use_contextual_retrieval=True)),
    AblationConfig("proposition", Settings(profile=Profile.TEST, chunk_strategy="proposition")),
    AblationConfig("multi_query", Settings(profile=Profile.TEST, use_query_transform=True)),
    AblationConfig("listwise", Settings(profile=Profile.TEST, reranker_kind="listwise")),
    AblationConfig("ctx_processing", Settings(profile=Profile.TEST, use_context_processing=True)),
]


def main() -> None:
    runner = AblationRunner(corpus=CORPUS, cases=CASES, k=5, mode=RetrievalMode.HYBRID)
    report = runner.run(CONFIGS)
    print(
        f"== ablation ({len(CONFIGS)} configs, {len(CASES)} cases, k={report.k}, "
        f"mode={report.mode}) ==\n"
    )
    print(report.comparison_table())
    print("\n== winner per metric ==")
    for metric in ("mean_recall_at_k", "mean_mrr", "mean_ndcg_at_k", "mean_context_precision"):
        print(f"  {metric:<24} {report.best_by(metric)}")


if __name__ == "__main__":
    main()
