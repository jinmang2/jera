"""AblationRunner — proves the harness measures a GENUINE technique difference (non-tautological).

Corpus mirrors the M6 contextual-retrieval fixture: the answer chunk never names the entity the
query asks for, so plain indexing ranks it below a rival, while Contextual Retrieval (which
prepends the document title) lifts it to rank 1. The ablation must SURFACE this as a higher MRR /
context-precision for the contextual config than the baseline — a real, measured gap.
"""

from __future__ import annotations

from jera.config.settings import Profile, Settings
from jera.domain.document import MediaType
from jera.domain.retrieval import RetrievalMode
from jera.evaluation.ablation import AblationCase, AblationConfig, AblationRunner
from jera.evaluation.dataset_builder import CaseSpec

CORPUS: dict[str, tuple[MediaType, str]] = {
    "acme": (
        MediaType.MARKDOWN,
        "# Acme Corporation Annual Report\n\n## Management Discussion\n\n"
        "The outlook for the coming year is positive, with sustained expansion anticipated.\n",
    ),
    "globex": (
        MediaType.MARKDOWN,
        "# Globex Briefing\n\n## Summary\n\nThe outlook here is uncertain given headwinds.\n",
    ),
}
CASES = [
    AblationCase(
        CaseSpec("c1", "Acme outlook", "outlook for the coming year"),
        reference_answer="The outlook for the coming year is positive.",
    )
]


def _runner() -> AblationRunner:
    return AblationRunner(corpus=CORPUS, cases=CASES, k=3, mode=RetrievalMode.SPARSE)


def _report():
    return _runner().run(
        [
            AblationConfig("baseline", Settings(profile=Profile.TEST)),
            AblationConfig(
                "contextual", Settings(profile=Profile.TEST, use_contextual_retrieval=True)
            ),
        ]
    )


def test_ablation_measures_a_real_contextual_lift() -> None:
    report = _report()
    base = next(e for e in report.entries if e.name == "baseline")
    ctx = next(e for e in report.entries if e.name == "contextual")
    # Genuine, measured gap: the entity-less gold chunk is ranked #1 only with contextual retrieval.
    assert ctx.mean_mrr > base.mean_mrr
    assert ctx.mean_context_precision > base.mean_context_precision
    assert report.best_by("mean_mrr") == "contextual"


def test_ablation_report_structure() -> None:
    report = _report()
    assert {e.name for e in report.entries} == {"baseline", "contextual"}
    assert report.k == 3 and report.mode == "sparse"
    table = report.comparison_table()
    assert "baseline" in table and "contextual" in table and "recall@k" in table
    # all metric means are in valid ranges
    for e in report.entries:
        for v in (e.mean_recall_at_k, e.mean_mrr, e.mean_faithfulness, e.mean_claim_precision):
            assert 0.0 <= v <= 1.0


def test_ablation_handles_multiple_configs_including_chunk_strategy() -> None:
    # proposition changes chunk ids → gold must be rebuilt per config; assert it runs + scores.
    report = _runner().run(
        [
            AblationConfig("baseline", Settings(profile=Profile.TEST)),
            AblationConfig(
                "proposition", Settings(profile=Profile.TEST, chunk_strategy="proposition")
            ),
            AblationConfig("ctx_proc", Settings(profile=Profile.TEST, use_context_processing=True)),
        ]
    )
    assert len(report.entries) == 3
    # every config retrieves the gold chunk within k (recall 1.0) — the corpus is small
    assert all(e.mean_recall_at_k == 1.0 for e in report.entries)
