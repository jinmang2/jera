"""Technique profiling — proves the profile DIFFERENTIATES technique niches (non-tautological).

The point of profiling across difficulty scenarios is that different techniques win on different
corpora. The test builds an "easy" corpus (everything ties → a technique adds no value) and an
"entity_less" corpus (the answer chunk never names the queried entity → Contextual Retrieval wins),
and asserts the profile reports DIFFERENT winners per scenario — a real, measured distinction, not
an assumed one.
"""

from __future__ import annotations

from jera.config.settings import Profile, Settings
from jera.domain.document import MediaType
from jera.domain.retrieval import RetrievalMode
from jera.evaluation.ablation import AblationCase, AblationConfig
from jera.evaluation.dataset_builder import CaseSpec
from jera.evaluation.profiling import CorpusScenario, run_profile

_EASY = CorpusScenario(
    name="easy",
    tag="easy",
    corpus={
        "d": (MediaType.MARKDOWN, "# Notes\n\nHybrid retrieval uses reciprocal rank fusion.\n")
    },
    cases=[AblationCase(CaseSpec("c", "what merges rankings?", "reciprocal rank fusion"))],
)
_ENTITY_LESS = CorpusScenario(
    name="entity_less",
    tag="entity_less",
    corpus={
        "acme": (
            MediaType.MARKDOWN,
            "# Acme Corporation Annual Report\n\n## Discussion\n\n"
            "The outlook for the coming year is positive.\n",
        ),
        "globex": (
            MediaType.MARKDOWN,
            "# Globex Briefing\n\n## Summary\n\nThe outlook is uncertain.\n",
        ),
    },
    cases=[AblationCase(CaseSpec("c", "Acme outlook", "outlook for the coming year"))],
)
_CONFIGS = [
    AblationConfig("baseline", Settings(profile=Profile.TEST)),
    AblationConfig("contextual", Settings(profile=Profile.TEST, use_contextual_retrieval=True)),
]


def _profile():
    return run_profile([_EASY, _ENTITY_LESS], _CONFIGS, k=3, mode=RetrievalMode.SPARSE)


def test_profile_reports_different_winners_per_scenario() -> None:
    report = _profile()
    winners = report.winners("mean_mrr")
    # Contextual wins exactly where it should (entity-less), and adds NO value on the easy corpus.
    assert winners["entity_less"] == "contextual"
    assert winners["easy"] == "baseline"  # tie → baseline; contextual did not beat it
    # The headline property: the profile genuinely DIFFERENTIATES — not one technique everywhere.
    assert len(set(winners.values())) >= 2


def test_strength_summary_maps_techniques_to_their_niche() -> None:
    summary = _profile().strength_summary("mean_mrr")
    assert summary["contextual"] == ["entity_less"]
    assert summary["baseline"] == ["easy"]


def test_profile_report_structure_and_table() -> None:
    report = _profile()
    assert set(report.scenarios) == {"easy", "entity_less"}
    assert report.k == 3 and report.mode == "sparse"
    table = report.profile_table("mean_context_precision")
    assert "entity_less" in table and "contextual" in table and "*" in table
