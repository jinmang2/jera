"""Runnable technique profiling — which technique wins on WHAT kind of corpus?

    uv run python scripts/profile.py

Runs several named configurations across difficulty scenarios (easy / entity-less / multi-fact)
and prints, per metric, a scenario × config matrix plus a per-technique strength summary. Fully
offline (test profile); the same harness measures real models without code changes.
"""

from __future__ import annotations

from jera.config.settings import Profile, Settings
from jera.domain.document import MediaType
from jera.domain.retrieval import RetrievalMode
from jera.evaluation.ablation import AblationCase, AblationConfig
from jera.evaluation.dataset_builder import CaseSpec
from jera.evaluation.profiling import CorpusScenario, run_profile

SCENARIOS = [
    CorpusScenario(
        name="easy",
        tag="easy",
        corpus={
            "d": (MediaType.MARKDOWN, "# Notes\n\nHybrid retrieval uses reciprocal rank fusion.\n")
        },
        cases=[AblationCase(CaseSpec("c", "what merges rankings?", "reciprocal rank fusion"))],
    ),
    CorpusScenario(
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
    ),
    CorpusScenario(
        name="multi_fact",
        tag="multi_fact",
        corpus={
            "geo": (
                MediaType.MARKDOWN,
                "# Geography\n\nThe Eiffel Tower is 330 metres tall. "
                "The Seine river flows through Paris. Mont Blanc is the highest alp.\n",
            )
        },
        cases=[AblationCase(CaseSpec("c", "how tall is the Eiffel Tower?", "330 metres tall"))],
    ),
]

CONFIGS = [
    AblationConfig("baseline", Settings(profile=Profile.TEST)),
    AblationConfig("contextual", Settings(profile=Profile.TEST, use_contextual_retrieval=True)),
    AblationConfig("proposition", Settings(profile=Profile.TEST, chunk_strategy="proposition")),
]


def main() -> None:
    report = run_profile(SCENARIOS, CONFIGS, k=5, mode=RetrievalMode.SPARSE)
    for metric in ("mean_mrr", "mean_context_precision"):
        print(report.profile_table(metric))
        print()
    print("== technique strengths (by mean_mrr): which scenarios each config wins ==")
    for config, scenarios in report.strength_summary("mean_mrr").items():
        print(f"  {config:<14} {', '.join(scenarios)}")


if __name__ == "__main__":
    main()
