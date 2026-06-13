"""Technique profiling — run the ablation across MULTIPLE corpora/difficulty scenarios.

A single ablation answers "which config wins on this corpus?". A *profile* answers the more
useful question: "which technique helps on WHAT KIND of corpus?" — by running the same set of
configurations across several labeled scenarios (e.g. an easy corpus where everything ties, an
entity-less corpus where Contextual Retrieval wins, a multi-fact corpus where proposition chunking
wins) and summarizing, per configuration, which scenarios it wins.

Reuses ``AblationRunner`` per scenario; fully offline-deterministic under the test profile.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from jera.domain.document import MediaType
from jera.domain.retrieval import RetrievalMode
from jera.evaluation.ablation import AblationCase, AblationConfig, AblationReport, AblationRunner


@dataclass(frozen=True)
class CorpusScenario:
    """A named evaluation scenario: a corpus + its questions + a difficulty/characteristic tag."""

    name: str
    tag: str  # e.g. "easy", "entity_less", "multi_fact", "redundant"
    corpus: dict[str, tuple[MediaType, str]]
    cases: Sequence[AblationCase]


class ProfileReport(BaseModel):
    """Per-scenario ablation reports + cross-scenario technique-strength summaries."""

    model_config = {"frozen": True}

    k: int
    mode: str
    scenarios: dict[str, AblationReport]  # scenario name → its AblationReport

    def winners(self, metric: str = "mean_mrr") -> dict[str, str]:
        """Best config name per scenario, by ``metric``."""
        return {name: report.best_by(metric) for name, report in self.scenarios.items()}

    def strength_summary(self, metric: str = "mean_mrr") -> dict[str, list[str]]:
        """Per config: the scenarios it wins (by ``metric``). Reveals each technique's niche."""
        summary: dict[str, list[str]] = {}
        for scenario_name, winner in self.winners(metric).items():
            summary.setdefault(winner, []).append(scenario_name)
        return summary

    def profile_table(self, metric: str = "mean_mrr") -> str:
        """A scenario × config matrix of one metric, with the winner per row starred."""
        configs = [e.name for e in next(iter(self.scenarios.values())).entries]
        header = f"{'scenario':<16}" + "".join(f"{c:>14}" for c in configs)
        rows = [header, f"-- metric: {metric} --"]
        for scenario_name, report in self.scenarios.items():
            by_name = {e.name: getattr(e, metric) for e in report.entries}
            best = report.best_by(metric)
            cells = "".join(f"{by_name[c]:>13.3f}" + ("*" if c == best else " ") for c in configs)
            rows.append(f"{scenario_name:<16}{cells}")
        return "\n".join(rows)


def run_profile(
    scenarios: Sequence[CorpusScenario],
    configs: Sequence[AblationConfig],
    *,
    k: int = 5,
    mode: RetrievalMode = RetrievalMode.HYBRID,
) -> ProfileReport:
    """Run every configuration across every scenario; collect per-scenario ablation reports."""
    reports = {
        scenario.name: AblationRunner(
            corpus=scenario.corpus, cases=scenario.cases, k=k, mode=mode
        ).run(configs)
        for scenario in scenarios
    }
    return ProfileReport(k=k, mode=mode.value, scenarios=reports)
