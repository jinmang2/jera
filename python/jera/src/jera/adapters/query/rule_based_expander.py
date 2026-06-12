"""RuleBasedExpander — deterministic multi-query expansion, no LLM, CI-real.

Splits a compound query into its clauses so each sub-question can be retrieved independently and
the rankings fused. A query joined by a conjunction ("dense and sparse retrieval and how they
merge") dilutes any single chunk that answers only one part; retrieving for each clause and
RRF-merging surfaces all the parts. Non-compound queries expand to just themselves (a safe
no-op). Language-neutral conjunction set covers English + Korean.
"""

from __future__ import annotations

import re

# Conjunction / separator tokens that delimit independent clauses. Matched case-insensitively;
# Korean connectives are matched as substrings (no word boundaries in Korean script).
_EN_CONJ = re.compile(r"\s+(?:and|or|versus|vs\.?)\s+|[;,]\s*", re.IGNORECASE)
_KO_CONJ = re.compile(r"\s*(?:그리고|또는|및|이랑|랑)\s*")


class RuleBasedExpander:
    """Expand a compound query into [original, *clauses] (deterministic)."""

    strategy = "rule_based"
    version = "1.0"

    def __init__(self, *, min_clause_chars: int = 3) -> None:
        # Clauses shorter than this are noise (stray "or", punctuation) and are dropped.
        self._min_clause_chars = min_clause_chars

    def transform(self, query: str) -> list[str]:
        original = query.strip()
        variants: list[str] = [original] if original else []
        for piece in _split_clauses(original):
            clause = piece.strip()
            if len(clause) >= self._min_clause_chars and clause not in variants:
                variants.append(clause)
        return variants or [query]


def _split_clauses(text: str) -> list[str]:
    # Apply both conjunction patterns; only treat as compound if a split actually happens.
    parts = _EN_CONJ.split(text)
    out: list[str] = []
    for part in parts:
        out.extend(_KO_CONJ.split(part))
    # If neither pattern split anything, `out` is just [text] → not compound, no extra clauses.
    return out if len(out) > 1 else []
