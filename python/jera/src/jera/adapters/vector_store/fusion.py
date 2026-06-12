"""Fusion functions with the determinism rules frozen by the golden-file contract.

Frozen rules (two implementations must produce byte-identical orderings):
  1. Rank is 1-based from each modality's score-sorted ranking (best = rank 1).
  2. RRF: score = Σ_modalities 1/(k + rank_i), k = 60.
  3. Missing-modality rule: a chunk absent from a modality contributes 0 (explicit, not 1/(k+∞)).
  4. DBSF: per-modality min-max normalization to [0,1] then sum across modalities; a modality
     whose scores are all equal contributes 0 (min-max undefined → 0, deterministically).
  5. Tie-break: equal fused scores break by chunk_id lexicographic ascending.
"""

from __future__ import annotations

RRF_K = 60


def _finalize(scores: dict[str, float]) -> list[tuple[str, float]]:
    # Rule 5: sort by score desc, then chunk_id asc (stable, deterministic).
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Fuse per-modality ranked id lists (best-first) via RRF. Rules 1-3."""
    scores: dict[str, float] = {}
    for ranked in rankings.values():
        for rank, chunk_id in enumerate(ranked, start=1):  # rule 1: 1-based
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)  # rule 2/3
    return _finalize(scores)


def distribution_based_score_fusion(
    scores_by_modality: dict[str, dict[str, float]],
) -> list[tuple[str, float]]:
    """Fuse per-modality {chunk_id: raw_score} maps via min-max + sum. Rules 3-4."""
    fused: dict[str, float] = {}
    for smap in scores_by_modality.values():
        if not smap:
            continue
        values = smap.values()
        lo, hi = min(values), max(values)
        rng = hi - lo
        for chunk_id, s in smap.items():
            norm = 0.0 if rng == 0 else (s - lo) / rng  # rule 4
            fused[chunk_id] = fused.get(chunk_id, 0.0) + norm  # rule 3: missing contributes 0
    return _finalize(fused)
