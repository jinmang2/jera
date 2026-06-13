"""Fusion functions with the determinism rules frozen by the golden-file contract.

Frozen rules (two implementations must produce byte-identical orderings):
  1. Rank is 1-based from each modality's score-sorted ranking (best = rank 1).
  2. RRF: score = Σ_modalities 1/(k + rank_i), k = 60 (Cormack et al., SIGIR 2009).
  3. Missing-modality rule: a chunk absent from a modality contributes 0 (explicit, not 1/(k+∞)).
  4. DBSF (Distribution-Based Score Fusion): per-modality 3-sigma normalization
     ``ŝ = (s − (μ − 3σ)) / (6σ)`` using the *sample* standard deviation, then sum across
     modalities. This mirrors the production Qdrant DBSF path exactly (Qdrant hybrid-queries
     docs; Mazzeschi 2024). Scores are NOT clamped to [0, 1]. A modality with fewer than two
     points or zero variance (all scores equal) emits the constant 0.5 per point rather than
     dividing by zero — matching Qdrant's documented behaviour.
  5. Tie-break: equal fused scores break by chunk_id lexicographic ascending.
"""

from __future__ import annotations

import math

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
    """Fuse per-modality {chunk_id: raw_score} maps via 3-sigma normalization + sum. Rules 3-4.

    Canonical Qdrant DBSF: ``ŝ = (s − (μ − 3σ)) / (6σ)`` with the *sample* standard deviation
    σ. Scores are not clamped. A modality with < 2 points or zero variance emits 0.5 per point
    (Qdrant's divide-by-zero guard). Missing chunks contribute 0 to the fused sum (rule 3).
    """
    fused: dict[str, float] = {}
    for smap in scores_by_modality.values():
        if not smap:
            continue
        values = list(smap.values())
        n = len(values)
        mean = sum(values) / n
        # Sample standard deviation (ddof=1), matching Qdrant's DBSF.
        variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
        sigma = math.sqrt(variance)
        lo = mean - 3.0 * sigma
        denom = 6.0 * sigma
        for chunk_id, s in smap.items():
            # Zero variance / single point -> 0.5 (rule 4 divide-by-zero guard).
            norm = 0.5 if denom == 0.0 else (s - lo) / denom
            fused[chunk_id] = fused.get(chunk_id, 0.0) + norm  # rule 3: missing contributes 0
    return _finalize(fused)
