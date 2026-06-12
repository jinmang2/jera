"""Parser/OCR quality metrics — pure deterministic functions (stdlib only, no scipy).

Graded against independent gold labels (never the parser's own output). Used by the CI
plumbing-check bench and the M5b opt-in real-engine benchmark.
"""

from __future__ import annotations

from collections.abc import Sequence


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate = edit_distance(ref, hyp) / max(len(ref), 1). 0.0 = perfect.

    Standard CER: may exceed 1.0 when the hypothesis is much longer than the reference.
    """
    return _levenshtein(reference, hypothesis) / max(len(reference), 1)


TableCell = tuple[int, int, str]  # (row, col, normalized cell text)


def table_f1(predicted: Sequence[TableCell], gold: Sequence[TableCell]) -> float:
    """F1 over the set of (row, col, cell-text) triples. 1.0 = exact cell-set match."""
    pred_set, gold_set = set(predicted), set(gold)
    if not pred_set and not gold_set:
        return 1.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def element_type_accuracy(predicted: dict[str, str], gold: dict[str, str]) -> float:
    """Fraction of gold elements (keyed by element_id) whose predicted type matches.

    Gold ids absent from ``predicted`` count as misses. 1.0 if gold is empty.
    """
    if not gold:
        return 1.0
    hits = sum(1 for eid, gtype in gold.items() if predicted.get(eid) == gtype)
    return hits / len(gold)


def reading_order_score(predicted_order: Sequence[str], gold_order: Sequence[str]) -> float:
    """Kendall tau over the COMMON-id intersection, rescaled to [0,1] (1.0 = same order).

    Defined as 1.0 when ≤1 shared element (no pair to compare) — never throws on count mismatch.
    """
    common = [eid for eid in gold_order if eid in set(predicted_order)]
    if len(common) <= 1:
        return 1.0
    pred_rank = {eid: i for i, eid in enumerate(predicted_order)}
    gold_rank = {eid: i for i, eid in enumerate(gold_order)}
    concordant = discordant = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            a, b = common[i], common[j]
            gp = gold_rank[a] - gold_rank[b]
            pp = pred_rank[a] - pred_rank[b]
            if gp * pp > 0:
                concordant += 1
            elif gp * pp < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    tau = (concordant - discordant) / total  # in [-1, 1]
    return (tau + 1.0) / 2.0  # rescale to [0, 1]
