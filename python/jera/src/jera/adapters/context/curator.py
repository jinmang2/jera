"""RedundancyCurator — greedy near-duplicate removal via token-Jaccard similarity.

Research basis
--------------
AdaGReS (Dec 2025, arXiv:2512.25052) demonstrated that 30–40 % of retrieved context
in production RAG systems is semantically redundant, and that greedy chunk selection
under a relevance-minus-redundancy objective (with adaptive lambda) consistently
outperforms fixed-lambda MMR on NQ and biomedical QA.

This adapter implements the simpler, standalone deduplication half of that idea:
a keep-first greedy pass that drops any chunk whose token-Jaccard similarity to an
already-accepted chunk meets or exceeds a configurable threshold.

Token-Jaccard is defined as:

    J(A, B) = |tokens(A) ∩ tokens(B)| / |tokens(A) ∪ tokens(B)|

where tokens are whitespace-split, lower-cased words.  This is sufficient to
detect near-duplicate retrieved passages without any external model.

The relative order of kept chunks is preserved (same as their input order), so
a preceding reranker's relevance ranking is maintained for kept items.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.chunk import Chunk


def _token_set(text: str) -> frozenset[str]:
    """Return the lower-cased whitespace-token set of *text*."""
    return frozenset(text.lower().split())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Token-Jaccard similarity; returns 0.0 when both sets are empty."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class RedundancyCurator:
    """Drop near-duplicate chunks via greedy keep-first token-Jaccard filtering.

    A chunk is dropped if its token-Jaccard similarity to **any** already-kept
    chunk is ≥ ``threshold``.  The first occurrence is always retained; later
    near-duplicates are discarded.  Relative order of kept chunks is preserved.

    Parameters
    ----------
    threshold:
        Jaccard similarity at or above which a chunk is considered a near-duplicate
        and dropped.  Default 0.8 matches the AdaGReS-inspired heuristic.
        Must be in (0.0, 1.0].
    """

    name: str = "redundancy_curator"

    def __init__(self, threshold: float = 0.8) -> None:
        if not (0.0 < threshold <= 1.0):
            raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")
        self.threshold = threshold

    def process(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]:  # noqa: ARG002
        """Return *chunks* with near-duplicates removed.

        Parameters
        ----------
        query:
            Unused by this stage; present for Protocol compatibility.
        chunks:
            Chunks in any order.  Typically descending relevance order so the
            best chunk of each near-duplicate cluster is the one retained.

        Returns
        -------
        list[Chunk]
            De-duplicated list.  Each retained chunk is identical to the input
            (no field is mutated).
        """
        kept: list[Chunk] = []
        kept_tokens: list[frozenset[str]] = []

        for chunk in chunks:
            tokens = _token_set(chunk.text)
            redundant = any(
                _jaccard(tokens, kept_tok) >= self.threshold for kept_tok in kept_tokens
            )
            if not redundant:
                kept.append(chunk)
                kept_tokens.append(tokens)

        return kept
