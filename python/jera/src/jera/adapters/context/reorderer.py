"""LostInTheMiddleReorderer — context reordering to counter the "lost in the middle" effect.

Research basis
--------------
Liu et al. (2023) "Lost in the Middle: How Language Models Use Long Contexts" showed that
LLMs attend most strongly to tokens at the beginning and end of the context window due to
RoPE long-term decay.  Information placed in the middle is systematically under-utilised,
causing 20 %+ accuracy drops (GPT-3.5-Turbo benchmark).

The effect was confirmed still present in 2025–2026 by LongBench-v2, HELMET, RULER (17
models), and the "Attention Basin" work (arXiv:2508.05128, Aug 2025), which proposes
mapping documents to the empirically highest-attention positions rather than naively placing
them first and last.

This adapter implements the alternating-edges interleave: rank-1 → position 0 (first),
rank-2 → position -1 (last), rank-3 → position 1, rank-4 → position -2, … matching
LangChain's ``LongContextReorder`` and the original Liu et al. fix.

The fix is zero-dependency, fully deterministic, and the highest-ROI context-engineering
intervention per token budget.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.chunk import Chunk


class LostInTheMiddleReorderer:
    """Reorder chunks so the highest-ranked items land at the edges of the context.

    Input chunks must be in descending relevance order (best first), as produced
    by the reranker.  The reorderer places:

    * rank 1  → position 0   (start of context)
    * rank 2  → position -1  (end of context)
    * rank 3  → position 1
    * rank 4  → position -2
    * …

    This alternating-edge interleave ensures the generator's primacy and recency
    attention peaks both fall on highly-relevant material, while the lowest-ranked
    chunks occupy the middle positions where attention is weakest.

    Provenance (``chunk_id``, ``char_span``, etc.) is never modified — only the
    order of the list changes.
    """

    name: str = "lost_in_the_middle_reorderer"

    def process(self, query: str, chunks: Sequence[Chunk]) -> list[Chunk]:  # noqa: ARG002
        """Reorder *chunks* by alternating from both edges inward.

        Parameters
        ----------
        query:
            Unused by this stage; present for Protocol compatibility.
        chunks:
            Chunks in descending relevance order (best first).

        Returns
        -------
        list[Chunk]
            Same chunks in edge-biased order.  The list length is unchanged.
        """
        n = len(chunks)
        if n <= 1:
            return list(chunks)

        result: list[Chunk | None] = [None] * n
        left = 0
        right = n - 1
        for i, chunk in enumerate(chunks):
            if i % 2 == 0:
                result[left] = chunk
                left += 1
            else:
                result[right] = chunk
                right -= 1

        # All slots are filled by construction (left and right converge to the same
        # midpoint after n steps covering all indices).
        return result  # type: ignore[return-value]
