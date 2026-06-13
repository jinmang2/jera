"""FollowupController port — decides the next retrieval query in an iterative loop.

This port is the "reasoning" step in the Search-R1 / IRCoT inference pattern:

    think → search → think → search → … → stop

After each retrieval round the controller inspects accumulated evidence and either
returns the next query string or returns ``None`` to signal STOP.

References
----------
* Search-R1 (Jin et al., 2025): arXiv:2503.09516
* A-RAG hierarchical retrieval (Li et al., 2026): arXiv:2602.03442
* IRCoT (Trivedi et al., 2022): https://arxiv.org/abs/2212.10509
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jera.domain.chunk import Chunk


@runtime_checkable
class FollowupController(Protocol):
    """Determines the next retrieval query given accumulated evidence.

    The controller is called once per completed retrieval round.  It receives:

    * ``original_query`` — the user's original (normalized) question, unchanged
      throughout the loop.
    * ``accumulated_chunks`` — all unique chunks gathered so far (across all
      previous rounds), in accumulation order (first-occurrence wins on dedup).
    * ``round_index`` — 0-based index of the round *just completed*.  On the
      first call this is ``0`` (after round 0 finished).

    Return value
    ------------
    * ``str`` — the query to use for the *next* retrieval round.
    * ``None`` — STOP: the loop terminates and the generator is called with the
      currently accumulated chunks.

    Implementations MUST be deterministic given the same inputs so that
    ``IterativeRetrievalPipeline`` behaves reproducibly in tests and CI.

    LLM-backed controllers (future opt-in)
    ---------------------------------------
    A controller that calls an LLM to generate the next query (the true IRCoT /
    Search-R1 "thinking" step) can implement this protocol by wrapping a
    ``GeneratorLLM``-style adapter.  It should be constructed only when cloud
    access is available and its ``next_query`` method should be gated behind
    ``settings.enable_cloud`` — the same guard pattern used by
    ``claude_generator.py`` and ``claude_hypothesis_llm.py``.
    """

    def next_query(
        self,
        original_query: str,
        accumulated_chunks: Sequence[Chunk],
        round_index: int,
    ) -> str | None:
        """Return the next retrieval query, or None to stop the loop."""
        ...
