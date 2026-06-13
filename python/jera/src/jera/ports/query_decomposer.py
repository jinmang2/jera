"""QueryDecomposer port.

Sub-question decomposition for multi-hop RAG (Pereira et al., ACL-SRW 2025, arXiv:2507.00355):
instead of issuing a single compound query, split it into an *ordered* sequence of
sub-questions, retrieve for each in turn, accumulate unique chunks across all hops, then
generate a single answer from the union.

This is distinct from multi-query (parallel RRF variants): decomposition is SEQUENTIAL — each
sub-question can depend on the answer chunk from the previous hop (the "bridge entity" pattern).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class QueryDecomposer(Protocol):
    """Splits a query into an ordered list of sub-questions for sequential retrieval.

    Contract:
    - A non-compound query MUST return ``[query]`` unchanged (safe no-op).
    - A multi-hop query returns ``[sub_q1, sub_q2, ...]`` in dependency order.
    - The list is never empty.
    """

    strategy: str
    version: str

    def decompose(self, query: str) -> list[str]:
        """Return ordered sub-questions; a simple query returns ``[query]`` unchanged."""
        ...
