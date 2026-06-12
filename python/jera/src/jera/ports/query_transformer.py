"""QueryTransformer port.

Multi-query retrieval: instead of retrieving for a single query string, expand the query into
several variants (sub-questions, a keyword form, or a HyDE hypothetical answer), retrieve for
each, and fuse the rankings (RRF). A chunk that the original phrasing buries can surface via a
variant. The transformer returns the variants to retrieve for; the original is always included
first so behavior degrades gracefully to single-query when no useful variant exists.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class QueryTransformer(Protocol):
    """Expands one query string into an ordered, de-duplicated list of query variants."""

    strategy: str
    version: str

    def transform(self, query: str) -> list[str]:
        """Return query variants to retrieve for, original first, never empty."""
        ...
