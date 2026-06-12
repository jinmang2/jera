"""Deterministic ID helpers.

All IDs are content/identity-derived so that re-ingesting the same source with the
same strategy yields identical IDs across runs (required by the chunking-stability gate).
"""

from __future__ import annotations

import hashlib

_SEP = "\x1f"  # unit separator, unlikely to appear in inputs


def stable_id(*parts: str, length: int = 16) -> str:
    """Return a deterministic short hex id derived from ``parts``.

    The same parts in the same order always produce the same id; different parts
    (or order) produce a different id with overwhelming probability.
    """
    digest = hashlib.sha256(_SEP.join(parts).encode("utf-8")).hexdigest()
    return digest[:length]
