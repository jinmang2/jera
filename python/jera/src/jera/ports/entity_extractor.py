"""EntityExtractor port.

Defines the contract for extracting entity strings from text. Implementations
range from deterministic regex heuristics (offline, CI-safe) to LLM-based
Open Information Extraction (opt-in, requires cloud credentials).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EntityExtractor(Protocol):
    """Returns a list of normalized (lowercased) entity strings found in *text*.

    Implementations must be deterministic given the same input so that graph
    indexing is reproducible across runs.
    """

    def extract(self, text: str) -> list[str]: ...
