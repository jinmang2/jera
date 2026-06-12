"""GeneratorLLM port."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from jera.domain.answer import Answer
from jera.domain.chunk import Chunk


@runtime_checkable
class GeneratorLLM(Protocol):
    """Produces a cited answer from a query and retrieved context chunks."""

    model_id: str

    def generate(self, query: str, contexts: Sequence[Chunk]) -> Answer: ...
