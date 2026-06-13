"""Instruction-tuned embedding wrapper (E5-Instruct / Qwen3-Embedding convention).

Modern instruction-tuned embedders (E5-instruct, gte-Qwen2-instruct, Qwen3-Embedding — #1 MTEB
multilingual, June 2025) steer retrieval with a natural-language task instruction prepended to
the QUERY only; documents are embedded as-is (an asymmetric convention). Changing the instruction
re-targets what the same query text retrieves.

This wraps ANY ``EmbeddingProvider`` and applies that convention. With a real instruction-tuned
base (Qwen3-Embedding via fastembed/ONNX) it is the production path; with the deterministic hash
embedding it is offline-testable — the instruction tokens genuinely shift the query's bag-of-tokens
vector, so the steering effect is real and provable in CI.

Prompt format (Qwen3 / E5): ``Instruct: {task}\nQuery: {text}``.
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.vectors import DenseVector
from jera.ports.embedding import EmbeddingProvider

DEFAULT_TASK = "Given a search query, retrieve relevant passages that answer it"


class InstructionEmbedding:
    """Wrap an EmbeddingProvider; prepend a task instruction to queries (documents unchanged).

    Parameters
    ----------
    base:
        The underlying EmbeddingProvider (hash in CI; an instruction-tuned model in prod).
    task:
        The natural-language retrieval instruction prepended to every query.
    """

    def __init__(self, base: EmbeddingProvider, task: str = DEFAULT_TASK) -> None:
        self._base = base
        self.task = task
        self.model_id: str = f"{base.model_id}-instruct"
        self.dimensions: int = base.dimensions
        self.context_limit: int | None = base.context_limit

    def _format_query(self, text: str) -> str:
        return f"Instruct: {self.task}\nQuery: {text}"

    def embed(self, texts: Sequence[str]) -> list[DenseVector]:
        """Embed documents as-is — the instruction convention is query-side only."""
        return self._base.embed(texts)

    def embed_query(self, text: str) -> DenseVector:
        """Embed the query with the task instruction prepended, steering retrieval."""
        return self._base.embed_query(self._format_query(text))
