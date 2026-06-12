"""HydeTransformer — HyDE (Hypothetical Document Embeddings) query expansion (opt-in).

HyDE (Gao et al., 2022): ask an LLM to write a hypothetical answer to the query, then retrieve
using that answer instead of (here: in addition to) the question — the hypothetical shares far
more vocabulary with real answer passages than the question does. The LLM is injected behind the
``HypothesisLLM`` protocol so this adapter is offline-testable with a deterministic fake; the
real Claude implementation (``ClaudeHypothesisLLM``) is built only with cloud enabled + a key.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class HypothesisLLM(Protocol):
    """Writes a short hypothetical answer passage for a query."""

    model_id: str

    def hypothesize(self, query: str) -> str: ...


class HydeTransformer:
    """Expand a query into [original, hypothetical-answer] via an injected ``HypothesisLLM``."""

    strategy = "hyde"
    version = "1.0"

    def __init__(self, llm: HypothesisLLM) -> None:
        self._llm = llm

    def transform(self, query: str) -> list[str]:
        original = query.strip()
        hypothesis = self._llm.hypothesize(original).strip()
        variants = [original] if original else []
        if hypothesis and hypothesis not in variants:
            variants.append(hypothesis)
        return variants or [query]
