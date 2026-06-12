"""Tool-augmented generator — wraps a ToolUseRuntime to answer numeric/table questions.

Two entry points
----------------
``generate(query, contexts) -> Answer``
    Port-conformant ``GeneratorLLM`` method.  Citations are *chunk* citations only —
    the calculator is never cited as a chunk, so the citation-resolution assertion in
    ``query.py`` is preserved.  The numeric result is embedded in the answer text but
    is NOT added as a field on the frozen ``Answer`` model.

``run(query, contexts) -> RunResult``
    Typed numeric path.  Returns the ``ToolUseRuntime``'s ``RunResult`` directly so
    callers (e.g. ``evaluation/computation.py``) can read ``result.final_value`` as a
    ``float | None`` without regex-extracting it from prose (Principle 3: ground truth
    independent of the LLM).

Design notes
------------
- ``generate`` delegates to ``run`` and wraps the result in an ``Answer``.
- Chunk context is injected into the query string so the LLM sees the retrieved
  passages inside the tool-use conversation.
- Only chunk citations are emitted; the calculator tool is a helper, not a source.
- The generator is disabled-by-default when a real paid LLM is supplied; pass
  ``enabled=True`` to unlock (same pattern as ``ClaudeGenerator``).
"""

from __future__ import annotations

from collections.abc import Sequence

from jera.domain.answer import Answer, Citation
from jera.domain.chunk import Chunk
from jera.tooluse.llm import ToolUseLLM
from jera.tooluse.runtime import RunResult, ToolUseRuntime
from jera.tooluse.tools import Tool

_SNIPPET_CHARS = 240


def _snippet(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= _SNIPPET_CHARS:
        return text
    return text[:_SNIPPET_CHARS].rstrip() + "…"


def _build_context_prompt(query: str, contexts: Sequence[Chunk]) -> str:
    """Embed retrieved chunks into the query string for the tool-use LLM."""
    if not contexts:
        return query
    passages = "\n\n".join(
        f"[{i + 1}] (chunk {c.chunk_id})\n{c.text}" for i, c in enumerate(contexts)
    )
    return (
        "Answer the following question using the numbered passages below. "
        "Use the calculator tool when numeric computation is needed.\n\n"
        f"Passages:\n{passages}\n\nQuestion: {query}"
    )


class ToolAugmentedGenerator:
    """GeneratorLLM port adapter backed by a transparent tool-use loop.

    Parameters
    ----------
    llm:
        Any ``ToolUseLLM`` — ``FakeToolUseLLM`` for offline tests,
        ``ClaudeToolUseGenerator`` for real paid calls.
    tools:
        Tools to register in the runtime (default: empty — caller supplies).
    max_rounds:
        Safety cap passed to ``ToolUseRuntime``.

    Example (offline, zero API calls)::

        from jera.tooluse import FakeToolUseLLM, CalculatorTool
        from jera.adapters.generator.tool_augmented_generator import ToolAugmentedGenerator

        gen = ToolAugmentedGenerator(
            llm=FakeToolUseLLM(mode="single"),
            tools=[CalculatorTool()],
        )
        result = gen.run("What is 1+1?", contexts=[])
        assert result.final_value == 2.0
    """

    model_id: str  # set from llm.model_id in __init__

    def __init__(
        self,
        llm: ToolUseLLM,
        tools: list[Tool] | None = None,
        max_rounds: int = 10,
    ) -> None:
        self._llm = llm
        self._tools: list[Tool] = tools if tools is not None else []
        self._max_rounds = max_rounds
        self._runtime = ToolUseRuntime(
            llm=llm,
            tools=self._tools,
            max_rounds=max_rounds,
        )
        # Satisfy the GeneratorLLM port's ``model_id`` attribute.
        self.model_id = llm.model_id

    # ------------------------------------------------------------------
    # Typed numeric path (called by computation eval)
    # ------------------------------------------------------------------

    def run(self, query: str, contexts: Sequence[Chunk]) -> RunResult:
        """Execute the tool-use loop and return a typed ``RunResult``.

        ``result.final_value`` carries the numeric answer as a ``float | None``
        so callers never need to regex-extract numbers from prose.
        """
        # Each call gets a fresh runtime so state doesn't bleed between calls.
        runtime = ToolUseRuntime(
            llm=self._llm,
            tools=self._tools,
            max_rounds=self._max_rounds,
        )
        augmented_query = _build_context_prompt(query, contexts)
        return runtime.run(augmented_query)

    # ------------------------------------------------------------------
    # Port-conformant path (GeneratorLLM protocol)
    # ------------------------------------------------------------------

    def generate(self, query: str, contexts: Sequence[Chunk]) -> Answer:
        """Run the tool-use loop and wrap the result as a port-conformant ``Answer``.

        Citations are chunk citations only — the calculator is never listed as
        a source chunk.  The numeric result (if any) is present in ``answer_text``
        but is not added as a field on the frozen ``Answer``.
        """
        result = self.run(query, contexts)

        # Build chunk citations from the *passed-in* contexts only.
        citations = [
            Citation(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                snippet=_snippet(c.text),
                score=0.0,
                page_span=(c.page_span.start_page, c.page_span.end_page),
                section_path=c.section_path,
            )
            for c in contexts
        ]

        return Answer(query=query, text=result.answer_text, citations=citations)
