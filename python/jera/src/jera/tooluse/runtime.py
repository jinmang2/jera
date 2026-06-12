"""Tool-use agentic loop runtime.

How the loop works
------------------
The ``ToolUseRuntime`` implements the transparent model-native tool-use loop
documented in the Anthropic API guide:

  1. Send ``messages`` + tool schemas to the LLM.
  2. If the model stops with ``stop_reason == "tool_use"``:
       a. Append the assistant turn's **raw** content verbatim to ``messages``
          (preserves opaque blocks like extended thinking).
       b. Dispatch every ``ToolUseBlock`` in the turn to its registered ``Tool``.
       c. Collect results as ``tool_result`` blocks keyed by ``tool_use_id``.
       d. Append a ``{"role": "user", "content": [tool_result, ...]}`` message.
       e. Loop back to step 1.
  3. When the model returns ``stop_reason == "end_turn"``, extract the final
     answer text and any ``float`` found in the last tool result, then return
     a ``RunResult``.

``RunResult`` fields
--------------------
- ``answer_text``  — concatenation of all ``TextBlock`` texts in the final turn
- ``tool_calls``   — list of ``(tool_name, input_dict, result_str)`` triples
- ``final_value``  — last numeric tool result parsed as ``float | None``
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from jera.tooluse.llm import AssistantTurn, TextBlock, ToolUseBlock, ToolUseLLM
from jera.tooluse.tools import Tool

# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    """Outcome of a single ``ToolUseRuntime.run()`` call."""

    answer_text: str
    tool_calls: list[tuple[str, dict[str, Any], str]] = field(default_factory=list)
    #: The numeric value of the **last** tool result dispatched before ``end_turn``,
    #: parsed as ``float``.  Contract: prompts and gold datasets must be designed so
    #: the final calculator call produces the answer — intermediate results from
    #: earlier tool calls in the same loop are overwritten.  ``None`` if no tool was
    #: called or the last tool result was not parseable as a float.
    final_value: float | None = None


# ---------------------------------------------------------------------------
# ToolUseRuntime
# ---------------------------------------------------------------------------


class ToolUseRuntime:
    """Transparent model-native tool-use loop.

    Parameters
    ----------
    llm:
        Any ``ToolUseLLM`` — real (``ClaudeToolUseGenerator``) or fake
        (``FakeToolUseLLM``) — as long as it satisfies the protocol.
    tools:
        Sequence of ``Tool`` instances to register.  Names must be unique.
    max_rounds:
        Safety cap on loop iterations to prevent infinite loops if the model
        keeps requesting tools.
    """

    def __init__(
        self,
        llm: ToolUseLLM,
        tools: list[Tool],
        max_rounds: int = 10,
    ) -> None:
        self._llm = llm
        self._tools: dict[str, Tool] = {t.name: t for t in tools}
        self._max_rounds = max_rounds

    # ------------------------------------------------------------------
    # Tool schema helpers
    # ------------------------------------------------------------------

    def _tool_schemas(self) -> list[dict[str, Any]]:
        """Build the ``tools`` list the LLM API expects."""
        return [
            {
                "name": t.name,
                "description": f"Tool: {t.name}",
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def run(self, query: str) -> RunResult:
        """Execute the tool-use loop for *query* and return a ``RunResult``.

        The mutable ``messages`` list is local to this call so concurrent
        ``run()`` invocations do not share state.
        """
        # Seed the conversation with the user query.
        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
        tool_schemas = self._tool_schemas()

        all_tool_calls: list[tuple[str, dict[str, Any], str]] = []
        last_tool_result: str | None = None
        final_turn: AssistantTurn | None = None

        for _ in range(self._max_rounds):
            # --- Step 1: call the LLM ---
            turn = self._llm.call(messages, tool_schemas)

            if turn.stop == "end_turn":
                # Model is done — capture the final turn and exit.
                final_turn = turn
                break

            # --- Step 2a: append assistant raw content verbatim ---
            # We MUST use ``turn.raw``, not the parsed ``turn.blocks``, so
            # that opaque blocks (e.g., extended thinking) are preserved.
            messages.append({"role": "assistant", "content": turn.raw})

            # --- Step 2b: collect every tool_use block ---
            tool_use_blocks = [b for b in turn.blocks if isinstance(b, ToolUseBlock)]

            # --- Step 2c: dispatch each tool and collect tool_result dicts ---
            tool_results: list[dict[str, Any]] = []
            for blk in tool_use_blocks:
                tool = self._tools.get(blk.name)
                if tool is None:
                    result_str = f"Error: unknown tool '{blk.name}'"
                else:
                    result_str = tool.execute(**blk.input)

                all_tool_calls.append((blk.name, dict(blk.input), result_str))
                last_tool_result = result_str

                # tool_use_id must match the block's id exactly.
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": blk.id,  # ← must match blk.id
                        "content": result_str,
                    }
                )

            # --- Step 2d: append tool results as next user message ---
            messages.append({"role": "user", "content": tool_results})

        # --- Step 3: extract answer from the final turn ---
        if final_turn is None:
            # Hit max_rounds without end_turn — treat last text as answer.
            answer_text = "(max rounds reached without end_turn)"
        else:
            answer_text = " ".join(b.text for b in final_turn.blocks if isinstance(b, TextBlock))

        # Parse last numeric tool result as final_value.
        # Contract: the LAST tool result dispatched before end_turn is treated as
        # the numeric answer.  Prompts and gold datasets must ensure the final
        # calculator call is the one that produces the answer; earlier intermediate
        # results are overwritten here by design.
        final_value: float | None = None
        if last_tool_result is not None:
            with contextlib.suppress(ValueError, TypeError):
                final_value = float(last_tool_result)

        return RunResult(
            answer_text=answer_text,
            tool_calls=all_tool_calls,
            final_value=final_value,
        )
