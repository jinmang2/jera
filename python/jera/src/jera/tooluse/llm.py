"""ToolUseLLM protocol, turn data-classes, and implementations.

The turn-based contract
-----------------------
``ToolUseLLM.call(messages)`` takes the current message list and returns an
``AssistantTurn``.  The caller (``ToolUseRuntime``) appends the turn's raw
content verbatim as the next assistant message — this preserves Opus extended
thinking blocks and any other blocks the model may emit.

``AssistantTurn`` fields
------------------------
- ``blocks``   — parsed list of ``TextBlock | ToolUseBlock``
- ``stop``     — model stop reason (``"tool_use"`` or ``"end_turn"``)
- ``raw``      — **verbatim** content list from the API response; the runtime
                 appends this to the conversation, never the parsed blocks.

``FakeToolUseLLM``
------------------
Two built-in fixture behaviours driven by ``mode``:

``"single"``  — single ``tool_use`` block (calculator, ``"1 + 1"``), then
               ``end_turn`` text reply on the second call.

``"multi"``   — first call returns text + *two* tool_use blocks; second call
               returns text + one more tool_use; third call returns end_turn.
               This exercises the multi-block branch of the runtime loop.

``ClaudeToolUseGenerator`` (opt-in / ``cloud`` extra)
------------------------------------------------------
Wraps the real Anthropic client.  Never enabled in automated tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable  # noqa: F401

# ---------------------------------------------------------------------------
# Block types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextBlock:
    """A plain-text content block in an assistant turn."""

    type: Literal["text"] = field(default="text", init=False)
    text: str


@dataclass(frozen=True)
class ToolUseBlock:
    """A tool-call content block in an assistant turn."""

    type: Literal["tool_use"] = field(default="tool_use", init=False)
    id: str
    name: str
    input: dict[str, Any]


# ---------------------------------------------------------------------------
# AssistantTurn
# ---------------------------------------------------------------------------


@dataclass
class AssistantTurn:
    """One turn produced by the LLM during the tool-use loop.

    ``raw`` must be appended to the conversation verbatim so that the API
    receives the exact content it produced (including opaque thinking blocks
    or any future block types).
    """

    blocks: list[TextBlock | ToolUseBlock]
    stop: str  # "tool_use" | "end_turn" | vendor-specific value
    raw: list[Any]  # verbatim API content — append this, not ``blocks``


# ---------------------------------------------------------------------------
# ToolUseLLM protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ToolUseLLM(Protocol):
    """Thin LLM façade for the tool-use runtime.

    ``call`` receives the *full* current message list (including any
    ``tool_result`` messages appended so far) and returns the next turn.
    """

    model_id: str

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn: ...


# ---------------------------------------------------------------------------
# FakeToolUseLLM — offline test fixture
# ---------------------------------------------------------------------------

# Deterministic fake IDs for test assertions.
_FAKE_ID_A = "fake-tool-id-aaa"
_FAKE_ID_B = "fake-tool-id-bbb"
_FAKE_ID_C = "fake-tool-id-ccc"


class FakeToolUseLLM:
    """Deterministic fake LLM that exercises the tool-use loop offline.

    ``mode="single"``
        Round 1 → one ``tool_use`` block (stop=``"tool_use"``)
        Round 2 → one ``text`` block  (stop=``"end_turn"``)

    ``mode="multi"``
        Round 1 → text + two ``tool_use`` blocks (stop=``"tool_use"``)
        Round 2 → text + one ``tool_use`` block  (stop=``"tool_use"``)
        Round 3 → text block only                (stop=``"end_turn"``)

    The ``raw`` field of each ``AssistantTurn`` mirrors the ``blocks`` list so
    that the runtime's verbatim-append invariant can be exercised without a
    real API connection.
    """

    model_id = "fake-tool-use-v1"

    def __init__(self, mode: Literal["single", "multi"] = "single") -> None:
        self._mode = mode
        self._call_count = 0

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        self._call_count += 1
        if self._mode == "single":
            return self._single_turn(self._call_count)
        return self._multi_turn(self._call_count)

    # ------------------------------------------------------------------
    # Single-mode fixture
    # ------------------------------------------------------------------

    def _single_turn(self, n: int) -> AssistantTurn:
        if n == 1:
            block = ToolUseBlock(
                id=_FAKE_ID_A,
                name="calculator",
                input={"expression": "1 + 1"},
            )
            raw = [
                {
                    "type": "tool_use",
                    "id": _FAKE_ID_A,
                    "name": "calculator",
                    "input": {"expression": "1 + 1"},
                },
            ]
            return AssistantTurn(blocks=[block], stop="tool_use", raw=raw)
        # Round 2 — final answer after tool result was appended
        text_block = TextBlock(text="The answer is 2.0.")
        raw = [{"type": "text", "text": "The answer is 2.0."}]
        return AssistantTurn(blocks=[text_block], stop="end_turn", raw=raw)

    # ------------------------------------------------------------------
    # Multi-block fixture (text + two tool_use blocks, then loop)
    # ------------------------------------------------------------------

    def _multi_turn(self, n: int) -> AssistantTurn:
        if n == 1:
            # First turn: a preamble text block + two tool_use blocks
            t = TextBlock(text="I will calculate both sub-expressions.")
            a = ToolUseBlock(id=_FAKE_ID_A, name="calculator", input={"expression": "3 + 4"})
            b = ToolUseBlock(id=_FAKE_ID_B, name="calculator", input={"expression": "10 * 2"})
            raw = [
                {"type": "text", "text": "I will calculate both sub-expressions."},
                {
                    "type": "tool_use",
                    "id": _FAKE_ID_A,
                    "name": "calculator",
                    "input": {"expression": "3 + 4"},
                },
                {
                    "type": "tool_use",
                    "id": _FAKE_ID_B,
                    "name": "calculator",
                    "input": {"expression": "10 * 2"},
                },
            ]
            return AssistantTurn(blocks=[t, a, b], stop="tool_use", raw=raw)

        if n == 2:
            # Second turn: another tool_use to combine the results
            c = ToolUseBlock(id=_FAKE_ID_C, name="calculator", input={"expression": "7.0 + 20.0"})
            raw = [
                {
                    "type": "tool_use",
                    "id": _FAKE_ID_C,
                    "name": "calculator",
                    "input": {"expression": "7.0 + 20.0"},
                },
            ]
            return AssistantTurn(blocks=[c], stop="tool_use", raw=raw)

        # Round 3 — end_turn
        block = TextBlock(text="The combined result is 27.0.")
        raw = [{"type": "text", "text": "The combined result is 27.0."}]
        return AssistantTurn(blocks=[block], stop="end_turn", raw=raw)


# ---------------------------------------------------------------------------
# ClaudeToolUseGenerator — opt-in, requires ``cloud`` extra + API key
# ---------------------------------------------------------------------------


class ClaudeToolUseGenerator:
    """Real Anthropic tool-use LLM.  Disabled by default; never called in CI.

    Requires:
      - ``uv sync --extra cloud`` (pulls in the ``anthropic`` package)
      - A valid ``api_key``
      - ``enabled=True``

    The ``model_id`` defaults to the latest Opus model.
    """

    _DEFAULT_MODEL = "claude-opus-4-8"
    _MAX_TOKENS = 4096

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        enabled: bool = False,
        max_tokens: int = _MAX_TOKENS,
    ) -> None:
        if not enabled:
            raise RuntimeError(
                "ClaudeToolUseGenerator is disabled by default. "
                "Pass enabled=True and an api_key (paid live calls; "
                "never enabled in automated tests)."
            )
        if not api_key:
            raise RuntimeError("ClaudeToolUseGenerator requires an api_key when enabled.")
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ClaudeToolUseGenerator requires the 'cloud' extra: `uv sync --extra cloud`."
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self.model_id = model
        self._max_tokens = max_tokens

    #: Maximum internal iterations to resolve consecutive ``pause_turn`` responses before
    #: raising ``RuntimeError``.  Each iteration re-sends the conversation with the
    #: accumulated content appended so adaptive thinking can continue.
    _MAX_PAUSE_ITERATIONS = 8

    def call(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AssistantTurn:
        """Invoke the Anthropic Messages API and wrap the response as an ``AssistantTurn``.

        ``pause_turn`` handling (adaptive thinking on Opus 4.8)
        --------------------------------------------------------
        When extended/adaptive thinking is enabled the model may return
        ``stop_reason="pause_turn"`` to signal it wants to continue reasoning in a
        subsequent call.  This method handles that transparently: it appends the
        partial ``resp.content`` to the *local* copy of ``messages`` and re-invokes
        ``messages.create`` in a loop until ``stop_reason != "pause_turn"``, then maps
        the final response as normal.  The outer ``ToolUseRuntime`` loop never sees
        ``"pause_turn"`` as a stop reason.

        The loop is capped at ``_MAX_PAUSE_ITERATIONS`` (default 8) to guard against
        a runaway model; a ``RuntimeError`` is raised if the cap is reached.
        """
        # Work on a local copy so repeated ``pause_turn`` appends do not mutate
        # the caller's list — the runtime owns that list between turns.
        working_messages = list(messages)

        resp = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            tools=tools,
            messages=working_messages,
        )

        for _pause_iter in range(self._MAX_PAUSE_ITERATIONS):
            if resp.stop_reason != "pause_turn":
                break
            # Append the partial content verbatim and re-invoke so the model can
            # continue its adaptive thinking chain.
            working_messages.append({"role": "assistant", "content": resp.content})
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                tools=tools,
                messages=working_messages,
            )
        else:
            raise RuntimeError(
                f"ClaudeToolUseGenerator.call: model returned 'pause_turn' for "
                f"{self._MAX_PAUSE_ITERATIONS} consecutive iterations without resolving. "
                "Check extended-thinking configuration."
            )

        blocks: list[TextBlock | ToolUseBlock] = []
        for blk in resp.content:
            if blk.type == "text":
                blocks.append(TextBlock(text=blk.text))
            elif blk.type == "tool_use":
                blocks.append(ToolUseBlock(id=blk.id, name=blk.name, input=blk.input))
            # Other block types (e.g., thinking) are preserved in ``raw`` only.

        return AssistantTurn(
            blocks=blocks,
            stop=resp.stop_reason or "end_turn",
            # verbatim content list — the runtime appends this to messages
            raw=resp.content,
        )
