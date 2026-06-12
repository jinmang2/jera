"""SDK-boundary tests for every opt-in Anthropic adapter.

All tests run fully offline — no network, no API key, no ``anthropic`` package
installed.  A fake ``anthropic`` module is injected into ``sys.modules`` before
each adapter is constructed so the real ``self._client = anthropic.Anthropic(...)``
line executes and the request-building + response-parsing logic is exercised
deterministically.

Coverage targets
----------------
- ``ClaudeGenerator.generate``           (adapters/generator/claude_generator.py)
- ``ClaudeSituateLLM.situate``           (adapters/contextual/claude_situate_llm.py)
- ``ClaudeHypothesisLLM.hypothesize``    (adapters/query/claude_hypothesis_llm.py)
- ``ClaudeToolUseGenerator.call``        (tooluse/llm.py) — incl. ``pause_turn`` loop
- ``ClaudeGoldGenerator.generate_cases`` (evaluation/gold_builder.py)
"""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# SDK-boundary mock helpers
# ---------------------------------------------------------------------------


def _make_text_block(text: str) -> Any:
    return types.SimpleNamespace(type="text", text=text)


def _make_tool_use_block(
    id: str,
    name: str,
    input: dict[str, Any],  # noqa: A002
) -> Any:
    return types.SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def _make_response(
    content: list[Any],
    stop_reason: str = "end_turn",
) -> Any:
    return types.SimpleNamespace(content=content, stop_reason=stop_reason)


def _install_fake_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any],
    responses: list[Any],
) -> None:
    """Inject a fake ``anthropic`` module that records ``messages.create`` kwargs.

    *responses* is consumed left-to-right across successive ``create`` calls so
    multi-round scenarios (``pause_turn`` loop) can be driven by a simple list.
    """
    call_index: list[int] = [0]

    mod = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **kwargs: Any) -> Any:
            # Capture every call; later calls overwrite earlier ones for simple
            # single-call adapters, but the full sequence is in ``captured["calls"]``.
            captured.update(kwargs)
            captured.setdefault("calls", []).append(dict(kwargs))
            idx = call_index[0]
            call_index[0] += 1
            return responses[idx]

    class Anthropic:
        def __init__(self, **kwargs: Any) -> None:
            self.messages = _Messages()

    mod.Anthropic = Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", mod)


# ---------------------------------------------------------------------------
# Helpers to build domain objects used by ClaudeGenerator
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "c1",
    document_id: str = "d1",
    text: str = "Revenue was 1,234 billion KRW.",
    start_page: int = 1,
    end_page: int = 1,
    section_path: tuple[str, ...] = ("Section 1",),
) -> Any:
    """Construct a ``Chunk`` domain object without touching test fixtures."""
    from jera.domain.chunk import Chunk, PageSpan

    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_id="src1",
        text=text,
        page_span=PageSpan(start_page=start_page, end_page=end_page),
        section_path=section_path,
        element_ids=("e1",),
        char_span=(0, len(text)),
        token_count=8,
        chunk_strategy="heading_aware",
        chunk_version="v1",
    )


# ===========================================================================
# 1. ClaudeGenerator
# ===========================================================================


class TestClaudeGeneratorRequest:
    """Verify the request ClaudeGenerator sends to messages.create."""

    def test_model_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("The revenue was 1,234 billion KRW.")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.generator.claude_generator import ClaudeGenerator

        gen = ClaudeGenerator(model="claude-test-model", api_key="test-key", enabled=True)
        gen.generate("What is the revenue?", [_make_chunk()])

        assert captured["model"] == "claude-test-model"

    def test_max_tokens_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("answer")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.generator.claude_generator import ClaudeGenerator

        gen = ClaudeGenerator(api_key="test-key", enabled=True, max_tokens=512)
        gen.generate("q?", [_make_chunk()])

        assert captured["max_tokens"] == 512

    def test_messages_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single user message whose content includes context block and question."""
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("answer")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.generator.claude_generator import ClaudeGenerator

        chunk = _make_chunk(chunk_id="c42", text="Chunk text here.")
        gen = ClaudeGenerator(api_key="test-key", enabled=True)
        gen.generate("What happened?", [chunk])

        msgs = captured["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        prompt = msgs[0]["content"]
        # Context marker [1] must appear
        assert "[1]" in prompt
        # chunk_id embedded
        assert "c42" in prompt
        # chunk text embedded
        assert "Chunk text here." in prompt
        # question embedded
        assert "What happened?" in prompt

    def test_numbered_context_for_multiple_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("combined answer")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.generator.claude_generator import ClaudeGenerator

        chunks = [_make_chunk(chunk_id=f"c{i}", text=f"Text {i}.") for i in range(3)]
        gen = ClaudeGenerator(api_key="test-key", enabled=True)
        gen.generate("Query?", chunks)

        prompt = captured["messages"][0]["content"]
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt


class TestClaudeGeneratorResponse:
    """Verify ClaudeGenerator parses the response into a correct Answer."""

    def test_text_blocks_joined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("First part. "), _make_text_block("Second part.")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.generator.claude_generator import ClaudeGenerator

        gen = ClaudeGenerator(api_key="test-key", enabled=True)
        answer = gen.generate("q?", [_make_chunk()])

        assert answer.text == "First part. Second part."

    def test_non_text_blocks_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Blocks with type != 'text' (e.g. thinking) must not appear in answer.text."""
        captured: dict[str, Any] = {}
        thinking = types.SimpleNamespace(type="thinking", thinking="<inner>")
        resp = _make_response([thinking, _make_text_block("Real answer.")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.generator.claude_generator import ClaudeGenerator

        gen = ClaudeGenerator(api_key="test-key", enabled=True)
        answer = gen.generate("q?", [_make_chunk()])

        assert answer.text == "Real answer."

    def test_one_citation_per_context(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("answer")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.generator.claude_generator import ClaudeGenerator

        chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(3)]
        gen = ClaudeGenerator(api_key="test-key", enabled=True)
        answer = gen.generate("q?", chunks)

        assert len(answer.citations) == 3
        ids = [cit.chunk_id for cit in answer.citations]
        assert ids == ["c0", "c1", "c2"]

    def test_citation_snippet_max_240_chars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("answer")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.generator.claude_generator import ClaudeGenerator

        long_text = "X" * 500
        chunk = _make_chunk(text=long_text)
        gen = ClaudeGenerator(api_key="test-key", enabled=True)
        answer = gen.generate("q?", [chunk])

        assert len(answer.citations[0].snippet) == 240

    def test_answer_query_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("ans")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.generator.claude_generator import ClaudeGenerator

        gen = ClaudeGenerator(api_key="test-key", enabled=True)
        answer = gen.generate("My exact query", [_make_chunk()])

        assert answer.query == "My exact query"

    def test_no_contexts_means_no_citations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("answer")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.generator.claude_generator import ClaudeGenerator

        gen = ClaudeGenerator(api_key="test-key", enabled=True)
        answer = gen.generate("q?", [])

        assert answer.citations == []


# ===========================================================================
# 2. ClaudeSituateLLM
# ===========================================================================


class TestClaudeSituateLLMRequest:
    """Verify the request ClaudeSituateLLM sends — especially the cache_control breakpoint."""

    def test_model_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("context sentence")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.contextual.claude_situate_llm import ClaudeSituateLLM

        llm = ClaudeSituateLLM(model="claude-situate-test", api_key="test-key", enabled=True)
        llm.situate("full doc", "chunk excerpt")

        assert captured["model"] == "claude-situate-test"

    def test_two_content_blocks_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The user message must have exactly two content blocks."""
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("ctx")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.contextual.claude_situate_llm import ClaudeSituateLLM

        llm = ClaudeSituateLLM(api_key="test-key", enabled=True)
        llm.situate("DOC_TEXT", "CHUNK_TEXT")

        msgs = captured["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert len(content) == 2

    def test_first_block_has_cache_control_ephemeral(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("ctx")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.contextual.claude_situate_llm import ClaudeSituateLLM

        llm = ClaudeSituateLLM(api_key="test-key", enabled=True)
        llm.situate("DOC_TEXT", "CHUNK_TEXT")

        doc_block = captured["messages"][0]["content"][0]
        assert doc_block["cache_control"] == {"type": "ephemeral"}

    def test_first_block_contains_document_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("ctx")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.contextual.claude_situate_llm import ClaudeSituateLLM

        llm = ClaudeSituateLLM(api_key="test-key", enabled=True)
        llm.situate("MY_DOCUMENT_BODY", "chunk")

        doc_block = captured["messages"][0]["content"][0]
        assert "MY_DOCUMENT_BODY" in doc_block["text"]

    def test_second_block_contains_chunk_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("ctx")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.contextual.claude_situate_llm import ClaudeSituateLLM

        llm = ClaudeSituateLLM(api_key="test-key", enabled=True)
        llm.situate("doc", "MY_CHUNK_EXCERPT")

        chunk_block = captured["messages"][0]["content"][1]
        assert "MY_CHUNK_EXCERPT" in chunk_block["text"]
        # Second block must NOT have cache_control
        assert "cache_control" not in chunk_block

    def test_both_blocks_are_type_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("ctx")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.contextual.claude_situate_llm import ClaudeSituateLLM

        llm = ClaudeSituateLLM(api_key="test-key", enabled=True)
        llm.situate("doc", "chunk")

        content = captured["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "text"


class TestClaudeSituateLLMResponse:
    def test_text_blocks_joined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("Part A. "), _make_text_block("Part B.")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.contextual.claude_situate_llm import ClaudeSituateLLM

        llm = ClaudeSituateLLM(api_key="test-key", enabled=True)
        result = llm.situate("doc", "chunk")

        assert result == "Part A. Part B."

    def test_non_text_blocks_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        thinking = types.SimpleNamespace(type="thinking", thinking="<inner>")
        resp = _make_response([thinking, _make_text_block("Good context.")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.contextual.claude_situate_llm import ClaudeSituateLLM

        llm = ClaudeSituateLLM(api_key="test-key", enabled=True)
        result = llm.situate("doc", "chunk")

        assert result == "Good context."


# ===========================================================================
# 3. ClaudeHypothesisLLM
# ===========================================================================


class TestClaudeHypothesisLLMRequest:
    def test_model_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("hypothesis text")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.query.claude_hypothesis_llm import ClaudeHypothesisLLM

        llm = ClaudeHypothesisLLM(model="claude-hypo-test", api_key="test-key", enabled=True)
        llm.hypothesize("What is GDP?")

        assert captured["model"] == "claude-hypo-test"

    def test_max_tokens_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("hypo")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.query.claude_hypothesis_llm import ClaudeHypothesisLLM

        llm = ClaudeHypothesisLLM(api_key="test-key", enabled=True, max_tokens=128)
        llm.hypothesize("query")

        assert captured["max_tokens"] == 128

    def test_messages_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single user message whose content is the prompt prefix + the query."""
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("hypo")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.query.claude_hypothesis_llm import (  # noqa: PLC0415
            _PROMPT,
            ClaudeHypothesisLLM,
        )

        llm = ClaudeHypothesisLLM(api_key="test-key", enabled=True)
        llm.hypothesize("MY_QUERY_TEXT")

        msgs = captured["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        # Must contain the full prompt prefix
        assert _PROMPT in content
        # Must end with the query
        assert content.endswith("MY_QUERY_TEXT")


class TestClaudeHypothesisLLMResponse:
    def test_text_blocks_joined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response(
            [_make_text_block("GDP refers to "), _make_text_block("total output.")]
        )
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.query.claude_hypothesis_llm import ClaudeHypothesisLLM

        llm = ClaudeHypothesisLLM(api_key="test-key", enabled=True)
        result = llm.hypothesize("What is GDP?")

        assert result == "GDP refers to total output."

    def test_non_text_blocks_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        thinking = types.SimpleNamespace(type="thinking", thinking="<>")
        resp = _make_response([thinking, _make_text_block("Hypothesis.")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.adapters.query.claude_hypothesis_llm import ClaudeHypothesisLLM

        llm = ClaudeHypothesisLLM(api_key="test-key", enabled=True)
        result = llm.hypothesize("q?")

        assert result == "Hypothesis."


# ===========================================================================
# 4. ClaudeToolUseGenerator
# ===========================================================================


class TestClaudeToolUseGeneratorRequest:
    def test_model_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("done")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.tooluse.llm import ClaudeToolUseGenerator

        gen = ClaudeToolUseGenerator(model="claude-tu-test", api_key="test-key", enabled=True)
        gen.call([{"role": "user", "content": "q"}], [])

        assert captured["model"] == "claude-tu-test"

    def test_max_tokens_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("done")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.tooluse.llm import ClaudeToolUseGenerator

        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True, max_tokens=2048)
        gen.call([{"role": "user", "content": "q"}], [])

        assert captured["max_tokens"] == 2048

    def test_tools_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("done")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.tooluse.llm import ClaudeToolUseGenerator

        tool_schema = [
            {
                "name": "calculator",
                "description": "Tool: calculator",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            }
        ]
        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        gen.call([{"role": "user", "content": "1+1"}], tool_schema)

        assert captured["tools"] == tool_schema

    def test_messages_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("done")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.tooluse.llm import ClaudeToolUseGenerator

        msgs = [{"role": "user", "content": "hello"}]
        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        gen.call(msgs, [])

        assert captured["messages"] == msgs


class TestClaudeToolUseGeneratorResponse:
    def test_text_block_mapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("Final answer.")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.tooluse.llm import ClaudeToolUseGenerator, TextBlock

        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        turn = gen.call([{"role": "user", "content": "q"}], [])

        assert len(turn.blocks) == 1
        assert isinstance(turn.blocks[0], TextBlock)
        assert turn.blocks[0].text == "Final answer."

    def test_tool_use_block_mapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response(
            [_make_tool_use_block("tid1", "calculator", {"expression": "1+1"})],
            stop_reason="tool_use",
        )
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.tooluse.llm import ClaudeToolUseGenerator, ToolUseBlock

        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        turn = gen.call([{"role": "user", "content": "q"}], [])

        assert turn.stop == "tool_use"
        assert len(turn.blocks) == 1
        blk = turn.blocks[0]
        assert isinstance(blk, ToolUseBlock)
        assert blk.id == "tid1"
        assert blk.name == "calculator"
        assert blk.input == {"expression": "1+1"}

    def test_multi_block_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Text block + tool_use block in the same response."""
        captured: dict[str, Any] = {}
        resp = _make_response(
            [
                _make_text_block("I will calculate."),
                _make_tool_use_block("tid2", "calculator", {"expression": "3*4"}),
            ],
            stop_reason="tool_use",
        )
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.tooluse.llm import ClaudeToolUseGenerator, TextBlock, ToolUseBlock

        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        turn = gen.call([{"role": "user", "content": "q"}], [])

        assert len(turn.blocks) == 2
        assert isinstance(turn.blocks[0], TextBlock)
        assert isinstance(turn.blocks[1], ToolUseBlock)

    def test_raw_equals_resp_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``turn.raw`` must be the verbatim content list from the API response."""
        captured: dict[str, Any] = {}
        content = [_make_text_block("answer")]
        resp = _make_response(content)
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.tooluse.llm import ClaudeToolUseGenerator

        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        turn = gen.call([{"role": "user", "content": "q"}], [])

        assert turn.raw is content

    def test_stop_reason_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("done")], stop_reason="end_turn")
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.tooluse.llm import ClaudeToolUseGenerator

        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        turn = gen.call([{"role": "user", "content": "q"}], [])

        assert turn.stop == "end_turn"

    def test_none_stop_reason_defaults_to_end_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("done")], stop_reason=None)  # type: ignore[arg-type]
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.tooluse.llm import ClaudeToolUseGenerator

        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        turn = gen.call([{"role": "user", "content": "q"}], [])

        assert turn.stop == "end_turn"

    def test_thinking_block_in_raw_but_not_blocks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Blocks with unknown type must pass through to raw but not appear in blocks."""
        captured: dict[str, Any] = {}
        thinking = types.SimpleNamespace(type="thinking", thinking="<step>")
        content = [thinking, _make_text_block("answer")]
        resp = _make_response(content)
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.tooluse.llm import ClaudeToolUseGenerator, TextBlock

        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        turn = gen.call([{"role": "user", "content": "q"}], [])

        # Only the text block appears in parsed blocks
        assert len(turn.blocks) == 1
        assert isinstance(turn.blocks[0], TextBlock)
        # But raw contains both
        assert len(turn.raw) == 2


# ===========================================================================
# 5. ClaudeToolUseGenerator — pause_turn loop
# ===========================================================================


class TestClaudeToolUseGeneratorPauseTurn:
    def test_pause_turn_loops_and_resolves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two consecutive ``pause_turn`` responses then ``end_turn`` — adapter must loop."""
        captured: dict[str, Any] = {}
        partial1 = _make_response(
            [types.SimpleNamespace(type="thinking", thinking="step1")],
            stop_reason="pause_turn",
        )
        partial2 = _make_response(
            [types.SimpleNamespace(type="thinking", thinking="step2")],
            stop_reason="pause_turn",
        )
        final = _make_response(
            [_make_text_block("Final answer after thinking.")], stop_reason="end_turn"
        )
        _install_fake_anthropic(monkeypatch, captured, [partial1, partial2, final])

        from jera.tooluse.llm import ClaudeToolUseGenerator, TextBlock

        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        msgs = [{"role": "user", "content": "compute something"}]
        turn = gen.call(msgs, [])

        # Must have consumed all three responses (3 create() calls)
        assert len(captured["calls"]) == 3
        # Final turn is correctly parsed
        assert turn.stop == "end_turn"
        assert len(turn.blocks) == 1
        assert isinstance(turn.blocks[0], TextBlock)
        assert turn.blocks[0].text == "Final answer after thinking."

    def test_pause_turn_appends_partial_content_to_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each ``pause_turn`` content list must be appended as an assistant message."""
        captured: dict[str, Any] = {}
        partial_content = [types.SimpleNamespace(type="thinking", thinking="thinking...")]
        partial = _make_response(partial_content, stop_reason="pause_turn")
        final = _make_response([_make_text_block("done")], stop_reason="end_turn")
        _install_fake_anthropic(monkeypatch, captured, [partial, final])

        from jera.tooluse.llm import ClaudeToolUseGenerator

        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        msgs = [{"role": "user", "content": "q"}]
        gen.call(msgs, [])

        # Second call must have had the partial content appended
        second_call_msgs = captured["calls"][1]["messages"]
        assert len(second_call_msgs) == 2
        assert second_call_msgs[1]["role"] == "assistant"
        assert second_call_msgs[1]["content"] is partial_content

    def test_pause_turn_does_not_mutate_caller_messages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The caller's messages list must not be mutated by pause_turn loop."""
        captured: dict[str, Any] = {}
        partial = _make_response(
            [types.SimpleNamespace(type="thinking", thinking="t")],
            stop_reason="pause_turn",
        )
        final = _make_response([_make_text_block("done")], stop_reason="end_turn")
        _install_fake_anthropic(monkeypatch, captured, [partial, final])

        from jera.tooluse.llm import ClaudeToolUseGenerator

        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        original_msgs = [{"role": "user", "content": "q"}]
        original_len = len(original_msgs)
        gen.call(original_msgs, [])

        assert len(original_msgs) == original_len  # caller list unchanged

    def test_pause_turn_cap_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """If all responses are ``pause_turn``, RuntimeError must be raised after the cap."""
        captured: dict[str, Any] = {}
        # Supply more than _MAX_PAUSE_ITERATIONS pause responses
        from jera.tooluse.llm import ClaudeToolUseGenerator

        cap = ClaudeToolUseGenerator._MAX_PAUSE_ITERATIONS
        responses = [
            _make_response(
                [types.SimpleNamespace(type="thinking", thinking=f"t{i}")],
                stop_reason="pause_turn",
            )
            for i in range(cap + 2)
        ]
        _install_fake_anthropic(monkeypatch, captured, responses)

        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        with pytest.raises(RuntimeError, match="pause_turn"):
            gen.call([{"role": "user", "content": "q"}], [])

    def test_single_pause_then_tool_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """pause_turn followed by tool_use — adapter returns correct ToolUseBlock turn."""
        captured: dict[str, Any] = {}
        partial = _make_response(
            [types.SimpleNamespace(type="thinking", thinking="reasoning")],
            stop_reason="pause_turn",
        )
        final = _make_response(
            [_make_tool_use_block("tid9", "calculator", {"expression": "2+2"})],
            stop_reason="tool_use",
        )
        _install_fake_anthropic(monkeypatch, captured, [partial, final])

        from jera.tooluse.llm import ClaudeToolUseGenerator, ToolUseBlock

        gen = ClaudeToolUseGenerator(api_key="test-key", enabled=True)
        turn = gen.call([{"role": "user", "content": "q"}], [])

        assert turn.stop == "tool_use"
        assert len(turn.blocks) == 1
        assert isinstance(turn.blocks[0], ToolUseBlock)
        assert turn.blocks[0].id == "tid9"


# ===========================================================================
# 6. ClaudeGoldGenerator
# ===========================================================================


class TestClaudeGoldGeneratorRequest:
    def _make_valid_json_response(self, n: int = 1) -> str:
        """Build a minimal valid JSON array that passes all guards."""
        cases = [
            {
                "kind": "retrieval",
                "query": f"Question {i}?",
                "supporting_chunk_ids": ["ch1"],
            }
            for i in range(n)
        ]
        return json.dumps(cases)

    def test_model_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        body = self._make_valid_json_response()
        resp = _make_response([_make_text_block(body)])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.evaluation.gold_builder import ClaudeGoldGenerator

        gen = ClaudeGoldGenerator(api_key="test-key", model="claude-gold-test", enabled=True)
        gen.generate_cases(
            [{"chunk_id": "ch1", "text": "some text"}],
            source_inst="Test",
            source_url="http://example.com",
            license_="MIT",
        )

        assert captured["model"] == "claude-gold-test"

    def test_max_tokens_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        body = self._make_valid_json_response()
        resp = _make_response([_make_text_block(body)])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.evaluation.gold_builder import ClaudeGoldGenerator

        gen = ClaudeGoldGenerator(api_key="test-key", enabled=True, max_tokens=2048)
        gen.generate_cases(
            [{"chunk_id": "ch1", "text": "some text"}],
            source_inst="T",
            source_url="http://x.com",
            license_="MIT",
        )

        assert captured["max_tokens"] == 2048

    def test_messages_contain_chunk_id_and_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        body = self._make_valid_json_response()
        resp = _make_response([_make_text_block(body)])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.evaluation.gold_builder import ClaudeGoldGenerator

        gen = ClaudeGoldGenerator(api_key="test-key", enabled=True)
        gen.generate_cases(
            [{"chunk_id": "SPECIAL_CHUNK_99", "text": "UNIQUE_CHUNK_TEXT"}],
            source_inst="T",
            source_url="http://x.com",
            license_="MIT",
        )

        prompt = captured["messages"][0]["content"]
        assert "SPECIAL_CHUNK_99" in prompt
        assert "UNIQUE_CHUNK_TEXT" in prompt

    def test_n_cases_in_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        body = self._make_valid_json_response(3)
        resp = _make_response([_make_text_block(body)])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.evaluation.gold_builder import ClaudeGoldGenerator

        gen = ClaudeGoldGenerator(api_key="test-key", enabled=True)
        gen.generate_cases(
            [{"chunk_id": "ch1", "text": "text"}],
            source_inst="T",
            source_url="http://x.com",
            license_="MIT",
            n_cases=7,
        )

        assert "7" in captured["messages"][0]["content"]


class TestClaudeGoldGeneratorResponse:
    def test_retrieval_case_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        cases = [{"kind": "retrieval", "query": "What?", "supporting_chunk_ids": ["ch1"]}]
        resp = _make_response([_make_text_block(json.dumps(cases))])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.evaluation.gold_builder import ClaudeGoldGenerator
        from jera.evaluation_contracts.dataset import CaseKind

        gen = ClaudeGoldGenerator(api_key="test-key", enabled=True)
        results = gen.generate_cases(
            [{"chunk_id": "ch1", "text": "some text"}],
            source_inst="Inst",
            source_url="http://url.com",
            license_="CC-BY",
        )

        assert len(results) == 1
        case = results[0]
        assert case.kind == CaseKind.RETRIEVAL
        assert case.query == "What?"
        assert len(case.gold) == 1
        assert case.gold[0].chunk_id == "ch1"

    def test_attribution_propagated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        cases = [{"kind": "retrieval", "query": "Q?", "supporting_chunk_ids": ["ch1"]}]
        resp = _make_response([_make_text_block(json.dumps(cases))])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.evaluation.gold_builder import ClaudeGoldGenerator

        gen = ClaudeGoldGenerator(api_key="test-key", enabled=True)
        results = gen.generate_cases(
            [{"chunk_id": "ch1", "text": "text"}],
            source_inst="MyInstitution",
            source_url="http://myurl.org",
            license_="Apache-2.0",
        )

        case = results[0]
        assert case.source_inst == "MyInstitution"
        assert case.source_url == "http://myurl.org"
        assert case.license == "Apache-2.0"

    def test_computation_case_with_valid_operands(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Computation case with operands that DO appear in the chunk text is accepted."""
        captured: dict[str, Any] = {}
        cases = [
            {
                "kind": "computation",
                "query": "Sum?",
                "supporting_chunk_ids": ["ch1"],
                "cited_numbers": [100.0, 200.0],
                "operation": "100.0 + 200.0",
            }
        ]
        resp = _make_response([_make_text_block(json.dumps(cases))])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.evaluation.gold_builder import ClaudeGoldGenerator
        from jera.evaluation_contracts.dataset import CaseKind

        gen = ClaudeGoldGenerator(api_key="test-key", enabled=True)
        results = gen.generate_cases(
            # chunk text must contain 100 and 200 for provenance guard to pass
            [{"chunk_id": "ch1", "text": "First value is 100 and second is 200."}],
            source_inst="T",
            source_url="http://x.com",
            license_="MIT",
        )

        assert len(results) == 1
        case = results[0]
        assert case.kind == CaseKind.COMPUTATION
        assert case.expected_value == pytest.approx(300.0)
        assert case.formula == "100.0 + 200.0"

    def test_computation_case_rejected_when_operand_hallucinated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Computation case whose operand does NOT appear in the chunk must be rejected."""
        captured: dict[str, Any] = {}
        cases = [
            {
                "kind": "computation",
                "query": "Sum?",
                "supporting_chunk_ids": ["ch1"],
                "cited_numbers": [999.0, 888.0],  # these numbers are NOT in chunk text
                "operation": "999.0 + 888.0",
            }
        ]
        resp = _make_response([_make_text_block(json.dumps(cases))])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.evaluation.gold_builder import ClaudeGoldGenerator

        gen = ClaudeGoldGenerator(api_key="test-key", enabled=True)
        results = gen.generate_cases(
            [{"chunk_id": "ch1", "text": "Unrelated text without the numbers."}],
            source_inst="T",
            source_url="http://x.com",
            license_="MIT",
        )

        assert results == []

    def test_json_in_markdown_fence_parsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Response wrapped in markdown fences must still be parsed correctly."""
        captured: dict[str, Any] = {}
        cases = [{"kind": "retrieval", "query": "Q?", "supporting_chunk_ids": ["ch1"]}]
        fenced = f"```json\n{json.dumps(cases)}\n```"
        resp = _make_response([_make_text_block(fenced)])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.evaluation.gold_builder import ClaudeGoldGenerator

        gen = ClaudeGoldGenerator(api_key="test-key", enabled=True)
        results = gen.generate_cases(
            [{"chunk_id": "ch1", "text": "text"}],
            source_inst="T",
            source_url="http://x.com",
            license_="MIT",
        )

        assert len(results) == 1

    def test_case_with_unknown_supporting_chunk_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}
        cases = [{"kind": "retrieval", "query": "Q?", "supporting_chunk_ids": ["NONEXISTENT_ID"]}]
        resp = _make_response([_make_text_block(json.dumps(cases))])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.evaluation.gold_builder import ClaudeGoldGenerator

        gen = ClaudeGoldGenerator(api_key="test-key", enabled=True)
        results = gen.generate_cases(
            [{"chunk_id": "ch1", "text": "text"}],
            source_inst="T",
            source_url="http://x.com",
            license_="MIT",
        )

        assert results == []

    def test_no_json_array_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        resp = _make_response([_make_text_block("No JSON here, just prose.")])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.evaluation.gold_builder import ClaudeGoldGenerator

        gen = ClaudeGoldGenerator(api_key="test-key", enabled=True)
        with pytest.raises(ValueError, match="No JSON array"):
            gen.generate_cases(
                [{"chunk_id": "ch1", "text": "text"}],
                source_inst="T",
                source_url="http://x.com",
                license_="MIT",
            )

    def test_multiple_text_blocks_joined(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multi-block response text is joined before JSON extraction."""
        captured: dict[str, Any] = {}
        cases = [{"kind": "retrieval", "query": "Q?", "supporting_chunk_ids": ["ch1"]}]
        half = json.dumps(cases)
        # Split the JSON across two text blocks
        mid = len(half) // 2
        resp = _make_response([_make_text_block(half[:mid]), _make_text_block(half[mid:])])
        _install_fake_anthropic(monkeypatch, captured, [resp])

        from jera.evaluation.gold_builder import ClaudeGoldGenerator

        gen = ClaudeGoldGenerator(api_key="test-key", enabled=True)
        results = gen.generate_cases(
            [{"chunk_id": "ch1", "text": "text"}],
            source_inst="T",
            source_url="http://x.com",
            license_="MIT",
        )

        assert len(results) == 1
