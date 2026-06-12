"""Unit tests for jera.tooluse — runs fully offline, zero API calls."""

from __future__ import annotations

import pytest

from jera.adapters.generator.tool_augmented_generator import ToolAugmentedGenerator
from jera.tooluse import (
    CalculatorTool,
    FakeToolUseLLM,
    RunResult,
    TextBlock,
    ToolUseBlock,
    ToolUseRuntime,
    safe_eval,
)

# ---------------------------------------------------------------------------
# safe_eval / CalculatorTool
# ---------------------------------------------------------------------------


class TestSafeEval:
    def test_addition(self) -> None:
        assert safe_eval("1 + 1") == pytest.approx(2.0)

    def test_subtraction(self) -> None:
        assert safe_eval("10 - 3") == pytest.approx(7.0)

    def test_multiplication(self) -> None:
        assert safe_eval("4 * 5") == pytest.approx(20.0)

    def test_division(self) -> None:
        assert safe_eval("10 / 4") == pytest.approx(2.5)

    def test_power(self) -> None:
        assert safe_eval("2 ** 8") == pytest.approx(256.0)

    def test_unary_minus(self) -> None:
        assert safe_eval("-3 + 5") == pytest.approx(2.0)

    def test_nested_parens(self) -> None:
        assert safe_eval("(3 + 4) * 2 / 7") == pytest.approx(2.0)

    def test_float_literals(self) -> None:
        assert safe_eval("1.5 * 2.0") == pytest.approx(3.0)

    def test_modulo(self) -> None:
        assert safe_eval("17 % 5") == pytest.approx(2.0)

    def test_division_by_zero(self) -> None:
        with pytest.raises((ZeroDivisionError, ValueError)):
            safe_eval("1 / 0")

    def test_rejects_function_call(self) -> None:
        with pytest.raises(ValueError):
            safe_eval("__import__('os').system('ls')")

    def test_rejects_name(self) -> None:
        with pytest.raises(ValueError):
            safe_eval("x + 1")

    def test_rejects_string_literal(self) -> None:
        with pytest.raises(ValueError):
            safe_eval("'hello'")

    def test_syntax_error(self) -> None:
        with pytest.raises(ValueError):
            safe_eval("3 +* 2")


class TestCalculatorTool:
    def setup_method(self) -> None:
        self.calc = CalculatorTool()

    def test_name(self) -> None:
        assert self.calc.name == "calculator"

    def test_schema_has_expression(self) -> None:
        assert "expression" in self.calc.input_schema["properties"]

    def test_execute_returns_string(self) -> None:
        result = self.calc.execute(expression="2 + 2")
        assert result == "4.0"

    def test_execute_missing_expression(self) -> None:
        result = self.calc.execute()
        assert result.startswith("Error")

    def test_execute_invalid_expression(self) -> None:
        result = self.calc.execute(expression="import os")
        assert result.startswith("Error")


# ---------------------------------------------------------------------------
# FakeToolUseLLM — single mode
# ---------------------------------------------------------------------------


class TestFakeToolUseLLMSingle:
    def setup_method(self) -> None:
        self.llm = FakeToolUseLLM(mode="single")

    def test_model_id(self) -> None:
        assert self.llm.model_id == "fake-tool-use-v1"

    def test_first_call_is_tool_use(self) -> None:
        turn = self.llm.call([], [])
        assert turn.stop == "tool_use"
        tool_blocks = [b for b in turn.blocks if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) == 1
        assert tool_blocks[0].name == "calculator"
        assert tool_blocks[0].input == {"expression": "1 + 1"}

    def test_first_call_raw_matches_blocks(self) -> None:
        turn = self.llm.call([], [])
        assert len(turn.raw) == 1
        assert turn.raw[0]["type"] == "tool_use"

    def test_second_call_is_end_turn(self) -> None:
        self.llm.call([], [])
        turn = self.llm.call([], [])
        assert turn.stop == "end_turn"
        text_blocks = [b for b in turn.blocks if isinstance(b, TextBlock)]
        assert len(text_blocks) == 1
        assert "2.0" in text_blocks[0].text


# ---------------------------------------------------------------------------
# FakeToolUseLLM — multi mode
# ---------------------------------------------------------------------------


class TestFakeToolUseLLMMulti:
    def setup_method(self) -> None:
        self.llm = FakeToolUseLLM(mode="multi")

    def test_first_call_has_text_and_two_tool_blocks(self) -> None:
        turn = self.llm.call([], [])
        assert turn.stop == "tool_use"
        text_blocks = [b for b in turn.blocks if isinstance(b, TextBlock)]
        tool_blocks = [b for b in turn.blocks if isinstance(b, ToolUseBlock)]
        assert len(text_blocks) == 1
        assert len(tool_blocks) == 2

    def test_second_call_has_one_tool_block(self) -> None:
        self.llm.call([], [])
        turn = self.llm.call([], [])
        assert turn.stop == "tool_use"
        tool_blocks = [b for b in turn.blocks if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) == 1

    def test_third_call_is_end_turn(self) -> None:
        for _ in range(2):
            self.llm.call([], [])
        turn = self.llm.call([], [])
        assert turn.stop == "end_turn"


# ---------------------------------------------------------------------------
# ToolUseRuntime — single mode (full loop)
# ---------------------------------------------------------------------------


class TestToolUseRuntimeSingle:
    def setup_method(self) -> None:
        self.runtime = ToolUseRuntime(
            llm=FakeToolUseLLM(mode="single"),
            tools=[CalculatorTool()],
        )

    def test_returns_run_result(self) -> None:
        result = self.runtime.run("What is 1 + 1?")
        assert isinstance(result, RunResult)

    def test_answer_text_not_empty(self) -> None:
        result = self.runtime.run("What is 1 + 1?")
        assert result.answer_text.strip() != ""

    def test_tool_calls_recorded(self) -> None:
        result = self.runtime.run("What is 1 + 1?")
        assert len(result.tool_calls) == 1
        name, inp, out = result.tool_calls[0]
        assert name == "calculator"
        assert inp == {"expression": "1 + 1"}
        assert out == "2.0"

    def test_final_value_is_float(self) -> None:
        result = self.runtime.run("What is 1 + 1?")
        assert result.final_value == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# ToolUseRuntime — multi mode (multi-block turn)
# ---------------------------------------------------------------------------


class TestToolUseRuntimeMulti:
    def setup_method(self) -> None:
        self.runtime = ToolUseRuntime(
            llm=FakeToolUseLLM(mode="multi"),
            tools=[CalculatorTool()],
        )

    def test_three_tool_calls_recorded(self) -> None:
        result = self.runtime.run("Compute something complex.")
        # Round 1: two tool_use blocks; Round 2: one tool_use block
        assert len(result.tool_calls) == 3

    def test_final_value_from_last_tool(self) -> None:
        result = self.runtime.run("Compute something complex.")
        # Last tool call is 7.0 + 20.0 = 27.0
        assert result.final_value == pytest.approx(27.0)

    def test_answer_text_present(self) -> None:
        result = self.runtime.run("Compute something complex.")
        assert "27.0" in result.answer_text


# ---------------------------------------------------------------------------
# ToolUseRuntime — unknown tool
# ---------------------------------------------------------------------------


class TestToolUseRuntimeUnknownTool:
    def test_unknown_tool_returns_error_string(self) -> None:
        # FakeToolUseLLM calls "calculator" but we register nothing
        runtime = ToolUseRuntime(
            llm=FakeToolUseLLM(mode="single"),
            tools=[],  # no tools registered
        )
        result = runtime.run("anything")
        assert len(result.tool_calls) == 1
        _, _, out = result.tool_calls[0]
        assert "Error" in out
        # final_value must be None because output is an error string
        assert result.final_value is None


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_calculator_tool_satisfies_protocol(self) -> None:
        from jera.tooluse import Tool

        assert isinstance(CalculatorTool(), Tool)

    def test_fake_llm_satisfies_protocol(self) -> None:
        from jera.tooluse import ToolUseLLM

        assert isinstance(FakeToolUseLLM(), ToolUseLLM)


# ---------------------------------------------------------------------------
# ToolAugmentedGenerator
# ---------------------------------------------------------------------------


class TestToolAugmentedGenerator:
    """Tests for ToolAugmentedGenerator — zero API calls, FakeToolUseLLM only."""

    def _make_gen(self, mode: str = "single") -> ToolAugmentedGenerator:
        return ToolAugmentedGenerator(
            llm=FakeToolUseLLM(mode=mode),  # type: ignore[arg-type]
            tools=[CalculatorTool()],
        )

    def test_model_id_matches_llm(self) -> None:
        gen = self._make_gen()
        assert gen.model_id == FakeToolUseLLM().model_id

    # ------------------------------------------------------------------
    # run() — typed numeric path
    # ------------------------------------------------------------------

    def test_run_returns_run_result(self) -> None:
        gen = self._make_gen()
        result = gen.run("What is 1 + 1?", contexts=[])
        assert isinstance(result, RunResult)

    def test_run_single_tool_call(self) -> None:
        gen = self._make_gen(mode="single")
        result = gen.run("What is 1 + 1?", contexts=[])
        assert len(result.tool_calls) == 1
        assert result.final_value == pytest.approx(2.0)

    def test_run_multi_tool_calls(self) -> None:
        gen = self._make_gen(mode="multi")
        result = gen.run("Compute something.", contexts=[])
        assert len(result.tool_calls) == 3
        assert result.final_value == pytest.approx(27.0)

    # ------------------------------------------------------------------
    # generate() — port-conformant Answer
    # ------------------------------------------------------------------

    def test_generate_returns_answer(self) -> None:
        from jera.domain.answer import Answer

        gen = self._make_gen()
        answer = gen.generate("What is 1 + 1?", contexts=[])
        assert isinstance(answer, Answer)

    def test_generate_answer_text_not_empty(self) -> None:
        gen = self._make_gen()
        answer = gen.generate("What is 1 + 1?", contexts=[])
        assert answer.text.strip() != ""

    def test_generate_citations_are_chunk_only(self) -> None:
        """Citations must come from contexts, not from the calculator tool."""
        from jera.domain.answer import Answer
        from jera.domain.chunk import Chunk, PageSpan

        chunk = Chunk(
            chunk_id="c1",
            document_id="d1",
            source_id="src1",
            text="Revenue was 100 billion.",
            page_span=PageSpan(start_page=1, end_page=1),
            section_path=("Section 1",),
            element_ids=("e1",),
            char_span=(0, 24),
            token_count=8,
            chunk_strategy="heading_aware",
            chunk_version="v1",
        )
        gen = self._make_gen()
        answer = gen.generate("What is the revenue?", contexts=[chunk])
        assert isinstance(answer, Answer)
        # Exactly one citation from the passed-in chunk
        assert len(answer.citations) == 1
        assert answer.citations[0].chunk_id == "c1"

    def test_generate_no_contexts_means_no_citations(self) -> None:
        gen = self._make_gen()
        answer = gen.generate("What is 1 + 1?", contexts=[])
        assert answer.citations == []

    def test_generate_satisfies_generator_llm_protocol(self) -> None:
        from jera.ports.generator import GeneratorLLM

        gen = self._make_gen()
        assert isinstance(gen, GeneratorLLM)
