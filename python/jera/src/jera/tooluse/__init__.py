"""Tool-use runtime package for transparent model-native tool-use loops.

Public surface
--------------
Tools:
    Tool                    — protocol every tool must satisfy
    CalculatorTool          — safe AST-based arithmetic evaluator

LLM / turn types:
    ToolUseLLM              — protocol for any tool-use LLM
    AssistantTurn           — one LLM turn (blocks + stop + verbatim raw)
    TextBlock               — plain-text block
    ToolUseBlock            — tool-call block

Fake / real implementations:
    FakeToolUseLLM          — offline fixture; zero API calls
    ClaudeToolUseGenerator  — real Anthropic client (opt-in, cloud extra)

Runtime:
    ToolUseRuntime          — agentic loop (trigger → dispatch → result)
    RunResult               — outcome of one run() call
"""

from jera.tooluse.llm import (
    AssistantTurn,
    ClaudeToolUseGenerator,
    FakeToolUseLLM,
    TextBlock,
    ToolUseBlock,
    ToolUseLLM,
)
from jera.tooluse.runtime import RunResult, ToolUseRuntime
from jera.tooluse.tools import CalculatorTool, Tool, safe_eval

__all__ = [
    # protocol
    "Tool",
    "ToolUseLLM",
    # tools
    "CalculatorTool",
    "safe_eval",
    # turn types
    "AssistantTurn",
    "TextBlock",
    "ToolUseBlock",
    # LLM implementations
    "FakeToolUseLLM",
    "ClaudeToolUseGenerator",
    # runtime
    "ToolUseRuntime",
    "RunResult",
]
