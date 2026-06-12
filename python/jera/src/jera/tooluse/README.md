# `jera.tooluse` — Transparent Model-Native Tool-Use Runtime

This package implements a **transparent, documented tool-use loop** so the
trigger → dispatch → result mechanism is legible for learning — not hidden
behind a framework.

## Concepts

### Trigger → Dispatch → Result

```
User query
    │
    ▼
ToolUseRuntime.run(query)
    │
    │  ┌─────────────────────────────────────────────────────────┐
    │  │  LOOP (until stop_reason == "end_turn")                 │
    │  │                                                         │
    │  │  1. LLM.call(messages, tool_schemas)                    │
    │  │       → AssistantTurn{blocks, stop, raw}                │
    │  │                                                         │
    │  │  2. Append turn.raw verbatim to messages                │
    │  │     (preserves opaque blocks, e.g., extended thinking)  │
    │  │                                                         │
    │  │  3. For each ToolUseBlock in turn.blocks:               │
    │  │       • Dispatch → Tool.execute(**block.input)          │
    │  │       • Collect tool_result keyed by tool_use_id        │
    │  │                                                         │
    │  │  4. Append {"role": "user",                             │
    │  │              "content": [tool_result, ...]}             │
    │  │                                                         │
    │  └─────────────────────────────────────────────────────────┘
    │
    ▼
RunResult{answer_text, tool_calls, final_value}
```

### Verbatim-Raw-Replay Invariant

The runtime appends `turn.raw` (the **verbatim** API response content list)
back into `messages`, **never** the parsed `turn.blocks`.  This is critical
because:

- The Anthropic API requires the assistant message to contain the *exact*
  content it produced — including any opaque block types (extended thinking,
  etc.).
- Re-serialising from parsed blocks would lose unknown fields and could cause
  a validation error on the next request.

### Model-Native vs Harness-Side Triggers

| Approach | How tool calls are detected | Who builds the loop |
|---|---|---|
| **Model-native** (this package) | Model emits `tool_use` blocks; `stop_reason == "tool_use"` | Your code (transparent loop) |
| **Harness-side** | Framework intercepts text patterns (e.g., `<tool>` tags) | Framework (opaque) |

`jera.tooluse` uses **model-native** function calling: the LLM decides *when*
to call a tool and emits a structured `tool_use` block.  The runtime simply
responds to that signal.

## Components

### `CalculatorTool`

Evaluates arithmetic expressions through Python's `ast` module.  **No
`eval()` / `exec()`** — the expression is parsed to an AST and only numeric
literals plus `+`, `-`, `*`, `/`, `**`, unary `+/-` are allowed.

```python
from jera.tooluse import CalculatorTool

calc = CalculatorTool()
print(calc.execute(expression="(3 + 4) * 2 / 7"))  # "2.0"
```

### `FakeToolUseLLM`

Offline fixture for testing.  Zero API calls.  Two modes:

- `mode="single"` — one `tool_use` block, then `end_turn`
- `mode="multi"`  — text + two `tool_use` blocks, then one more, then `end_turn`

```python
from jera.tooluse import FakeToolUseLLM, CalculatorTool, ToolUseRuntime

runtime = ToolUseRuntime(
    llm=FakeToolUseLLM(mode="single"),
    tools=[CalculatorTool()],
)
result = runtime.run("What is 1 + 1?")
print(result.answer_text)   # "The answer is 2.0."
print(result.final_value)   # 2.0
```

### `ClaudeToolUseGenerator` (opt-in)

Real Anthropic client.  Requires `uv sync --extra cloud` and an API key.
**Never called in automated CI.**

```python
from jera.tooluse import ClaudeToolUseGenerator, CalculatorTool, ToolUseRuntime

runtime = ToolUseRuntime(
    llm=ClaudeToolUseGenerator(enabled=True, api_key="sk-ant-..."),
    tools=[CalculatorTool()],
)
result = runtime.run("A company had revenue of 123.4 billion and costs of 98.7 billion. What is the profit margin?")
print(result.answer_text)
print(result.tool_calls)   # [(name, input, result), ...]
```

## Testing

```bash
# Run only tooluse tests (offline, no API key needed):
uv run pytest python/jera/tests/unit/test_tooluse.py -v

# Full gate suite:
bash scripts/gates.sh
```
