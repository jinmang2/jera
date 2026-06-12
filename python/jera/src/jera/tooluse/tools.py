"""Tool protocol and built-in tools for the tool-use runtime.

Each Tool exposes a JSON schema that the LLM sees, and an ``execute`` method
that the runtime dispatches to after the model emits a ``tool_use`` block.

The ``CalculatorTool`` evaluates arithmetic expressions through Python's AST
module — no ``eval()`` / ``exec()`` so there is no code-injection surface.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Tool protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Tool(Protocol):
    """Minimal contract every tool must satisfy."""

    #: Name used in tool-schema ``name`` field and model ``tool_use`` blocks.
    name: str

    #: JSON-serialisable schema sent to the LLM in the ``tools`` list.
    input_schema: dict[str, Any]

    def execute(self, **kwargs: Any) -> str:
        """Dispatch the tool call and return a string result."""
        ...


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator (AST-based)
# ---------------------------------------------------------------------------

_SAFE_OPS: dict[type[Any], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

_SAFE_UNARY: dict[type[Any], Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.expr) -> float:
    """Recursively evaluate a numeric AST node.

    Only numeric literals, binary arithmetic, and unary +/- are allowed.
    Any other node type raises ``ValueError``.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Non-numeric constant: {node.value!r}")

    if isinstance(node, ast.BinOp):
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return float(op_fn(left, right))

    if isinstance(node, ast.UnaryOp):
        op_fn = _SAFE_UNARY.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return float(op_fn(_eval_node(node.operand)))

    raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def safe_eval(expression: str) -> float:
    """Parse and evaluate an arithmetic *expression* string without ``eval``.

    Raises ``ValueError`` if the expression is invalid or contains non-numeric
    constructs (function calls, attribute access, names, etc.).
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in expression: {expression!r}") from exc

    return _eval_node(tree.body)


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------

_CALCULATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": (
                "An arithmetic expression using +, -, *, /, %, ** and numeric literals. "
                "No function calls or variable names are allowed."
            ),
        }
    },
    "required": ["expression"],
}


class CalculatorTool:
    """Evaluates arithmetic expressions using a safe AST-based evaluator.

    The tool never calls ``eval()`` or ``exec()``, so arbitrary code cannot
    be injected through the expression argument.

    Usage::

        calc = CalculatorTool()
        result = calc.execute(expression="(3 + 4) * 2 / 7")
        # result == "2.0"
    """

    name: str = "calculator"
    input_schema: dict[str, Any] = _CALCULATOR_SCHEMA

    def execute(self, **kwargs: Any) -> str:
        expression: str = kwargs.get("expression", "")
        if not expression:
            return "Error: 'expression' argument is required."
        try:
            value = safe_eval(expression)
        except (ValueError, ZeroDivisionError) as exc:
            return f"Error: {exc}"
        # Return a compact string; the runtime stores it as the tool_result content.
        return str(value)
