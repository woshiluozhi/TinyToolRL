"""Safe calculator tool used by the agent environment.

真实 agent 往往能调用搜索、代码解释器、数据库等工具。为了让项目能在
普通电脑上稳定复现，这里只实现一个安全版计算器 CALC[...]。

重点：不要直接对模型输出使用 ``eval``。模型输出是不可信文本，所以我们
用 Python 的 ``ast`` 解析表达式，只允许加减乘除、幂和少数数学函数。
"""

from __future__ import annotations

import ast
import math
from typing import Any


class CalculatorError(ValueError):
    """Raised when a calculator expression is invalid or unsafe."""


ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
}

ALLOWED_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
}

ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}

ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


def safe_calculate(expression: str) -> float:
    """Evaluate a small arithmetic expression safely."""

    expression = expression.strip()
    if not expression:
        raise CalculatorError("empty expression")
    if len(expression) > 120:
        raise CalculatorError("expression is too long")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"syntax error: {exc}") from exc

    result = _eval_node(tree.body)
    if not math.isfinite(result):
        raise CalculatorError("non-finite result")
    return float(result)


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate only whitelisted AST nodes."""

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise CalculatorError("only numeric constants are allowed")

    if isinstance(node, ast.Name):
        if node.id in ALLOWED_NAMES:
            return float(ALLOWED_NAMES[node.id])
        raise CalculatorError(f"name not allowed: {node.id}")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_BINOPS:
            raise CalculatorError(f"operator not allowed: {op_type.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        try:
            return float(ALLOWED_BINOPS[op_type](left, right))
        except ZeroDivisionError as exc:
            raise CalculatorError("division by zero") from exc

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in ALLOWED_UNARYOPS:
            raise CalculatorError(f"unary operator not allowed: {op_type.__name__}")
        return float(ALLOWED_UNARYOPS[op_type](_eval_node(node.operand)))

    if isinstance(node, ast.Call):
        function = _resolve_function(node.func)
        args = [_eval_node(arg) for arg in node.args]
        if node.keywords:
            raise CalculatorError("keyword arguments are not allowed")
        try:
            return float(function(*args))
        except (TypeError, ValueError) as exc:
            raise CalculatorError(f"bad function call: {exc}") from exc

    raise CalculatorError(f"node not allowed: {type(node).__name__}")


def _resolve_function(node: ast.AST) -> Any:
    if isinstance(node, ast.Name) and node.id in ALLOWED_FUNCTIONS:
        return ALLOWED_FUNCTIONS[node.id]
    raise CalculatorError("function not allowed")

