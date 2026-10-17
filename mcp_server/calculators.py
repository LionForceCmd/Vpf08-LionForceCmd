"""Вычисления для MCP-инструментов: безопасный калькулятор и финансы кино.

Ключевой момент урока — ``eval()`` использовать нельзя: он умеет исполнять
произвольный код на сервере. Здесь выражения разбираются деревом ``ast`` и
вычисляются вручную по белому списку операций.
"""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

MAX_EXPRESSION_LENGTH = 200
MAX_POWER_EXPONENT = 16
MAX_ABS_VALUE = 1e15


class CalculationError(ValueError):
    """Некорректное или недопустимое выражение."""


# Двоичные операторы: только арифметика, никаких побитовых и сравнений.
_BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Разрешённые математические функции: имя → вычисление.
_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": lambda *args: sum(args),
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pow": math.pow,
}

# Константы, доступные без объявления.
_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}

_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    *tuple(_BINARY_OPS),
    *tuple(_UNARY_OPS),
)


def _check_node(node: ast.AST) -> None:
    if not isinstance(node, _ALLOWED_NODES):
        raise CalculationError(
            f"запрещённый элемент выражения: {type(node).__name__}"
        )


def _evaluate(node: ast.AST) -> Any:
    _check_node(node)

    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalculationError("разрешены только числовые константы")
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in _CONSTANTS:
            raise CalculationError(f"неизвестная константа: {node.id}")
        return _CONSTANTS[node.id]

    if isinstance(node, ast.UnaryOp):
        handler = _UNARY_OPS.get(type(node.op))
        if handler is None:
            raise CalculationError("запрещённая унарная операция")
        return handler(_evaluate(node.operand))

    if isinstance(node, ast.BinOp):
        handler = _BINARY_OPS.get(type(node.op))
        if handler is None:
            raise CalculationError("запрещённая бинарная операция")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            _guard_power(left, right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise CalculationError("деление на ноль")
        result = handler(left, right)
        return _guard_result(result)

    if isinstance(node, ast.Call):
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in _FUNCTIONS:
            raise CalculationError("вызов разрешённых функций: " + ", ".join(sorted(_FUNCTIONS)))
        if node.keywords:
            raise CalculationError("именованные аргументы не поддерживаются")
        args = [_evaluate(arg) for arg in node.args]
        if not args:
            raise CalculationError("функции нужны аргументы")
        try:
            return _guard_result(_FUNCTIONS[func.id](*args))
        except CalculationError:
            raise
        except Exception as exc:  # noqa: BLE001 - приводим к доменной ошибке
            raise CalculationError(f"не удалось вычислить {func.id}: {exc}") from exc

    raise CalculationError(f"неподдерживаемое выражение: {type(node).__name__}")


def _guard_power(base: Any, exponent: Any) -> None:
    if abs(exponent) > MAX_POWER_EXPONENT:
        raise CalculationError(
            f"показатель степени должен быть не больше {MAX_POWER_EXPONENT}"
        )
    if abs(base) > 10:
        raise CalculationError("слишком большое основание для возведения в степень")


def _guard_result(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CalculationError("результат должен быть числом")
    if math.isinf(value) or math.isnan(value):
        raise CalculationError("результат не является конечным числом")
    if abs(value) > MAX_ABS_VALUE:
        raise CalculationError("результат слишком большой")
    return value


def safe_eval(expression: str) -> dict[str, Any]:
    """Безопасно вычисляет арифметическое выражение.

    Разрешены: ``+ - * / // % **``, скобки, унарный минус, константы
    ``pi``, ``e``, ``tau`` и функции ``sqrt``, ``abs``, ``round``, ``min``,
    ``max``, ``log``, ``floor``, ``ceil`` и другие из белого списка.

    Возвращает ``{"expression": ..., "result": ..., "formatted": ...}``.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise CalculationError("пустое выражение")

    cleaned = expression.strip()
    if len(cleaned) > MAX_EXPRESSION_LENGTH:
        raise CalculationError(
            f"слишком длинное выражение (максимум {MAX_EXPRESSION_LENGTH} символов)"
        )

    # Русские и альтернативные обозначения — частая ошибка пользователей в чате.
    normalized = cleaned.lower().replace("плюс", "+").replace("минус", "-")
    # «2х2» / «2 x 2» → «2*2», но только между числами, чтобы не сломать max(...)
    normalized = re.sub(r"(?<=\d)\s*[xх]\s*(?=\d)", "*", normalized)
    # Запятая как десятичный разделитель («2,5»), но не разделитель аргументов max(1, 2)
    normalized = re.sub(r"(?<=\d),(?=\d)", ".", normalized)

    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        raise CalculationError(f"не удалось разобрать выражение: {exc.msg}") from exc

    result = _evaluate(tree)
    if isinstance(result, float) and result.is_integer():
        result = int(result)

    return {
        "expression": cleaned,
        "result": result,
        "formatted": f"{cleaned} = {result}",
    }


def box_office_financials(
    budget_usd: float | None,
    revenue_usd: float | None,
    title: str | None = None,
) -> dict[str, Any]:
    """Считает прибыль, рентабельность и окупаемость по бюджету и сборам."""
    if budget_usd is None and revenue_usd is None:
        raise CalculationError("нужны бюджет или кассовые сборы фильма")

    budget = float(budget_usd or 0)
    revenue = float(revenue_usd or 0)

    profit = revenue - budget
    roi = (profit / budget * 100) if budget > 0 else None
    markup = (revenue / budget) if budget > 0 else None
    breakeven_multiple = (budget / revenue) if revenue > 0 else None

    result: dict[str, Any] = {
        "title": title,
        "budget_usd": budget,
        "revenue_usd": revenue,
        "profit_usd": profit,
        "budget_formatted": format_money(budget),
        "revenue_formatted": format_money(revenue),
        "profit_formatted": format_money(profit),
        "profit_sign": "прибыль" if profit >= 0 else "убыток",
    }
    if roi is not None:
        result["roi_percent"] = round(roi, 1)
        result["verdict"] = "окупился в прокате" if profit >= 0 else "не окупился в прокате"
    if markup is not None:
        result["box_office_to_budget"] = round(markup, 2)
    if breakeven_multiple is not None:
        result["revenue_per_budget_dollar"] = round(revenue / budget, 2) if budget else None
    return result


def format_money(amount: float | None) -> str:
    """Человекочитаемый формат долларов: ``$1.05 млрд`` / ``$45.0 млн``."""
    if amount is None:
        return "н/д"
    value = float(amount)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if value >= 1_000_000_000:
        return f"{sign}${value / 1_000_000_000:.2f} млрд"
    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.1f} млн"
    if value >= 1_000:
        return f"{sign}${value / 1_000:.1f} тыс."
    return f"{sign}${value:,.0f}"


def format_runtime(minutes: int | None) -> str:
    """``157`` → ``2 ч 37 мин``."""
    if not minutes:
        return "н/д"
    hours, mins = divmod(int(minutes), 60)
    if hours and mins:
        return f"{hours} ч {mins} мин"
    if hours:
        return f"{hours} ч"
    return f"{mins} мин"


def format_votes(votes: int | None) -> str:
    """``2321391`` → ``2,32 млн``, ``12000`` → ``12 тыс.``."""
    if not votes:
        return "н/д"
    value = float(votes)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} млн".replace(".", ",")
    if value >= 1_000:
        return f"{int(value // 1_000)} тыс."
    return str(int(value))
