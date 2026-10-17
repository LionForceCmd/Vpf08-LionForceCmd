"""Тесты безопасного калькулятора и финансовых расчётов."""

from __future__ import annotations

import pytest

from calculators import (
    CalculationError,
    box_office_financials,
    format_money,
    format_runtime,
    format_votes,
    safe_eval,
)


class TestSafeEval:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("2+2", 4),
            ("(2+2)*10", 40),
            ("10/4", 2.5),
            ("2**10", 1024),
            ("10 % 3", 1),
            ("-5 + 3", -2),
            ("sqrt(144)", 12),
            ("max(1, 7, 3)", 7),
            ("abs(-3.5)", 3.5),
            ("round(3.14159, 2)", 3.14),
            ("2^2", None),  # «^» — это XOR в Python, выражением не является
        ],
    )
    def test_basic_arithmetic(self, expression: str, expected: object) -> None:
        if expected is None:
            with pytest.raises(CalculationError):
                safe_eval(expression)
        else:
            assert safe_eval(expression)["result"] == expected

    def test_russian_and_comma_notation(self) -> None:
        # Пользователи часто пишут «15% от 2500», «2х3» и запятую вместо точки
        assert safe_eval("2500 * 0.15")["result"] == 375
        assert safe_eval("2,5 + 2,5")["result"] == 5
        assert safe_eval("2х3")["result"] == 6
        assert safe_eval("2 x 3")["result"] == 6
        assert safe_eval("2 плюс 3")["result"] == 5
        assert safe_eval("5 минус 8")["result"] == -3

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('dir')",
            "open('/etc/passwd').read()",
            "[].__class__.__mro__[1].__subclasses__()",
            "().__class__.__bases__[0].__subclasses__()",
            "eval('1+1')",
            "exec('x=1')",
            "lambda: 1",
            "1 if True else 2",
            "x = 5",
        ],
    )
    def test_rejects_dangerous_expressions(self, expression: str) -> None:
        with pytest.raises(CalculationError):
            safe_eval(expression)

    def test_rejects_division_by_zero(self) -> None:
        with pytest.raises(CalculationError, match="деление на ноль"):
            safe_eval("1/0")

    def test_rejects_huge_power(self) -> None:
        with pytest.raises(CalculationError):
            safe_eval("2**1000")
        with pytest.raises(CalculationError):
            safe_eval("10**100")

    def test_rejects_too_long_expression(self) -> None:
        with pytest.raises(CalculationError, match="слишком длинное"):
            safe_eval("1+" * 300 + "1")

    def test_rejects_empty(self) -> None:
        for value in ("", "   ", None):
            with pytest.raises(CalculationError):
                safe_eval(value)  # type: ignore[arg-type]

    def test_constants_available(self) -> None:
        assert safe_eval("round(pi * 100) / 100")["result"] == 3.14

    def test_unknown_function_rejected(self) -> None:
        with pytest.raises(CalculationError, match="разрешённых функций"):
            safe_eval("exec('1')")

    def test_formatted_output(self) -> None:
        result = safe_eval("(2+2)*10")
        assert result["formatted"] == "(2+2)*10 = 40"
        assert result["expression"] == "(2+2)*10"


class TestBoxOffice:
    def test_profitable_movie(self) -> None:
        stats = box_office_financials(165_000_000, 434_000_000, "Дюна")
        assert stats["profit_usd"] == 269_000_000
        assert stats["roi_percent"] == 163.0
        assert stats["box_office_to_budget"] == 2.63
        assert stats["profit_sign"] == "прибыль"
        assert "окупился" in stats["verdict"]

    def test_loss_movie(self) -> None:
        stats = box_office_financials(200_000_000, 40_000_000, "Убыточный")
        assert stats["profit_usd"] == -160_000_000
        assert stats["profit_sign"] == "убыток"
        assert stats["roi_percent"] == -80.0

    def test_missing_budget(self) -> None:
        stats = box_office_financials(None, 1000, "Только сборы")
        assert stats["budget_usd"] == 0
        assert "roi_percent" not in stats

    def test_requires_some_numbers(self) -> None:
        with pytest.raises(CalculationError):
            box_office_financials(None, None, "Пусто")


class TestFormatters:
    @pytest.mark.parametrize(
        ("amount", "expected"),
        [
            (2_320_000_000, "$2.32 млрд"),
            (434_000_000, "$434.0 млн"),
            (45_000, "$45.0 тыс."),
            (999, "$999"),
            (0, "$0"),
            (None, "н/д"),
        ],
    )
    def test_format_money(self, amount: int | None, expected: str) -> None:
        assert format_money(amount) == expected

    @pytest.mark.parametrize(
        ("minutes", "expected"),
        [(157, "2 ч 37 мин"), (120, "2 ч"), (45, "45 мин"), (None, "н/д")],
    )
    def test_format_runtime(self, minutes: int | None, expected: str) -> None:
        assert format_runtime(minutes) == expected

    @pytest.mark.parametrize(
        ("votes", "expected"),
        [(2_321_391, "2,32 млн"), (500_000, "500 тыс."), (999, "999"), (None, "н/д")],
    )
    def test_format_votes(self, votes: int | None, expected: str) -> None:
        assert format_votes(votes) == expected
