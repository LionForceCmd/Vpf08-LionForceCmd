"""Обёртка над LLM (OpenAI и совместимые провайдеры).

Класс ``OpenAILLM`` намеренно тонкий: он умеет только превратить сообщения и
список инструментов в один вызов Chat Completions и вернуть нормализованный ответ.
Вся логика выбора инструментов живёт в ``agent.py``, поэтому агента легко
тестировать с подставной «моделью» без обращения к сети.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Ошибка обращения к языковой модели."""


@dataclass(slots=True)
class ToolCall:
    """Запрос модели на вызов инструмента."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""


@dataclass(slots=True)
class LLMResponse:
    """Нормализованный ответ модели."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    model: str | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLM(Protocol):
    """Минимальный интерфейс, который нужен агенту."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        ...


def parse_tool_arguments(raw: str | None) -> dict[str, Any]:
    """Разбирает аргументы вызова: модель иногда присылает невалидный JSON."""
    if not raw:
        return {}
    text = raw.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Некоторые модели оборачивают JSON в markdown-блок ```json ... ```
        cleaned = text.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Не удалось разобрать аргументы инструмента: {text[:120]}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("Аргументы инструмента должны быть объектом JSON")
    return parsed


class OpenAILLM:
    """Реализация ``LLM`` поверх AsyncOpenAI."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise LLMError("Не задан OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def aclose(self) -> None:
        await self._client.close()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            completion = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - приводим к доменной ошибке
            raise LLMError(f"Запрос к LLM не удался: {exc}") from exc

        choice = completion.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        for raw_call in message.tool_calls or []:
            raw_arguments = getattr(raw_call.function, "arguments", "") or ""
            try:
                arguments = parse_tool_arguments(raw_arguments)
            except LLMError as exc:
                logger.warning("Аргументы инструмента %s не разобраны: %s", raw_call.function.name, exc)
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=raw_call.id,
                    name=raw_call.function.name,
                    arguments=arguments,
                    raw_arguments=raw_arguments,
                )
            )

        return LLMResponse(
            content=(message.content or "").strip() or None,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model=completion.model,
        )
