"""Агент: цикл «LLM → выбор MCP-инструмента → вызов → ответ пользователю».

Это и есть то место, где MCP окупается: модель не знает ничего о базе, но
получает список инструментов с сервера и сама решает, что вызвать.

Агент умеет два режима выбора инструмента:
1. native function calling — основной путь (OpenAI ``tools``);
2. резервный JSON-план ``{"tool": ..., "arguments": {...}}`` из урока — если
   модель вернула такой объект обычным текстом.

Класс не зависит от Telegram и от OpenAI напрямую, поэтому его целиком
покрывают тесты с подставными ``LLM`` и ``MCPClient``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from llm import LLM, LLMError, LLMResponse, ToolCall
from mcp_client import MCPClient, MCPError
from prompts import build_system_prompt, format_tool_call_log

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def assistant_tool_message(response: LLMResponse) -> dict[str, Any]:
    """Собирает сообщение assistant с вызовами инструментов в формате OpenAI."""
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments or json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in response.tool_calls
        ],
    }


class AgentError(RuntimeError):
    """Ошибка, из-за которой нельзя получить ответ пользователю."""


@dataclass(slots=True)
class AgentReply:
    """Ответ агента вместе с технической информацией для логов и тестов."""

    text: str
    tool_calls: list[str] = field(default_factory=list)
    rounds: int = 0
    used_fallback_plan: bool = False

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)


class MovieAgent:
    """Связывает LLM и MCP-сервер кино-каталога."""

    def __init__(
        self,
        llm: LLM,
        mcp: MCPClient,
        *,
        max_tool_rounds: int = 6,
        history_turns: int = 12,
        system_prompt: str | None = None,
    ) -> None:
        self.llm = llm
        self.mcp = mcp
        self.max_tool_rounds = max_tool_rounds
        self.history_turns = history_turns
        self._system_prompt = system_prompt
        self._tool_names: list[str] | None = None
        self._openai_tools: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------- инициализация

    async def prepare(self) -> list[str]:
        """Рукопожатие с MCP-сервером и кэширование списка инструментов."""
        await self.mcp.initialize()
        self._tool_names = await self.mcp.tool_names()
        self._openai_tools = await self.mcp.openai_tools()
        return self._tool_names

    @property
    def system_prompt(self) -> str:
        if self._system_prompt is not None:
            return self._system_prompt
        return build_system_prompt(self._tool_names)

    # ------------------------------------------------------------------ запуск

    async def answer(
        self, user_text: str, history: list[dict[str, Any]] | None = None, *, user_id: str = "anonymous"
    ) -> AgentReply:
        """Отвечает на одно сообщение пользователя."""
        if self._openai_tools is None:
            await self.prepare()

        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self._trim_history(history or []))
        messages.append({"role": "user", "content": user_text})

        executed: list[str] = []
        used_fallback = False

        for round_number in range(1, self.max_tool_rounds + 1):
            try:
                response = await self.llm.complete(messages, self._openai_tools)
            except LLMError as exc:
                raise AgentError(f"Языковая модель недоступна: {exc}") from exc

            if response.wants_tools:
                messages.append(assistant_tool_message(response))
                for tool_call in response.tool_calls:
                    label = format_tool_call_log(tool_call.name, tool_call.arguments)
                    tool_message = await self._run_tool(tool_call, user_id)
                    executed.append(label)
                    messages.append(tool_message)
                logger.info("Раунд %d: вызваны инструменты %s", round_number, executed)
                continue

            answer_text = (response.content or "").strip()

            # Модель без native function calling вернула план инструмента текстом.
            plan = self._extract_plan(answer_text)
            if plan is not None:
                used_fallback = True
                tool_name, arguments = plan
                label = format_tool_call_log(tool_name, arguments)
                executed.append(label)
                rendered = await self._render_tool_result(
                    tool_name,
                    ToolCall(id="call_plan", name=tool_name, arguments=arguments),
                    executed,
                )
                logger.info("Резервный JSON-план: %s", label)
                return AgentReply(
                    text=rendered,
                    tool_calls=executed,
                    rounds=round_number,
                    used_fallback_plan=True,
                )

            if not answer_text:
                answer_text = "Не понял запрос. Уточни, пожалуйста, что именно нужно найти."

            logger.info("Ответ без инструментов после %d раундов: %s", round_number, executed or "[]")
            return AgentReply(
                text=answer_text,
                tool_calls=executed,
                rounds=round_number,
                used_fallback_plan=used_fallback,
            )

        # Лимит раундов исчерпан — просим модель ответить на имеющихся данных.
        try:
            final = await self.llm.complete(
                messages + [{"role": "user", "content": "Сделай итоговый вывод по этим данным."}],
                None,
            )
            text = (final.content or "").strip() or self._fallback_text(executed)
        except LLMError:
            text = self._fallback_text(executed)

        return AgentReply(
            text=text,
            tool_calls=executed,
            rounds=self.max_tool_rounds,
            used_fallback_plan=used_fallback,
        )

    # ------------------------------------------------------------- внутреннее

    def _trim_history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Оставляет последние N сообщений диалога и не начинает с assistant."""
        trimmed = [message for message in history if message.get("role") in {"user", "assistant"}]
        trimmed = trimmed[-self.history_turns * 2 :] if self.history_turns > 0 else []
        while trimmed and trimmed[0]["role"] == "assistant":
            trimmed.pop(0)
        return trimmed

    def _assistant_message(self, response: LLMResponse) -> dict[str, Any]:
        """Сообщение assistant для истории диалога (без служебных вызовов)."""
        return {"role": "assistant", "content": (response.content or "").strip()}

    async def _run_tool(self, tool_call: ToolCall, user_id: str) -> dict[str, Any]:
        """Вызывает MCP-инструмент и возвращает сообщение ``role="tool"``.

        ``tool_call_id`` обязан совпадать с id вызова из последнего сообщения
        assistant, иначе OpenAI отвергнет всю переписку.
        """
        try:
            result = await self.mcp.call_tool(tool_call.name, tool_call.arguments, user_id=user_id)
        except MCPError as exc:
            logger.warning("Инструмент %s недоступен: %s", tool_call.name, exc)
            content: dict[str, Any] = {
                "error": f"Инструмент «{tool_call.name}» вернул ошибку: {exc}",
                "is_error": True,
            }
            is_error = True
        else:
            is_error = result.is_error
            content = {"is_error": is_error, **(result.data or {})}

        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.name,
            "content": json.dumps(content, ensure_ascii=False),
        }

    async def _render_tool_result(
        self,
        tool_name: str,
        tool_call: ToolCall,
        executed: list[str],
    ) -> str:
        """После резервного JSON-плана просит модель оформить результат."""
        tool_message = await self._run_tool(tool_call, user_id="plan")
        try:
            final = await self.llm.complete(
                [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": f"Вызови инструмент {tool_name} и оформи результат для пользователя.",
                    },
                    assistant_tool_message(
                        LLMResponse(tool_calls=[tool_call])
                    ),
                    tool_message,
                ],
                None,
            )
            text = (final.content or "").strip()
            if text:
                return text
        except LLMError as exc:
            logger.warning("Не удалось оформить результат инструмента: %s", exc)

        try:
            data = json.loads(tool_message["content"])
        except json.JSONDecodeError:
            return "Инструмент отработал, но вернул нечитаемый ответ."
        if isinstance(data, dict) and data.get("error"):
            return f"Не получилось выполнить запрос: {data['error']}"
        return self._fallback_text(executed)

    @staticmethod
    def _extract_plan(text: str) -> tuple[str, dict[str, Any]] | None:
        """Пытается достать из текста план ``{"tool": ..., "arguments": {...}}``."""
        if not text or "tool" not in text:
            return None

        candidates: list[str] = []
        match = _JSON_BLOCK.search(text)
        if match:
            candidates.append(match.group(1))
        candidates.append(text)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])

        for candidate in candidates:
            try:
                parsed = json.loads(candidate.strip())
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            tool_name = parsed.get("tool")
            if not isinstance(tool_name, str) or not tool_name:
                continue
            arguments = parsed.get("arguments") or {}
            if not isinstance(arguments, dict):
                continue
            return tool_name, arguments
        return None

    @staticmethod
    def _fallback_text(executed: list[str]) -> str:
        if executed:
            return (
                "Я собрал данные по запросу, но не смог красиво их оформить.\n"
                f"Использованные инструменты: {', '.join(executed)}."
            )
        return "Не получилось получить данные из каталога. Попробуй переформулировать запрос."
