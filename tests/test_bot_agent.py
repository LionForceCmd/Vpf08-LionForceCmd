"""Тесты Telegram-бота: MCP-клиент, преобразование схем, цикл агента, бот-логика.

Ключевая идея: LLM подменяется скриптом, а MCP-клиент работает с настоящим
сервером. Поэтому проверяется весь путь «сообщение → выбор инструмента →
вызов MCP → ответ», который и демонстрируется в домашнем задании.
"""

from __future__ import annotations

import json

import pytest

from agent import MovieAgent, assistant_tool_message
from bot import split_message
from llm import LLMResponse, parse_tool_arguments
from mcp_client import MCPError, MCPTool
from prompts import build_system_prompt, format_tool_call_log


# --------------------------------------------------------------------------- #
# MCP-клиент
# --------------------------------------------------------------------------- #

class TestMCPClient:
    async def test_initialize(self, mcp_client) -> None:
        info = await mcp_client.initialize()
        assert info["name"] == "movie-mcp"
        assert info["version"]
        assert mcp_client.protocol_version
        assert "каталог" in mcp_client.instructions.casefold()
        assert "movie-mcp" in mcp_client.instructions.casefold()

    async def test_list_tools(self, mcp_client) -> None:
        await mcp_client.initialize()
        tools = await mcp_client.list_tools()
        assert len(tools) >= 15
        assert {"find_movie", "list_movies", "calculate"} <= {tool.name for tool in tools}

    async def test_tools_are_cached(self, mcp_client) -> None:
        await mcp_client.initialize()
        first = await mcp_client.list_tools()
        second = await mcp_client.list_tools()
        assert first is second

    async def test_openai_tools_format(self, mcp_client) -> None:
        await mcp_client.initialize()
        functions = await mcp_client.openai_tools()
        for function in functions:
            assert function["type"] == "function"
            spec = function["function"]
            assert spec["name"]
            assert spec["description"]
            assert spec["parameters"]["type"] == "object"
            assert isinstance(spec["parameters"]["properties"], dict)

    async def test_call_tool(self, mcp_client) -> None:
        await mcp_client.initialize()
        result = await mcp_client.call_tool("find_movie", {"query": "нач"})
        assert result.is_error is False
        assert result.data["movies"][0]["title"] == "Начало"
        assert json.loads(result.text)["found"] >= 1

    async def test_call_tool_reports_errors(self, mcp_client) -> None:
        await mcp_client.initialize()
        result = await mcp_client.call_tool("find_movie", {"query": ""})
        assert result.is_error is True
        assert "error" in result.data

    async def test_user_id_is_forwarded(self, mcp_client) -> None:
        await mcp_client.initialize()
        await mcp_client.call_tool("rate_movie", {"title": "Дюна", "score": 8}, user_id="carol")
        ratings = await mcp_client.call_tool("get_user_ratings", {}, user_id="carol")
        assert ratings.data["count"] == 1

    async def test_unknown_tool_raises(self, mcp_client) -> None:
        await mcp_client.initialize()
        with pytest.raises(MCPError):
            await mcp_client.call_tool("нет_такого", {})

    async def test_unreachable_server(self) -> None:
        from mcp_client import MCPClient

        client = MCPClient("http://127.0.0.1:1/mcp", timeout=1.0)
        with pytest.raises(MCPError):
            await client.initialize()
        await client.aclose()


class TestToolSchemaConversion:
    def test_input_schema_is_mapped_to_parameters(self) -> None:
        tool = MCPTool.from_mcp(
            {
                "name": "find_movie",
                "description": "Ищет фильм",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Запрос"}},
                    "required": ["query"],
                    "$schema": "https://json-schema.org/draft-07/schema",
                    "additionalProperties": False,
                },
            }
        )
        function = tool.to_openai_function()["function"]
        assert function["name"] == "find_movie"
        assert function["parameters"]["required"] == ["query"]
        assert function["parameters"]["properties"]["query"]["type"] == "string"
        # Служебные ключи JSON Schema в OpenAI не передаём
        assert "$schema" not in function["parameters"]
        assert "additionalProperties" not in function["parameters"]

    def test_missing_schema_is_tolerated(self) -> None:
        tool = MCPTool.from_mcp({"name": "ping_tool", "description": "d"})
        assert tool.to_openai_function()["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }


# --------------------------------------------------------------------------- #
# Разбор аргументов инструментов
# --------------------------------------------------------------------------- #

class TestParseToolArguments:
    def test_plain_json(self) -> None:
        assert parse_tool_arguments('{"query": "нач"}') == {"query": "нач"}

    def test_markdown_wrapped(self) -> None:
        assert parse_tool_arguments('```json\n{"query": "x"}\n```') == {"query": "x"}

    def test_empty(self) -> None:
        assert parse_tool_arguments("") == {}
        assert parse_tool_arguments(None) == {}

    def test_invalid_raises(self) -> None:
        from llm import LLMError

        with pytest.raises(LLMError):
            parse_tool_arguments("{не json}")

    def test_non_object_raises(self) -> None:
        from llm import LLMError

        with pytest.raises(LLMError):
            parse_tool_arguments("[1, 2]")


# --------------------------------------------------------------------------- #
# Цикл агента
# --------------------------------------------------------------------------- #

@pytest.fixture
async def agent(mcp_client, scripted_llm):
    """Готовый агент, подключённый к тестовому MCP-серверу."""

    async def _factory(responses):
        llm = scripted_llm(responses)
        instance = MovieAgent(llm, mcp_client, max_tool_rounds=5, history_turns=6)
        await instance.prepare()
        return instance, llm

    return _factory


class TestAgentLoop:
    async def test_direct_answer_without_tools(self, agent) -> None:
        instance, llm = await agent(["Привет! Я — Кино-Навигатор, чем помочь?"])
        reply = await instance.answer("привет", user_id="u1")
        assert reply.text.startswith("Привет")
        assert reply.used_tools is False
        assert llm.calls[0]["tools"]  # модель получила список инструментов

    async def test_tool_call_then_answer(self, agent) -> None:
        instance, llm = await agent(
            [
                ("find_movie", {"query": "нач"}),
                "Нашёл: «Начало» (2010), рейтинг IMDb 8.8, режиссёр Кристофер Нолан.",
            ]
        )
        reply = await instance.answer("найди Начало", user_id="u1")
        assert reply.used_tools is True
        assert reply.tool_calls == ['find_movie({"query": "нач"})']
        assert "Начало" in reply.text
        assert reply.rounds == 2

    async def test_arguments_are_passed_to_server(self, agent) -> None:
        instance, llm = await agent(
            [("list_movies", {"kind": "anime", "limit": 3}), "Вот три аниме."]
        )
        await instance.answer("покажи аниме", user_id="u1")

        tool_message = llm.calls[1]["messages"][-1]
        payload = json.loads(tool_message["content"])
        assert payload["total_found"] >= 3
        assert all(movie["kind"] == "anime" for movie in payload["movies"])

    async def test_several_tool_calls_in_one_round(self, agent) -> None:
        instance, llm = await agent(
            [
                {
                    "tool_calls": [
                        {"id": "c1", "name": "find_movie", "arguments": {"query": "Дюна"}},
                        {"id": "c2", "name": "get_movie_stats", "arguments": {}},
                    ]
                },
                "Собрал всё вместе.",
            ]
        )
        reply = await instance.answer("дай инфу про Дюну и статистику", user_id="u1")
        assert len(reply.tool_calls) == 2
        assert "Собрал всё вместе." in reply.text

    async def test_multiple_rounds(self, agent) -> None:
        instance, _ = await agent(
            [
                ("find_movie", {"query": "Начало"}),
                ("box_office_stats", {"title": "Начало"}),
                "«Начало» заработало $679 млн.",
            ]
        )
        reply = await instance.answer("сколько заработало Начало", user_id="u1")
        assert len(reply.tool_calls) == 2
        assert reply.rounds == 3

    async def test_tool_error_is_passed_to_model(self, agent) -> None:
        instance, llm = await agent(
            [("find_movie", {"query": ""}), "Не понял запрос, уточни название."]
        )
        reply = await instance.answer("ну фильм", user_id="u1")
        payload = json.loads(llm.calls[1]["messages"][-1]["content"])
        assert payload["is_error"] is True
        assert "уточни" in reply.text

    async def test_history_is_passed(self, agent) -> None:
        instance, llm = await agent(["Ок"])
        history = [
            {"role": "user", "content": "привет"},
            {"role": "assistant", "content": "Привет!"},
            {"role": "system", "content": "мусорный промпт"},
        ]
        await instance.answer("а теперь топ", history, user_id="u1")

        messages = llm.calls[0]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "привет"}
        assert messages[2] == {"role": "assistant", "content": "Привет!"}
        assert messages[-1] == {"role": "user", "content": "а теперь топ"}
        assert all(message["role"] != "system" for message in messages[1:])

    async def test_history_starts_not_with_assistant(self, agent) -> None:
        instance, llm = await agent(["Ок"])
        history = [{"role": "assistant", "content": "старый ответ"}]
        await instance.answer("вопрос", history, user_id="u1")
        assert llm.calls[0]["messages"][1]["role"] == "user"

    async def test_round_limit_is_respected(self, agent) -> None:
        # Модель бесконечно просит инструменты — агент обязан остановиться
        instance, _ = await agent([("get_movie_stats", {})] * 10 + ["Итог."])
        reply = await instance.answer("цикл", user_id="u1")
        assert reply.rounds == 5
        assert len(reply.tool_calls) == 5

    async def test_legacy_json_plan(self, agent) -> None:
        """Модель без function calling возвращает план обычным текстом."""
        instance, _ = await agent(
            [
                '{"tool": "find_movies_by_genre", "arguments": {"genre": "Драма", "limit": 2}}',
                "Две драмы: «Побег из Шоушенка» и «Форрест Гамп».",
            ]
        )
        reply = await instance.answer("покажи драмы", user_id="u1")
        assert reply.used_fallback_plan is True
        assert reply.tool_calls == ['find_movies_by_genre({"genre": "Драма", "limit": 2})']
        assert "драмы" in reply.text

    async def test_json_plan_in_markdown(self, agent) -> None:
        instance, _ = await agent(
            [
                'Сейчас посмотрю.\n```json\n{"tool": "get_movie_stats", "arguments": {}}\n```',
                "В каталоге 151 фильм.",
            ]
        )
        reply = await instance.answer("сколько фильмов", user_id="u1")
        assert reply.used_fallback_plan is True
        assert reply.tool_calls == ["get_movie_stats"]

    async def test_plain_text_with_word_tool_is_not_a_plan(self, agent) -> None:
        instance, _ = await agent(["Инструменты доступны, но вопросов нет."])
        reply = await instance.answer("сколько фильмов", user_id="u1")
        assert reply.used_fallback_plan is False
        assert reply.used_tools is False

    async def test_empty_answer_gets_fallback_text(self, agent) -> None:
        instance, _ = await agent([LLMResponse(content=None)])
        reply = await instance.answer("хм", user_id="u1")
        assert reply.text
        assert "Уточни" in reply.text or "уточни" in reply.text

    async def test_system_prompt_lists_tools(self, agent) -> None:
        instance, _ = await agent(["Ок"])
        prompt = instance.system_prompt
        assert "Кино-Навигатор" in prompt
        assert "list_movies" in prompt
        assert "find_movie" in prompt
        assert "calculate" in prompt


class TestAgentHelpers:
    def test_assistant_tool_message_shape(self) -> None:
        from llm import ToolCall

        response = LLMResponse(
            tool_calls=[ToolCall(id="call_1", name="find_movie", arguments={"query": "x"})]
        )
        message = assistant_tool_message(response)
        assert message["role"] == "assistant"
        assert message["tool_calls"][0]["id"] == "call_1"
        assert message["tool_calls"][0]["type"] == "function"
        assert message["tool_calls"][0]["function"]["name"] == "find_movie"
        # arguments должны быть строкой JSON, как требует OpenAI
        assert isinstance(message["tool_calls"][0]["function"]["arguments"], str)
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"query": "x"}

    def test_prompt_mentions_fallback_format(self) -> None:
        from prompts import legacy_plan_prompt

        prompt = legacy_plan_prompt()
        assert '{"tool":' in prompt
        assert "arguments" in prompt

    def test_format_tool_call_log(self) -> None:
        assert format_tool_call_log("calculate", {"expression": "2+2"}) == (
            'calculate({"expression": "2+2"})'
        )
        assert format_tool_call_log("get_movie_stats") == "get_movie_stats"
        long_call = format_tool_call_log("find_movie", {"query": "д" * 300})
        assert len(long_call) < 200
        assert "..." in long_call

    def test_build_system_prompt_with_tools(self) -> None:
        prompt = build_system_prompt(["find_movie", "calculate"])
        assert "Инструменты, доступные в текущей сессии MCP" in prompt
        assert "find_movie" in prompt


# --------------------------------------------------------------------------- #
# Логика бота
# --------------------------------------------------------------------------- #

class TestSplitMessage:
    def test_short_message_is_single_chunk(self) -> None:
        assert split_message("привет", 100) == ["привет"]

    def test_splits_on_line_boundaries(self) -> None:
        text = "\n".join("строка " * 5 for _ in range(10))
        chunks = split_message(text, 100)
        assert len(chunks) > 1
        assert all(len(chunk) <= 100 for chunk in chunks)
        assert "\n".join(chunks) == text

    def test_splits_very_long_line(self) -> None:
        chunks = split_message("я" * 350, 100)
        assert all(len(chunk) <= 100 for chunk in chunks)
        assert "".join(chunks) == "я" * 350

    def test_empty_message(self) -> None:
        assert split_message("", 100) == [""]


class TestBotConfig:
    def test_describe_hides_secrets(self) -> None:
        from config import BotSettings

        described = BotSettings().describe()
        assert described["telegram_token"] in {"задан", "НЕ ЗАДАН"}
        assert described["openai_api_key"] in {"задан", "НЕ ЗАДАН"}
        # Значения токенов в описании не утекают
        assert not any("sk-" in str(value) for value in described.values())

    def test_missing_tokens_detected(self) -> None:
        from config import BotSettings

        settings = BotSettings()
        settings.telegram_token = ""
        settings.openai_api_key = ""
        assert settings.missing_telegram_token() is True
        assert settings.missing_openai_key() is True

    def test_defaults(self) -> None:
        from config import BotSettings

        settings = BotSettings()
        assert settings.llm_model
        assert settings.mcp_url.endswith("/mcp")
        assert settings.max_tool_rounds >= 1
        assert settings.max_message_length >= 1000
