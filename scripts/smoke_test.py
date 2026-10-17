"""Сквозная проверка проекта без Telegram: поднимает сервер и прогоняет сценарии.

Запуск::

    python scripts/smoke_test.py

Скрипт сам поднимает MCP-сервер на свободном порту (временная база), подключается
настоящим ``MCPClient`` из ``telegram_bot`` и проверяет:

* рукопожатие ``initialize`` и список инструментов;
* типовые сценарии из домашнего задания;
* полный цикл агента «вопрос → выбор инструмента → вызов → ответ»
  (с подставной LLM, чтобы не тратить токены; при наличии ключа и
  ``--live`` можно прогнать настоящую модель).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (PROJECT_ROOT, PROJECT_ROOT / "mcp_server", PROJECT_ROOT / "telegram_bot"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.server_runner import ServerHandle  # noqa: E402

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


def _print_ok(label: str, detail: str = "") -> None:
    print(f"{GREEN}OK{RESET}   {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))


def _print_fail(label: str, detail: str = "") -> None:
    print(f"{RED}FAIL{RESET} {label}" + (f"  {detail}" if detail else ""))


def _print_skip(label: str, detail: str = "") -> None:
    print(f"{YELLOW}SKIP{RESET} {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))


def _brief(data: object, limit: int = 90) -> str:
    text = str(data)
    return text[: limit - 3] + "..." if len(text) > limit else text


# --------------------------------------------------------------------------- #
# Проверки
# --------------------------------------------------------------------------- #

async def check_tools_discovery(mcp_client) -> None:
    info = await mcp_client.initialize()
    _print_ok("initialize", f"{info['name']} v{info['version']}, протокол {mcp_client.protocol_version}")

    tools = await mcp_client.list_tools()
    _print_ok("tools/list", f"{len(tools)} инструментов")

    openai_tools = await mcp_client.openai_tools()
    assert all(tool["function"]["parameters"]["type"] == "object" for tool in openai_tools)
    _print_ok("конвертация схем в формат OpenAI", f"{len(openai_tools)} шт.")


async def check_scenarios(mcp_client) -> None:
    scenarios: list[tuple[str, dict]] = [
        ("покажи все товары → list_movies", ("list_movies", {"limit": 3})),
        ("найди фильм → find_movie", ("find_movie", {"query": "нач"})),
        ("найди по id → get_movie", ("get_movie", {"movie_id": 1})),
        ("поиск по жанру → find_movies_by_genre", ("find_movies_by_genre", {"genre": "Драма", "limit": 3})),
        ("фильмография → get_person_filmography", ("get_person_filmography", {"name": "Кристофер Нолан"})),
        ("топ по сборам → get_top_movies", ("get_top_movies", {"by": "revenue", "limit": 3})),
        ("похожие → get_similar_movies", ("get_similar_movies", {"title": "Начало", "limit": 3})),
        ("статистика → get_movie_stats", ("get_movie_stats", {})),
        ("кассовые сборы → box_office_stats", ("box_office_stats", {"title": "Дюна"})),
        ("калькулятор → calculate", ("calculate", {"expression": "(2+2)*10"})),
        ("оценка → rate_movie", ("rate_movie", {"title": "Начало", "score": 9})),
        ("избранное → toggle_favorite", ("toggle_favorite", {"title": "Дюна"})),
        ("мои оценки → get_user_ratings", ("get_user_ratings", {})),
        ("рекомендации → get_recommendations", ("get_recommendations", {"limit": 3})),
        ("добавление → add_movie", ("add_movie", {"title": "Дым над Уоквиллом", "year": 1955, "genres": "Драма", "rating_imdb": 8.2})),
    ]

    for label, (name, arguments) in scenarios:
        result = await mcp_client.call_tool(name, arguments, user_id="smoke-user")
        assert not result.is_error, result.text
        _print_ok(label, _brief(result.data))


async def check_write_cycle(mcp_client) -> None:
    created = await mcp_client.call_tool(
        "add_movie", {"title": "Дым над Уоквиллом", "year": 1955, "genres": "драма"}, user_id="smoke-user"
    )
    movie_id = created.data["movie"]["id"]

    duplicate = await mcp_client.call_tool("add_movie", {"title": "Дым над Уоквиллом"})
    assert duplicate.is_error, "дубликат должен отклоняться"
    _print_ok("защита от дублей", duplicate.data["error"])

    updated = await mcp_client.call_tool(
        "update_movie", {"movie_id": movie_id, "fields": {"rating_kp": 8.0}}
    )
    assert not updated.is_error
    _print_ok("update_movie", f"id={movie_id}, поля: {updated.data['updated_fields']}")

    deleted = await mcp_client.call_tool("delete_movie", {"movie_id": movie_id})
    assert not deleted.is_error
    _print_ok("delete_movie", deleted.data["message"])


async def check_calculator_security(mcp_client) -> None:
    attempts = [
        "__import__('os').system('echo hacked')",
        "open('C:/Windows/win.ini').read()",
        "[].__class__.__base__.__subclasses__()",
    ]
    for expression in attempts:
        result = await mcp_client.call_tool("calculate", {"expression": expression})
        assert result.is_error, f"выражение не отклонено: {expression}"
    _print_ok("калькулятор не выполняет произвольный код", f"{len(attempts)} попыток отклонено")

    good = await mcp_client.call_tool("calculate", {"expression": "sqrt(144) + 2**3"})
    assert good.data["result"] == 20
    _print_ok("калькулятор считает", good.data["formatted"])


class _ScriptedLLM:
    """Подставная модель: имитирует выбор инструмента обычным текстом."""

    def __init__(self, plans: list[str]) -> None:
        self.plans = list(plans)
        self.step = 0

    async def complete(self, messages, tools=None, *, temperature=None, max_tokens=None):
        from llm import LLMResponse

        if self.step < len(self.plans):
            content = self.plans[self.step]
            self.step += 1
            return LLMResponse(content=content)
        return LLMResponse(content="Готово, вот что нашёл в каталоге.")


async def check_agent_loop(mcp_client) -> None:
    from agent import MovieAgent

    llm = _ScriptedLLM(
        [
            '{"tool": "find_movie", "arguments": {"query": "нач"}}',
            '«Начало» (2010) — Кристофер Нолан, IMDb 8.8.',
        ]
    )
    agent = MovieAgent(llm, mcp_client, max_tool_rounds=4, history_turns=6)
    await agent.prepare()

    reply = await agent.answer("найди фильм Начало", user_id="smoke-user")
    assert reply.used_tools, "агент должен был вызвать инструмент"
    _print_ok(
        "цикл агента: LLM → MCP → ответ",
        f"инструменты: {reply.tool_calls}",
    )
    print(f"     {DIM}ответ бота: {reply.text}{RESET}")


def _brief(data: object, limit: int = 90) -> str:
    text = str(data)
    return text[: limit - 3] + "..." if len(text) > limit else text


# --------------------------------------------------------------------------- #
# Live-прогон с настоящей моделью
# --------------------------------------------------------------------------- #

async def check_live_agent(mcp_client, url: str) -> None:
    from agent import MovieAgent
    from config import BotSettings
    from llm import OpenAILLM

    settings = BotSettings()
    if settings.missing_openai_key():
        _print_skip("живой прогон с LLM", "нет OPENAI_API_KEY в .env")
        return

    llm = OpenAILLM(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        base_url=settings.openai_base_url,
    )
    agent = MovieAgent(llm, mcp_client, max_tool_rounds=settings.max_tool_rounds)
    await agent.prepare()

    questions = [
        "покажи топ-5 фильмов по рейтингу",
        "найди Начало и расскажи про него",
        "сколько заработал Начало в прокате",
        "что посмотреть похоже на Дюну?",
        "сколько будет 15 процентов от 2500",
    ]
    for question in questions:
        reply = await agent.answer(question, user_id="live-user")
        print(f"\n     {DIM}Q: {question}{RESET}")
        print(f"     {DIM}A: {reply.text}{RESET}")
        if reply.tool_calls:
            print(f"     {DIM}инструменты: {', '.join(reply.tool_calls)}{RESET}")
    _print_ok("живой прогон с настоящей LLM завершён")
    await llm.aclose()


# --------------------------------------------------------------------------- #

async def main() -> int:
    parser = argparse.ArgumentParser(description="Сквозная проверка movie-mcp")
    parser.add_argument("--live", action="store_true", help="прогнать настоящую модель (нужен OPENAI_API_KEY)")
    args = parser.parse_args()

    from mcp_client import MCPClient

    failures = 0

    with ServerHandle(db_filename="movie-mcp-smoke.db") as handle:
        print(f"\nMCP-сервер: {handle.url}\n")

        async with MCPClient(handle.url) as mcp_client:
            steps: list[tuple[str, object]] = [
                ("Обнаружение инструментов", check_tools_discovery(mcp_client)),
                ("Сценарии из задания", check_scenarios(mcp_client)),
                ("Цикл create → update → delete", check_write_cycle(mcp_client)),
                ("Безопасность калькулятора", check_calculator_security(mcp_client)),
                ("Цикл агента", check_agent_loop(mcp_client)),
            ]
            if args.live:
                steps.append(("Живой прогон с LLM", check_live_agent(mcp_client, handle.url)))

            for label, coro in steps:
                print(f"\n--- {label} ---")
                try:
                    await coro
                except AssertionError as exc:
                    failures += 1
                    _print_fail(label, str(exc) or "проверка не пройдена")
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    _print_fail(label, f"{type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"{RED}Провалено проверок: {failures}{RESET}")
        return 1
    print(f"{GREEN}Все проверки пройдены.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
