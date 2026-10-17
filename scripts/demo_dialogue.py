"""Демонстрация проекта без единого ключа.

Запуск::

    python scripts/demo_dialogue.py

Скрипт поднимает MCP-сервер на свободном порту (временная база), подключается
настоящим ``MCPClient`` и прогоняет диалог через ``MovieAgent`` с офлайн-LLM
(``OfflineLLM``) — то есть работает вообще без ``OPENAI_API_KEY`` и без Telegram.

Показывает весь путь агента: фраза → выбор MCP-инструмента → вызов → ответ.
Такой вывод удобно приложить к домашнему заданию как доказательство работоспособности.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (PROJECT_ROOT, PROJECT_ROOT / "mcp_server", PROJECT_ROOT / "telegram_bot"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.server_runner import ServerHandle  # noqa: E402

CYAN = "\033[96m"
GREEN = "\033[92m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Сценарий: обычные фразы пользователя, как в реальном чате.
DIALOGUE: list[str] = [
    "привет",
    "покажи топ-5 фильмов",
    "найди Начало",
    "что посмотреть похоже на Начало",
    "сколько заработал Начало",
    "сколько фильм заработала Дюна",
    "покажи аниме",
    "что снял Кристофер Нолан",
    "поставь Началу 9",
    "мои оценки",
    "добавь Начало в избранное",
    "что в избранном",
    "что мне посмотреть",
    "сколько всего фильмов в базе",
    "посчитай 15% от 2500",
    "посчитай sqrt(144) + 2**3",
    "непонятный запрос",
]


async def main() -> int:
    from agent import MovieAgent
    from mcp_client import MCPClient
    from offline_llm import OfflineLLM

    user_id = "demo-user"

    with ServerHandle() as handle:
        print(f"{BOLD}Демонстрация movie-mcp без ключей{RESET}")
        print(f"{DIM}MCP-сервер: {handle.url}  |  LLM: OfflineLLM (правила, без API){RESET}\n")

        async with MCPClient(handle.url) as mcp_client:
            agent = MovieAgent(OfflineLLM(), mcp_client, max_tool_rounds=4, history_turns=6)
            tools = await agent.prepare()
            print(f"{DIM}сервер отдал {len(tools)} MCP-инструментов: "
                  f"{', '.join(tools)}{RESET}\n")

            history: list[dict[str, str]] = []
            for question in DIALOGUE:
                print(f"{CYAN}Вы: {question}{RESET}")
                reply = await agent.answer(question, history, user_id=user_id)
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": reply.text})
                for line in reply.text.splitlines():
                    print(f"   {line}")
                if reply.tool_calls:
                    print(f"{DIM}   ↳ MCP: {', '.join(reply.tool_calls)}{RESET}")
                print()

    print(f"{GREEN}Диалог завершён.{RESET}")
    print(f"{DIM}Ключ OpenAI не требовался: в telegram_bot/config.py есть режим LLM_MODE=offline.")
    print(f"Чтобы получить настоящие диалоги от GPT/DeepSeek/Ollama — заполни .env и запусти "
          f"telegram_bot/bot.py.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
