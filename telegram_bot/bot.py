"""Telegram-бот кино-каталога на aiogram 3.

Логика работы (по материалам урока):

    сообщение пользователя
        → LLM получает текст и список MCP-инструментов
        → LLM решает, нужен ли инструмент
        → бот вызывает инструмент на MCP-сервере и подставляет результат
        → LLM формулирует финальный ответ
        → бот отправляет его в чат

Запуск::

    python telegram_bot/bot.py

Перед запуском подними MCP-сервер (``python mcp_server/server.py``) и заполни ``.env``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

# Бот запускается как `python bot.py` из своей папки — добавляем её в sys.path,
# чтобы импорты config/prompts/mcp_client работали независимо от cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from aiogram import Bot, Dispatcher, F  # noqa: E402
from aiogram.client.default import DefaultBotProperties  # noqa: E402
from aiogram.enums import ParseMode  # noqa: E402
from aiogram.filters import Command  # noqa: E402
from aiogram.fsm.context import FSMContext  # noqa: E402
from aiogram.fsm.storage.memory import MemoryStorage  # noqa: E402
from aiogram.types import BotCommand, Message  # noqa: E402

from agent import AgentError, MovieAgent  # noqa: E402
from config import settings  # noqa: E402
from llm import LLMError, OpenAILLM  # noqa: E402
from mcp_client import MCPClient, MCPError  # noqa: E402
from prompts import HELP_MESSAGE, START_MESSAGE  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("movie-mcp-bot")

HISTORY_KEY = "chat_history"

AGENT: MovieAgent | None = None
LLM: OpenAILLM | None = None
MCP: MCPClient | None = None
HISTORY_LIMIT = 20


# --------------------------------------------------------------------------- #
# Вспомогательные функции
# --------------------------------------------------------------------------- #

def split_message(text: str, limit: int) -> list[str]:
    """Режет длинный ответ на куски по границам строк, чтобы влезть в лимит Telegram."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for line in text.splitlines():
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(line) > limit:
            chunks.append(line[:limit])
            line = line[limit:]
        current = line
    if current:
        chunks.append(current)
    return chunks


async def send_answer(message: Message, text: str) -> None:
    for chunk in split_message(text, settings.max_message_length):
        await message.answer(chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True)


async def push_history(state: FSMContext, role: str, text: str) -> None:
    history = state.get_data().get(HISTORY_KEY, [])
    history.append({"role": role, "content": text})
    if len(history) > HISTORY_LIMIT:
        history = history[-HISTORY_LIMIT:]
    await state.update_data(data={HISTORY_KEY: history})


# --------------------------------------------------------------------------- #
# Обработчики
# --------------------------------------------------------------------------- #

async def cmd_start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(START_MESSAGE)
    logger.info("Новый пользователь: %s", message.from_user.id if message.from_user else "?")


async def cmd_help(message: Message) -> None:
    await message.answer(HELP_MESSAGE)


async def cmd_reset(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Переписка очищена, начинаем заново 👌")


async def cmd_tools(message: Message) -> None:
    if MCP is None:
        await message.answer("MCP-клиент не инициализирован.")
        return
    try:
        names = await MCP.tool_names()
    except MCPError as exc:
        await message.answer(f"Не удалось получить список инструментов: {exc}")
        return
    listing = "\n".join(f"• {name}" for name in names)
    await message.answer(f"Инструменты MCP-сервера ({len(names)}):\n{listing}")


async def handle_message(message: Message, state: FSMContext) -> None:
    if AGENT is None or LLM is None:  # pragma: no cover - защита от гонок при старте
        await message.answer("Бот ещё запускается, попробуй через пару секунд.")
        return

    text = (message.text or "").strip()
    if not text:
        return

    user_id = str(message.from_user.id) if message.from_user else "anonymous"
    history = state.get_data().get(HISTORY_KEY, [])

    await state.set_state(None)
    try:
        await message.bot.send_chat_action(chat_id=message.chat.id, action="typing")
    except Exception:  # noqa: BLE001 - не критично
        pass

    try:
        reply = await AGENT.answer(text, history, user_id=user_id)
    except AgentError as exc:
        logger.warning("Ошибка агента: %s", exc)
        await message.answer(
            "Не получилось получить ответ. Проверь, что MCP-сервер запущен и ключи верные.\n"
            f"Подробности: {exc}"
        )
        return
    except LLMError as exc:  # pragma: no cover - уже обёрнуто в AgentError
        await message.answer(f"Языковая модель недоступна: {exc}")
        return

    await push_history(state, "user", text)
    await push_history(state, "assistant", reply.text)
    await send_answer(message, reply.text)
    logger.info(
        "user=%s rounds=%d tools=%s", user_id, reply.rounds, reply.tool_calls or "[]"
    )


# --------------------------------------------------------------------------- #
# Запуск
# --------------------------------------------------------------------------- #

async def build_agent() -> tuple[MovieAgent, OpenAILLM, MCPClient]:
    """Создаёт LLM, MCP-клиент и агента, выполняя рукопожатие с сервером."""
    llm = OpenAILLM(
        api_key=settings.openai_api_key,
        model=settings.llm_model,
        base_url=settings.openai_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
    )
    mcp = MCPClient(
        settings.mcp_url,
        timeout=settings.mcp_timeout,
        token=settings.mcp_token,
        client_name=settings.mcp_client_name,
    )
    agent = MovieAgent(
        llm,
        mcp,
        max_tool_rounds=settings.max_tool_rounds,
        history_turns=settings.history_turns,
    )
    tools = await agent.prepare()
    logger.info("Подключено к MCP-серверу, инструментов: %d", len(tools))
    return agent, llm, mcp


async def on_startup(bot: Bot) -> None:
    """Регистрирует команды в Telegram и поднимает агента."""
    global AGENT, LLM, MCP
    for command in (
        BotCommand(command="start", description="Начать общение с ботом"),
        BotCommand(command="help", description="Что умеет бот"),
        BotCommand(command="tools", description="Список MCP-инструментов"),
        BotCommand(command="reset", description="Очистить переписку"),
    ):
        try:
            await bot.set_my_commands([command])
        except Exception as exc:  # noqa: BLE001 - старые версии Bot API
            logger.debug("Не удалось зарегистрировать команду %s: %s", command.command, exc)

    AGENT, LLM, MCP = await build_agent()
    logger.info("Бот готов: %s", settings.describe())


async def on_shutdown(bot: Bot) -> None:
    if MCP is not None:
        await MCP.aclose()
    if LLM is not None:
        await LLM.aclose()
    logger.info("Бот остановлен")


def check_config() -> list[str]:
    problems: list[str] = []
    if settings.missing_telegram_token():
        problems.append("не задан TELEGRAM_BOT_TOKEN")
    if settings.missing_openai_key():
        problems.append("не задан OPENAI_API_KEY")
    return problems


async def main() -> None:
    problems = check_config()
    if problems:
        print("Не удалось запустить бота. Проблемы с конфигурацией:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nСкопируйте .env.example в .env, заполните значения и повторите запуск.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    bot = Bot(token=settings.telegram_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dispatcher = Dispatcher(storage=MemoryStorage())
    dispatcher.startup.register(on_startup)
    dispatcher.shutdown.register(on_shutdown)

    dispatcher.message.register(cmd_start, Command("start"))
    dispatcher.message.register(cmd_help, Command("help"))
    dispatcher.message.register(cmd_reset, Command("reset"))
    dispatcher.message.register(cmd_tools, Command("tools"))
    dispatcher.message.register(handle_message, F.text)

    try:
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("Запускаю long polling…")
        await dispatcher.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановлено пользователем (Ctrl+C)")
