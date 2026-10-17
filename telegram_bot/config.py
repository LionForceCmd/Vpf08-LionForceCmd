"""Конфигурация Telegram-бота.

Секреты читаются из переменных окружения, а не хардкодятся в коде. Файл ``.env``
лежит в корне проекта — его нужно заполнить по образцу ``.env.example``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class BotSettings:
    """Все настройки бота и LLM."""

    def __init__(self) -> None:
        # Telegram
        self.telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

        # LLM (OpenAI и любые OpenAI-совместимые провайдеры)
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
        self.openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None
        self.llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.2)
        self.llm_max_tokens: int = _env_int("LLM_MAX_TOKENS", 1200)
        self.llm_timeout: float = _env_float("LLM_TIMEOUT", 60.0)

        # MCP-сервер
        self.mcp_url: str = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")
        self.mcp_timeout: float = _env_float("MCP_TIMEOUT", 20.0)
        self.mcp_token: str = os.getenv("MCP_API_TOKEN", "").strip() or None
        self.mcp_client_name: str = os.getenv("MCP_CLIENT_NAME", "movie-mcp-telegram-bot")

        # Поведение агента
        self.max_tool_rounds: int = _env_int("MAX_TOOL_ROUNDS", 6)
        self.history_turns: int = _env_int("HISTORY_TURNS", 12)
        self.max_message_length: int = _env_int("MAX_MESSAGE_LENGTH", 3500)

    # -------------------------------------------------------------- проверки

    def missing_telegram_token(self) -> bool:
        return not self.telegram_token

    def missing_openai_key(self) -> bool:
        return not self.openai_api_key

    def describe(self) -> dict[str, object]:
        """Состояние конфигурации для /help и логов (без самих секретов)."""
        return {
            "telegram_token": "задан" if self.telegram_token else "НЕ ЗАДАН",
            "openai_api_key": "задан" if self.openai_api_key else "НЕ ЗАДАН",
            "llm_model": self.llm_model,
            "openai_base_url": self.openai_base_url or "https://api.openai.com/v1",
            "mcp_server_url": self.mcp_url,
            "max_tool_rounds": self.max_tool_rounds,
            "history_turns": self.history_turns,
        }


settings = BotSettings()
