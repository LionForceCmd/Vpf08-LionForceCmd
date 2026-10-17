"""Конфигурация MCP-сервера кино-каталога.

Все параметры читаются из переменных окружения (файл ``.env`` в корне проекта),
поэтому секреты и адреса не хардкодятся в коде.
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
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class ServerSettings:
    """Настройки MCP-сервера."""

    def __init__(self) -> None:
        self.server_name: str = os.getenv("MCP_SERVER_NAME", "movie-mcp")
        self.server_version: str = os.getenv("MCP_SERVER_VERSION", "1.0.0")

        # Путь к базе. По умолчанию рядом с сервером, файл создаётся при старте.
        self.db_path: Path = Path(
            os.getenv("MOVIE_DB_PATH", str(BASE_DIR / "movies.db"))
        ).expanduser()

        # Заполнять таблицу демо-данными, если она пустая.
        self.seed_on_startup: bool = _env_flag("MOVIE_DB_SEED", True)

        # Сколько фильмов создавать при первом запуске.
        self.seed_limit: int = _env_int("MOVIE_DB_SEED_LIMIT", 0) or 0

        # HTTP
        self.host: str = os.getenv("MCP_HOST", "127.0.0.1")
        self.port: int = _env_int("MCP_PORT", 8000)
        self.log_level: str = os.getenv("MCP_LOG_LEVEL", "info")

        # Необязательный токен: если задан, клиент обязан прислать
        # заголовок Authorization: Bearer <token>. По умолчанию проверка выключена,
        # чтобы сервер можно было запустить "из коробки" на локальной машине.
        self.api_token: str | None = os.getenv("MCP_API_TOKEN") or None

    @property
    def instructions(self) -> str:
        return (
            "Кино-каталог movie-mcp: поиск фильмов и сериалов по названию, жанру, "
            "году и рейтингу, карточки фильмов, фильмография актёров и режиссёров, "
            "топ рейтингов и кассовых сборов, статистика коллекции, безопасный "
            "математический калькулятор и расчёт окупаемости кассовых сборов."
        )


settings = ServerSettings()
