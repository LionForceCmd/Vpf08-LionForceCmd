"""Общие фикстуры для тестов movie-mcp.

Тесты не требуют ни Telegram, ни ключей OpenAI: LLM подменяется скриптом, а
MCP-клиент работает с настоящим ASGI-приложением сервера через ASGI-транспорт.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER_DIR = PROJECT_ROOT / "mcp_server"
TELEGRAM_BOT_DIR = PROJECT_ROOT / "telegram_bot"

for path in (str(MCP_SERVER_DIR), str(TELEGRAM_BOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


# --------------------------------------------------------------------------- #
# MCP-сервер
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Путь к временной базе, общей для всех тестов сервера."""
    return tmp_path_factory.mktemp("movie-db") / "movies.db"


@pytest.fixture(scope="session", autouse=True)
def server_env(db_path: Path) -> Path:
    """Готовит окружение до того, как кто-либо импортирует ``server``.

    Важно: переменные окружения обязаны быть выставлены *до* первого импорта
    ``server``, иначе модуль навсегда свяжется с боевым путём к базе.
    """
    import os

    os.environ["MOVIE_DB_PATH"] = str(db_path)
    os.environ["MOVIE_DB_SEED"] = "true"
    return db_path


@pytest.fixture(scope="session")
def server_app(server_env: Path):
    """FastAPI-приложение movie-mcp с подставленной временной базой."""
    import server  # noqa: PLC0415
    from db import MovieDB  # noqa: PLC0415

    server.settings.db_path = server_env
    server.db = MovieDB(server_env)
    return server.app


@pytest.fixture(scope="session")
def seeded_db(server_app):
    """Готовая база со схемой и демо-данными."""
    import server  # noqa: PLC0415

    server.bootstrap_database()
    return server.db


@pytest.fixture
def client(server_app):
    """httpx-клиент поверх ASGI-приложения (lifespan выполняется)."""
    from fastapi.testclient import TestClient

    with TestClient(server_app) as test_client:
        yield test_client


class RpcHelper:
    """Хелпер для JSON-RPC вызовов ``POST /mcp``.

    ``call``   — сырой ``httpx.Response``;
    ``result`` — разобранное поле ``result`` (падает, если сервер вернул error);
    ``error``  — разобранное поле ``error``;
    ``tool``   — ``tools/call`` с возвратом ``structuredContent``.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        user_id: str | None = None,
        request_id: Any = 1,
    ):
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            body["params"] = params
        if user_id:
            body.setdefault("params", {})["_meta"] = {"user_id": user_id}
        return self._client.post("/mcp", json=body)

    def result(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        response = self.call(method, params, user_id=user_id)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert "error" not in payload, payload["error"]
        return payload["result"]

    def error(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        payload = self.call(method, params, user_id=user_id).json()
        assert "error" in payload, payload
        return payload["error"]

    def tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        result = self.result(
            "tools/call", {"name": name, "arguments": arguments or {}}, user_id=user_id
        )
        return result.get("structuredContent")

    def tool_call_result(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Полный результат ``tools/call`` (нужен ``isError``)."""
        return self.result(
            "tools/call", {"name": name, "arguments": arguments or {}}, user_id=user_id
        )

    def __call__(
        self, name: str, arguments: dict[str, Any] | None = None, *, user_id: str | None = None
    ) -> dict[str, Any]:
        """Короткая запись ``rpc("find_movie", {...})`` = ``rpc.tool(...)``."""
        return self.tool(name, arguments, user_id=user_id)


@pytest.fixture
def rpc(client) -> RpcHelper:
    return RpcHelper(client)


# --------------------------------------------------------------------------- #
# Телеграм-бот: MCP-клиент поверх ASGI-приложения
# --------------------------------------------------------------------------- #

class ASGITransportClient:
    """Подменяет HTTP-вызовы MCP-клиента обращением к ASGI-приложению.

    Позволяет проверять весь код ``telegram_bot`` (клиент, агент, цикл LLM)
    без запуска uvicorn и без сети.
    """

    def __init__(self, app: Any) -> None:
        self._app = app
        self.requests: list[dict[str, Any]] = []

    async def post(self, url: str, *, content: str | bytes, headers: dict[str, str]) -> Any:
        import json as _json

        from fastapi.testclient import TestClient

        body = _json.loads(content) if isinstance(content, str) else content
        self.requests.append({"method": "POST", "url": url, "body": body, "headers": headers})

        with TestClient(self._app) as test_client:
            response = test_client.post("/mcp", content=content, headers=headers)
        return _ResponseShim(response)

    async def get(self, url: str, *, headers: dict[str, str] | None = None) -> Any:
        from fastapi.testclient import TestClient

        with TestClient(self._app) as test_client:
            response = test_client.get("/health", headers=headers or {})
        return _ResponseShim(response)

    async def aclose(self) -> None:
        return None


class _ResponseShim:
    """Минимальный асинхронный аналог ``httpx.Response``."""

    def __init__(self, response: Any) -> None:
        self.status_code = response.status_code
        self.text = response.text
        self.headers = dict(response.headers)

    def json(self) -> Any:
        import json as _json

        return _json.loads(self.text)


@pytest.fixture
def mcp_client(server_app, monkeypatch):
    """Настоящий MCPClient, направленный в ASGI-приложение сервера."""
    from mcp_client import MCPClient

    client = MCPClient("http://testserver/mcp")
    transport = ASGITransportClient(server_app)
    monkeypatch.setattr(client, "_client", transport)
    return client


# --------------------------------------------------------------------------- #
# Подставная LLM
# --------------------------------------------------------------------------- #

class ScriptedLLM:
    """LLM, которая выдаёт заранее заданные ответы по очереди.

    Позволяет проверить цикл агента целиком: выбор инструмента, передачу
    аргументов, обработку результата и формирование финального ответа.
    """

    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        from llm import LLMResponse, ToolCall

        self.calls.append({"messages": list(messages), "tools": tools})
        if not self.responses:
            return LLMResponse(content="Скрипт LLM исчерпан.")

        item = self.responses.pop(0)
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, str):
            return LLMResponse(content=item)
        if isinstance(item, tuple):
            name, arguments = item
            return LLMResponse(
                tool_calls=[ToolCall(id=f"call_{name}", name=name, arguments=arguments)]
            )
        if isinstance(item, dict):
            return LLMResponse(
                tool_calls=[
                    ToolCall(id=call.get("id", "call_1"), name=call["name"], arguments=call.get("arguments", {}))
                    for call in item["tool_calls"]
                ],
                content=item.get("content"),
            )
        raise TypeError(f"Не понимаю тип ответа скрипта: {type(item)!r}")


@pytest.fixture
def scripted_llm():
    def _factory(responses: list[Any]) -> ScriptedLLM:
        return ScriptedLLM(responses)

    return _factory


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
