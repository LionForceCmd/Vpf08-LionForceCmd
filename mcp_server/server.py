"""MCP-сервер кино-каталога: ``product-mcp`` из урока, адаптированный под кино.

Транспорт — Streamable HTTP (обычный ``POST /mcp`` с JSON-RPC 2.0), поэтому
сервер совместим и с нашим Telegram-клиентом, и с официальным SDK ``mcp``
(проверяется тестами в ``tests/test_spec_compliance.py``).

Запуск::

    python mcp_server/server.py            # http://127.0.0.1:8000
    curl http://127.0.0.1:8000/health

Полезные вспомогательные ручки (для отладки без MCP-клиента):
``GET /`` — описание сервера, ``GET /tools`` — список инструментов.
"""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from db import MovieDB
from seed_data import build_people_and_credits, seed_movies
from settings import settings
from tools import HANDLERS, MCP_RESOURCES, MCP_TOOLS, read_resource

JSONRPC_VERSION = "2.0"
PROTOCOL_VERSION_HEADER = "mcp-protocol-version"
SESSION_ID_HEADER = "mcp-session-id"
USER_ID_HEADER = "x-user-id"
SUPPORTED_PROTOCOL_VERSIONS = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
    "2026-07-28",
)
DEFAULT_PROTOCOL_VERSION = "2025-11-25"

# Коды ошибок JSON-RPC 2.0 и расширения MCP.
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

db = MovieDB(settings.db_path)


# --------------------------------------------------------------------------- #
# Инициализация базы
# --------------------------------------------------------------------------- #

def bootstrap_database() -> int:
    """Создаёт схему и наполняет базу демо-данными, если она пустая.

    Возвращает число добавленных фильмов.
    """
    db.init_schema()

    if not settings.seed_on_startup or db.count_movies() > 0:
        return 0

    movies = seed_movies(settings.seed_limit)
    people, credits = build_people_and_credits()
    return db.bulk_seed(movies, people, credits)


# --------------------------------------------------------------------------- #
# JSON-RPC helpers
# --------------------------------------------------------------------------- #

def _result(request_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": payload}


def _error(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error}


def _text_content(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text}]


def _text_result(text: str, structured: Any = None, is_error: bool = False) -> dict[str, Any]:
    """Собирает результат ``tools/call`` в формате MCP."""
    payload: dict[str, Any] = {
        "content": _text_content(text),
        "isError": is_error,
    }
    if structured is not None:
        payload["structuredContent"] = structured
    return payload


# --------------------------------------------------------------------------- #
# Обработчики методов MCP
# --------------------------------------------------------------------------- #

MethodHandler = Callable[[dict[str, Any], str], Awaitable[dict[str, Any]]]


async def _handle_initialize(params: dict[str, Any], session_id: str) -> dict[str, Any]:
    requested = str(params.get("protocolVersion") or "")
    protocol_version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else DEFAULT_PROTOCOL_VERSION
    return {
        "protocolVersion": protocol_version,
        "capabilities": {
            "tools": {"listChanged": False},
            "resources": {"subscribe": False, "listChanged": False},
        },
        "serverInfo": {"name": settings.server_name, "version": settings.server_version},
        "instructions": settings.instructions,
    }


def _extract_user_id(params: dict[str, Any]) -> str:
    """Достаёт идентификатор пользователя из ``params._meta``.

    Telegram-бот кладёт туда свой ``user_id``, чтобы персональные данные
    (оценки, избранное) не путались между разными пользователями.
    """
    meta = params.get("_meta") or params.get("meta")
    if isinstance(meta, dict):
        for key in ("user_id", "userId", "client_user_id"):
            value = meta.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


async def _handle_tools_list(params: dict[str, Any], session_id: str) -> dict[str, Any]:
    return {"tools": MCP_TOOLS}


async def _handle_tools_call(params: dict[str, Any], user_id: str) -> dict[str, Any]:
    name = params.get("name")
    if not isinstance(name, str) or not name:
        raise RpcError(INVALID_PARAMS, "Не указано имя инструмента (name)")

    handler = HANDLERS.get(name)
    if handler is None:
        available = ", ".join(sorted(HANDLERS))
        raise RpcError(
            INVALID_PARAMS,
            f"Инструмент «{name}» не найден. Доступные инструменты: {available}",
        )

    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise RpcError(INVALID_PARAMS, "arguments должен быть объектом JSON")

    try:
        structured = await handler(db, arguments, user_id)
    except Exception as exc:  # noqa: BLE001 - инструмент не должен ронять сервер
        return _text_result(
            f"Инструмент «{name}» завершился ошибкой: {exc}",
            {"error": str(exc), "tool": name},
            is_error=True,
        )

    is_error = isinstance(structured, dict) and "error" in structured
    text = json.dumps(structured, ensure_ascii=False, indent=2)
    return _text_result(text, structured, is_error=is_error)


async def _handle_resources_list(params: dict[str, Any], session_id: str) -> dict[str, Any]:
    return {"resources": MCP_RESOURCES}


async def _handle_resources_read(params: dict[str, Any], session_id: str) -> dict[str, Any]:
    uri = params.get("uri")
    if not isinstance(uri, str) or not uri:
        raise RpcError(INVALID_PARAMS, "Не указан uri ресурса")

    payload = read_resource(db, uri)
    if payload is None:
        available = ", ".join(resource["uri"] for resource in MCP_RESOURCES)
        raise RpcError(INVALID_PARAMS, f"Ресурс «{uri}» не найден. Доступные: {available}")

    return {"contents": [payload]}


async def _handle_ping(params: dict[str, Any], session_id: str) -> dict[str, Any]:
    return {}


METHODS: dict[str, MethodHandler] = {
    "initialize": _handle_initialize,
    "tools/list": _handle_tools_list,
    "tools/call": _handle_tools_call,
    "resources/list": _handle_resources_list,
    "resources/read": _handle_resources_read,
    "ping": _handle_ping,
}

NOTIFICATION_METHODS = {"notifications/initialized", "notifications/cancelled"}


class RpcError(Exception):
    """Ошибка, которую нужно вернуть клиенту как JSON-RPC error."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# --------------------------------------------------------------------------- #
# FastAPI приложение
# --------------------------------------------------------------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    inserted = bootstrap_database()
    total = db.count_movies()
    app.state.db_ready = True
    app.state.movie_count = total
    print(
        f"[{settings.server_name}] база: {settings.db_path} | фильмов в каталоге: {total}"
        + (f" (+{inserted} добавлено при первом запуске)" if inserted else "")
    )
    yield


app = FastAPI(
    title=f"{settings.server_name} — MCP-сервер кино-каталога",
    description=settings.instructions,
    version=settings.server_version,
    lifespan=lifespan,
)


@app.get("/health", tags=["служебные"])
async def health() -> dict[str, Any]:
    """Проверка живости сервера и состояния базы."""
    return {
        "status": "ok",
        "server": settings.server_name,
        "version": settings.server_version,
        "protocol_versions": list(SUPPORTED_PROTOCOL_VERSIONS),
        "movies_in_db": db.count_movies(),
        "tools_count": len(MCP_TOOLS),
        "resources_count": len(MCP_RESOURCES),
    }


@app.get("/tools", tags=["служебные"])
async def list_tools_debug() -> dict[str, Any]:
    """Список MCP-инструментов в их JSON Schema — удобно для отладки."""
    return {"count": len(MCP_TOOLS), "tools": MCP_TOOLS}


@app.get("/", tags=["служебные"])
async def index() -> dict[str, Any]:
    """Описание сервера и список поддерживаемых методов MCP."""
    return {
        "server": settings.server_name,
        "version": settings.server_version,
        "description": settings.instructions,
        "mcp_endpoint": "/mcp",
        "protocol_versions": list(SUPPORTED_PROTOCOL_VERSIONS),
        "methods": sorted(METHODS),
        "resources": [resource["uri"] for resource in MCP_RESOURCES],
        "tools": [tool["name"] for tool in MCP_TOOLS],
    }


@app.post("/mcp", tags=["mcp"])
async def mcp_endpoint(request: Request) -> Response:
    """Основная точка MCP: JSON-RPC 2.0 поверх Streamable HTTP."""
    if settings.api_token:
        header = request.headers.get("authorization", "")
        if header != f"Bearer {settings.api_token}":
            return JSONResponse(
                _error(None, INVALID_REQUEST, "Требуется заголовок Authorization: Bearer <token>"),
                status_code=401,
            )

    try:
        payload = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse(_error(None, PARSE_ERROR, "Некорректный JSON"), status_code=400)

    batch = isinstance(payload, list)
    messages = payload if batch else [payload]
    if not messages:
        return JSONResponse(_error(None, INVALID_REQUEST, "Пустой запрос"), status_code=400)

    protocol_version = request.headers.get(PROTOCOL_VERSION_HEADER) or DEFAULT_PROTOCOL_VERSION
    # Сессия MCP: клиент возвращает её в заголовке Mcp-Session-Id после initialize.
    # Если клиент сессию не ведёт (как наш простой бот), идентификатором
    # пользователя служит заголовок X-User-Id либо params._meta.user_id.
    session_id = request.headers.get(SESSION_ID_HEADER) or uuid.uuid4().hex
    default_user_id = request.headers.get(USER_ID_HEADER) or ""

    responses: list[dict[str, Any]] = []
    is_initialize = False
    negotiated_version = protocol_version

    for message in messages:
        if not isinstance(message, dict):
            responses.append(_error(None, INVALID_REQUEST, "Сообщение должно быть объектом"))
            continue

        method = message.get("method")
        request_id = message.get("id")

        if not isinstance(method, str) or not method:
            responses.append(
                _error(request_id, INVALID_REQUEST, "Не указано поле method")
            )
            continue

        # Уведомления не требуют ответа (JSON-RPC 2.0 и спецификация MCP).
        if method in NOTIFICATION_METHODS or request_id is None:
            continue

        params = message.get("params") or {}
        if not isinstance(params, dict):
            responses.append(_error(request_id, INVALID_PARAMS, "params должен быть объектом"))
            continue

        handler = METHODS.get(method)
        if handler is None:
            responses.append(
                _error(request_id, METHOD_NOT_FOUND, f"Метод «{method}» не поддерживается")
            )
            continue

        if method == "initialize":
            is_initialize = True
            requested = str(params.get("protocolVersion") or "")
            if requested in SUPPORTED_PROTOCOL_VERSIONS:
                negotiated_version = requested

        # Пользовательские данные привязываем к пользователю, а не к сессии.
        user_id = _extract_user_id(params) or default_user_id or session_id

        try:
            result = await handler(params, user_id)
        except RpcError as exc:
            responses.append(_error(request_id, exc.code, exc.message))
        except Exception as exc:  # noqa: BLE001
            responses.append(_error(request_id, INTERNAL_ERROR, f"Внутренняя ошибка сервера: {exc}"))
        else:
            responses.append(_result(request_id, result))

    headers = {PROTOCOL_VERSION_HEADER: negotiated_version}
    if is_initialize:
        headers[SESSION_ID_HEADER] = session_id

    if not responses:
        # Только уведомления — по спецификации отвечаем 202 Accepted без тела.
        return Response(status_code=202, headers=headers)

    return JSONResponse(responses if batch else responses[0], headers=headers)


def main() -> None:
    import uvicorn

    print(f"Запуск {settings.server_name} v{settings.server_version}")
    print(f"MCP endpoint: http://{settings.host}:{settings.port}/mcp")
    print(f"Проверка живости: http://{settings.host}:{settings.port}/health")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
