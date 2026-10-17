"""Проверка соответствия официальной спецификации Model Context Protocol.

Сервер поднимается на реальном порту, и к нему подключается клиент из
официального SDK ``mcp`` (``mcp.client.streamable_http``). Это доказывает, что
наш самописный JSON-RPC сервер — не «похожее на MCP», а настоящий MCP-сервер:
с тем же транспортом, тем же рукопожатием и той же схемой ответов.

Если пакет ``mcp`` не установлен, тест пропускается (``pytest.skip``), а не
падает: основная логика покрыта тестами в ``test_mcp_protocol.py``.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp", reason="официальный SDK mcp не установлен")

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def live_server(tmp_path_factory: pytest.TempPathFactory):
    """Поднимает uvicorn в отдельном потоке на свободном порту."""
    import uvicorn

    server_dir = Path(__file__).resolve().parent.parent / "mcp_server"
    if str(server_dir) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(server_dir))

    db_file = tmp_path_factory.mktemp("spec-db") / "movies.db"

    import os

    os.environ["MOVIE_DB_PATH"] = str(db_file)
    os.environ["MOVIE_DB_SEED"] = "true"

    import server

    server.settings.db_path = db_file
    server.db = __import__("db").MovieDB(db_file)

    port = _free_port()
    config = uvicorn.Config(server.app, host="127.0.0.1", port=port, log_level="warning")
    uvicorn_server = uvicorn.Server(config)
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                break
        except OSError:
            time.sleep(0.1)
    else:  # pragma: no cover - сервер не поднялся
        pytest.fail("MCP-сервер не запустился")

    yield f"http://127.0.0.1:{port}/mcp"

    uvicorn_server.should_exit = True
    thread.join(timeout=10)


def _run(coro):
    """Синхронная обёртка для async-тестов этого модуля."""
    return asyncio.run(coro)


class TestOfficialSdkClient:
    def test_initialize_handshake(self, live_server: str) -> None:
        async def scenario():
            async with streamable_http_client(live_server) as streams:
                read, write = streams[0], streams[1]
                async with ClientSession(read, write) as session:
                    return await session.initialize()

        result = _run(scenario())
        assert result.server_info.name == "movie-mcp"
        assert result.server_info.version
        assert result.instructions

    def test_tools_list(self, live_server: str) -> None:
        async def scenario():
            async with streamable_http_client(live_server) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    return await session.list_tools()

        result = _run(scenario())
        names = {tool.name for tool in result.tools}
        assert {"find_movie", "list_movies", "calculate", "box_office_stats"} <= names
        for tool in result.tools:
            assert tool.description
            assert tool.input_schema["type"] == "object"

    def test_call_tool_text_content(self, live_server: str) -> None:
        async def scenario():
            async with streamable_http_client(live_server) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    return await session.call_tool("find_movie", {"query": "нач"})

        result = _run(scenario())
        assert result.is_error is False
        assert result.content
        assert result.content[0].type == "text"
        assert "Начало" in result.content[0].text

    def test_call_calculate(self, live_server: str) -> None:
        async def scenario():
            async with streamable_http_client(live_server) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    return await session.call_tool("calculate", {"expression": "2**8"})

        result = _run(scenario())
        assert "256" in result.content[0].text

    def test_call_reports_tool_error(self, live_server: str) -> None:
        async def scenario():
            async with streamable_http_client(live_server) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    return await session.call_tool("find_movie", {"query": ""})

        result = _run(scenario())
        # Инструмент отработал, но сообщил об ошибке в своих данных
        assert result.is_error is True

    def test_unknown_tool_raises_protocol_error(self, live_server: str) -> None:
        async def scenario():
            async with streamable_http_client(live_server) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    await session.call_tool("no_such_tool", {})

        # SDK оборачивает ошибку в ExceptionGroup, поэтому смотрим repr целиком
        with pytest.raises(BaseException) as excinfo:
            _run(scenario())
        rendered = repr(excinfo.value)
        assert "no_such_tool" in rendered
        assert "-32602" in rendered

    def test_resources(self, live_server: str) -> None:
        async def scenario():
            async with streamable_http_client(live_server) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    listed = await session.list_resources()
                    uri = str(listed.resources[0].uri)
                    read = await session.read_resource(uri)
                    return listed, read

        listed, read = _run(scenario())
        assert listed.resources
        assert str(listed.resources[0].uri).startswith("movie://")
        assert read.contents[0].text
