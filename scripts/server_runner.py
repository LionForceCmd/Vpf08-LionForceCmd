"""Подъём MCP-сервера movie-mcp в фоновом потоке — для скриптов проверки.

Используется ``scripts/smoke_test.py`` и ``scripts/demo_dialogue.py``: оба поднимают
настоящий сервер на свободном порту с временной базой, чтобы прогнать клиента
и агента по реальному протоколу MCP, а не по мокам.
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def free_port() -> int:
    """Возвращает свободный TCP-порт на localhost."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ServerHandle:
    """Контекстный менеджер: поднимает uvicorn и гасит его на выходе."""

    def __init__(self, db_filename: str = "movie-mcp-script.db", seed: bool = True) -> None:
        import uvicorn

        if str(PROJECT_ROOT / "mcp_server") not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / "mcp_server"))

        db_file = Path(tempfile.gettempdir()) / db_filename
        for suffix in ("", "-wal", "-shm"):
            stale = Path(str(db_file) + suffix)
            if stale.exists():
                stale.unlink()

        os.environ["MOVIE_DB_PATH"] = str(db_file)
        os.environ["MOVIE_DB_SEED"] = "true" if seed else "false"

        import server
        from db import MovieDB

        server.settings.db_path = db_file
        server.db = MovieDB(db_file)

        self.db_path = db_file
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}/mcp"
        config = uvicorn.Config(server.app, host="127.0.0.1", port=self.port, log_level="warning")
        self._uvicorn = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._uvicorn.run, daemon=True)

    def __enter__(self) -> ServerHandle:
        self._thread.start()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if getattr(self._uvicorn, "started", False):
                return self
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.3):
                    time.sleep(0.2)  # даём завершиться lifespan (заливка базы)
                    return self
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("MCP-сервер не запустился за 30 секунд")

    def __exit__(self, *exc_info: object) -> None:
        self._uvicorn.should_exit = True
        self._thread.join(timeout=10)
        for suffix in ("", "-wal", "-shm"):
            stale = Path(str(self.db_path) + suffix)
            if stale.exists():
                stale.unlink()
