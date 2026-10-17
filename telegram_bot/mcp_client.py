"""MCP-клиент: обращается к нашему MCP-серверу по протоколу JSON-RPC 2.0.

Отвечает за рукопожатие ``initialize``, получение списка инструментов
(``tools/list``) и их вызов (``tools/call``). Схемы инструментов сразу
преобразуются в формат OpenAI function calling — так бот отдаёт модели то,
что она умеет понимать, ничего не дублируя вручную.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
SESSION_ID_HEADER = "mcp-session-id"
PROTOCOL_VERSION_HEADER = "mcp-protocol-version"
USER_ID_HEADER = "x-user-id"

JSONRPC_VERSION = "2.0"

SUPPORTED_SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "description",
    "default",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "anyOf",
    "oneOf",
    "format",
}


class MCPError(RuntimeError):
    """Ошибка обращения к MCP-серверу."""


class MCPTool:
    """Описание одного MCP-инструмента в виде, удобном для LLM."""

    __slots__ = ("name", "description", "input_schema", "title")

    def __init__(self, name: str, description: str, input_schema: dict[str, Any], title: str = "") -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema or {"type": "object", "properties": {}}
        self.title = title

    @classmethod
    def from_mcp(cls, payload: dict[str, Any]) -> MCPTool:
        return cls(
            name=payload.get("name", ""),
            description=payload.get("description", "") or "",
            input_schema=payload.get("inputSchema") or {"type": "object", "properties": {}},
            title=payload.get("title", "") or "",
        )

    def to_openai_function(self) -> dict[str, Any]:
        """Схема MCP-инструмента в формате OpenAI ``tools``."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._clean_schema(self.input_schema),
            },
        }

    @classmethod
    def _clean_schema(cls, schema: dict[str, Any]) -> dict[str, Any]:
        """Убирает из схемы ключи, которые OpenAI не принимает."""
        if not isinstance(schema, dict):
            return {"type": "object", "properties": {}}
        cleaned: dict[str, Any] = {}
        for key, value in schema.items():
            if key not in SUPPORTED_SCHEMA_KEYS:
                continue
            if key in {"properties", "$defs"} and isinstance(value, dict):
                cleaned["properties"] = {
                    name: cls._clean_schema(sub) if isinstance(sub, dict) else sub
                    for name, sub in value.items()
                }
            elif key in {"items", "anyOf", "oneOf"} and isinstance(value, list):
                cleaned[key] = [
                    cls._clean_schema(sub) if isinstance(sub, dict) else sub for sub in value
                ]
            elif isinstance(value, dict):
                cleaned[key] = cls._clean_schema(value)
            else:
                cleaned[key] = value
        cleaned.setdefault("type", "object")
        cleaned.setdefault("properties", {})
        return cleaned

    def __repr__(self) -> str:  # pragma: no cover - удобство отладки
        return f"MCPTool({self.name})"


class MCPToolResult:
    """Результат вызова инструмента: текст для LLM и разобранные данные."""

    __slots__ = ("text", "data", "is_error")

    def __init__(self, text: str, data: Any, is_error: bool) -> None:
        self.text = text
        self.data = data
        self.is_error = is_error

    def __repr__(self) -> str:  # pragma: no cover
        return f"MCPToolResult(is_error={self.is_error}, text={self.text[:60]!r})"


class MCPClient:
    """Асинхронный клиент нашего MCP-сервера поверх Streamable HTTP."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 20.0,
        token: str | None = None,
        client_name: str = "movie-mcp-telegram-bot",
    ) -> None:
        self.url = url
        self.endpoint = url
        self.base_url = url.rsplit("/", 1)[0]
        self.token = token
        self.client_name = client_name
        self.protocol_version = PROTOCOL_VERSION
        self.server_info: dict[str, Any] = {}
        self.instructions: str = ""
        self.session_id: str | None = None
        self._tools: list[MCPTool] | None = None
        self._request_id = 0
        self._client = httpx.AsyncClient(timeout=timeout, headers=self._base_headers())

    def _base_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json, text/event-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> MCPClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ------------------------------------------------------- низкий уровень

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _headers(self, user_id: str | None = None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            PROTOCOL_VERSION_HEADER: self.protocol_version,
        }
        if self.session_id:
            headers[SESSION_ID_HEADER] = self.session_id
        if user_id:
            headers[USER_ID_HEADER] = str(user_id)
        return headers

    @staticmethod
    def _parse_response(raw: str, content_type: str) -> Any:
        """Понимает и ``application/json``, и ``text/event-stream``."""
        content_type = (content_type or "").lower()
        if content_type.startswith("text/event-stream"):
            chunks: list[str] = []
            for line in raw.splitlines():
                if line.startswith("data:"):
                    chunks.append(line[5:].strip())
            if not chunks:
                raise MCPError("Сервер вернул пустой SSE-поток")
            return json.loads(chunks[-1])
        return json.loads(raw)

    async def request(
        self, method: str, params: dict[str, Any] | None = None, *, user_id: str | None = None
    ) -> dict[str, Any]:
        """Один JSON-RPC вызов. Возвращает поле ``result``."""
        body: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "id": self._next_id(),
            "method": method,
        }
        effective_params = dict(params or {})
        if user_id:
            meta = dict(effective_params.get("_meta") or {})
            meta["user_id"] = str(user_id)
            effective_params["_meta"] = meta
        if effective_params:
            body["params"] = effective_params

        try:
            response = await self._client.post(
                self.endpoint, content=json.dumps(body), headers=self._headers(user_id)
            )
        except httpx.HTTPError as exc:
            raise MCPError(f"Не удалось связаться с MCP-сервером {self.url}: {exc}") from exc

        if response.status_code == 202:
            return {}

        if response.status_code == 401:
            raise MCPError("MCP-сервер отклонил авторизацию: проверь MCP_API_TOKEN")

        if response.status_code >= 400:
            detail = response.text[:300]
            try:
                parsed = self._parse_response(response.text, response.headers.get("content-type", ""))
                if isinstance(parsed, dict) and "error" in parsed:
                    detail = str(parsed["error"].get("message", detail))
            except (json.JSONDecodeError, MCPError):
                pass
            raise MCPError(f"MCP-сервер ответил кодом {response.status_code}: {detail}")

        try:
            payload = self._parse_response(response.text, response.headers.get("content-type", ""))
        except (json.JSONDecodeError, MCPError) as exc:
            raise MCPError(f"Не удалось разобрать ответ MCP-сервера: {exc}") from exc

        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        if not isinstance(payload, dict):
            raise MCPError("Неожиданный формат ответа MCP-сервера")

        if "error" in payload:
            error = payload["error"]
            raise MCPError(f"{error.get('message', 'ошибка MCP')} (code={error.get('code')})")

        return payload.get("result") or {}

    # ------------------------------------------------------------ публичное API

    async def initialize(self) -> dict[str, Any]:
        """Рукопожатие MCP. Вызывается один раз перед первым использованием."""
        result = await self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"roots": {"listChanged": False}},
                "clientInfo": {"name": self.client_name, "version": "1.0.0"},
            },
        )
        self.protocol_version = result.get("protocolVersion", PROTOCOL_VERSION)
        self.server_info = result.get("serverInfo") or {}
        self.instructions = result.get("instructions", "")

        # Сервер может потребовать подтверждение инициализации.
        try:
            await self._client.post(
                self.endpoint,
                content=json.dumps(
                    {"jsonrpc": JSONRPC_VERSION, "method": "notifications/initialized"}
                ),
                headers=self._headers(),
            )
        except httpx.HTTPError as exc:  # pragma: no cover - не критично
            logger.debug("Не удалось отправить notifications/initialized: %s", exc)

        logger.info(
            "MCP-сервер %s v%s, протокол %s",
            self.server_info.get("name"),
            self.server_info.get("version"),
            self.protocol_version,
        )
        return self.server_info

    async def list_tools(self, *, refresh: bool = False) -> list[MCPTool]:
        """Список инструментов сервера (кэшируется на время работы бота)."""
        if self._tools is not None and not refresh:
            return self._tools
        result = await self.request("tools/list")
        self._tools = [MCPTool.from_mcp(item) for item in result.get("tools", [])]
        logger.info("Получено %d инструментов от MCP-сервера", len(self._tools))
        return self._tools

    async def openai_tools(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        """Инструменты в формате OpenAI function calling."""
        return [tool.to_openai_function() for tool in await self.list_tools(refresh=refresh)]

    async def tool_names(self, *, refresh: bool = False) -> list[str]:
        return [tool.name for tool in await self.list_tools(refresh=refresh)]

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None, *, user_id: str | None = None
    ) -> MCPToolResult:
        """Вызывает MCP-инструмент и разбирает ответ в текст + данные."""
        result = await self.request(
            "tools/call", {"name": name, "arguments": arguments or {}}, user_id=user_id
        )

        text_parts: list[str] = []
        for block in result.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        text = "\n".join(part for part in text_parts if part)

        data = result.get("structuredContent")
        if data is None and text:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = None
        if data is None:
            data = {"text": text} if text else {}

        return MCPToolResult(text=text, data=data, is_error=bool(result.get("isError")))

    async def list_resources(self) -> list[dict[str, Any]]:
        result = await self.request("resources/list")
        return list(result.get("resources", []))

    async def read_resource(self, uri: str) -> str:
        result = await self.request("resources/read", {"uri": uri})
        contents = result.get("contents") or []
        return "\n".join(str(item.get("text", "")) for item in contents)

    async def health(self) -> dict[str, Any]:
        """Проверка живости сервера (вспомогательный ``GET /health``)."""
        response = await self._client.get(f"{self.base_url}/health")
        if response.status_code >= 400:
            raise MCPError(f"Сервер недоступен: код {response.status_code}")
        return response.json()
