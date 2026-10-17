"""Тесты MCP-протокола: рукопожатие, tools/list, tools/call, resources, ошибки."""

from __future__ import annotations

from typing import Any

import pytest

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602


class TestInitialize:
    def test_handshake(self, rpc) -> None:
        result = rpc.result(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
        )
        assert result["protocolVersion"] == "2025-06-18"
        assert result["serverInfo"]["name"] == "movie-mcp"
        assert result["serverInfo"]["version"]
        assert result["instructions"]
        assert "tools" in result["capabilities"]
        assert "resources" in result["capabilities"]

    def test_negotiates_supported_version(self, rpc) -> None:
        result = rpc.result(
            "initialize",
            {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "t"}},
        )
        assert result["protocolVersion"] == "2024-11-05"

    def test_falls_back_for_unknown_version(self, rpc) -> None:
        result = rpc.result(
            "initialize",
            {"protocolVersion": "1999-01-01", "capabilities": {}, "clientInfo": {"name": "t"}},
        )
        assert result["protocolVersion"] in {
            "2024-11-05",
            "2025-03-26",
            "2025-06-18",
            "2025-11-25",
            "2026-07-28",
        }

    def test_returns_session_and_protocol_headers(self, rpc) -> None:
        response = rpc.call(
            "initialize",
            {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}},
        )
        assert response.headers["mcp-session-id"]
        assert response.headers["mcp-protocol-version"] == "2025-06-18"

    def test_initialized_notification_is_accepted(self, rpc) -> None:
        response = rpc.call("notifications/initialized")
        assert response.status_code == 202


class TestToolsList:
    def test_tools_are_exposed(self, rpc) -> None:
        tools = rpc.result("tools/list")["tools"]
        names = {tool["name"] for tool in tools}
        assert {
            "list_movies",
            "find_movie",
            "get_movie",
            "find_movies_by_genre",
            "add_movie",
            "calculate",
            "box_office_stats",
        } <= names

    def test_every_tool_has_valid_schema(self, rpc) -> None:
        for tool in rpc.result("tools/list")["tools"]:
            assert tool["name"]
            assert len(tool["description"]) > 40, tool["name"]
            schema = tool["inputSchema"]
            assert schema["type"] == "object"
            assert isinstance(schema["properties"], dict)
            for required in schema.get("required", []):
                assert required in schema["properties"], tool["name"]

    def test_required_arguments_exist(self, rpc) -> None:
        schemas = {tool["name"]: tool["inputSchema"] for tool in rpc.result("tools/list")["tools"]}
        assert schemas["find_movie"]["required"] == ["query"]
        assert "query" in schemas["find_movie"]["properties"]
        assert schemas["get_movie"]["required"] == ["movie_id"]
        assert schemas["calculate"]["required"] == ["expression"]


class TestToolsCall:
    def test_find_movie(self, rpc) -> None:
        data = rpc("find_movie", {"query": "нач"})
        assert data["found"] >= 1
        assert data["movies"][0]["title"] == "Начало"

    def test_get_movie_includes_cast(self, rpc) -> None:
        data = rpc("get_movie", {"movie_id": 1})
        assert data["title"]
        assert "cast" in data
        assert "synopsis" in data

    def test_list_movies_filters(self, rpc) -> None:
        data = rpc("list_movies", {"kind": "anime", "min_rating": 8.0, "limit": 5})
        assert data["returned"] <= 5
        assert data["total_found"] >= data["returned"]
        for movie in data["movies"]:
            assert movie["kind"] == "anime"
            assert movie["rating_imdb"] >= 8.0

    def test_top_movies(self, rpc) -> None:
        data = rpc("get_top_movies", {"by": "rating", "limit": 5})
        ratings = [movie["rating_imdb"] for movie in data["movies"]]
        assert ratings == sorted(ratings, reverse=True)

    def test_stats(self, rpc) -> None:
        data = rpc("get_movie_stats")
        assert data["total_movies"] >= 100
        assert data["by_kind"]
        assert data["genres"]

    def test_filmography(self, rpc) -> None:
        data = rpc("get_person_filmography", {"name": "Кристофер Нолан"})
        assert data["person"]["role"] == "director"
        assert data["total_works"] >= 3

    def test_similar_movies(self, rpc) -> None:
        data = rpc("get_similar_movies", {"title": "Начало", "limit": 3})
        assert data["based_on"]["title"] == "Начало"
        assert len(data["movies"]) <= 3

    def test_box_office(self, rpc) -> None:
        data = rpc("box_office_stats", {"title": "Дюна"})
        assert data["profit_usd"] > 0
        assert "profit_formatted" in data

    def test_calculate(self, rpc) -> None:
        assert rpc("calculate", {"expression": "2+2"})["result"] == 4

    def test_error_payload_marked_as_error(self, rpc) -> None:
        result = rpc.tool_call_result("find_movie", {"query": ""})
        assert result["isError"] is True
        assert "error" in result["structuredContent"]

    def test_unknown_tool_is_rpc_error(self, rpc) -> None:
        err = rpc.error("tools/call", {"name": "нет_такого", "arguments": {}})
        assert err["code"] == INVALID_PARAMS
        assert "Доступные инструменты" in err["message"]

    def test_missing_tool_name(self, rpc) -> None:
        assert rpc.error("tools/call", {"arguments": {}})["code"] == INVALID_PARAMS

    def test_arguments_must_be_object(self, rpc) -> None:
        err = rpc.error("tools/call", {"name": "find_movie", "arguments": "строка"})
        assert err["code"] == INVALID_PARAMS


class TestUserIsolation:
    def test_ratings_are_separate_per_user(self, rpc) -> None:
        first = rpc("rate_movie", {"title": "Матрица", "score": 10}, user_id="alice")
        second = rpc("rate_movie", {"title": "Матрица", "score": 3}, user_id="bob")

        assert first["your_score"] == 10
        assert second["your_score"] == 3
        assert rpc("get_user_ratings", {}, user_id="alice")["ratings"][0]["your_score"] == 10
        assert rpc("get_user_ratings", {}, user_id="bob")["ratings"][0]["your_score"] == 3

    def test_favorites_are_separate_per_user(self, rpc) -> None:
        rpc("toggle_favorite", {"title": "Унесённые призраками"}, user_id="alice")
        assert rpc("get_favorites", {}, user_id="alice")["count"] >= 1
        assert rpc("get_favorites", {}, user_id="bob")["count"] == 0

    def test_recommendations_become_personalized(self, rpc) -> None:
        user = "recommender"
        before = rpc("get_recommendations", {"limit": 3}, user_id=user)
        assert before["personalized"] is False

        rpc("rate_movie", {"title": "Паразиты", "score": 10}, user_id=user)
        after = rpc("get_recommendations", {"limit": 3}, user_id=user)
        assert after["personalized"] is True


class TestWrites:
    def test_add_movie(self, rpc) -> None:
        data = rpc("add_movie", {"title": "Кинотест", "year": 2026, "genres": "фантастика, драма"})
        assert data["movie"]["id"] > 0
        assert data["movie"]["genres"] == "Драма, Фантастика"

        found = rpc("find_movie", {"query": "Кинотест"})
        assert found["found"] == 1

        rpc("delete_movie", {"movie_id": data["movie"]["id"]})
        assert rpc("find_movie", {"query": "Кинотест"})["found"] == 0

    def test_add_duplicate_is_rejected(self, rpc) -> None:
        data = rpc("add_movie", {"title": "Начало"})
        assert "error" in data
        assert "уже есть" in data["error"]

    def test_add_requires_title(self, rpc) -> None:
        assert "error" in rpc("add_movie", {"year": 2026})

    def test_add_rejects_bad_rating(self, rpc) -> None:
        data = rpc("add_movie", {"title": "Плохой рейтинг", "rating_imdb": 42})
        assert "error" in data

    def test_update_movie(self, rpc) -> None:
        created = rpc("add_movie", {"title": "Обновляемый"})["movie"]
        data = rpc("update_movie", {"movie_id": created["id"], "fields": {"rating_kp": 8.8}})
        assert data["updated_fields"] == ["rating_kp"]
        assert rpc("get_movie", {"movie_id": created["id"]})["rating_kp"] == 8.8
        rpc("delete_movie", {"movie_id": created["id"]})

    def test_update_requires_fields(self, rpc) -> None:
        assert "error" in rpc("update_movie", {"movie_id": 1})

    def test_delete_missing_movie(self, rpc) -> None:
        assert "error" in rpc("delete_movie", {"movie_id": 10**9})


class TestResources:
    def test_list(self, rpc) -> None:
        uris = {resource["uri"] for resource in rpc.result("resources/list")["resources"]}
        assert "movie://catalog/stats" in uris
        assert "movie://catalog/genres" in uris

    def test_read(self, rpc) -> None:
        result = rpc.result("resources/read", {"uri": "movie://catalog/genres"})
        payload = result["contents"][0]
        assert payload["mimeType"] == "application/json"
        assert "Драма" in payload["text"]

    def test_read_missing(self, rpc) -> None:
        assert rpc.error("resources/read", {"uri": "movie://нет"})["code"] == INVALID_PARAMS


class TestProtocolErrors:
    def test_unknown_method(self, rpc) -> None:
        assert rpc.error("неизвестный/метод")["code"] == METHOD_NOT_FOUND

    def test_malformed_json(self, client) -> None:
        response = client.post(
            "/mcp",
            content="{не json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == PARSE_ERROR

    def test_missing_method(self, rpc) -> None:
        assert rpc.error("")["code"] == INVALID_REQUEST

    def test_batch_request(self, client) -> None:
        response = client.post(
            "/mcp",
            json=[
                {"jsonrpc": "2.0", "id": 1, "method": "ping"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ],
        )
        assert response.status_code == 200
        payload = response.json()
        assert isinstance(payload, list)
        assert len(payload) == 2

    def test_notification_only_returns_202(self, client) -> None:
        response = client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
        assert response.status_code == 202

    def test_ping(self, rpc) -> None:
        assert rpc.result("ping") == {}

    def test_id_is_echoed(self, client) -> None:
        response = client.post("/mcp", json={"jsonrpc": "2.0", "id": "abc-42", "method": "ping"})
        assert response.json()["id"] == "abc-42"


class TestHttpHelpers:
    def test_health(self, client) -> None:
        payload = client.get("/health").json()
        assert payload["status"] == "ok"
        assert payload["movies_in_db"] >= 100
        assert payload["tools_count"] >= 15

    def test_index(self, client) -> None:
        payload = client.get("/").json()
        assert payload["mcp_endpoint"] == "/mcp"
        assert "initialize" in payload["methods"]
        assert "calculate" in payload["tools"]

    def test_tools_debug_endpoint(self, client) -> None:
        payload = client.get("/tools").json()
        assert payload["count"] == len(payload["tools"])
