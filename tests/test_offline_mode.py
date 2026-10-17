"""Тесты офлайн-режима: разбор фраз в MCP-инструменты и оформление ответов.

Офлайн-LLM нужна, чтобы проект запускался вообще без ключей. Здесь проверяем,
что типовые фразы разбираются правильно и что агент проходит тот же цикл
«выбрал инструмент → вызвал MCP → получил ответ», что и с настоящей моделью.
"""

from __future__ import annotations

import pytest

from offline_llm import OfflineLLM


def plan_of(question: str):
    """Возвращает ``(tool, arguments)``, которые выберет OfflineLLM."""
    return OfflineLLM()._plan(question)


class TestIntentRouting:
    @pytest.mark.parametrize(
        ("question", "tool"),
        [
            ("покажи топ-5 фильмов", "get_top_movies"),
            ("лучшие фильмы", "get_top_movies"),
            ("самые кассовые", "get_top_movies"),
            ("новые фильмы", "get_top_movies"),
            ("найди Начало", "find_movie"),
            ("поищи Оппенгеймера", "find_movie"),
            ("покажи аниме", "find_movies_by_genre"),
            ("дай комедии", "find_movies_by_genre"),
            ("покажи фантастику", "find_movies_by_genre"),
            ("что посмотреть", "get_recommendations"),
            ("порекомендуй что-нибудь", "get_recommendations"),
            ("что посмотреть похоже на Дюну", "get_similar_movies"),
            ("покажи похожее на Начало", "get_similar_movies"),
            ("сколько заработал Начало", "box_office_stats"),
            ("сколько фильм заработала Дюна", "box_office_stats"),
            ("окупился ли Интерстеллар", "box_office_stats"),
            ("сколько фильмов в базе", "get_movie_stats"),
            ("покажи статистику", "get_movie_stats"),
            ("что снял Кристофер Нолан", "get_person_filmography"),
            ("фильмы с Леонардо ДиКаприо", "get_person_filmography"),
            ("кто такой Тарантино", "find_person"),
            ("покажи фильм с id 3", "get_movie"),
            ("мои оценки", "get_user_ratings"),
            ("поставь Началу 9", "rate_movie"),
            ("оцени Дюну на 8", "rate_movie"),
            ("добавь Начало в избранное", "toggle_favorite"),
            ("что в избранном", "get_favorites"),
            ("добавь фильм Дым 1955 драма 8.2", "add_movie"),
            ("удали фильм с id 42", "delete_movie"),
            ("посчитай 2+2", "calculate"),
            ("посчитай 15% от 2500", "calculate"),
            ("посчитай sqrt(144) + 2**3", "calculate"),
        ],
    )
    def test_routes_to_expected_tool(self, question: str, tool: str) -> None:
        plan = plan_of(question)
        assert plan is not None, f"«{question}» не распознана"
        assert plan[0] == tool, f"«{question}» → {plan}"

    @pytest.mark.parametrize(
        "question",
        ["привет", "непонятный запрос", "как дела", "ааа", "погода в москве"],
    )
    def test_small_talk_needs_no_tool(self, question: str) -> None:
        assert plan_of(question) is None


class TestTitleExtraction:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("найди Начало", "Начал"),
            ("поищи Дюну", "Дюн"),
            ("найди Оппенгеймера", "Оппенгеймер"),
            ("поставь Началу 9", "Начал"),
            ("оцени Интерстеллар на 8", "Интерстеллар"),
            ("найди «Однажды в Голливуде»", "Однажды в Голливуде"),
        ],
    )
    def test_stems_russian_endings(self, question: str, expected: str) -> None:
        plan = plan_of(question)
        assert plan is not None
        title = plan[1].get("title") or plan[1].get("query")
        assert title == expected

    def test_normal_form_unchanged(self) -> None:
        # Слова без русских окончаний не меняются
        assert OfflineLLM._stem_title("Форрест Гамп") == "Форрест Гамп"

    @pytest.mark.parametrize("word", ["Дюна", "Дюну", "Началу", "Дюны"])
    def test_stemmed_title_still_finds_movie(self, rpc, word: str) -> None:
        """Основа слова должна находить фильм: поиск на сервере подстрочный."""
        data = rpc("find_movie", {"query": OfflineLLM._stem_title(word)})
        assert data["movies"], f"{word!r} не нашлась в базе"
        assert data["movies"][0]["title"] in {"Дюна", "Дюна: Часть вторая", "Начало", "Начало 2"}

    def test_stop_word_not_matched_inside_word(self) -> None:
        # «на» внутри «Начало» не должно превращаться в «чало»
        plan = plan_of("что посмотреть похоже на Начало")
        assert plan[1]["title"] == "Начал"

    def test_stemmed_title_is_found_by_server(self, rpc) -> None:
        """«Начал» должен реально находиться в базе (поиск подстрочный)."""
        assert rpc("find_movie", {"query": "Начал"})["movies"][0]["title"] == "Начало"


class TestExpressionExtraction:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("посчитай 2+2", "2+2"),
            ("посчитай 15% от 2500", "2500 * 0.15"),
            ("сколько будет 30 процентов от 800", "800 * 0.3"),
            ("посчитай sqrt(144) + 2**3", "sqrt(144) + 2**3"),
            ("посчитай (2+2)*10", "(2+2)*10"),
        ],
    )
    def test_extraction(self, question: str, expected: str) -> None:
        assert OfflineLLM._extract_expression(question) == expected

    @pytest.mark.parametrize(
        "question",
        ["покажи фильмы", "привет", "сколько фильмов"],
    )
    def test_no_expression_in_prose(self, question: str) -> None:
        assert OfflineLLM._extract_expression(question) == ""


class TestAddMovieParsing:
    def test_full_payload(self) -> None:
        payload = OfflineLLM._parse_add_movie("добавь фильм Дым над Уоквиллом 1955 драма 8.2")
        assert payload["year"] == 1955
        assert payload["rating_imdb"] == 8.2
        assert payload["title"] == "Дым"
        assert "драма" in payload["genres"].casefold()

    def test_minimal_payload(self) -> None:
        payload = OfflineLLM._parse_add_movie("добавь фильм Начало")
        assert payload["title"] == "Начало"
        assert "year" not in payload


class TestResultRendering:
    def test_movies_list(self) -> None:
        rendered = OfflineLLM()._render_one(
            "find_movie",
            {
                "query": "нач",
                "found": 1,
                "movies": [
                    {
                        "title": "Начало",
                        "year": 2010,
                        "rating_imdb": 8.8,
                        "runtime": "2 ч 28 мин",
                        "genres": "Боевик, Триллер",
                    }
                ],
            },
        )
        assert "Начало (2010)" in rendered
        assert "IMDb 8.8" in rendered
        assert rendered.count("\n") == 1  # заголовок + одна строка

    def test_empty_result(self) -> None:
        rendered = OfflineLLM()._render_one("find_movie", {"movies": [], "found": 0})
        assert "Ничего не нашёл" in rendered

    def test_error_is_reported(self) -> None:
        rendered = OfflineLLM()._render_one("find_movie", {"is_error": True, "error": "пусто"})
        assert rendered.startswith("⚠️")
        assert "пусто" in rendered

    def test_box_office(self) -> None:
        rendered = OfflineLLM()._render_one(
            "box_office_stats",
            {
                "movie": {"title": "Дюна"},
                "budget_formatted": "$165.0 млн",
                "revenue_formatted": "$434.0 млн",
                "profit_formatted": "$269.0 млн",
                "profit_sign": "прибыль",
                "roi_percent": 163.0,
                "box_office_to_budget": 2.63,
                "verdict": "окупился в прокате",
            },
        )
        assert "Дюна" in rendered
        assert "$269.0 млн" in rendered
        assert "163.0%" in rendered

    def test_stats(self) -> None:
        rendered = OfflineLLM()._render_one(
            "get_movie_stats",
            {
                "total_movies": 151,
                "people_count": 242,
                "avg_rating": 7.91,
                "first_year": 1954,
                "last_year": 2024,
                "total_revenue_formatted": "$50.26 млрд",
                "by_kind": [{"name": "movie", "count": 105}],
                "by_genre": [{"name": "Драма", "count": 83}],
                "by_decade": [{"decade": 2010, "count": 64}],
            },
        )
        assert "151" in rendered
        assert "Драма (83)" in rendered
        assert "2010s" in rendered

    def test_no_double_punctuation(self) -> None:
        rendered = OfflineLLM()._render_one(
            "get_recommendations",
            {"reason": "Подобрано по жанрам.", "movies": [{"title": "X", "year": 2020}]},
        )
        assert "..:" not in rendered
        assert not rendered.splitlines()[0].endswith(".:")

    def test_collects_several_tool_results(self) -> None:
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "name": "find_movie", "content": '{"movies": [{"title": "A"}]}'},
            {"role": "tool", "name": "get_movie_stats", "content": '{"total_movies": 7}'},
        ]
        collected = OfflineLLM._collect_tool_results(messages)
        assert [name for name, _ in collected] == ["find_movie", "get_movie_stats"]
        assert collected[1][1]["total_movies"] == 7

    def test_stops_at_previous_assistant_turn(self) -> None:
        messages = [
            {"role": "user", "content": "старый вопрос"},
            {"role": "tool", "name": "old", "content": "{}"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "2"}]},
            {"role": "user", "content": "новый вопрос"},
        ]
        assert OfflineLLM._collect_tool_results(messages) == []


class TestOfflineAgentLoop:
    """Полный цикл агента с настоящим MCP-сервером и без единого ключа."""

    @pytest.fixture
    async def offline_agent(self, mcp_client):
        from agent import MovieAgent

        agent = MovieAgent(OfflineLLM(), mcp_client, max_tool_rounds=4, history_turns=6)
        await agent.prepare()
        return agent

    @pytest.mark.parametrize(
        ("question", "tool", "must_contain"),
        [
            ("покажи топ-3 фильмов", "get_top_movies", "IMDb"),
            ("найди Начало", "find_movie", "Начало"),
            ("что посмотреть похоже на Начало", "get_similar_movies", "IMDb"),
            ("сколько заработал Начало", "box_office_stats", "бюджет"),
            ("сколько фильмов в базе", "get_movie_stats", "Статистика"),
            ("что снял Кристофер Нолан", "get_person_filmography", "Нолан"),
            ("покажи аниме", "find_movies_by_genre", "Анимация"),
            ("посчитай 2+2", "calculate", "4"),
            ("посчитай 15% от 2500", "calculate", "375"),
        ],
    )
    async def test_question_produces_tool_call_and_answer(
        self, offline_agent, question: str, tool: str, must_contain: str
    ) -> None:
        reply = await offline_agent.answer(question, user_id="offline-user")
        assert reply.used_tools, f"«{question}»: инструмент не вызван"
        assert reply.tool_calls[0].startswith(tool), reply.tool_calls
        assert must_contain in reply.text, reply.text
        assert reply.text.strip()

    async def test_writes_persist(self, offline_agent) -> None:
        await offline_agent.answer("поставь Началу 9", user_id="offline-user")
        reply = await offline_agent.answer("мои оценки", user_id="offline-user")
        assert "9" in reply.text
        assert "Начало" in reply.text

    async def test_favorite_round_trip(self, offline_agent) -> None:
        await offline_agent.answer("добавь Дюна в избранное", user_id="offline-user")
        reply = await offline_agent.answer("что в избранном", user_id="offline-user")
        assert "Дюна" in reply.text

    async def test_recommendations_become_personal(self, offline_agent) -> None:
        await offline_agent.answer("поставь Началу 10", user_id="offline-user")
        await offline_agent.answer("поставь Дюне 9", user_id="offline-user")
        reply = await offline_agent.answer("что мне посмотреть", user_id="offline-user")
        assert reply.used_tools
        assert "Начало" not in reply.text.split("\n")[0]

    async def test_unknown_question_answers_in_words(self, offline_agent) -> None:
        reply = await offline_agent.answer("привет", user_id="offline-user")
        assert reply.used_tools is False
        assert "Привет" in reply.text

    async def test_history_is_used(self, offline_agent) -> None:
        await offline_agent.answer("поставь Началу 7", user_id="offline-user")
        reply = await offline_agent.answer(
            "мои оценки",
            [{"role": "user", "content": "поставь Началу 7"}, {"role": "assistant", "content": "ок"}],
            user_id="offline-user",
        )
        assert "Начало" in reply.text


class TestOfflineConfig:
    def test_offline_mode_flag(self) -> None:
        from config import BotSettings

        settings = BotSettings()
        settings.llm_mode = "offline"
        settings.openai_api_key = ""
        assert settings.offline_mode is True
        assert settings.missing_openai_key() is False

    @pytest.mark.parametrize("mode", ["auto", "openai", ""])
    def test_auto_mode_requires_key(self, mode: str) -> None:
        from config import BotSettings

        settings = BotSettings()
        settings.llm_mode = mode
        settings.openai_api_key = ""
        assert settings.offline_mode is False
        assert settings.missing_openai_key() is True

    def test_describe_reports_offline(self) -> None:
        from config import BotSettings

        settings = BotSettings()
        settings.llm_mode = "offline"
        settings.openai_api_key = ""
        described = settings.describe()
        assert "offline" in str(described["llm_mode"])
        assert described["openai_api_key"] == "не нужен в offline-режиме"

    def test_bot_config_check_allows_offline(self) -> None:
        import bot
        from config import BotSettings

        original_mode = bot.settings.llm_mode
        original_key = bot.settings.openai_api_key
        original_token = bot.settings.telegram_token
        try:
            bot.settings.llm_mode = "offline"
            bot.settings.openai_api_key = ""
            bot.settings.telegram_token = "123:ABC"
            problems = bot.check_config()
            assert problems == []

            bot.settings.telegram_token = ""
            problems = bot.check_config()
            assert len(problems) == 1
            assert "TELEGRAM_BOT_TOKEN" in problems[0]
        finally:
            bot.settings.llm_mode = original_mode
            bot.settings.openai_api_key = original_key
            bot.settings.telegram_token = original_token
