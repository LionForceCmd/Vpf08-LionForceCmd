"""Офлайн-режим: агент, который обходится без LLM и без ключей.

Нужен в двух случаях:

* **демонстрация и сдача задания** — проект запускается вообще без
  ``OPENAI_API_KEY``: правила разбирают типовые фразы и превращают их в вызовы
  MCP-инструментов, а ответы собираются из результатов инструментов;
* **отладка** — видно, что именно отдаёт каждый инструмент, без «шумного» текста
  от языковой модели.

Класс реализует тот же интерфейс ``LLM``, что и ``OpenAILLM``, поэтому
``MovieAgent`` работает с ним без единого изменения: цикл «выбрал инструмент →
вызвал MCP → получил данные → сформулировал ответ» остаётся тем же самым.
Качество ответов, конечно, ниже, чем у настоящей модели, — зато нулевая
зависимость от внешних сервисов.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from llm import LLMResponse, ToolCall

logger = logging.getLogger(__name__)

_MATH_RE = re.compile(r"^[\d\s()+\-*/%.^]+$")
_QUOTED_RE = re.compile(r"[«\"']([^»\"']+)[»\"']")
_PERCENT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:%|процент\w*)\s*(?:от|из|с)\s*(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_FAVORITE_RE = re.compile(r"(?:добавь|добав)\s+(.+?)\s+в\s+избранн", re.IGNORECASE)
_RATE_RE = re.compile(
    r"(?:оцен\w*|постав[ьи])\s+(.+?)\s*(?:на\s+|=|:)?\s*(\d{1,2})\b",
    re.IGNORECASE,
)

# Русские падежные окончания: поиск на сервере подстрочный, поэтому достаточно
# убрать окончание — «Началу» превратится в «Начал» и найдёт «Начало».
_RU_ENDINGS = (
    "ому", "ему", "ыми", "ими", "ах", "ях", "ов", "ев", "ей", "ой", "ым", "им",
    "ам", "ям", "ом", "ем", "ую", "юю", "ая", "яя", "ое", "ее", "ые", "ие",
    "ый", "ий", "ая", "яя", "а", "я", "е", "ы", "и", "о", "у", "ю", "ь",
)
_RU_ENDINGS_BY_LENGTH = tuple(sorted(set(_RU_ENDINGS), key=len, reverse=True))

#: Жанры, которые встречаются в демо-каталоге. Держим в одном месте с остальными
#: правилами: полный набор лежит в mcp_server/seed_data.py.
GENRES = (
    "Драма", "Комедия", "Боевик", "Фантастика", "Триллер", "Криминал",
    "Приключения", "Фэнтези", "Ужасы", "Мелодрама", "Детектив", "Вестерн",
    "Семейный", "Романтика", "Исторический", "Биография", "Мистика", "Война",
    "Психология", "Документальный", "Наука", "Природа", "Короткометражка", "Анимация",
)

#: Разговорные названия, которых нет в каталоге буквально.
GENRE_ALIASES = {
    "документал": "Документальный",
    "аниме": "Анимация",
    "мульт": "Анимация",
    "хоррор": "Ужасы",
    "страшн": "Ужасы",
    "приключенч": "Приключения",
    "криминал": "Криминал",
    "детектив": "Детектив",
    "романт": "Романтика",
    "мелодрам": "Мелодрама",
}

CALCULATE_FUNCTIONS = (
    "sqrt",
    "abs",
    "round",
    "min",
    "max",
    "log",
    "floor",
    "ceil",
    "pow",
    "exp",
    "sin",
    "cos",
    "tan",
    "pi",
    "e",
)


class OfflineLLM:
    """Правиловый «мозг» агента: фраза → MCP-инструмент → текстовый ответ."""

    def __init__(self) -> None:
        self._last_tool_name = ""
        self._tool_calls = 0

    # ------------------------------------------------------------------ LLM

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Тот же контракт, что у ``OpenAILLM.complete``."""
        del temperature, max_tokens  # офлайн-режиму они не нужны

        results = self._collect_tool_results(messages)
        if results:
            return LLMResponse(content=self._render_results(results))

        question = self._last_user_text(messages)
        plan = self._plan(question)
        if plan is None:
            return LLMResponse(content=self._chitchat(question))

        name, arguments = plan
        self._last_tool_name = name
        self._tool_calls += 1
        logger.info("Офлайн-режим: %s(%s)", name, json.dumps(arguments, ensure_ascii=False))
        return LLMResponse(
            tool_calls=[
                ToolCall(id=f"offline_{self._tool_calls}", name=name, arguments=arguments)
            ]
        )

    async def aclose(self) -> None:
        return None

    # ------------------------------------------------------------ разбор фразы

    @staticmethod
    def _last_user_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content") or "").strip()
        return ""

    def _plan(self, question: str) -> tuple[str, dict[str, Any]] | None:
        """Возвращает ``(инструмент, аргументы)`` или ``None``, если инструмент не нужен."""
        text = question.casefold().strip()
        if not text:
            return None

        # Явная оценка: «поставь Началу 9», «оцени Дюну на 8»
        rate = _RATE_RE.search(question)
        if rate:
            score = int(rate.group(2))
            title = self._stem_title(rate.group(1))
            if 1 <= score <= 10 and title:
                return ("rate_movie", {"title": title, "score": score})

        if any(word in text for word in ("мои оценки", "что я оценил", "моя оценка")):
            return ("get_user_ratings", {})

        if "избранн" in text:
            favorite = _FAVORITE_RE.search(question)
            if favorite:
                return ("toggle_favorite", {"title": self._stem_title(favorite.group(1))})
            return ("get_favorites", {})

        # Похожее проверяем раньше рекомендаций: «что посмотреть похоже на X»
        # должно уходить в get_similar_movies, а не в get_recommendations.
        if any(word in text for word in ("похож", "в духе", "в стиле", "похоже на")):
            title = self._extract_title(question, ("похоже на", "похожая на", "на", "похож", "в духе", "в стиле"))
            return ("get_similar_movies", {"title": title, "limit": 5} if title else {})

        if any(word in text for word in ("рекоменд", "что посмотреть", "что мне")):
            return ("get_recommendations", {"limit": 5})

        if any(word in text for word in ("удали", "убери из каталога")):
            id_match = re.search(r"id\s*(\d+)", text)
            title = self._extract_title(question, ("фильм", "фильма"))
            arguments: dict[str, Any] = {}
            if id_match:
                arguments["movie_id"] = int(id_match.group(1))
            elif title:
                arguments["title"] = title
            return ("delete_movie", arguments)

        if any(word in text for word in ("добавь фильм", "добавить фильм", "добавь в каталог")):
            return ("add_movie", self._parse_add_movie(question))

        # Кассовые сборы проверяем раньше статистики: «сколько фильм заработал»
        # — это про деньги, а не про количество записей. Слова «кассовые» и «бюджет»
        # намеренно не здесь: «самые кассовые» — это запрос топа, а не одного фильма.
        if any(word in text for word in ("заработал", "заработала", "заработало", "заработали",
                                         "окупил", "окупился", "окупилась", "прибыл", "прибыль")):
            title = self._extract_title(question, ("заработал", "заработала", "заработало", "фильм", "у"))
            return ("box_office_stats", {"title": title} if title else {})

        if any(word in text for word in ("сколько фильмов", "сколько всего", "сколько записей",
                                         "статистик", "что в базе", "что в каталоге")):
            return ("get_movie_stats", {})

        if any(word in text for word in ("съемал", "снимал", "фильмография", "фильмы с", "что снял")):
            name = self._extract_person(question)
            if name:
                return ("get_person_filmography", {"name": name, "limit": 20})
            return ("find_person", {"query": name or ""})

        if any(word in text for word in ("кто такой", "кто это", "найди актёра", "найди режиссёра")):
            name = self._extract_person(question)
            return ("find_person", {"query": name})

        if "топ" in text or any(
            word in text
            for word in ("лучшие", "лучшее", "лучшая", "рейтинг", "самые", "самый",
                         "новые", "последние", "недавние", "бюджет")
        ):
            arguments: dict[str, Any] = {"by": "rating", "limit": 5}
            genre = self._extract_genre(question)
            if genre:
                arguments["genre"] = genre
            if any(word in text for word in ("кассов", "сбор", "прибыльн")):
                arguments["by"] = "revenue"
            if any(word in text for word in ("новые", "последние", "недавние")):
                arguments["by"] = "year"
            if "бюджет" in text:
                arguments["by"] = "budget"
            return ("get_top_movies", arguments)

        id_match = re.search(r"(?:id|айди)\s*(\d+)", text)
        if id_match:
            return ("get_movie", {"movie_id": int(id_match.group(1))})

        genre = self._extract_genre(question)
        if genre and any(word in text for word in ("покажи", "дай", "подбери", "найди", "фильмы", "кино")):
            return ("find_movies_by_genre", {"genre": genre, "limit": 10})

        title = self._extract_title(question, ("найди", "найди фильм", "поищи", "что за"))
        if title:
            return ("find_movie", {"query": title})

        expression = self._extract_expression(question)
        if expression:
            return ("calculate", {"expression": expression})

        return None

    #: Слова, которые не являются частью названия и отбрасываются с начала фразы.
    LEADING_NOISE = frozenset(
        {
            "сколько", "фильм", "фильма", "фильму", "фильмы", "кино", "у", "в", "и", "а",
            "заработал", "заработала", "заработало", "заработали",
            "посмотреть", "посмотрим", "смотреть", "похоже", "похож", "покажи",
            "найди", "дай", "подбери", "что", "е", "мне", "нам",
            "в духе", "в стиле", "на", "по",
        }
    )

    @classmethod
    def _stem_title(cls, text: str) -> str:
        """Убирает русское окончание у последнего слова названия.

        Поиск на сервере подстрочный, поэтому «Началу» → «Начал» и находит «Начало»,
        а «Интерстеллара» → «Интерстеллар». Нормальные формы не меняются.
        """
        cleaned = text.strip(" .,:;!?—–-\"'«»")
        if not cleaned:
            return ""
        head, _, last = cleaned.rpartition(" ")
        for ending in _RU_ENDINGS_BY_LENGTH:
            if last.casefold().endswith(ending) and len(last) - len(ending) >= 3:
                last = last[: len(last) - len(ending)]
                break
        return f"{head} {last}".strip() if head else last

    @classmethod
    def _strip_leading_noise(cls, text: str) -> str:
        """Отбрасывает служебные слова в начале: «заработала Дюна» → «Дюна»."""
        words = [word for word in text.split() if word]
        while words:
            head = words[0].strip(" .,:;!?—–-\"'«»").casefold()
            if head in cls.LEADING_NOISE:
                words.pop(0)
                continue
            break
        return " ".join(words).strip(" .,:;!?—–-\"'«»")

    @classmethod
    def _extract_title(cls, question: str, stop_words: tuple[str, ...]) -> str:
        """Достаёт название фильма: из кавычек либо после служебного слова.

        Служебные слова ищутся строго по границам слова, иначе «на» нашлось бы
        внутри «Начало» и название испортилось бы.
        """
        quoted = _QUOTED_RE.findall(question)
        if quoted:
            # Явное название в кавычках не stemмим: пользователь написал его сам.
            return quoted[-1].strip(" .,:;!?-\"'")

        best = ""
        for stop in sorted(stop_words, key=len, reverse=True):
            pattern = rf"(?<!\w){re.escape(stop)}(?!\w)"
            matches = list(re.finditer(pattern, question, flags=re.IGNORECASE))
            if not matches:
                continue
            candidate = question[matches[-1].end() :]
            # отрезаем хвост-комментарий вроде «заработало в прокате»
            for tail in (" в прокате", " в мире", " в сериале", " в кино"):
                cut = candidate.casefold().find(tail)
                if cut > 0:
                    candidate = candidate[:cut]
            candidate = cls._strip_leading_noise(candidate.strip(" .,:!?—–-"))
            if len(candidate) > len(best):
                best = candidate
        return cls._stem_title(best)

    @staticmethod
    def _extract_person(question: str) -> str:
        quoted = _QUOTED_RE.findall(question)
        if quoted:
            return quoted[-1].strip()
        for marker in ("кто такой", "кто это", "фильмография", "фильмы с", "что снял", "съемал", "снимал"):
            position = question.casefold().rfind(marker)
            if position != -1:
                return question[position + len(marker) :].strip(" ?!.,:—-«»\"'")
        return ""

    @classmethod
    def _extract_genre(cls, question: str) -> str:
        """Ищет в тексте один из жанров каталога с учётом падежных форм.

        «покажи комедии» и «покажи комедия» должны найти один жанр, поэтому
        сравниваем не строки целиком, а основы отдельных слов.
        """
        lowered = question.casefold()

        for genre in GENRES:
            if genre.casefold() in lowered:
                return genre

        genre_stems = {cls._stem_ru(genre).casefold() for genre in GENRES}
        for word in re.findall(r"[^\W\d_]+", lowered, flags=re.UNICODE):
            stem = cls._stem_ru(word).casefold()
            if stem in genre_stems:
                return next(
                    genre for genre in GENRES if cls._stem_ru(genre).casefold() == stem
                )

        # Словоформы, которых нет в списке жанров
        for alias, genre in GENRE_ALIASES.items():
            if alias in lowered:
                return genre
        return ""

    @staticmethod
    def _stem_ru(word: str) -> str:
        """Отбрасывает одно русское окончание, если от слова остаётся >= 3 букв."""
        for ending in _RU_ENDINGS_BY_LENGTH:
            if word.casefold().endswith(ending) and len(word) - len(ending) >= 3:
                return word[: len(word) - len(ending)]
        return word

    @staticmethod
    def _extract_expression(question: str) -> str:
        """Достаёт арифметику: «15% от 2500», «sqrt(144) + 2**3», «2+2»."""
        percent = _PERCENT_RE.search(question)
        if percent:
            share = float(percent.group(1).replace(",", ".")) / 100
            total = percent.group(2).replace(",", ".")
            return f"{total} * {share}"

        candidate = question
        for filler in ("посчитай", "посчитайте", "сколько будет", "сколько получится", "калькулятор", "что будет"):
            candidate = re.sub(re.escape(filler), " ", candidate, flags=re.IGNORECASE)
        candidate = candidate.strip(" .,:!?")
        if not candidate:
            return ""

        # Убираем имена функций и константы, чтобы проверить «чистую» арифметику.
        bare = candidate
        for name in sorted(CALCULATE_FUNCTIONS, key=len, reverse=True):
            bare = re.sub(rf"\b{name}\b", " ", bare, flags=re.IGNORECASE)
        if not _MATH_RE.match(bare.strip()):
            return ""
        if not re.search(r"\d", candidate):
            return ""
        if not re.search(r"[+\-*/%^]|\b(?:sqrt|abs|round|min|max|log|floor|ceil|pow|exp)\b", candidate, re.I):
            return ""
        return candidate

    @staticmethod
    def _parse_add_movie(question: str) -> dict[str, Any]:
        """Разбирает «добавь фильм Название 2024 фантастика 8.7»."""
        body = re.sub(r"добавь\s+фильм|добавить\s+фильм|добавь\s+в\s+каталог", " ", question, flags=re.IGNORECASE)
        parts = body.strip(" .!?")
        payload: dict[str, Any] = {}

        year_match = re.search(r"\b(19|20)\d{2}\b", parts)
        if year_match:
            payload["year"] = int(year_match.group(0))
            parts = parts.replace(year_match.group(0), " ")

        rating_match = re.search(r"\b(\d[.,]\d)\b", parts)
        if rating_match:
            payload["rating_imdb"] = float(rating_match.group(1).replace(",", "."))
            parts = parts.replace(rating_match.group(0), " ")

        words = [word.strip(" .,:!?") for word in parts.split() if word.strip(" .,:!?")]
        if words:
            payload["title"] = words[0]
            rest = words[1:]
            if rest:
                payload["genres"] = ", ".join(rest)

        return payload

    # ------------------------------------------------------------- оформление

    @staticmethod
    def _collect_tool_results(
        messages: list[dict[str, Any]],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Собирает все сообщения ``role="tool"`` после последнего assistant-сообщения."""
        collected: list[tuple[str, dict[str, Any]]] = []
        for message in reversed(messages):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                break
            if message.get("role") == "tool":
                try:
                    data = json.loads(message.get("content") or "{}")
                except json.JSONDecodeError:
                    data = {}
                collected.append((str(message.get("name") or ""), data if isinstance(data, dict) else {}))
            elif message.get("role") == "user":
                break
        collected.reverse()
        return collected

    def _render_results(self, results: list[tuple[str, dict[str, Any]]]) -> str:
        blocks = [self._render_one(name, data) for name, data in results]
        return "\n\n".join(block for block in blocks if block)

    def _render_one(self, name: str, data: dict[str, Any]) -> str:
        if data.get("is_error") or data.get("error"):
            message = data.get("error") or "инструмент вернул ошибку"
            return f"⚠️ {message}"

        renderers = {
            "find_movie": self._render_movies,
            "list_movies": self._render_movies,
            "find_movies_by_genre": self._render_movies,
            "get_top_movies": self._render_movies,
            "get_similar_movies": self._render_movies,
            "get_recommendations": self._render_movies,
            "get_movie": self._render_single_movie,
            "box_office_stats": self._render_box_office,
            "calculate": self._render_calculate,
            "get_movie_stats": self._render_stats,
            "get_person_filmography": self._render_filmography,
            "find_person": self._render_person,
            "rate_movie": self._render_rate,
            "get_user_ratings": self._render_user_ratings,
            "toggle_favorite": self._render_toggle_favorite,
            "get_favorites": self._render_favorites,
            "add_movie": self._render_add_movie,
            "delete_movie": self._render_delete_movie,
        }
        renderer = renderers.get(name)
        if renderer is None:
            return ""
        return renderer(data)

    @staticmethod
    def _render_movies(data: dict[str, Any]) -> str:
        movies = data.get("movies") or []
        if not movies:
            hint = data.get("hint") or "Ничего не нашлось. Попробуй другой запрос."
            return f"Ничего не нашёл. {hint}"

        if data.get("genre"):
            header = f"Подборка по жанру «{data['genre']}»"
        elif data.get("criterion"):
            header = f"Топ по критерию: {data['criterion']}"
        elif data.get("reason"):
            header = str(data["reason"])
        else:
            header = "Вот что нашёл"
        header = header.rstrip(" .:!") + ":"
        if data.get("total_found"):
            header = f"{header[:-1]} (всего найдено {data['total_found']}):"
        lines = [header]
        for movie in movies[:10]:
            genres = movie.get("genres") or ""
            rating = movie.get("rating_imdb")
            parts = [f"• {movie.get('title', '?')} ({movie.get('year') or '—'})"]
            meta = [f"IMDb {rating}"] if rating is not None else []
            if movie.get("runtime") and movie["runtime"] != "н/д":
                meta.append(movie["runtime"])
            if genres:
                meta.append(genres)
            if meta:
                parts.append(" — " + ", ".join(meta))
            lines.append("".join(parts))
        return "\n".join(lines)

    @staticmethod
    def _render_single_movie(data: dict[str, Any]) -> str:
        lines = [
            f"{data.get('title')} ({data.get('original_title') or data.get('year') or '—'})",
        ]
        facts = []
        if data.get("year"):
            facts.append(f"год: {data['year']}")
        if data.get("genres"):
            facts.append(f"жанры: {data['genres']}")
        if data.get("director"):
            facts.append(f"режиссёр: {data['director']}")
        if data.get("runtime") and data["runtime"] != "н/д":
            facts.append(f"длительность: {data['runtime']}")
        if data.get("rating_imdb") is not None:
            facts.append(f"IMDb: {data['rating_imdb']} / КП: {data.get('rating_kp') or '—'}")
        if data.get("votes") and data["votes"] != "н/д":
            facts.append(f"голосов: {data['votes']}")
        if data.get("budget") and data["budget"] != "н/д":
            facts.append(f"бюджет: {data['budget']}")
        if data.get("revenue") and data["revenue"] != "н/д":
            facts.append(f"сборы: {data['revenue']}")
        if data.get("age_rating"):
            facts.append(f"возраст: {data['age_rating']}")
        for fact in facts:
            lines.append(f"  {fact}")
        if data.get("synopsis"):
            lines.append(f"\n{data['synopsis']}")
        cast = data.get("cast") or []
        actors = [person for person in cast if person.get("role") == "actor"][:6]
        if actors:
            lines.append("\nВ ролях: " + ", ".join(
                f"{person['name']}" + (f" ({person['character']})" if person.get("character") else "")
                for person in actors
            ))
        if data.get("is_favorite"):
            lines.append("\n⭐ Этот фильм у тебя в избранном.")
        return "\n".join(lines)

    @staticmethod
    def _render_box_office(data: dict[str, Any]) -> str:
        movie = data.get("movie") or {}
        lines = [f"Финансы фильма «{movie.get('title') or data.get('title') or '—'}»:"]
        lines.append(f"  бюджет: {data.get('budget_formatted')}")
        lines.append(f"  сборы: {data.get('revenue_formatted')}")
        lines.append(f"  {data.get('profit_sign')}: {data.get('profit_formatted')}")
        if data.get("roi_percent") is not None:
            lines.append(f"  рентабельность: {data['roi_percent']}%")
        if data.get("box_office_to_budget"):
            lines.append(f"  сборы превысили бюджет в {data['box_office_to_budget']} раз")
        if data.get("verdict"):
            lines.append(f"  вывод: {data['verdict']}")
        return "\n".join(lines)

    @staticmethod
    def _render_calculate(data: dict[str, Any]) -> str:
        return str(data.get("formatted") or f"{data.get('expression')} = {data.get('result')}")

    @staticmethod
    def _render_stats(data: dict[str, Any]) -> str:
        lines = [
            "Статистика каталога:",
            f"  всего записей: {data.get('total_movies')}",
            f"  персон: {data.get('people_count')}",
            f"  средний рейтинг IMDb: {data.get('avg_rating')}",
            f"  годы: {data.get('first_year')}–{data.get('last_year')}",
            f"  суммарные сборы: {data.get('total_revenue_formatted')}",
        ]
        by_kind = data.get("by_kind") or []
        if by_kind:
            lines.append("  по типам: " + ", ".join(f"{item['name']} — {item['count']}" for item in by_kind))
        top_genres = (data.get("by_genre") or [])[:8]
        if top_genres:
            lines.append("  жанры: " + ", ".join(f"{item['name']} ({item['count']})" for item in top_genres))
        decades = data.get("by_decade") or []
        if decades:
            lines.append(
                "  по десятилетиям: "
                + ", ".join(f"{item['decade']}s — {item['count']}" for item in decades)
            )
        return "\n".join(lines)

    @staticmethod
    def _render_filmography(data: dict[str, Any]) -> str:
        person = data.get("person") or {}
        works = data.get("works") or []
        if not works:
            return f"У {person.get('name')} пока нет работ в каталоге."
        lines = [
            f"{person.get('name')} — {person.get('role')}, "
            f"{len(works)} работ в каталоге:"
        ]
        for work in works[:20]:
            rating = f"IMDb {work['rating_imdb']}" if work.get("rating_imdb") is not None else "без рейтинга"
            role = f" — {work['character']}" if work.get("character") else ""
            lines.append(f"  • {work.get('title')} ({work.get('year') or '—'}), {rating}{role}")
        return "\n".join(lines)

    @staticmethod
    def _render_person(data: dict[str, Any]) -> str:
        people = data.get("people") or []
        if not people:
            return "Никого не нашёл. Проверь имя или попробуй поиск по фильму."
        lines = [f"Нашёл персон по запросу «{data.get('query')}»:"]
        for person in people[:10]:
            details = [f"  • {person['name']}"]
            if person.get("birth_year"):
                details.append(f", {person['birth_year']} г.р.")
            if person.get("country"):
                details.append(f", {person['country']}")
            details.append(f" — {person['role']}, работ в каталоге: {person.get('works_in_catalog')}")
            lines.append("".join(details))
        return "\n".join(lines)

    @staticmethod
    def _render_rate(data: dict[str, Any]) -> str:
        average = data.get("your_average_score")
        tail = f" Твоя средняя оценка: {average}." if average is not None else ""
        return f"{data.get('message')}{tail} Оценено фильмов: {data.get('your_ratings_count')}."

    @staticmethod
    def _render_user_ratings(data: dict[str, Any]) -> str:
        ratings = data.get("ratings") or []
        if not ratings:
            return "Ты пока ничего не оценил. Скажи «поставь Началу 9»."
        average = data.get("average_score")
        lines = [f"Твои оценки ({len(ratings)} шт., средняя {average}):"]
        for item in ratings[:20]:
            note = f" — {item['note']}" if item.get("note") else ""
            lines.append(f"  • {item['movie']['title']} ({item['movie'].get('year') or '—'}) — {item['your_score']}/10{note}")
        return "\n".join(lines)

    @staticmethod
    def _render_toggle_favorite(data: dict[str, Any]) -> str:
        return str(data.get("message") or "Готово.")

    @staticmethod
    def _render_favorites(data: dict[str, Any]) -> str:
        movies = data.get("movies") or []
        if not movies:
            return "В избранном пока пусто. Скажи «добавь Начало в избранное»."
        lines = ["В избранном:"]
        for movie in movies[:20]:
            rating = f", IMDb {movie['rating_imdb']}" if movie.get("rating_imdb") is not None else ""
            lines.append(f"  • {movie.get('title')} ({movie.get('year') or '—'}){rating}")
        return "\n".join(lines)

    @staticmethod
    def _render_add_movie(data: dict[str, Any]) -> str:
        movie = data.get("movie") or {}
        return f"{data.get('message')} id={movie.get('id')}, жанры: {movie.get('genres') or '—'}"

    @staticmethod
    def _render_delete_movie(data: dict[str, Any]) -> str:
        return str(data.get("message") or "Фильм удалён.")

    # -------------------------------------------------------------small talk

    def _chitchat(self, question: str) -> str:
        text = question.casefold()
        if any(word in text for word in ("привет", "здоров", "хай", "добрый")):
            return (
                "Привет! Я работаю на MCP-инструментах, поэтому знаю только то, что лежит "
                "в каталоге.\nСпроси: «покажи топ-5», «найди Начало», «сколько заработала Дюна», "
                "«что посмотреть похоже на Интерстеллар», «посчитай 15% от 2500»."
            )
        if any(word in text for word in ("помощь", "умеешь", "что делать", "команды")):
            return (
                "Я умею:\n"
                "• искать фильмы по названию и жанру;\n"
                "• показывать топ по рейтингу и кассовым сборам;\n"
                "• подбирать похожее на конкретный фильм;\n"
                "• считать прибыль и окупаемость;\n"
                "• запоминать твои оценки и избранное;\n"
                "• считать в калькуляторе.\n\n"
                "Команды: /help, /tools, /reset"
            )
        return (
            "Не совсем понял запрос. Я работаю только через MCP-инструменты, поэтому умею "
            "искать по каталогу, считать статистику и считать в калькуляторе.\n"
            "Примеры: «покажи топ-10», «найди Дюну», «сколько фильм заработал», "
            "«посчитай 2+2». Список команд — /help"
        )

