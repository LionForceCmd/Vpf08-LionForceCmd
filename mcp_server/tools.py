"""MCP-инструменты кино-каталога.

Каждый инструмент описан в формате MCP JSON Schema (``name``, ``description``,
``inputSchema``) — именно эти описания LLM получает через ``tools/list`` и по ним
выбирает нужную функцию. Реализация лежит рядом, в ``HANDLERS``.

Базовый набор урока (``list_products``, ``find_product``, ``add_product``,
``calculate``) перенесён и доработан под кино: добавлены поиск по id и жанрам,
фильмография, топы, статистика, рекомендации, пользовательские оценки и избранное.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from calculators import (
    CalculationError,
    box_office_financials,
    format_money,
    format_runtime,
    format_votes,
    safe_eval,
)
from db import MOVIE_KINDS, MovieDB

# --------------------------------------------------------------------------- #
# Описания инструментов (MCP)
# --------------------------------------------------------------------------- #

_KIND_ENUM = list(MOVIE_KINDS)
_SORT_ENUM = ["rating", "year", "title", "votes", "budget", "revenue", "runtime"]

MCP_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_movies",
        "description": (
            "Возвращает список фильмов и сериалов из каталога с фильтрами и сортировкой. "
            "Используй, когда просят показать/отфильтровать/отсортировать каталог: "
            "«покажи топ фильмов», «дай все аниме», «что вышло после 2020», "
            "«покажи фантастику с рейтингом выше 8»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": _KIND_ENUM,
                    "description": "Тип записи: movie (фильм), series (сериал), anime, documentary, short",
                },
                "genre": {
                    "type": "string",
                    "description": "Жанр, например «Драма», «Боевик», «Комедия», «Фантастика»",
                },
                "year_from": {"type": "integer", "description": "Год не раньше указанного"},
                "year_to": {"type": "integer", "description": "Год не позже указанного"},
                "min_rating": {"type": "number", "description": "Минимальный рейтинг IMDb, 0–10"},
                "max_budget": {
                    "type": "integer",
                    "description": "Максимальный бюджет в долларах (например, 100000000 = 100 млн $)",
                },
                "sort": {
                    "type": "string",
                    "enum": _SORT_ENUM,
                    "description": "Поле сортировки. По умолчанию rating.",
                },
                "order": {"type": "string", "enum": ["asc", "desc"], "description": "Направление сортировки"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Сколько записей вернуть (по умолчанию 10, максимум 50)"},
                "offset": {"type": "integer", "minimum": 0, "description": "Смещение для постраничного вывода"},
            },
            "required": [],
        },
    },
    {
        "name": "find_movie",
        "description": (
            "Ищет фильмы, сериалы или аниме по названию (полному или частичному), "
            "в том числе по оригинальному названию. Используй для запросов вида "
            "«найди Начало», «есть ли что-то про динозавров», «что за фильм про матрицу»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Название или его часть для поиска"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25, "description": "Сколько результатов вернуть"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_movie",
        "description": (
            "Возвращает полную карточку фильма по его id: год, жанры, режиссёр, страна, "
            "хронометраж, рейтинги, бюджет, сборы, описание и список актёров. "
            "Используй, когда пользователь ссылается на конкретный фильм по номеру."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"movie_id": {"type": "integer", "description": "Числовой id фильма из каталога"}},
            "required": ["movie_id"],
        },
    },
    {
        "name": "find_movies_by_genre",
        "description": (
            "Быстрый поиск по одному жанру с сортировкой по рейтингу. "
            "Используй для запросов «покажи фильмы в категории фантастика», "
            "«что посмотреть из аниме», «дай комедии»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "genre": {"type": "string", "description": "Название жанра с заглавной буквы, например «Драма»"},
                "kind": {"type": "string", "enum": _KIND_ENUM, "description": "Ограничить тип записи"},
                "min_rating": {"type": "number", "description": "Минимальный рейтинг IMDb"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Сколько результатов вернуть"},
            },
            "required": ["genre"],
        },
    },
    {
        "name": "add_movie",
        "description": (
            "Добавляет новый фильм в каталог и возвращает его id. Используй, когда "
            "пользователь просит добавить фильм: «добавь фильм Интерстеллар 2014 фантастика 8.7»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Название на русском языке"},
                "original_title": {"type": "string", "description": "Оригинальное название, если отличается"},
                "year": {"type": "integer", "description": "Год выхода"},
                "kind": {"type": "string", "enum": _KIND_ENUM, "description": "Тип записи, по умолчанию movie"},
                "genres": {"type": "string", "description": "Жанры через запятую, например «Драма, Боевик»"},
                "director": {"type": "string", "description": "Режиссёр"},
                "country": {"type": "string", "description": "Страна производства"},
                "runtime_min": {"type": "integer", "description": "Хронометраж в минутах"},
                "rating_imdb": {"type": "number", "description": "Рейтинг IMDb от 0 до 10"},
                "rating_kp": {"type": "number", "description": "Рейтинг Кинопоиска от 0 до 10"},
                "votes": {"type": "integer", "description": "Число голосов"},
                "budget_usd": {"type": "integer", "description": "Бюджет в долларах"},
                "revenue_usd": {"type": "integer", "description": "Кассовые сборы в долларах"},
                "age_rating": {"type": "string", "description": "Возрастной рейтинг, например «16+»"},
                "synopsis": {"type": "string", "description": "Краткое описание сюжета"},
            },
            "required": ["title"],
        },
    },
    {
        "name": "update_movie",
        "description": (
            "Обновляет поля существующего фильма по его id. Используй, когда пользователь "
            "просит исправить данные: «поставь Начату рейтинг 9.5», «добавь фильму Гравитация жанр Триллер»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "movie_id": {"type": "integer", "description": "id фильма для обновления"},
                "fields": {
                    "type": "object",
                    "description": "Поля, которые нужно изменить (те же имена, что у add_movie)",
                    "properties": {
                        "title": {"type": "string"},
                        "year": {"type": "integer"},
                        "kind": {"type": "string", "enum": _KIND_ENUM},
                        "genres": {"type": "string"},
                        "director": {"type": "string"},
                        "country": {"type": "string"},
                        "runtime_min": {"type": "integer"},
                        "rating_imdb": {"type": "number"},
                        "rating_kp": {"type": "number"},
                        "votes": {"type": "integer"},
                        "budget_usd": {"type": "integer"},
                        "revenue_usd": {"type": "integer"},
                        "age_rating": {"type": "string"},
                        "synopsis": {"type": "string"},
                    },
                },
            },
            "required": ["movie_id", "fields"],
        },
    },
    {
        "name": "delete_movie",
        "description": (
            "Удаляет фильм из каталога по id. Используй только когда пользователь явно "
            "просит удалить конкретный фильм. Перед вызовом стоит уточнить название фильма."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"movie_id": {"type": "integer", "description": "id удаляемого фильма"}},
            "required": ["movie_id"],
        },
    },
    {
        "name": "find_person",
        "description": (
            "Ищет актёра или режиссёра по имени и возвращает краткую справку. "
            "Используй для запросов «кто такой Кристофер Нолан», «найди актёра ДиКаприо»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Имя или часть имени персоны"},
                "role": {"type": "string", "enum": ["actor", "director"], "description": "Ограничить ролью"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25, "description": "Сколько персон вернуть"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_person_filmography",
        "description": (
            "Возвращает фильмографию персоны (актёра или режиссёра) с её ролью в каждой работе. "
            "Используй для запросов «что снял Нолан», «фильмы с Киллианом Мёрфи»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Имя персоны"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Сколько работ вернуть"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_top_movies",
        "description": (
            "Возвращает топ фильмов по одному из критериев: рейтинг IMDb, кассовые сборы, "
            "бюджет или число голосов. Используй для запросов «топ-10 фильмов», "
            "«самые кассовые», «лучшее по версии IMDb»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "by": {
                    "type": "string",
                    "enum": ["rating", "revenue", "budget", "votes", "year"],
                    "description": "Критерий топа. По умолчанию rating.",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Размер топа (по умолчанию 10)"},
                "genre": {"type": "string", "description": "Ограничить топ одним жанром"},
                "kind": {"type": "string", "enum": _KIND_ENUM, "description": "Ограничить типом записи"},
            },
            "required": [],
        },
    },
    {
        "name": "get_movie_stats",
        "description": (
            "Возвращает статистику всей коллекции: сколько всего фильмов, распределение "
            "по типам, жанрам, десятилетиям и странам, средний рейтинг и суммарные сборы. "
            "Используй для вопросов «что у тебя в базе», «сколько фильмов», «какие жанры есть»."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_similar_movies",
        "description": (
            "Подбирает фильмы, похожие на указанный: по общим жанрам, режиссёру, "
            "типу и близости года. Используй для запросов «что посмотреть похоже на Начало», "
            "«дай что-то в духе Дюны»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Название фильма-образца"},
                "movie_id": {"type": "integer", "description": "Либо id фильма-образца"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Сколько аналогов вернуть"},
            },
            "required": [],
        },
    },
    {
        "name": "rate_movie",
        "description": (
            "Сохраняет личную оценку пользователя фильму по шкале от 1 до 10. "
            "Используй для запросов «поставь Началу 9», «оценил Дюну на 8»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Название фильма"},
                "movie_id": {"type": "integer", "description": "Либо id фильма"},
                "score": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Оценка от 1 до 10"},
                "note": {"type": "string", "description": "Необязательный комментарий"},
            },
            "required": ["score"],
        },
    },
    {
        "name": "get_user_ratings",
        "description": (
            "Возвращает фильмы, которые пользователь уже оценил, с их оценками. "
            "Используй для запросов «что я оценил», «какие у меня оценки», «покажи мою оценку Начала»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Сколько оценок вернуть"}},
            "required": [],
        },
    },
    {
        "name": "toggle_favorite",
        "description": (
            "Добавляет фильм в избранное или убирает оттуда и сообщает текущее состояние. "
            "Используй для запросов «добавь в избранное», «убери из избранного», «мой список»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Название фильма"},
                "movie_id": {"type": "integer", "description": "Либо id фильма"},
            },
            "required": [],
        },
    },
    {
        "name": "get_favorites",
        "description": (
            "Возвращает список избранного пользователя. Используй для запросов "
            "«что у меня в избранном», «покажи мой список фильмов»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Сколько записей вернуть"}},
            "required": [],
        },
    },
    {
        "name": "get_recommendations",
        "description": (
            "Подбирает фильмы к вкусам пользователя на основе его оценок и избранного. "
            "Используй для запросов «что мне посмотреть», «порекомендуй что-нибудь»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Сколько рекомендаций вернуть"}},
            "required": [],
        },
    },
    {
        "name": "box_office_stats",
        "description": (
            "Считает финансовую картину фильма по бюджету и кассовым сборам: прибыль или убыток, "
            "рентабельность в процентах, во сколько раз сборы превысили бюджет. "
            "Используй для запросов «сколько заработал Начало», «окупился ли фильм», «прибыль Дюны»."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Название фильма"},
                "movie_id": {"type": "integer", "description": "Либо id фильма"},
            },
            "required": [],
        },
    },
    {
        "name": "calculate",
        "description": (
            "Безопасный математический калькулятор. Используй для любых вычислений: "
            "«сколько будет 2+2», «посчитай 15% от 2500», «корень из 144». "
            "Поддерживает + - * / // % ** и функции sqrt, abs, round, min, max, log."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Математическое выражение, например «(2+2)*10»"}},
            "required": ["expression"],
        },
    },
]


# --------------------------------------------------------------------------- #
# Представление карточек для LLM
# --------------------------------------------------------------------------- #

def movie_brief(movie: dict[str, Any]) -> dict[str, Any]:
    """Короткая карточка фильма — то, что LLM видит в выдаче инструмента."""
    return {
        "id": movie["id"],
        "title": movie["title"],
        "original_title": movie.get("original_title"),
        "year": movie.get("year"),
        "kind": movie.get("kind"),
        "genres": movie.get("genres"),
        "director": movie.get("director"),
        "country": movie.get("country"),
        "runtime": format_runtime(movie.get("runtime_min")),
        "rating_imdb": movie.get("rating_imdb"),
        "rating_kp": movie.get("rating_kp"),
        "votes": format_votes(movie.get("votes")),
        "budget": format_money(movie.get("budget_usd")),
        "revenue": format_money(movie.get("revenue_usd")),
        "age_rating": movie.get("age_rating"),
    }


def movie_full(movie: dict[str, Any], cast: list[dict[str, Any]], db: MovieDB, user_id: str) -> dict[str, Any]:
    """Полная карточка фильма с актёрами и персональными отметками."""
    payload = movie_brief(movie)
    payload["synopsis"] = movie.get("synopsis")
    payload["cast"] = [
        {
            "name": person["name"],
            "role": person["role"],
            "character": person.get("character"),
            "country": person.get("country"),
        }
        for person in cast
    ]
    payload["is_favorite"] = db.is_favorite(user_id, movie["id"])
    return payload


# --------------------------------------------------------------------------- #
# Обработчики инструментов
# --------------------------------------------------------------------------- #

ToolHandler = Callable[[MovieDB, dict[str, Any], str], Awaitable[dict[str, Any]]]


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    number = _to_float(value)
    return int(number) if number is not None else None


def _resolve_movie(db: MovieDB, args: dict[str, Any]) -> dict[str, Any] | None:
    """Находит фильм по ``movie_id`` либо по названию."""
    movie_id = _to_int(args.get("movie_id"))
    if movie_id is not None:
        return db.get_movie(movie_id)
    title = (args.get("title") or "").strip()
    if title:
        return db.get_movie_by_title(title)
    return None


async def _list_movies(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    kind = args.get("kind")
    if kind and kind not in MOVIE_KINDS:
        return {"error": f"Неизвестный тип «{kind}». Допустимые: {', '.join(MOVIE_KINDS)}"}

    movies, total = db.list_movies(
        kind=kind or None,
        genre=(args.get("genre") or None),
        year_from=_to_int(args.get("year_from")),
        year_to=_to_int(args.get("year_to")),
        min_rating=_to_float(args.get("min_rating")),
        max_budget=_to_int(args.get("max_budget")),
        sort=str(args.get("sort") or "rating"),
        order=str(args.get("order") or "desc"),
        limit=_clamp(args.get("limit"), 1, 50, 10),
        offset=_clamp(args.get("offset"), 0, 10_000, 0),
    )

    filters = {
        key: args.get(key)
        for key in ("kind", "genre", "year_from", "year_to", "min_rating", "max_budget")
        if args.get(key) is not None
    }
    return {
        "total_found": total,
        "returned": len(movies),
        "filters": filters,
        "movies": [movie_brief(movie) for movie in movies],
    }


async def _find_movie(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "Нужно передать непустой поисковый запрос (query)"}

    movies = db.find_movies_by_title(query, limit=_clamp(args.get("limit"), 1, 25, 10))
    if not movies:
        return {
            "query": query,
            "found": 0,
            "movies": [],
            "hint": "Ничего не найдено. Попробуй уточнить запрос или поищи по жанру через list_movies.",
        }
    return {
        "query": query,
        "found": len(movies),
        "movies": [movie_brief(movie) for movie in movies],
    }


async def _get_movie(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    movie = _resolve_movie(db, args)
    if movie is None:
        return {"error": "Фильм не найден. Проверь id или найди фильм через find_movie."}
    return movie_full(movie, db.get_cast(movie["id"]), db, user_id)


async def _find_by_genre(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    genre = (args.get("genre") or "").strip()
    if not genre:
        return {"error": "Нужно указать жанр (genre)"}

    movies, total = db.list_movies(
        kind=args.get("kind") if args.get("kind") in MOVIE_KINDS else None,
        genre=genre,
        min_rating=_to_float(args.get("min_rating")),
        sort="rating",
        order="desc",
        limit=_clamp(args.get("limit"), 1, 50, 10),
    )
    known = db.all_genres()
    return {
        "genre": genre,
        "total_found": total,
        "returned": len(movies),
        "movies": [movie_brief(movie) for movie in movies],
        "available_genres": known,
    }


async def _add_movie(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    title = (args.get("title") or "").strip()
    if not title:
        return {"error": "Нужно указать название фильма (title)"}

    existing = db.get_movie_by_title(title)
    if existing:
        return {
            "error": f"Фильм «{title}» уже есть в каталоге (id={existing['id']}).",
            "movie": movie_brief(existing),
            "hint": "Предложи обновить данные через update_movie или добавить другой фильм.",
        }

    kind = args.get("kind") if args.get("kind") in MOVIE_KINDS else "movie"
    rating = _to_float(args.get("rating_imdb"))
    if rating is not None and not 0 <= rating <= 10:
        return {"error": "Рейтинг должен быть в диапазоне от 0 до 10"}

    movie_id = db.add_movie(
        {
            "title": title,
            "original_title": args.get("original_title"),
            "year": _to_int(args.get("year")),
            "kind": kind,
            "genres": args.get("genres"),
            "director": args.get("director"),
            "country": args.get("country"),
            "runtime_min": _to_int(args.get("runtime_min")),
            "rating_imdb": rating,
            "rating_kp": _to_float(args.get("rating_kp")),
            "votes": _to_int(args.get("votes")),
            "budget_usd": _to_int(args.get("budget_usd")),
            "revenue_usd": _to_int(args.get("revenue_usd")),
            "age_rating": args.get("age_rating"),
            "synopsis": args.get("synopsis"),
        }
    )
    movie = db.get_movie(movie_id)
    return {
        "message": f"Фильм «{title}» добавлен в каталог.",
        "movie": movie_brief(movie) if movie else {"id": movie_id, "title": title},
    }


async def _update_movie(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    movie_id = _to_int(args.get("movie_id"))
    if movie_id is None:
        return {"error": "Нужно указать movie_id фильма для обновления"}

    fields = args.get("fields")
    if not isinstance(fields, dict) or not fields:
        return {"error": "Нужно передать объект fields с полями, которые нужно изменить"}

    if not db.get_movie(movie_id):
        return {"error": f"Фильм с id={movie_id} не найден"}

    if not db.update_movie(movie_id, fields):
        return {"error": "Не удалось обновить фильм: переданы неизвестные поля"}

    movie = db.get_movie(movie_id)
    return {
        "message": f"Фильм «{movie['title']}» обновлён.",
        "updated_fields": list(fields),
        "movie": movie_brief(movie) if movie else None,
    }


async def _delete_movie(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    movie_id = _to_int(args.get("movie_id"))
    if movie_id is None:
        return {"error": "Нужно указать movie_id фильма для удаления"}

    movie = db.get_movie(movie_id)
    if movie is None:
        return {"error": f"Фильм с id={movie_id} не найден"}

    db.delete_movie(movie_id)
    return {"message": f"Фильм «{movie['title']}» (id={movie_id}) удалён из каталога."}


async def _find_person(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    query = (args.get("query") or "").strip()
    if not query:
        return {"error": "Нужно указать имя персоны (query)"}

    people = db.search_people(
        query=query,
        role=args.get("role") if args.get("role") in {"actor", "director"} else None,
        limit=_clamp(args.get("limit"), 1, 25, 10),
    )
    return {
        "query": query,
        "found": len(people),
        "people": [
            {
                "name": person["name"],
                "role": person["role"],
                "birth_year": person.get("birth_year"),
                "country": person.get("country"),
                "works_in_catalog": person.get("works_count"),
            }
            for person in people
        ],
    }


async def _get_filmography(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    name = (args.get("name") or "").strip()
    if not name:
        return {"error": "Нужно указать имя персоны (name)"}

    person = db.get_person_by_name(name)
    if person is None:
        return {"error": f"Персона «{name}» не найдена в каталоге"}

    works = db.get_person_filmography(person["id"], limit=_clamp(args.get("limit"), 1, 50, 50))
    return {
        "person": {
            "name": person["name"],
            "role": person["role"],
            "birth_year": person.get("birth_year"),
            "country": person.get("country"),
        },
        "total_works": len(works),
        "works": [
            {
                **movie_brief(work),
                "character": work.get("character"),
                "as": "режиссёр" if person["role"] == "director" else "актёр",
            }
            for work in works
        ],
    }


async def _get_top_movies(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    by = args.get("by") if args.get("by") in {"rating", "revenue", "budget", "votes", "year"} else "rating"
    movies, _ = db.list_movies(
        kind=args.get("kind") if args.get("kind") in MOVIE_KINDS else None,
        genre=args.get("genre") or None,
        sort=by,
        order="desc",
        limit=_clamp(args.get("limit"), 1, 50, 10),
    )
    criteria = {
        "rating": "рейтинг IMDb",
        "revenue": "кассовые сборы",
        "budget": "бюджет",
        "votes": "число голосов",
        "year": "год выхода",
    }
    return {
        "by": by,
        "criterion": criteria[by],
        "count": len(movies),
        "movies": [movie_brief(movie) for movie in movies],
    }


async def _get_movie_stats(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    stats = db.get_stats()
    stats["total_revenue_formatted"] = format_money(stats.get("total_revenue_usd"))
    stats["total_budget_formatted"] = format_money(stats.get("total_budget_usd"))
    stats["genres"] = db.all_genres()
    return stats


async def _get_similar_movies(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    base = _resolve_movie(db, args)
    if base is None:
        return {"error": "Не удалось найти фильм-образец. Передай название (title) или id."}

    similar = db.get_similar_movies(base["id"], limit=_clamp(args.get("limit"), 1, 20, 5))
    return {
        "based_on": movie_brief(base),
        "count": len(similar),
        "movies": [
            {
                **movie_brief(movie),
                "shared_genres": movie.get("shared_genres"),
                "match_score": movie.get("match_score"),
            }
            for movie in similar
        ],
    }


async def _rate_movie(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    movie = _resolve_movie(db, args)
    if movie is None:
        return {"error": "Фильм не найден. Передай название (title) или id."}

    score = _to_int(args.get("score"))
    if score is None or not 1 <= score <= 10:
        return {"error": "Оценка должна быть целым числом от 1 до 10"}

    db.set_rating(user_id, movie["id"], score, args.get("note"))
    ratings = db.get_user_ratings(user_id, limit=100)
    average = round(sum(r["score"] for r in ratings) / len(ratings), 2) if ratings else None
    return {
        "message": f"Оценка «{score}» сохранена для фильма «{movie['title']}».",
        "movie": movie_brief(movie),
        "your_score": score,
        "your_ratings_count": len(ratings),
        "your_average_score": average,
    }


async def _get_user_ratings(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    ratings = db.get_user_ratings(user_id, limit=_clamp(args.get("limit"), 1, 50, 20))
    average = round(sum(r["score"] for r in ratings) / len(ratings), 2) if ratings else None
    return {
        "count": len(ratings),
        "average_score": average,
        "ratings": [
            {
                "movie": movie_brief(rating),
                "your_score": rating["score"],
                "note": rating.get("note"),
                "rated_at": rating.get("created_at"),
            }
            for rating in ratings
        ],
    }


async def _toggle_favorite(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    movie = _resolve_movie(db, args)
    if movie is None:
        return {"error": "Фильм не найден. Передай название (title) или id."}

    is_favorite = db.toggle_favorite(user_id, movie["id"])
    return {
        "movie": movie_brief(movie),
        "is_favorite": is_favorite,
        "message": (
            f"«{movie['title']}» добавлен в избранное."
            if is_favorite
            else f"«{movie['title']}» убран из избранного."
        ),
    }


async def _get_favorites(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    favorites = db.get_favorites(user_id, limit=_clamp(args.get("limit"), 1, 50, 20))
    return {
        "count": len(favorites),
        "movies": [movie_brief(movie) for movie in favorites],
    }


async def _get_recommendations(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    limit = _clamp(args.get("limit"), 1, 20, 5)
    recommendations = db.get_recommendations(user_id, limit=limit)

    if not recommendations:
        # оценок ещё нет — предлагаем самый рейтинговый жанр коллекции
        popular, _ = db.list_movies(sort="rating", order="desc", limit=limit)
        return {
            "count": len(popular),
            "personalized": False,
            "reason": "Ты пока ничего не оценил, поэтому показываю топ каталога.",
            "movies": [movie_brief(movie) for movie in popular],
            "hint": "Оцени пару фильмов через rate_movie, и я подберу рекомендации под тебя.",
        }

    return {
        "count": len(recommendations),
        "personalized": True,
        "reason": "Подобрано по жанрам, которые ты оценил выше среднего.",
        "movies": [
            {**movie_brief(movie), "reason": movie.get("reason")} for movie in recommendations
        ],
    }


async def _box_office_stats(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    movie = _resolve_movie(db, args)
    if movie is None:
        return {"error": "Фильм не найден. Передай название (title) или id."}

    try:
        stats = box_office_financials(
            movie.get("budget_usd"), movie.get("revenue_usd"), movie["title"]
        )
    except CalculationError as exc:
        return {"error": str(exc), "movie": movie_brief(movie)}

    stats["movie"] = movie_brief(movie)
    return stats


async def _calculate(db: MovieDB, args: dict[str, Any], user_id: str) -> dict[str, Any]:
    try:
        return safe_eval(args.get("expression", ""))
    except CalculationError as exc:
        return {"error": str(exc)}


HANDLERS: dict[str, ToolHandler] = {
    "list_movies": _list_movies,
    "find_movie": _find_movie,
    "get_movie": _get_movie,
    "find_movies_by_genre": _find_by_genre,
    "add_movie": _add_movie,
    "update_movie": _update_movie,
    "delete_movie": _delete_movie,
    "find_person": _find_person,
    "get_person_filmography": _get_filmography,
    "get_top_movies": _get_top_movies,
    "get_movie_stats": _get_movie_stats,
    "get_similar_movies": _get_similar_movies,
    "rate_movie": _rate_movie,
    "get_user_ratings": _get_user_ratings,
    "toggle_favorite": _toggle_favorite,
    "get_favorites": _get_favorites,
    "get_recommendations": _get_recommendations,
    "box_office_stats": _box_office_stats,
    "calculate": _calculate,
}


# --------------------------------------------------------------------------- #
# Ресурсы MCP (контекст, который сервер умеет отдавать целиком)
# --------------------------------------------------------------------------- #

MCP_RESOURCES: list[dict[str, Any]] = [
    {
        "uri": "movie://catalog/stats",
        "name": "Статистика каталога",
        "title": "Сводка по кино-каталогу",
        "description": "Количество фильмов, распределение по жанрам и типам, средний рейтинг.",
        "mimeType": "application/json",
    },
    {
        "uri": "movie://catalog/genres",
        "name": "Жанры каталога",
        "title": "Список доступных жанров",
        "description": "Все жанры, которые встречаются в каталоге, с количеством фильмов.",
        "mimeType": "application/json",
    },
    {
        "uri": "movie://catalog/top",
        "name": "Топ каталога",
        "title": "Топ-10 по рейтингу IMDb",
        "description": "Десять лучших записей каталога по версии IMDb.",
        "mimeType": "application/json",
    },
]


def read_resource(db: MovieDB, uri: str) -> dict[str, Any] | None:
    """Возвращает содержимое ресурса по ``uri`` или ``None``, если ресурса нет."""
    if uri == "movie://catalog/stats":
        stats = db.get_stats()
        stats["total_revenue_formatted"] = format_money(stats.get("total_revenue_usd"))
        return {"uri": uri, "mimeType": "application/json", "text": _dumps(stats)}
    if uri == "movie://catalog/genres":
        genres = db.all_genres()
        return {
            "uri": uri,
            "mimeType": "application/json",
            "text": _dumps({"count": len(genres), "genres": genres}),
        }
    if uri == "movie://catalog/top":
        movies, _ = db.list_movies(sort="rating", order="desc", limit=10)
        return {
            "uri": uri,
            "mimeType": "application/json",
            "text": _dumps({"count": len(movies), "movies": [movie_brief(m) for m in movies]}),
        }
    return None


def _dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
