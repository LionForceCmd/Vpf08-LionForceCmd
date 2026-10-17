"""Слой работы с базой данных кино-каталога (SQLite3).

Отвечает за схему БД, первичное наполнение демо-данными и все SQL-запросы,
которые вызывают MCP-инструменты из ``tools.py``.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS movies (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    title          TEXT    NOT NULL,
    original_title TEXT,
    year           INTEGER,
    kind           TEXT    NOT NULL DEFAULT 'movie',
    genres         TEXT    NOT NULL DEFAULT '',
    director       TEXT,
    country        TEXT,
    runtime_min    INTEGER,
    rating_imdb    REAL,
    rating_kp      REAL,
    votes          INTEGER,
    budget_usd     INTEGER,
    revenue_usd    INTEGER,
    age_rating     TEXT,
    synopsis       TEXT,
    created_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS people (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT    NOT NULL UNIQUE,
    role       TEXT    NOT NULL DEFAULT 'actor',
    birth_year INTEGER,
    country    TEXT
);

CREATE TABLE IF NOT EXISTS credits (
    movie_id  INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    person_id INTEGER NOT NULL REFERENCES people(id)  ON DELETE CASCADE,
    character TEXT,
    PRIMARY KEY (movie_id, person_id)
);

CREATE TABLE IF NOT EXISTS user_ratings (
    user_id    TEXT    NOT NULL,
    movie_id   INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    score      INTEGER NOT NULL CHECK (score BETWEEN 1 AND 10),
    note       TEXT,
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, movie_id)
);

CREATE TABLE IF NOT EXISTS favorites (
    user_id    TEXT    NOT NULL,
    movie_id   INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    created_at TEXT    NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_movies_title   ON movies (title);
CREATE INDEX IF NOT EXISTS idx_movies_year    ON movies (year);
CREATE INDEX IF NOT EXISTS idx_movies_genres  ON movies (genres);
CREATE INDEX IF NOT EXISTS idx_movies_kind    ON movies (kind);
CREATE INDEX IF NOT EXISTS idx_movies_rating  ON movies (rating_imdb);
CREATE INDEX IF NOT EXISTS idx_credits_person ON credits (person_id);
"""

MOVIE_KINDS = ("movie", "series", "anime", "documentary")

MOVIE_COLUMNS = (
    "title",
    "original_title",
    "year",
    "kind",
    "genres",
    "director",
    "country",
    "runtime_min",
    "rating_imdb",
    "rating_kp",
    "votes",
    "budget_usd",
    "revenue_usd",
    "age_rating",
    "synopsis",
)

# Белый список полей сортировки: значение приходит от LLM, в SQL подставляется
# только проверенный идентификатор колонки.
SORT_COLUMNS = {
    "rating": "rating_imdb",
    "year": "year",
    "title": "title",
    "votes": "votes",
    "budget": "budget_usd",
    "revenue": "revenue_usd",
    "runtime": "runtime_min",
}


def _lower_u(value: Any) -> str:
    """Unicode-версия LOWER(): casefold() корректно работает и с кириллицей."""
    return value.casefold() if isinstance(value, str) else ""


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _rows_to_dicts(rows: Sequence[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def normalize_genres(genres: str | None) -> str:
    """Приводит жанры к единому виду: ``"драма, боевик"`` → ``"Боевик, Драма"``."""
    if not genres:
        return ""
    if isinstance(genres, (list, tuple, set)):
        parts = [str(item) for item in genres]
    else:
        parts = re.split(r"[,;/]| и ", str(genres))
    cleaned = {_capitalize(part.strip()) for part in parts if part and part.strip()}
    return ", ".join(sorted(cleaned, key=str.casefold))


def _capitalize(value: str) -> str:
    """Первая буква заглавная, остальные как есть: ``"драма"`` → ``"Драма"``."""
    return value[:1].upper() + value[1:] if value else value


class MovieDB:
    """Тонкая обёртка над sqlite3 с набором методов для MCP-инструментов."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    # ------------------------------------------------------------------ core

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Открывает соединение. Для каждого вызова — своё соединение."""
        conn = sqlite3.connect(self.db_path, timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        # Штатная lower_u() в SQLite работает только с ASCII, поэтому регистронезависимый
        # поиск по кириллице делаем через свою функцию на основе str.casefold().
        conn.create_function("lower_u", 1, _lower_u)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        parent = Path(self.db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def count_movies(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0])

    # --------------------------------------------------------------- выборки

    def list_movies(
        self,
        *,
        kind: str | None = None,
        genre: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        min_rating: float | None = None,
        max_budget: int | None = None,
        sort: str = "rating",
        order: str = "desc",
        limit: int = 10,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        where: list[str] = []
        params: list[Any] = []

        if kind:
            where.append("kind = ?")
            params.append(kind)
        if genre:
            where.append("lower_u(genres) LIKE ?")
            params.append(f"%{genre.casefold()}%")
        if year_from is not None:
            where.append("year >= ?")
            params.append(year_from)
        if year_to is not None:
            where.append("year <= ?")
            params.append(year_to)
        if min_rating is not None:
            where.append("rating_imdb >= ?")
            params.append(min_rating)
        if max_budget is not None:
            where.append("budget_usd <= ?")
            params.append(max_budget)

        clause = f" WHERE {' AND '.join(where)}" if where else ""
        column = SORT_COLUMNS.get(sort, "rating_imdb")
        direction = "ASC" if str(order).lower() in {"asc", "ascending", "1"} else "DESC"

        with self.connect() as conn:
            total = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM movies{clause}", params
                ).fetchone()[0]
            )
            rows = conn.execute(
                f"SELECT * FROM movies{clause} "
                f"ORDER BY {column} IS NULL, {column} {direction}, title ASC "
                f"LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return _rows_to_dicts(rows), total

    def find_movies_by_title(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        pattern = f"%{query.strip().casefold()}%"
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT *,
                       CASE
                           WHEN lower_u(title) = ?             THEN 0
                           WHEN lower_u(title) LIKE ?          THEN 1
                           WHEN lower_u(original_title) LIKE ? THEN 2
                           ELSE 3
                       END AS match_rank
                FROM movies
                WHERE lower_u(title) LIKE ?
                   OR lower_u(IFNULL(original_title, '')) LIKE ?
                ORDER BY match_rank, rating_imdb DESC NULLS LAST, year DESC
                LIMIT ?
                """,
                (query.strip().casefold(), pattern, pattern, pattern, pattern, limit),
            ).fetchall()
        return _rows_to_dicts(rows)

    def get_movie(self, movie_id: int) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM movies WHERE id = ?", (movie_id,)).fetchone()
        return _row_to_dict(row)

    def get_movie_by_title(self, title: str) -> dict[str, Any] | None:
        movies = self.find_movies_by_title(title, limit=1)
        return movies[0] if movies else None

    def get_cast(self, movie_id: int, limit: int = 15) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT p.name, p.role, p.birth_year, p.country, c.character
                FROM credits c
                JOIN people p ON p.id = c.person_id
                WHERE c.movie_id = ?
                ORDER BY CASE p.role WHEN 'director' THEN 0 ELSE 1 END, p.name
                LIMIT ?
                """,
                (movie_id, limit),
            ).fetchall()
        return _rows_to_dicts(rows)

    # ------------------------------------------------------------- персоналии

    def search_people(
        self, query: str | None = None, role: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        where: list[str] = []
        params: list[Any] = []
        if query:
            where.append("lower_u(name) LIKE ?")
            params.append(f"%{query.strip().casefold()}%")
        if role:
            where.append("role = ?")
            params.append(role)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT p.*,
                       (SELECT COUNT(*) FROM credits c
                         JOIN movies m ON m.id = c.movie_id
                        WHERE c.person_id = p.id) AS works_count
                FROM people p{clause}
                ORDER BY works_count DESC, p.name
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return _rows_to_dicts(rows)

    def get_person_by_name(self, name: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM people WHERE lower_u(name) = ?", (name.strip().casefold(),)
            ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT * FROM people WHERE lower_u(name) LIKE ? LIMIT 1",
                    (f"%{name.strip().casefold()}%",),
                ).fetchone()
        return _row_to_dict(row)

    def get_person_filmography(self, person_id: int, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT m.*, c.character
                FROM credits c
                JOIN movies m ON m.id = c.movie_id
                WHERE c.person_id = ?
                ORDER BY m.year DESC NULLS LAST, m.rating_imdb DESC NULLS LAST
                LIMIT ?
                """,
                (person_id, limit),
            ).fetchall()
        return _rows_to_dicts(rows)

    # ---------------------------------------------------------------- запись

    @staticmethod
    def _movie_fields(data: dict[str, Any]) -> dict[str, Any]:
        """Готовит набор колонок для вставки фильма из словаря."""
        fields = {
            key: data.get(key)
            for key in MOVIE_COLUMNS
            if key in data
        }
        fields["genres"] = normalize_genres(fields.get("genres"))
        fields.setdefault("kind", "movie")
        fields["created_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return fields

    @staticmethod
    def _insert_sql(fields: dict[str, Any]) -> str:
        columns = ", ".join(fields)
        placeholders = ", ".join(f":{name}" for name in fields)
        return f"INSERT INTO movies ({columns}) VALUES ({placeholders})"

    def add_movie(self, data: dict[str, Any]) -> int:
        fields = self._movie_fields(data)
        with self.connect() as conn:
            cursor = conn.execute(self._insert_sql(fields), fields)
            return int(cursor.lastrowid)

    def bulk_seed(
        self,
        movies: Sequence[dict[str, Any]],
        people: Sequence[dict[str, Any]],
        credits: Sequence[tuple[str, str, str | None]],
    ) -> int:
        """Наполняет все таблицы одной транзакцией.

        Атомарность важна: если сервер прервут посреди заливки, база останется
        пустой и данные дозаполнятся при следующем запуске, а не «наполовину».
        """
        with self.connect() as conn:
            # На старте скорость важнее долговечности: данных много, они одноразовые.
            conn.execute("PRAGMA synchronous = OFF")

            movie_ids: dict[str, int] = {}
            for movie in movies:
                fields = self._movie_fields(movie)
                cursor = conn.execute(self._insert_sql(fields), fields)
                movie_ids[movie["title"]] = int(cursor.lastrowid)

            for person in people:
                conn.execute(
                    "INSERT OR IGNORE INTO people (name, role, birth_year, country) "
                    "VALUES (:name, :role, :birth_year, :country)",
                    person,
                )
            person_ids = {
                row["name"]: int(row["id"])
                for row in conn.execute("SELECT id, name FROM people").fetchall()
            }

            links = [
                (movie_ids[movie_title], person_ids[person_name], character)
                for movie_title, person_name, character in credits
                if movie_title in movie_ids and person_name in person_ids
            ]
            conn.executemany(
                "INSERT OR IGNORE INTO credits (movie_id, person_id, character) VALUES (?, ?, ?)",
                links,
            )
        return len(movies)

    def update_movie(self, movie_id: int, data: dict[str, Any]) -> bool:
        allowed = {
            "title",
            "original_title",
            "year",
            "kind",
            "genres",
            "director",
            "country",
            "runtime_min",
            "rating_imdb",
            "rating_kp",
            "votes",
            "budget_usd",
            "revenue_usd",
            "age_rating",
            "synopsis",
        }
        fields = {k: v for k, v in data.items() if k in allowed}
        if not fields:
            return False
        if "genres" in fields:
            fields["genres"] = normalize_genres(fields["genres"])
        assignments = ", ".join(f"{name} = :{name}" for name in fields)
        with self.connect() as conn:
            cursor = conn.execute(
                f"UPDATE movies SET {assignments} WHERE id = :__id",
                {**fields, "__id": movie_id},
            )
            return cursor.rowcount > 0

    def delete_movie(self, movie_id: int) -> bool:
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM movies WHERE id = ?", (movie_id,))
            return cursor.rowcount > 0

    # -------------------------------------------------- пользовательские данные

    def set_rating(
        self, user_id: str, movie_id: int, score: int, note: str | None = None
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO user_ratings (user_id, movie_id, score, note)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, movie_id)
                DO UPDATE SET score = excluded.score,
                              note   = excluded.note,
                              created_at = datetime('now')
                """,
                (str(user_id), movie_id, int(score), note),
            )

    def get_user_ratings(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT r.score, r.note, r.created_at, m.*
                FROM user_ratings r
                JOIN movies m ON m.id = r.movie_id
                WHERE r.user_id = ?
                ORDER BY r.created_at DESC, m.title
                LIMIT ?
                """,
                (str(user_id), limit),
            ).fetchall()
        return _rows_to_dicts(rows)

    def toggle_favorite(self, user_id: str, movie_id: int) -> bool:
        """Переключает избранное. Возвращает True, если фильм теперь в избранном."""
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM favorites WHERE user_id = ? AND movie_id = ?",
                (str(user_id), movie_id),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO favorites (user_id, movie_id) VALUES (?, ?)",
                    (str(user_id), movie_id),
                )
                return True
            conn.execute(
                "DELETE FROM favorites WHERE user_id = ? AND movie_id = ?",
                (str(user_id), movie_id),
            )
            return False

    def is_favorite(self, user_id: str, movie_id: int) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM favorites WHERE user_id = ? AND movie_id = ?",
                (str(user_id), movie_id),
            ).fetchone()
        return row is not None

    def get_favorites(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT m.*, f.created_at AS added_at
                FROM favorites f
                JOIN movies m ON m.id = f.movie_id
                WHERE f.user_id = ?
                ORDER BY f.created_at DESC
                LIMIT ?
                """,
                (str(user_id), limit),
            ).fetchall()
        return _rows_to_dicts(rows)

    def get_recommendations(self, user_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Рекомендации по жанрам, которые пользователь оценил выше среднего."""
        with self.connect() as conn:
            genre_rows = conn.execute(
                """
                SELECT m.genres, AVG(r.score) AS avg_score, COUNT(*) AS cnt
                FROM user_ratings r
                JOIN movies m ON m.id = r.movie_id
                WHERE r.user_id = ?
                GROUP BY m.genres
                ORDER BY avg_score DESC, cnt DESC
                LIMIT 3
                """,
                (str(user_id),),
            ).fetchall()

            # Уже оценённые и добавленные в избранное не рекомендуем повторно.
            seen = {
                int(row["movie_id"])
                for row in conn.execute(
                    "SELECT movie_id FROM user_ratings WHERE user_id = ?", (str(user_id),)
                ).fetchall()
            }
            seen |= {
                int(row["movie_id"])
                for row in conn.execute(
                    "SELECT movie_id FROM favorites WHERE user_id = ?", (str(user_id),)
                ).fetchall()
            }

            result: list[dict[str, Any]] = []
            for genre_row in genre_rows:
                genres = [g.strip() for g in (genre_row["genres"] or "").split(",") if g.strip()]
                if not genres:
                    continue
                condition = " OR ".join(["lower_u(m.genres) LIKE ?"] * len(genres))
                params = [f"%{genre.casefold()}%" for genre in genres]
                rows = conn.execute(
                    f"""
                    SELECT m.* FROM movies m
                    WHERE ({condition}) AND m.rating_imdb IS NOT NULL
                    ORDER BY m.rating_imdb DESC
                    LIMIT ?
                    """,
                    [*params, limit * 3],
                ).fetchall()
                for row in rows:
                    if int(row["id"]) in seen:
                        continue
                    seen.add(int(row["id"]))
                    result.append(
                        {
                            **dict(row),
                            "matched_genres": ", ".join(genres),
                            "reason": f"похоже на ваши оценки в жанрах: {', '.join(genres)}",
                        }
                    )
                    if len(result) >= limit:
                        return result
        return result

    # ----------------------------------------------------------- аналитика

    def get_similar_movies(self, movie_id: int, limit: int = 5) -> list[dict[str, Any]]:
        base = self.get_movie(movie_id)
        if base is None:
            return []
        base_genres = {g.strip() for g in (base["genres"] or "").split(",") if g.strip()}
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM movies
                WHERE id != ? AND rating_imdb IS NOT NULL
                ORDER BY rating_imdb DESC
                LIMIT 400
                """,
                (movie_id,),
            ).fetchall()

        scored: list[tuple[float, dict[str, Any]]] = []
        base_year = base.get("year") or 0
        for row in rows:
            item = dict(row)
            item_genres = {g.strip() for g in (item["genres"] or "").split(",") if g.strip()}
            shared = base_genres & item_genres
            if not shared:
                continue
            score = 3.0 * len(shared)
            if base.get("director") and base["director"] == item.get("director"):
                score += 2.0
            if base.get("kind") == item.get("kind"):
                score += 1.0
            year_gap = abs((item.get("year") or base_year) - base_year)
            score += max(0.0, 2.0 - year_gap / 15.0)
            score += (item.get("rating_imdb") or 0) / 2.5
            item["shared_genres"] = ", ".join(sorted(shared))
            item["match_score"] = round(score, 2)
            scored.append((score, item))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:limit]]

    @staticmethod
    def _group_by_column(conn: sqlite3.Connection, column: str) -> list[dict[str, Any]]:
        rows = conn.execute(
            f"""
            SELECT {column} AS name, COUNT(*) AS count
            FROM movies WHERE {column} IS NOT NULL AND {column} != ''
            GROUP BY {column} ORDER BY count DESC
            """
        ).fetchall()
        return _rows_to_dicts(rows)

    def _genre_counter(self) -> dict[str, int]:
        counter: dict[str, int] = {}
        with self.connect() as conn:
            for row in conn.execute("SELECT genres FROM movies").fetchall():
                for genre in (row["genres"] or "").split(","):
                    genre = genre.strip()
                    if genre:
                        counter[genre] = counter.get(genre, 0) + 1
        return counter

    def get_stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0])

            by_kind = conn.execute(
                """
                SELECT kind AS name, COUNT(*) AS count FROM movies
                GROUP BY kind ORDER BY count DESC
                """
            ).fetchall()

            agg = conn.execute(
                """
                SELECT COUNT(*)                        AS total,
                       ROUND(AVG(rating_imdb), 2)       AS avg_rating,
                       MIN(year)                        AS first_year,
                       MAX(year)                        AS last_year,
                       SUM(IFNULL(revenue_usd, 0))      AS total_revenue,
                       SUM(IFNULL(budget_usd, 0))       AS total_budget,
                       ROUND(AVG(NULLIF(runtime_min, 0)), 1) AS avg_runtime
                FROM movies
                """
            ).fetchone()

            decade_rows = conn.execute(
                """
                SELECT (year / 10) * 10 AS decade, COUNT(*) AS count,
                       ROUND(AVG(rating_imdb), 2) AS avg_rating
                FROM movies WHERE year IS NOT NULL
                GROUP BY decade ORDER BY decade
                """
            ).fetchall()

            people_count = int(conn.execute("SELECT COUNT(*) FROM people").fetchone()[0])
            by_director = self._group_by_column(conn, "director")[:10]
            by_country = self._group_by_column(conn, "country")[:10]

        genre_counter = self._genre_counter()

        return {
            "total_movies": total,
            "people_count": people_count,
            "by_kind": _rows_to_dicts(by_kind),
            "by_genre": [
                {"name": name, "count": count}
                for name, count in sorted(genre_counter.items(), key=lambda kv: (-kv[1], kv[0]))
            ],
            "by_director": by_director,
            "by_country": by_country,
            "by_decade": _rows_to_dicts(decade_rows),
            "avg_rating": agg["avg_rating"],
            "avg_runtime_min": agg["avg_runtime"],
            "first_year": agg["first_year"],
            "last_year": agg["last_year"],
            "total_revenue_usd": agg["total_revenue"],
            "total_budget_usd": agg["total_budget"],
        }

    def all_genres(self) -> list[str]:
        counter = self._genre_counter()
        return [name for name, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))]

