"""Тесты слоя данных: схема, демо-данные, поиск, аналитика, пользовательские записи."""

from __future__ import annotations

import pytest

from db import MovieDB, normalize_genres


@pytest.fixture
def db(seeded_db: MovieDB) -> MovieDB:
    return seeded_db


class TestSchemaAndSeed:
    def test_database_is_seeded(self, db: MovieDB) -> None:
        # В уроке требовалось наполнить базу 100+ тестовыми записями при первом запуске
        assert db.count_movies() >= 100

    def test_bulk_seed_adds_rows(self, db: MovieDB) -> None:
        from seed_data import build_people_and_credits, seed_movies

        before = db.count_movies()
        people, credits = build_people_and_credits()
        inserted = db.bulk_seed(seed_movies(5), people[:2], credits[:2])
        assert inserted == 5
        assert db.count_movies() == before + 5

    def test_bulk_seed_links_cast(self, tmp_path) -> None:
        db = MovieDB(tmp_path / "seed.db")
        db.init_schema()
        from seed_data import build_people_and_credits, seed_movies

        people, credits = build_people_and_credits()
        db.bulk_seed(seed_movies(10), people, credits)
        inception = db.get_movie_by_title("Начало")
        cast = {person["name"] for person in db.get_cast(inception["id"])}
        assert "Леонардо ДиКаприо" in cast

    def test_bootstrap_is_idempotent(self, tmp_path, monkeypatch) -> None:
        """Повторный запуск сервера не должен дублировать данные."""
        import os

        import server

        db_file = tmp_path / "bootstrap.db"
        monkeypatch.setenv("MOVIE_DB_PATH", str(db_file))
        monkeypatch.setenv("MOVIE_DB_SEED", "true")
        monkeypatch.setattr(server.settings, "db_path", db_file)
        monkeypatch.setattr(server, "db", MovieDB(db_file))

        first = server.bootstrap_database()
        total = server.db.count_movies()
        second = server.bootstrap_database()

        assert first > 0
        assert second == 0
        assert server.db.count_movies() == total
        del os

    def test_all_movies_have_required_fields(self, db: MovieDB) -> None:
        movies, _ = db.list_movies(limit=1000)
        for movie in movies:
            assert movie["title"], movie
            assert movie["kind"] in {"movie", "series", "anime", "documentary", "short"}
            if movie["rating_imdb"] is not None:
                assert 0 <= movie["rating_imdb"] <= 10


class TestNormalizeGenres:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Драма, Боевик", "Боевик, Драма"),
            ("боевик,драма", "Боевик, Драма"),
            ("Драма; Боевик", "Боевик, Драма"),
            ("Драма / Боевик", "Боевик, Драма"),
            ("", ""),
            (None, ""),
            (["Драма", "боевик"], "Боевик, Драма"),
        ],
    )
    def test_normalize(self, raw: object, expected: str) -> None:
        assert normalize_genres(raw) == expected  # type: ignore[arg-type]


class TestSearch:
    @pytest.mark.parametrize("query", ["нач", "НАЧ", "Начало", "начало", " Inception "])
    def test_find_movie_is_case_insensitive(self, db: MovieDB, query: str) -> None:
        results = db.find_movies_by_title(query)
        assert results
        assert results[0]["title"] == "Начало"

    def test_find_by_original_title(self, db: MovieDB) -> None:
        results = db.find_movies_by_title("inception")
        assert results[0]["title"] == "Начало"

    def test_exact_match_ranks_first(self, db: MovieDB) -> None:
        results = db.find_movies_by_title("Дюна")
        assert results[0]["title"] == "Дюна"

    def test_unknown_query_returns_empty(self, db: MovieDB) -> None:
        assert db.find_movies_by_title("фильма-которого-нет") == []

    def test_get_movie_by_id(self, db: MovieDB) -> None:
        movie = db.get_movie(1)
        assert movie is not None
        assert movie["id"] == 1

    def test_get_movie_missing(self, db: MovieDB) -> None:
        assert db.get_movie(10**9) is None


class TestFilters:
    def test_filter_by_kind(self, db: MovieDB) -> None:
        anime, total = db.list_movies(kind="anime", limit=100)
        assert total > 0
        assert all(movie["kind"] == "anime" for movie in anime)

    def test_filter_by_genre_and_years(self, db: MovieDB) -> None:
        movies, total = db.list_movies(genre="Драма", year_from=2015, year_to=2024, limit=100)
        assert total > 0
        for movie in movies:
            assert 2015 <= movie["year"] <= 2024
            assert "драма" in movie["genres"].casefold()

    def test_filter_by_min_rating(self, db: MovieDB) -> None:
        movies, _ = db.list_movies(min_rating=9.0, limit=50)
        assert all(movie["rating_imdb"] >= 9.0 for movie in movies)

    def test_sort_order(self, db: MovieDB) -> None:
        desc, _ = db.list_movies(sort="rating", order="desc", limit=20)
        asc, _ = db.list_movies(sort="rating", order="asc", limit=20)
        ratings_desc = [m["rating_imdb"] for m in desc if m["rating_imdb"] is not None]
        ratings_asc = [m["rating_imdb"] for m in asc if m["rating_imdb"] is not None]
        assert ratings_desc == sorted(ratings_desc, reverse=True)
        assert ratings_asc == sorted(ratings_asc)

    def test_unknown_sort_falls_back(self, db: MovieDB) -> None:
        movies, _ = db.list_movies(sort="; DROP TABLE movies; --", limit=5)
        assert len(movies) == 5  # таблица на месте, запрос не выполнился как SQL

    def test_pagination(self, db: MovieDB) -> None:
        first, total = db.list_movies(limit=5, offset=0)
        second, _ = db.list_movies(limit=5, offset=5)
        assert total > 10
        assert {m["id"] for m in first}.isdisjoint({m["id"] for m in second})


class TestPeople:
    def test_search_people(self, db: MovieDB) -> None:
        people = db.search_people(query="Нолан")
        assert people
        assert people[0]["name"] == "Кристофер Нолан"
        assert people[0]["role"] == "director"

    def test_filmography(self, db: MovieDB) -> None:
        person = db.get_person_by_name("Кристофер Нолан")
        assert person is not None
        works = db.get_person_filmography(person["id"])
        titles = {work["title"] for work in works}
        assert {"Начало", "Интерстеллар", "Оппенгеймер", "Тёмный рыцарь"} <= titles

    def test_actor_filmography_has_characters(self, db: MovieDB) -> None:
        person = db.get_person_by_name("Леонардо ДиКаприо")
        assert person is not None
        works = db.get_person_filmography(person["id"])
        assert any(work.get("character") for work in works)

    def test_cast_of_movie(self, db: MovieDB) -> None:
        movie = db.get_movie_by_title("Тёмный рыцарь")
        cast = db.get_cast(movie["id"])
        names = {person["name"] for person in cast}
        assert {"Кристиан Бейл", "Хоакин Феникс", "Гэри Олдман"} <= names

    def test_unknown_person(self, db: MovieDB) -> None:
        assert db.get_person_by_name("Несуществующий Человек") is None


class TestWrites:
    def test_add_update_delete(self, tmp_path) -> None:
        db = MovieDB(tmp_path / "writes.db")
        db.init_schema()

        movie_id = db.add_movie(
            {"title": "Тестовый фильм", "year": 2026, "genres": "драма, боевик", "rating_imdb": 7.5}
        )
        created = db.get_movie(movie_id)
        assert created["title"] == "Тестовый фильм"
        assert created["genres"] == "Боевик, Драма"  # жанры нормализовались

        assert db.update_movie(movie_id, {"rating_imdb": 8.1}) is True
        assert db.get_movie(movie_id)["rating_imdb"] == 8.1

        assert db.update_movie(movie_id, {"нет_такого_поля": 1}) is False

        assert db.delete_movie(movie_id) is True
        assert db.get_movie(movie_id) is None

    def test_update_missing_movie(self, tmp_path) -> None:
        db = MovieDB(tmp_path / "writes2.db")
        db.init_schema()
        assert db.update_movie(999, {"title": "x"}) is False
        assert db.delete_movie(999) is False


class TestUserData:
    def test_ratings_are_per_user(self, tmp_path) -> None:
        db = MovieDB(tmp_path / "users.db")
        db.init_schema()
        movie_id = db.add_movie({"title": "Фильм"})

        db.set_rating("user-1", movie_id, 9)
        db.set_rating("user-2", movie_id, 4)

        assert db.get_user_ratings("user-1")[0]["score"] == 9
        assert db.get_user_ratings("user-2")[0]["score"] == 4

        # повторная оценка обновляет, а не добавляет дубликат
        db.set_rating("user-1", movie_id, 7)
        ratings = db.get_user_ratings("user-1")
        assert len(ratings) == 1
        assert ratings[0]["score"] == 7

    def test_toggle_favorite(self, tmp_path) -> None:
        db = MovieDB(tmp_path / "fav.db")
        db.init_schema()
        movie_id = db.add_movie({"title": "Любимый"})

        assert db.toggle_favorite("u1", movie_id) is True
        assert db.is_favorite("u1", movie_id) is True
        assert db.get_favorites("u1")[0]["title"] == "Любимый"

        assert db.toggle_favorite("u1", movie_id) is False
        assert db.is_favorite("u1", movie_id) is False
        assert db.get_favorites("u1") == []

    def test_recommendations_need_ratings(self, tmp_path) -> None:
        db = MovieDB(tmp_path / "rec.db")
        db.init_schema()
        first = db.add_movie({"title": "Любимая драма", "genres": "Драма", "rating_imdb": 7.0})
        db.add_movie({"title": "Похожая драма", "genres": "Драма", "rating_imdb": 8.5})
        db.add_movie({"title": "Комедия", "genres": "Комедия", "rating_imdb": 9.0})

        assert db.get_recommendations("u1") == []

        db.set_rating("u1", first, 10)
        recommendations = db.get_recommendations("u1")
        titles = {movie["title"] for movie in recommendations}
        assert "Похожая драма" in titles
        assert "Любимая драма" not in titles  # уже оценённый не предлагаем
        assert "Комедия" not in titles


class TestAnalytics:
    def test_stats(self, db: MovieDB) -> None:
        stats = db.get_stats()
        assert stats["total_movies"] == db.count_movies()
        assert stats["avg_rating"] is not None
        assert stats["by_kind"]
        assert stats["by_genre"]
        assert stats["by_director"]
        assert stats["by_decade"]
        assert stats["first_year"] < stats["last_year"]

    def test_similar_movies_share_genres(self, db: MovieDB) -> None:
        base = db.get_movie_by_title("Начало")
        similar = db.get_similar_movies(base["id"], limit=5)
        assert similar
        base_genres = {g.strip() for g in base["genres"].split(",")}
        for movie in similar:
            shared = {g.strip() for g in movie["shared_genres"].split(",")}
            assert shared & base_genres

    def test_similar_movies_excludes_itself(self, db: MovieDB) -> None:
        base = db.get_movie_by_title("Дюна")
        similar = db.get_similar_movies(base["id"], limit=10)
        assert base["id"] not in {movie["id"] for movie in similar}

    def test_all_genres_sorted_by_frequency(self, db: MovieDB) -> None:
        genres = db.all_genres()
        assert "Драма" in genres
        assert len(genres) == len(set(genres))
