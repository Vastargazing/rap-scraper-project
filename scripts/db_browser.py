"""
#!/usr/bin/env python3
🔍 Database Browser — просмотр данных PostgreSQL

НАЗНАЧЕНИЕ:
- Быстрый просмотр треков и анализов из базы данных без GUI

ИСПОЛЬЗОВАНИЕ:
python scripts/db_browser.py

ЗАВИСИМОСТИ:
- Python 3.8+
- src/database/postgres_adapter.py
- PostgreSQL база данных (rap_lyrics)

РЕЗУЛЬТАТ:
- Удобный текстовый интерфейс для базы данных

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""

import argparse
import asyncio
import json
import sys

sys.path.append(".")
from src.database.postgres_adapter import PostgreSQLManager


class DatabaseBrowser:
    """Браузер базы данных PostgreSQL"""

    def __init__(self):
        self.db = PostgreSQLManager()

    async def initialize(self):
        """Инициализация подключения"""
        await self.db.initialize()

    async def close(self):
        """Закрытие подключения"""
        await self.db.close()

    async def show_stats(self):
        """Показать общую статистику"""
        async with self.db.get_connection() as conn:
            # Общая статистика
            tracks_count = await conn.fetchval("SELECT COUNT(*) FROM tracks")
            analysis_count = await conn.fetchval(
                "SELECT COUNT(*) FROM analysis_results"
            )

            # Статистика по артистам
            artist_stats = await conn.fetch("""
                SELECT artist, COUNT(*) as track_count 
                FROM tracks 
                GROUP BY artist 
                ORDER BY track_count DESC 
                LIMIT 10
            """)

            # Статистика по анализаторам
            analyzer_stats = await conn.fetch("""
                SELECT analyzer_type, COUNT(*) as analysis_count 
                FROM analysis_results 
                GROUP BY analyzer_type 
                ORDER BY analysis_count DESC
            """)

            print("📊 СТАТИСТИКА БАЗЫ ДАННЫХ")
            print("=" * 50)
            print(f"🎵 Треков: {tracks_count:,}")
            print(f"🧠 Анализов: {analysis_count:,}")

            print("\n🎤 ТОП-10 АРТИСТОВ:")
            for artist in artist_stats:
                print(f"  {artist['artist']}: {artist['track_count']} треков")

            print("\n🤖 АНАЛИЗАТОРЫ:")
            for analyzer in analyzer_stats:
                print(
                    f"  {analyzer['analyzer_type']}: {analyzer['analysis_count']} анализов"
                )

    async def search_tracks(self, query: str, limit: int = 10):
        """Поиск треков"""
        async with self.db.get_connection() as conn:
            tracks = await conn.fetch(
                """
                SELECT id, title, artist, album, release_date, LENGTH(lyrics) as lyrics_length
                FROM tracks 
                WHERE title ILIKE $1 OR artist ILIKE $1 OR lyrics ILIKE $1
                ORDER BY 
                    CASE 
                        WHEN title ILIKE $1 THEN 1
                        WHEN artist ILIKE $1 THEN 2
                        ELSE 3
                    END
                LIMIT $2
            """,
                f"%{query}%",
                limit,
            )

            print(
                f"\n🔍 РЕЗУЛЬТАТЫ ПОИСКА '{query}' (показано {len(tracks)} из макс. {limit}):"
            )
            print("=" * 80)

            for track in tracks:
                print(f"🎵 ID: {track['id']}")
                print(f"   Название: {track['title']}")
                print(f"   Артист: {track['artist']}")
                print(f"   Альбом: {track['album'] or 'Неизвестно'}")
                print(f"   Дата выпуска: {track['release_date'] or 'Неизвестно'}")
                print(f"   Длина текста: {track['lyrics_length']} символов")
                print("-" * 40)

    async def show_track(self, track_id: int):
        """Показать полную информацию о треке"""
        async with self.db.get_connection() as conn:
            # Информация о треке
            track = await conn.fetchrow("SELECT * FROM tracks WHERE id = $1", track_id)
            if not track:
                print(f"❌ Трек с ID {track_id} не найден")
                return

            # Анализы трека
            analyses = await conn.fetch(
                """
                SELECT * FROM analysis_results 
                WHERE track_id = $1 
                ORDER BY created_at DESC
            """,
                track_id,
            )

            print(f"\n🎵 ТРЕК ID: {track_id}")
            print("=" * 60)
            print(f"Название: {track['title']}")
            print(f"Артист: {track['artist']}")
            print(f"Альбом: {track['album'] or 'Неизвестно'}")
            print(f"Дата выпуска: {track['release_date'] or 'Неизвестно'}")
            print(f"Жанр: {track['genre'] or 'Неизвестно'}")
            print(f"Популярность: {track['popularity_score'] or 'Неизвестно'}")

            print("\n📝 ТЕКСТ ПЕСНИ:")
            print("-" * 40)
            print(track["lyrics"][:500] + ("..." if len(track["lyrics"]) > 500 else ""))

            print(f"\n🧠 АНАЛИЗЫ ({len(analyses)} шт.):")
            print("-" * 40)
            for analysis in analyses:
                print(f"🤖 Анализатор: {analysis['analyzer_type']}")
                print(f"   Настроение: {analysis['sentiment'] or 'Не определено'}")
                print(f"   Уверенность: {analysis['confidence'] or 'Не определена'}")
                print(
                    f"   Сложность: {analysis['complexity_score'] or 'Не определена'}"
                )
                print(f"   Создан: {analysis['created_at']}")

                if analysis["analysis_data"]:
                    try:
                        # analysis_data может быть строкой JSON или уже объектом
                        if isinstance(analysis["analysis_data"], str):
                            data = json.loads(analysis["analysis_data"])
                        else:
                            data = analysis["analysis_data"]

                        print(f"   Жанр: {data.get('genre', 'Не определен')}")
                        print(
                            f"   Энергия: {data.get('energy_level', 'Не определена')}"
                        )
                        print(
                            f"   Качество: {data.get('overall_quality', 'Не определено')}"
                        )
                    except (json.JSONDecodeError, AttributeError):
                        print("   Дополнительные данные: ошибка парсинга")
                print("-" * 20)

    async def list_tracks(self, artist: str | None = None, limit: int = 20):
        """Список треков"""
        async with self.db.get_connection() as conn:
            if artist:
                tracks = await conn.fetch(
                    """
                    SELECT id, title, artist, album, release_date 
                    FROM tracks 
                    WHERE artist ILIKE $1
                    ORDER BY title
                    LIMIT $2
                """,
                    f"%{artist}%",
                    limit,
                )
                print(
                    f"\n🎤 ТРЕКИ АРТИСТА '{artist}' (показано {len(tracks)} из макс. {limit}):"
                )
            else:
                tracks = await conn.fetch(
                    """
                    SELECT id, title, artist, album, release_date 
                    FROM tracks 
                    ORDER BY id DESC
                    LIMIT $1
                """,
                    limit,
                )
                print(
                    f"\n🎵 ПОСЛЕДНИЕ ТРЕКИ (показано {len(tracks)} из макс. {limit}):"
                )

            print("=" * 80)
            for track in tracks:
                release_year = (
                    track["release_date"].year if track["release_date"] else "????"
                )
                print(
                    f"ID: {track['id']:6} | {track['artist'][:20]:20} | {track['title'][:40]:40} | {release_year}"
                )

    async def recent_analyses(self, limit: int = 10):
        """Последние анализы"""
        async with self.db.get_connection() as conn:
            analyses = await conn.fetch(
                """
                SELECT ar.*, t.title, t.artist 
                FROM analysis_results ar
                JOIN tracks t ON ar.track_id = t.id
                ORDER BY ar.created_at DESC
                LIMIT $1
            """,
                limit,
            )

            print(
                f"\n🧠 ПОСЛЕДНИЕ АНАЛИЗЫ (показано {len(analyses)} из макс. {limit}):"
            )
            print("=" * 90)

            for analysis in analyses:
                print(
                    f"🤖 {analysis['analyzer_type']:15} | {analysis['artist'][:20]:20} - {analysis['title'][:30]:30}"
                )
                print(
                    f"   Настроение: {analysis['sentiment'] or 'Не определено':15} | Создан: {analysis['created_at']}"
                )
                print("-" * 80)


async def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Database Browser для PostgreSQL")
    parser.add_argument(
        "command",
        choices=["stats", "search", "track", "list", "recent"],
        help="Команда для выполнения",
    )
    parser.add_argument("--query", "-q", help="Поисковый запрос")
    parser.add_argument("--id", "-i", type=int, help="ID трека")
    parser.add_argument("--artist", "-a", help="Имя артиста")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Лимит результатов")

    args = parser.parse_args()

    browser = DatabaseBrowser()

    try:
        await browser.initialize()

        if args.command == "stats":
            await browser.show_stats()

        elif args.command == "search":
            if not args.query:
                print("❌ Необходим параметр --query для поиска")
                return
            await browser.search_tracks(args.query, args.limit)

        elif args.command == "track":
            if not args.id:
                print("❌ Необходим параметр --id для просмотра трека")
                return
            await browser.show_track(args.id)

        elif args.command == "list":
            await browser.list_tracks(args.artist, args.limit)

        elif args.command == "recent":
            await browser.recent_analyses(args.limit)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        await browser.close()


if __name__ == "__main__":
    # Примеры использования
    if len(sys.argv) == 1:
        print("🔍 DATABASE BROWSER - Просмотр PostgreSQL базы данных")
        print("=" * 60)
        print("Примеры использования:")
        print("")
        print("📊 Общая статистика:")
        print("  python scripts/db_browser.py stats")
        print("")
        print("🔍 Поиск треков:")
        print("  python scripts/db_browser.py search --query 'eminem'")
        print("  python scripts/db_browser.py search -q 'love' --limit 5")
        print("")
        print("🎵 Просмотр трека:")
        print("  python scripts/db_browser.py track --id 12561")
        print("")
        print("📋 Список треков:")
        print("  python scripts/db_browser.py list --limit 20")
        print("  python scripts/db_browser.py list --artist 'drake' --limit 10")
        print("")
        print("🧠 Последние анализы:")
        print("  python scripts/db_browser.py recent --limit 15")
        print("")
        print("Запустите команду с параметрами для начала работы!")
    else:
        asyncio.run(main())
