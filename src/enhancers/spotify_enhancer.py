#!/usr/bin/env python3
"""
🎵 Pure PostgreSQL Spotify Enhancer

НАЗНАЧЕНИЕ:
- Интеграция с Spotify API для обогащения треков
- Получение метаданных и audio features
- Сохранение в PostgreSQL (НИКАКОГО SQLite!)
- Async/await архитектура

ИСПОЛЬЗОВАНИЕ:
from src.enhancers.spotify_enhancer import SpotifyEnhancer

ЗАВИСИМОСТИ:                # Pause between batches
                if i + batch_size < len(tracks):
                    await asyncio.sleep(0.5)

            logger.info(f"Enhancement completed: {stats}")
            return statstgreSQL через database/postgres_adapter.py
- spotipy для Spotify Web API
- SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET в env

РЕЗУЛЬТАТ:
- Обогащенные треки в spotify_data таблице
- Логи в logs/spotify_enhancement.log

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# Настройка путей
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Загрузка переменных окружения
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("WARNING: python-dotenv не установлен, используем системные переменные")

from src.database.postgres_adapter import PostgreSQLManager

# Простое логирование
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/spotify_enhancement.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("spotify_enhancer")

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False
    logger.warning("Spotipy не установлен: pip install spotipy")


class SpotifyEnhancer:
    """
    Pure PostgreSQL Spotify Enhancer

    🔥 ОСОБЕННОСТИ:
    - 100% PostgreSQL, никакого SQLite
    - Async/await операции
    - Spotify Web API интеграция
    - Rate limiting и кэширование
    """

    def __init__(self):
        # PostgreSQL менеджер
        self.db_manager = PostgreSQLManager()

        # Spotify API
        self.spotify = None
        if SPOTIPY_AVAILABLE:
            self.spotify = self._setup_spotify_client()

        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        self.max_requests_per_minute = 100

        # Кэширование поиска
        self.search_cache = {}

        logger.info("Pure PostgreSQL Spotify Enhancer готов к работе")

    def _setup_spotify_client(self):
        """Настройка Spotify API клиента"""
        try:
            client_id = os.getenv("SPOTIFY_CLIENT_ID")
            client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

            if not client_id or not client_secret:
                logger.warning("Spotify credentials не найдены в окружении")
                return None

            # Устанавливаем переменные окружения для управления кешем
            cache_dir = os.path.join(os.getcwd(), "data")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, ".cache")

            # Устанавливаем переменные окружения spotipy
            os.environ["SPOTIPY_CACHE_PATH"] = cache_path
            os.environ["SPOTIPY_CACHE_USERNAME"] = "spotify_enhancer"

            logger.info(f"🗂️ Настройка кеша: SPOTIPY_CACHE_PATH = {cache_path}")

            # Проверяем состояние до создания клиента
            root_cache = os.path.join(os.getcwd(), ".cache")
            logger.info(
                f"📍 Перед созданием клиента: .cache в корне {'ЕСТЬ' if os.path.exists(root_cache) else 'НЕТ'}"
            )

            # Удаляем существующий кеш в корне
            if os.path.exists(root_cache):
                logger.info("🗑️ Удаляем существующий .cache в корне")
                os.remove(root_cache)

            # Создаем spotify клиент
            logger.info("🔧 Создаем SpotifyClientCredentials...")
            auth_manager = SpotifyClientCredentials(
                client_id=client_id, client_secret=client_secret
            )

            logger.info("🎵 Создаем Spotify объект...")
            spotify = spotipy.Spotify(auth_manager=auth_manager)

            # Проверяем состояние после создания
            logger.info(
                f"📍 После создания клиента: .cache в корне {'ЕСТЬ' if os.path.exists(root_cache) else 'НЕТ'}"
            )
            logger.info(
                f"📍 После создания клиента: .cache в data/ {'ЕСТЬ' if os.path.exists(cache_path) else 'НЕТ'}"
            )

            # Тестовый API вызов
            logger.info("🧪 Делаем тестовый API вызов...")
            spotify.search(q="test", type="track", limit=1)

            # Проверяем состояние после API вызова
            logger.info(
                f"📍 После API вызова: .cache в корне {'ЕСТЬ' if os.path.exists(root_cache) else 'НЕТ'}"
            )
            logger.info(
                f"📍 После API вызова: .cache в data/ {'ЕСТЬ' if os.path.exists(cache_path) else 'НЕТ'}"
            )

            # Перемещаем кеш если он создался в корне
            if os.path.exists(root_cache):
                logger.warning(
                    f"⚠️ .cache создан в корне после API вызова! Перемещаем в {cache_path}"
                )
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                os.rename(root_cache, cache_path)
                logger.info("✅ Кеш перемещен в data/")

            # Создаем пустой файл-заглушку чтобы блокировать создание в корне
            try:
                with open(root_cache, "w") as f:
                    f.write(
                        "# Этот файл блокирует создание кеша в корне\n# Реальный кеш находится в data/.cache\n"
                    )
                os.chmod(root_cache, 0o444)  # Только для чтения
                logger.info("🔒 Создан файл-блокировщик .cache в корне")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось создать блокировщик: {e}")

            # Тест подключения
            spotify.search(q="test", type="track", limit=1)
            logger.info("Spotify API connected")

            return spotify

        except Exception as e:
            logger.error(f"Ошибка Spotify API: {e}")
            return None

    async def initialize(self):
        """Инициализация PostgreSQL соединения"""
        try:
            await self.db_manager.initialize()
            logger.info("PostgreSQL инициализирован")
        except Exception as e:
            logger.error(f"Ошибка PostgreSQL: {e}")
            raise

    def _apply_rate_limiting(self):
        """Rate limiting для Spotify API"""
        current_time = time.time()

        if self.request_count >= self.max_requests_per_minute:
            if current_time - self.last_request_time < 60:
                sleep_time = 60 - (current_time - self.last_request_time)
                logger.info(f"⏰ Rate limit: ждем {sleep_time:.1f}с")
                time.sleep(sleep_time)
                self.request_count = 0

        if current_time - self.last_request_time >= 60:
            self.request_count = 0
            self.last_request_time = current_time

        self.request_count += 1

    async def search_track(self, artist: str, title: str) -> dict[str, Any] | None:
        """Поиск трека в Spotify"""
        if not self.spotify:
            return None

        try:
            # Проверяем кэш
            cache_key = f"{artist.lower()}||{title.lower()}"
            if cache_key in self.search_cache:
                return self.search_cache[cache_key]

            # Rate limiting
            self._apply_rate_limiting()

            # Поиск
            query = f'artist:"{artist}" track:"{title}"'
            results = self.spotify.search(q=query, type="track", limit=5)

            if not results["tracks"]["items"]:
                # Упрощенный поиск
                results = self.spotify.search(
                    q=f"{artist} {title}", type="track", limit=5
                )

            if results["tracks"]["items"]:
                best_match = self._find_best_match(
                    results["tracks"]["items"], artist, title
                )
                self.search_cache[cache_key] = best_match
                return best_match

            return None

        except Exception as e:
            logger.error(f"Ошибка поиска {artist} - {title}: {e}")
            return None

    def _find_best_match(
        self, tracks: list[dict], target_artist: str, target_title: str
    ) -> dict[str, Any]:
        """Находит лучший матч из результатов поиска"""

        def similarity(s1: str, s2: str) -> float:
            s1, s2 = s1.lower().strip(), s2.lower().strip()
            if s1 == s2:
                return 1.0
            if s1 in s2 or s2 in s1:
                return 0.8

            words1, words2 = set(s1.split()), set(s2.split())
            common = len(words1.intersection(words2))
            total = len(words1.union(words2))
            return common / total if total > 0 else 0.0

        best_score = 0.0
        best_track = tracks[0]

        for track in tracks:
            artist_name = track["artists"][0]["name"] if track["artists"] else ""
            track_name = track["name"]

            artist_sim = similarity(target_artist, artist_name)
            title_sim = similarity(target_title, track_name)

            score = (artist_sim * 0.4) + (title_sim * 0.6)

            if score > best_score:
                best_score = score
                best_track = track

        return best_track

    async def enhance_song(self, song_id: int, artist: str, title: str) -> bool:
        """Обогащение одной песни"""
        try:
            # Проверяем, есть ли уже данные
            if await self._has_spotify_data(song_id):
                return True

            # Поиск трека
            track_data = await self.search_track(artist, title)
            if not track_data:
                return False

            # Сохранение в PostgreSQL (без audio_features)
            return await self._save_spotify_data(song_id, track_data)

        except Exception as e:
            logger.error(f"Ошибка обогащения песни {song_id}: {e}")
            return False

    async def _has_spotify_data(self, song_id: int) -> bool:
        """Проверка наличия Spotify данных"""
        try:
            async with self.db_manager.get_connection() as conn:
                query = "SELECT 1 FROM tracks WHERE id = $1 AND spotify_data IS NOT NULL LIMIT 1"
                result = await conn.fetch(query, song_id)
                return len(result) > 0
        except Exception:
            return False

    async def _save_spotify_data(self, song_id: int, track_data: dict) -> bool:
        """Сохранение в PostgreSQL tracks.spotify_data колонку (без audio_features)"""
        try:
            # Формируем полные данные Spotify
            spotify_data = {
                "track_id": track_data["id"],
                "album_name": track_data.get("album", {}).get("name"),
                "album_id": track_data.get("album", {}).get("id"),
                "release_date": track_data.get("album", {}).get("release_date"),
                "popularity": track_data.get("popularity", 0),
                "preview_url": track_data.get("preview_url"),
                "external_urls": track_data.get("external_urls", {}),
                "artists": track_data.get("artists", []),
            }

            query = """
                UPDATE tracks 
                SET spotify_data = $2
                WHERE id = $1
            """

            async with self.db_manager.get_connection() as conn:
                await conn.execute(query, song_id, json.dumps(spotify_data))

            return True

        except Exception as e:
            logger.error(f"Ошибка сохранения в PostgreSQL: {e}")
            return False

    async def get_tracks_for_enhancement(
        self, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Получение песен для обогащения"""
        try:
            query = """
                SELECT id, artist, title
                FROM tracks 
                WHERE spotify_data IS NULL
                ORDER BY id
                LIMIT $1
            """

            async with self.db_manager.get_connection() as conn:
                result = await conn.fetch(query, limit)
                return [dict(row) for row in result] if result else []
        except Exception as e:
            logger.error(f"Ошибка получения песен: {e}")
            return []

    async def bulk_enhance(
        self, batch_size: int = 50, max_tracks: int | None = None
    ) -> dict[str, int]:
        """Массовое обогащение песен"""
        logger.info("Начинаем массовое Spotify обогащение")

        stats = {"processed": 0, "enhanced": 0, "failed": 0, "skipped": 0}

        if not self.spotify:
            logger.error("Spotify API недоступен")
            return stats

        try:
            tracks = await self.get_tracks_for_enhancement(max_tracks or 10000)

            if not tracks:
                logger.info("Все песни уже обогащены")
                return stats

            logger.info(f"Found {len(tracks)} songs for enhancement")

            for i in range(0, len(tracks), batch_size):
                batch = tracks[i : i + batch_size]

                logger.info(
                    f"Batch {i // batch_size + 1}/{(len(tracks) - 1) // batch_size + 1}"
                )

                for track in batch:
                    success = await self.enhance_song(
                        track["id"], track["artist"], track["title"]
                    )

                    stats["processed"] += 1
                    if success:
                        stats["enhanced"] += 1
                    else:
                        stats["failed"] += 1

                    if stats["processed"] % 25 == 0:
                        logger.info(f"Progress: {stats}")

                # Пауза между батчами
                if i + batch_size < len(tracks):
                    await asyncio.sleep(1)

            logger.info(f"Обогащение завершено: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Ошибка массового обогащения: {e}")
            return stats

    async def get_enhancement_stats(self) -> dict[str, Any]:
        """Статистика обогащения"""
        try:
            query = """
                SELECT 
                    (SELECT COUNT(*) FROM tracks) as total_songs,
                    (SELECT COUNT(*) FROM tracks WHERE spotify_data IS NOT NULL) as enhanced_songs,
                    (SELECT AVG(CAST(spotify_data->>'popularity' AS NUMERIC)) 
                     FROM tracks 
                     WHERE spotify_data IS NOT NULL AND spotify_data->>'popularity' != '') as avg_popularity
            """

            async with self.db_manager.get_connection() as conn:
                result = await conn.fetch(query)

                if result:
                    stats = dict(result[0])
                    if stats["total_songs"] > 0:
                        stats["enhancement_percentage"] = round(
                            stats["enhanced_songs"] / stats["total_songs"] * 100, 2
                        )
                    return stats

            return {}

        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {}

    async def close(self):
        """Закрытие соединений"""
        try:
            if self.db_manager:
                await self.db_manager.close()
            logger.info("Spotify Enhancer closed")
        except Exception as e:
            logger.error(f"Ошибка закрытия: {e}")


def show_menu():
    """Показать интерактивное меню"""
    print("\n" + "=" * 50)
    print("🎵 SPOTIFY ENHANCER - ГЛАВНОЕ МЕНЮ")
    print("=" * 50)
    print("1. 📊 Показать статистику обогащения")
    print("2. 🚀 Запустить массовое обогащение")
    print("3. 🧪 Тестовое обогащение (100 треков)")
    print("4. 🔧 Настройки батча")
    print("5. ❌ Выход")
    print("=" * 50)
    return input("Выберите опцию (1-5): ").strip()


async def interactive_mode():
    """Интерактивный режим с меню"""
    enhancer = SpotifyEnhancer()

    try:
        await enhancer.initialize()

        while True:
            choice = show_menu()

            if choice == "1":
                print("\n📊 Загрузка статистики...")
                stats = await enhancer.get_enhancement_stats()
                print("\n" + "=" * 40)
                print("СТАТИСТИКА SPOTIFY ОБОГАЩЕНИЯ:")
                print("=" * 40)
                for key, value in stats.items():
                    if key == "enhancement_percentage":
                        print(f"  🎯 {key}: {value}%")
                    elif key == "total_songs":
                        print(f"  📀 {key}: {value:,}")
                    elif key == "enhanced_songs":
                        print(f"  ✅ {key}: {value:,}")
                    elif key == "avg_popularity":
                        print(f"  📈 {key}: {value:.2f}")
                    else:
                        print(f"  📋 {key}: {value}")

                remaining = stats.get("total_songs", 0) - stats.get("enhanced_songs", 0)
                print(f"  ⏳ Осталось треков: {remaining:,}")
                print("=" * 40)

            elif choice == "2":
                print("\n🚀 МАССОВОЕ ОБОГАЩЕНИЕ")
                print("⚠️  Это займет много времени!")
                confirm = input("Продолжить? (y/N): ").strip().lower()
                if confirm == "y":
                    batch_size = int(input("Размер батча (по умолчанию 50): ") or "50")
                    max_tracks = int(input("Максимум треков (0 = все): ") or "0")
                    max_tracks = max_tracks if max_tracks > 0 else None

                    print(
                        f"🔄 Запуск обогащения: батч={batch_size}, лимит={max_tracks or 'все'}"
                    )
                    stats = await enhancer.bulk_enhance(
                        batch_size=batch_size, max_tracks=max_tracks
                    )
                    print(f"✅ Результат обогащения: {stats}")
                else:
                    print("❌ Отменено")

            elif choice == "3":
                print("\n🧪 Тестовое обогащение (100 треков)")
                batch_size = int(input("Размер батча (по умолчанию 20): ") or "20")
                print("🔄 Запуск тестового обогащения...")
                stats = await enhancer.bulk_enhance(
                    batch_size=batch_size, max_tracks=100
                )
                print(f"✅ Тест завершен: {stats}")

            elif choice == "4":
                print("\n🔧 НАСТРОЙКИ БАТЧА")
                print("Рекомендуемые размеры:")
                print("  • 10-20: Безопасно, медленно")
                print("  • 30-50: Оптимально")
                print("  • 60+: Быстро, риск превышения лимитов")
                input("Нажмите Enter для возврата в меню...")

            elif choice == "5":
                print("👋 До свидания!")
                break

            else:
                print("❌ Неверный выбор! Попробуйте еще раз.")

            if choice in ["1", "2", "3"]:
                input("\nНажмите Enter для возврата в меню...")

    finally:
        await enhancer.close()


# CLI интерфейс
async def main():
    """CLI для Spotify Enhancer"""
    import argparse

    parser = argparse.ArgumentParser(description="PostgreSQL Spotify Enhancer")
    parser.add_argument("--enhance", action="store_true", help="Запустить обогащение")
    parser.add_argument("--stats", action="store_true", help="Показать статистику")
    parser.add_argument("--limit", type=int, default=1000, help="Лимит песен")
    parser.add_argument("--batch-size", type=int, default=50, help="Размер батча")
    parser.add_argument(
        "--interactive", action="store_true", help="Интерактивный режим"
    )

    args = parser.parse_args()

    # Если нет аргументов, запускаем интерактивный режим
    if not any([args.enhance, args.stats, args.interactive]):
        await interactive_mode()
        return

    # Интерактивный режим
    if args.interactive:
        await interactive_mode()
        return

    # Режим командной строки
    enhancer = SpotifyEnhancer()

    try:
        await enhancer.initialize()

        if args.stats:
            stats = await enhancer.get_enhancement_stats()
            print("Статистика Spotify обогащения:")
            for key, value in stats.items():
                print(f"  {key}: {value}")

        if args.enhance:
            stats = await enhancer.bulk_enhance(
                batch_size=args.batch_size, max_tracks=args.limit
            )
            print(f"Результат обогащения: {stats}")

    finally:
        await enhancer.close()


if __name__ == "__main__":
    asyncio.run(main())
