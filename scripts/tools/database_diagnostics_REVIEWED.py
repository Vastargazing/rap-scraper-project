#!/usr/bin/env python3
# TODO(CODE_REVIEW): Add proper copyright and license header (Apache 2.0 is Google standard)
# TODO(CODE_REVIEW): Add module-level docstring in English (Google Style Guide)
# TODO(CODE_REVIEW): Remove emojis from code - not professional for production
"""
🔍 Unified Database Diagnostics Tool (PostgreSQL)
Объединенный инструмент диагностики PostgreSQL базы данных

НАЗНАЧЕНИЕ:
- Общая диагностика базы данных (структура, размер, статистика)
- Проверка схемы таблиц и их структуры
- Анализ статуса AI-анализа и покрытия
- Поиск неанализированных записей
- Рекомендации для оптимизации

ИСПОЛЬЗОВАНИЕ:
python scripts/tools/database_diagnostics_postgres.py                # Полная диагностика
python scripts/tools/database_diagnostics_postgres.py --schema       # Только схема
python scripts/tools/database_diagnostics_postgres.py --analysis     # Только AI анализ
python scripts/tools/database_diagnostics_postgres.py --unanalyzed   # Неанализированные записи
python scripts/tools/database_diagnostics_postgres.py --quick        # Быстрая проверка

ЗАВИСИМОСТИ:
- PostgreSQL database
- Таблицы: tracks, artists, spotify_tracks, spotify_artists, spotify_audio_features
- Python библиотеки (psycopg2, argparse, pathlib)

РЕЗУЛЬТАТ:
- Консольный вывод с диагностической информацией
- Рекомендации для продолжения обработки данных
- Статистика покрытия AI анализа и Spotify обогащения

АВТОР: AI Assistant (обновлено для PostgreSQL)
ДАТА: Сентябрь 2025
"""

# TODO(CODE_REVIEW): Add type hints for all imports (from __future__ import annotations)
# TODO(CODE_REVIEW): Sort imports according to PEP 8: stdlib, third-party, local
# TODO(CODE_REVIEW): Add missing imports: logging, typing, dataclasses, time
import argparse
import os
import sys
from pathlib import Path
# TODO(CODE_REVIEW): Add these imports:
# from typing import Optional, Dict, Any, List, Tuple
# import logging
# import time
# from dataclasses import dataclass

import psycopg2
import psycopg2.extras

# TODO(CODE_REVIEW): Define module-level constants at the top after imports
# TODO(CODE_REVIEW): Add constants for table names to avoid magic strings:
#   TRACKS_TABLE = "tracks"
#   ANALYSIS_RESULTS_TABLE = "analysis_results"
#   AI_ANALYSIS_TABLE = "ai_analysis"
#   SPOTIFY_TRACKS_TABLE = "spotify_tracks"
#   SPOTIFY_ARTISTS_TABLE = "spotify_artists"

# TODO(CODE_REVIEW): Add constants for retry configuration:
#   MAX_RETRY_ATTEMPTS = 3
#   INITIAL_RETRY_DELAY_SECONDS = 1
#   MAX_RETRY_DELAY_SECONDS = 10
#   CONNECTION_TIMEOUT_SECONDS = 30

# TODO(CODE_REVIEW): Add constants for default limits:
#   DEFAULT_UNANALYZED_LIMIT = 10
#   DEFAULT_TOP_ARTISTS_LIMIT = 10
#   DEFAULT_RECENT_TRACKS_LIMIT = 5

# TODO(CODE_REVIEW): Configure logging at module level:
#   logging.basicConfig(
#       level=logging.INFO,
#       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#   )
#   logger = logging.getLogger(__name__)

# Добавляем корневую папку в path для доступа к src модулям
# TODO(CODE_REVIEW): Avoid modifying sys.path at runtime - use proper package structure
# TODO(CODE_REVIEW): Consider using setuptools/poetry for proper package installation
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


# TODO(CODE_REVIEW): Create @dataclass for database configuration
# @dataclass
# class DbConfig:
#     host: str
#     port: int
#     database: str
#     user: str
#     password: str
#     connect_timeout: int = 30


# TODO(CODE_REVIEW): Add comprehensive class docstring in Google style with Args, Attributes, Examples
# TODO(CODE_REVIEW): Consider using @dataclass for configuration storage
# TODO(CODE_REVIEW): Implement context manager protocol (__enter__, __exit__) for automatic connection cleanup
# TODO(CODE_REVIEW): Consider splitting this class - it has too many responsibilities (SRP violation)
#   - ConnectionManager for DB connection logic
#   - SchemaAnalyzer for schema operations
#   - AnalysisReporter for analysis status
#   - DiagnosticsAggregator to coordinate them all
class PostgreSQLDiagnostics:
    """Класс для диагностики PostgreSQL базы данных"""

    # TODO(CODE_REVIEW): Add type hints to __init__ -> None
    # TODO(CODE_REVIEW): Accept db_config as parameter instead of loading it in __init__ (dependency injection)
    # TODO(CODE_REVIEW): Add logger as a parameter with default value
    # TODO(CODE_REVIEW): Add docstring with Args section
    def __init__(self):
        # TODO(CODE_REVIEW): Add type hints for instance variables:
        #   self.project_root: Path
        #   self.conn: Optional[psycopg2.extensions.connection]
        #   self.db_config: Dict[str, Any]
        #   self.logger: logging.Logger
        self.project_root = project_root
        self.conn = None

        # Загружаем конфигурацию БД
        # TODO(CODE_REVIEW): Extract config loading to separate method _load_db_config()
        # TODO(CODE_REVIEW): Use specific exceptions instead of bare except
        # TODO(CODE_REVIEW): Add logging for which config source was used
        # TODO(CODE_REVIEW): Create a DbConfig dataclass/namedtuple instead of dict
        try:
            from config.config_loader import get_config
            config_obj = get_config()
            db_config = config_obj.database
            self.db_config = {
                "host": db_config.host,
                "port": db_config.port,
                "database": db_config.database,
                "user": db_config.username,
                "password": db_config.password,
            }
        except (ImportError, AttributeError):
            # Fallback конфигурация для Windows PostgreSQL
            try:
                from src.utils.config import get_db_config
                self.db_config = get_db_config()
            except ImportError:
                # TODO(CODE_REVIEW): Extract default values to module-level constants
                # TODO(CODE_REVIEW): Add validation for port number (handle ValueError from int())
                # TODO(CODE_REVIEW): Consider using pydantic for config validation
                self.db_config = {
                    "host": os.getenv("DB_HOST", "localhost"),
                    "port": int(os.getenv("DB_PORT", "5432")),
                    "database": os.getenv("DB_NAME", "postgres"),
                    "user": os.getenv("DB_USER", "postgres"),
                    "password": os.getenv("DB_PASSWORD", ""),
                }

    # TODO(CODE_REVIEW): Add type hint -> bool
    # TODO(CODE_REVIEW): Add docstring in Google style with Args, Returns, Raises
    # TODO(CODE_REVIEW): Add connection timeout parameter
    # TODO(CODE_REVIEW): Implement exponential backoff retry logic with tenacity library
    # TODO(CODE_REVIEW): Return connection object instead of bool, raise exception on failure
    # TODO(CODE_REVIEW): This method is too long (>100 lines) - violates SRP
    def connect(self):
        """Подключение к PostgreSQL базе данных"""
        # TODO(CODE_REVIEW): Use specific exception types (psycopg2.OperationalError, psycopg2.DatabaseError)
        # TODO(CODE_REVIEW): Replace all print with logging.info/error/warning
        # TODO(CODE_REVIEW): Add connection pooling for production use (psycopg2.pool)
        try:
            # TODO(CODE_REVIEW): Add connect_timeout to db_config
            self.conn = psycopg2.connect(**self.db_config)
            # Включаем автокоммит для диагностических запросов
            # TODO(CODE_REVIEW): Document why autocommit is needed
            self.conn.autocommit = True
            print("✅ Подключение к PostgreSQL успешно!")
            return True
        except Exception as e:
            # TODO(CODE_REVIEW): Use logging.error() with exc_info=True for stack trace
            print(f"❌ Ошибка подключения к PostgreSQL: {e}")
            # TODO(CODE_REVIEW): CRITICAL SECURITY ISSUE - Never print passwords in logs!
            # TODO(CODE_REVIEW): Sanitize db_config before printing (mask password with ***)
            print(f"🔧 Текущие настройки: {self.db_config}")

            # Предлагаем альтернативные варианты подключения
            # TODO(CODE_REVIEW): Extract retry logic to separate method _try_alternative_connections()
            # TODO(CODE_REVIEW): This retry logic is too complex - refactor into smaller methods
            # TODO(CODE_REVIEW): Use a list of alternative configs and iterate through them
            print("\n🔧💡 Попробуйте альтернативные варианты:")

            # Вариант 1: пустой пароль
            # TODO(CODE_REVIEW): Extract each connection attempt to a separate method
            if self.db_config["password"]:
                print("1️⃣ Попытка подключения без пароля...")
                try:
                    alt_config = self.db_config.copy()
                    alt_config["password"] = ""
                    self.conn = psycopg2.connect(**alt_config)
                    self.conn.autocommit = True
                    self.db_config = alt_config
                    print("✅ Подключение без пароля успешно!")
                    return True
                except Exception:
                    # TODO(CODE_REVIEW): Don't use bare 'pass' - at least log the failure
                    # TODO(CODE_REVIEW): Don't silently swallow exceptions - use logging
                    pass

            # Вариант 2: база данных postgres
            if self.db_config["database"] != "postgres":
                print("2️⃣ Попытка подключения к БД 'postgres'...")
                try:
                    alt_config = self.db_config.copy()
                    alt_config["database"] = "postgres"
                    alt_config["password"] = ""
                    self.conn = psycopg2.connect(**alt_config)
                    self.conn.autocommit = True
                    self.db_config = alt_config
                    print("✅ Подключение к БД 'postgres' успешно!")
                    return True
                except Exception:
                    # TODO(CODE_REVIEW): Log the exception
                    pass

            # Вариант 3: пользователь из .env
            # TODO(CODE_REVIEW): Extract env var names to constants
            env_user = os.getenv("POSTGRES_USERNAME")
            env_password = os.getenv("POSTGRES_PASSWORD")
            env_db = os.getenv("POSTGRES_DATABASE")

            if env_user and env_password and env_db:
                print("3️⃣ Попытка подключения с настройками из .env...")
                try:
                    alt_config = {
                        "host": self.db_config["host"],
                        "port": self.db_config["port"],
                        "database": env_db,
                        "user": env_user,
                        "password": env_password,
                    }
                    self.conn = psycopg2.connect(**alt_config)
                    self.conn.autocommit = True
                    self.db_config = alt_config
                    print("✅ Подключение с настройками из .env успешно!")
                    return True
                except Exception as e2:
                    # TODO(CODE_REVIEW): Use logging.debug for detailed error info
                    print(f"   ❌ Также не удалось: {e2}")

            # TODO(CODE_REVIEW): Extract help text to constant or separate method
            print("\n🛠️  Для решения проблемы:")
            print("   1. Проверьте, что PostgreSQL запущен")
            print("   2. Убедитесь, что пользователь и пароль корректны")
            print("   3. Проверьте настройки в .env файле")
            print("   4. Возможно, нужно создать базу данных и пользователя")

            return False

    # TODO(CODE_REVIEW): Add type hint -> None
    # TODO(CODE_REVIEW): Add docstring
    # TODO(CODE_REVIEW): Consider implementing __del__ method as backup for cleanup
    def close(self):
        """Закрытие соединения"""
        # TODO(CODE_REVIEW): Check if connection is still alive before closing
        # TODO(CODE_REVIEW): Handle exceptions during close
        # TODO(CODE_REVIEW): Log successful/failed closure
        if self.conn:
            self.conn.close()

    # TODO(CODE_REVIEW): Add type hint -> None
    # TODO(CODE_REVIEW): Add comprehensive docstring with description of what stats are shown
    # TODO(CODE_REVIEW): Method is too long (>120 lines) - split into smaller methods:
    #   _get_database_size(), _get_table_list(), _get_track_statistics(),
    #   _get_top_artists(), _get_recent_tracks()
    # TODO(CODE_REVIEW): Return structured data instead of just printing
    def check_general_status(self):
        """Общая диагностика PostgreSQL базы данных"""
        print("🔍 ОБЩАЯ ДИАГНОСТИКА POSTGRESQL БАЗЫ ДАННЫХ")
        print("=" * 50)

        if not self.conn:
            print("❌ Нет подключения к базе данных")
            return

        # TODO(CODE_REVIEW): Use try-except for each SQL query separately to show partial results
        # TODO(CODE_REVIEW): Create a Statistics dataclass to store and return results
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Размер базы данных
                # TODO(CODE_REVIEW): Extract SQL queries to module-level constants or separate file
                cur.execute(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                )
                db_size = cur.fetchone()[0]
                print(f"📁 Размер БД: {db_size}")

                # Список таблиц
                # TODO(CODE_REVIEW): Add schema parameter to make it configurable
                cur.execute("""
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    ORDER BY tablename
                """)
                tables = [row[0] for row in cur.fetchall()]
                print(f"📋 Таблицы в БД ({len(tables)}): {', '.join(tables)}")

                # Основная статистика
                print("\n📊 ОСНОВНАЯ СТАТИСТИКА:")

                # Треки
                # TODO(CODE_REVIEW): Use constants for table names instead of strings
                if self._table_exists("tracks"):
                    cur.execute("SELECT COUNT(*) FROM tracks")
                    total_tracks = cur.fetchone()[0]

                    # TODO(CODE_REVIEW): Combine these queries with a single query using COUNT(CASE WHEN ...)
                    cur.execute("""
                        SELECT COUNT(*) FROM tracks
                        WHERE lyrics IS NOT NULL AND lyrics != ''
                    """)
                    tracks_with_lyrics = cur.fetchone()[0]

                    # TODO(CODE_REVIEW): Add thousands separator formatting or use humanize library
                    print(f"🎵 Всего треков: {total_tracks:,}")
                    if total_tracks > 0:
                        # TODO(CODE_REVIEW): Extract percentage calculation to utility function
                        print(
                            f"🎵 С текстами: {tracks_with_lyrics:,} ({tracks_with_lyrics / total_tracks * 100:.1f}%)"
                        )

                    # Проверяем наличие AI анализов в разных возможных таблицах
                    # TODO(CODE_REVIEW): This logic is duplicated throughout the file - extract to method
                    # TODO(CODE_REVIEW): Use elif instead of nested if for analysis tables
                    analyzed_tracks = 0
                    if self._table_exists("analysis_results"):
                        cur.execute(
                            "SELECT COUNT(DISTINCT track_id) FROM analysis_results"
                        )
                        analyzed_tracks = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM analysis_results")
                        total_analyses = cur.fetchone()[0]
                        print(
                            f"🤖 С AI анализом: {analyzed_tracks:,} треков ({analyzed_tracks / total_tracks * 100:.1f}%)"
                        )
                        print(f"🤖 Всего анализов: {total_analyses:,}")
                    elif self._table_exists("ai_analysis"):
                        cur.execute("SELECT COUNT(DISTINCT track_id) FROM ai_analysis")
                        analyzed_tracks = cur.fetchone()[0]
                        if analyzed_tracks > 0:
                            print(
                                f"🤖 С AI анализом: {analyzed_tracks:,} ({analyzed_tracks / total_tracks * 100:.1f}%)"
                            )
                    else:
                        print("🤖 Таблица AI анализов не найдена")

                # Артисты (уникальные из треков)
                # TODO(CODE_REVIEW): Simplify this logic - check artists table first
                if self._table_exists("tracks"):
                    cur.execute("SELECT COUNT(DISTINCT artist) FROM tracks")
                    unique_artists = cur.fetchone()[0]
                    print(f"🎤 Уникальных артистов: {unique_artists:,}")
                elif self._table_exists("artists"):
                    cur.execute("SELECT COUNT(*) FROM artists")
                    total_artists = cur.fetchone()[0]
                    print(f"🎤 Всего артистов: {total_artists:,}")

                # Spotify данные
                # TODO(CODE_REVIEW): Combine these checks into a single method _get_spotify_stats()
                if self._table_exists("spotify_tracks"):
                    cur.execute("SELECT COUNT(*) FROM spotify_tracks")
                    spotify_tracks = cur.fetchone()[0]
                    print(f"🎵 Spotify треков: {spotify_tracks:,}")

                if self._table_exists("spotify_artists"):
                    cur.execute("SELECT COUNT(*) FROM spotify_artists")
                    spotify_artists = cur.fetchone()[0]
                    print(f"🎤 Spotify артистов: {spotify_artists:,}")

                # Топ артистов
                # TODO(CODE_REVIEW): Add LIMIT as a parameter/constant
                if self._table_exists("tracks"):
                    print("\n🏆 ТОП-10 АРТИСТОВ ПО КОЛИЧЕСТВУ ТРЕКОВ:")
                    # TODO(CODE_REVIEW): Handle NULL artists
                    cur.execute("""
                        SELECT artist, COUNT(*) as count
                        FROM tracks
                        GROUP BY artist
                        ORDER BY count DESC
                        LIMIT 10
                    """)
                    top_artists = cur.fetchall()

                    for i, (artist, count) in enumerate(top_artists, 1):
                        # TODO(CODE_REVIEW): Add NULL check for artist
                        print(f"  {i:2d}. {artist}: {count:,} треков")

                # Последние добавленные
                # TODO(CODE_REVIEW): Add LIMIT as parameter
                if self._table_exists("tracks"):
                    print("\n📅 ПОСЛЕДНИЕ ДОБАВЛЕННЫЕ ТРЕКИ:")
                    cur.execute("""
                        SELECT title, artist, created_at
                        FROM tracks
                        WHERE created_at IS NOT NULL
                        ORDER BY created_at DESC
                        LIMIT 5
                    """)
                    recent_tracks = cur.fetchall()

                    # TODO(CODE_REVIEW): Handle case when no recent tracks found
                    for title, artist, date in recent_tracks:
                        print(f"  • {artist} - {title} ({date})")

        except Exception as e:
            # TODO(CODE_REVIEW): Use specific exceptions (psycopg2.Error)
            # TODO(CODE_REVIEW): Add logging.error with exc_info=True
            # TODO(CODE_REVIEW): Return error status instead of just printing
            print(f"❌ Ошибка при получении статистики: {e}")

    # TODO(CODE_REVIEW): Add type hint -> None
    # TODO(CODE_REVIEW): Add comprehensive docstring
    # TODO(CODE_REVIEW): Method is too long - split into smaller methods
    # TODO(CODE_REVIEW): Return structured data (list of TableSchema objects)
    def check_schema(self):
        """Проверка схемы PostgreSQL таблиц"""
        print("🏗️ ПРОВЕРКА СХЕМЫ POSTGRESQL БАЗЫ ДАННЫХ")
        print("=" * 50)

        if not self.conn:
            print("❌ Нет подключения к базе данных")
            return

        # Основные таблицы для проверки
        # TODO(CODE_REVIEW): Move to module-level constant
        important_tables = [
            "tracks",
            "analysis_results",
            "ai_analysis",
            "spotify_tracks",
            "spotify_artists",
        ]

        # TODO(CODE_REVIEW): Wrap entire method in try-except is too broad
        # TODO(CODE_REVIEW): Handle exceptions per table to show partial results
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # TODO(CODE_REVIEW): Extract table inspection to separate method _inspect_table()
                for table in important_tables:
                    if self._table_exists(table):
                        print(f"\n📋 Таблица: {table}")

                        # Получаем информацию о колонках
                        # TODO(CODE_REVIEW): This query should be in a constant
                        cur.execute(
                            """
                            SELECT
                                column_name,
                                data_type,
                                is_nullable,
                                column_default,
                                character_maximum_length
                            FROM information_schema.columns
                            WHERE table_name = %s
                            AND table_schema = 'public'
                            ORDER BY ordinal_position
                        """,
                            (table,),
                        )

                        columns = cur.fetchall()
                        print(f"  Колонок: {len(columns)}")

                        # TODO(CODE_REVIEW): Extract column display logic to separate method
                        for col in columns:
                            col_name = col["column_name"]
                            col_type = col["data_type"]
                            is_nullable = col["is_nullable"]
                            default_val = col["column_default"]
                            max_length = col["character_maximum_length"]

                            # Формируем тип с длиной
                            if max_length and col_type == "character varying":
                                col_type = f"varchar({max_length})"

                            constraints = []
                            if is_nullable == "NO":
                                constraints.append("NOT NULL")
                            if default_val:
                                constraints.append(f"DEFAULT {default_val}")

                            constraint_str = (
                                f" ({', '.join(constraints)})" if constraints else ""
                            )
                            print(f"    {col_name}: {col_type}{constraint_str}")

                        # Индексы
                        # TODO(CODE_REVIEW): Extract to _get_table_indexes() method
                        cur.execute(
                            """
                            SELECT indexname, indexdef
                            FROM pg_indexes
                            WHERE tablename = %s
                            AND schemaname = 'public'
                        """,
                            (table,),
                        )

                        indexes = cur.fetchall()
                        if indexes:
                            print(f"  Индексы: {len(indexes)}")
                            for idx in indexes:
                                print(f"    {idx['indexname']}")

                        # Количество записей
                        # TODO(CODE_REVIEW): SQL INJECTION RISK - Never use f-string in SQL!
                        # TODO(CODE_REVIEW): Use parameterized query or psycopg2.sql.Identifier
                        # CRITICAL: This is a security vulnerability even for internal tools
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cur.fetchone()[0]
                        print(f"  Записей: {count:,}")

                        # Размер таблицы
                        cur.execute(
                            """
                            SELECT pg_size_pretty(pg_total_relation_size(%s))
                        """,
                            (table,),
                        )
                        size = cur.fetchone()[0]
                        print(f"  Размер: {size}")
                    else:
                        print(f"\n❌ Таблица {table} не найдена")

        except Exception as e:
            # TODO(CODE_REVIEW): Use specific exceptions
            # TODO(CODE_REVIEW): Add logging
            print(f"❌ Ошибка при проверке схемы: {e}")

    # TODO(CODE_REVIEW): Add type hint -> None
    # TODO(CODE_REVIEW): Add comprehensive docstring
    # TODO(CODE_REVIEW): Method is too long (>180 lines) - split into smaller methods
    # TODO(CODE_REVIEW): Too much duplicated code with check_general_status()
    # TODO(CODE_REVIEW): Return structured data instead of printing
    def check_analysis_status(self):
        """Проверка статуса AI анализа в PostgreSQL"""
        print("🤖 СТАТУС AI АНАЛИЗА")
        print("=" * 50)

        if not self.conn:
            print("❌ Нет подключения к базе данных")
            return

        if not self._table_exists("tracks"):
            print("❌ Таблица tracks не найдена")
            return

        # TODO(CODE_REVIEW): Handle exceptions per section to show partial results
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Общая статистика
                cur.execute("""
                    SELECT COUNT(*) FROM tracks
                    WHERE lyrics IS NOT NULL AND lyrics != ''
                """)
                total_tracks = cur.fetchone()[0]

                # Проверяем AI анализы в отдельной таблице
                # TODO(CODE_REVIEW): This logic is duplicated - extract to _get_analysis_table_name()
                # TODO(CODE_REVIEW): Use dictionary mapping for table-specific queries
                analyzed_tracks = 0
                total_analyses = 0

                if self._table_exists("analysis_results"):
                    cur.execute("SELECT COUNT(DISTINCT track_id) FROM analysis_results")
                    analyzed_tracks = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM analysis_results")
                    total_analyses = cur.fetchone()[0]
                elif self._table_exists("ai_analysis"):
                    cur.execute("SELECT COUNT(DISTINCT track_id) FROM ai_analysis")
                    analyzed_tracks = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM ai_analysis")
                    total_analyses = cur.fetchone()[0]

                print("📊 Общая статистика:")
                print(f"  🎵 Треков с текстами: {total_tracks:,}")
                print(f"  🤖 Проанализированных треков: {analyzed_tracks:,}")
                print(f"  📊 Всего анализов: {total_analyses:,}")

                if total_tracks > 0:
                    # TODO(CODE_REVIEW): Extract percentage calculation to utility function
                    # TODO(CODE_REVIEW): Handle division by zero
                    coverage = analyzed_tracks / total_tracks * 100
                    print(f"  📈 Покрытие: {coverage:.1f}%")
                    print(f"  📋 Неанализированных: {total_tracks - analyzed_tracks:,}")

                # Статистика по моделям (из отдельной таблицы)
                # TODO(CODE_REVIEW): Extract to separate method _get_analyzer_statistics()
                # TODO(CODE_REVIEW): Huge code duplication - use strategy pattern or dict mapping
                if analyzed_tracks > 0:
                    print("\n🧠 Статистика по анализаторам:")

                    if self._table_exists("analysis_results"):
                        cur.execute("""
                            SELECT
                                analyzer_type,
                                COUNT(*) as count,
                                COUNT(DISTINCT track_id) as unique_tracks
                            FROM analysis_results
                            WHERE analyzer_type IS NOT NULL
                            GROUP BY analyzer_type
                            ORDER BY count DESC
                        """)
                        models = cur.fetchall()

                        for model_data in models:
                            analyzer = model_data["analyzer_type"] or "Unknown"
                            count = model_data["count"]
                            unique_count = model_data["unique_tracks"]
                            percentage = (
                                count / total_analyses * 100
                                if total_analyses > 0
                                else 0
                            )
                            print(
                                f"  • {analyzer}: {count:,} анализов ({unique_count:,} треков, {percentage:.1f}%)"
                            )

                    elif self._table_exists("ai_analysis"):
                        cur.execute("""
                            SELECT
                                model_version,
                                COUNT(*) as count
                            FROM ai_analysis
                            WHERE model_version IS NOT NULL
                            GROUP BY model_version
                            ORDER by count DESC
                        """)
                        models = cur.fetchall()

                    # TODO(CODE_REVIEW): This loop is unreachable if ai_analysis table exists
                    # TODO(CODE_REVIEW): Fix indentation - this code seems to be in wrong place
                    for model_data in models:
                        model = model_data["model_version"] or "Unknown"
                        count = model_data["count"]
                        percentage = count / analyzed_tracks * 100
                        print(f"  • {model}: {count:,} ({percentage:.1f}%)")

                # Временная статистика
                # TODO(CODE_REVIEW): Extract to _get_time_statistics()
                if analyzed_tracks > 0:
                    print("\n📅 Временная статистика:")

                    if self._table_exists("analysis_results"):
                        cur.execute("""
                            SELECT
                                MIN(created_at) as first_analysis,
                                MAX(created_at) as last_analysis
                            FROM analysis_results
                        """)
                        time_stats = cur.fetchone()

                        if time_stats and time_stats["first_analysis"]:
                            print(f"  🏁 Первый анализ: {time_stats['first_analysis']}")
                            print(
                                f"  🏆 Последний анализ: {time_stats['last_analysis']}"
                            )

                    elif self._table_exists("ai_analysis"):
                        cur.execute("""
                            SELECT
                                MIN(analysis_date) as first_analysis,
                                MAX(analysis_date) as last_analysis
                            FROM ai_analysis
                        """)
                        time_stats = cur.fetchone()

                        if time_stats and time_stats["first_analysis"]:
                            print(f"  🏁 Первый анализ: {time_stats['first_analysis']}")
                            print(
                                f"  🏆 Последний анализ: {time_stats['last_analysis']}"
                            )

                # Последние анализы
                # TODO(CODE_REVIEW): Extract to _get_recent_analyses()
                # TODO(CODE_REVIEW): Add LIMIT as parameter
                if analyzed_tracks > 0:
                    print("\n🕐 Последние 5 анализов:")

                    if self._table_exists("analysis_results"):
                        cur.execute("""
                            SELECT
                                t.title,
                                t.artist,
                                a.sentiment,
                                a.analyzer_type,
                                a.created_at,
                                a.confidence
                            FROM analysis_results a
                            JOIN tracks t ON a.track_id = t.id
                            ORDER BY a.created_at DESC
                            LIMIT 5
                        """)
                        recent = cur.fetchall()

                        for track in recent:
                            artist = track["artist"]
                            title = track["title"]
                            sentiment = track["sentiment"] or "N/A"
                            analyzer = track["analyzer_type"] or "Unknown"
                            date = track["created_at"]
                            confidence = track["confidence"] or 0
                            # TODO(CODE_REVIEW): Format confidence better (handle None)
                            print(
                                f"  • {artist} - {title} | {sentiment} ({confidence:.1%}) | {analyzer} | {date}"
                            )

                    elif self._table_exists("ai_analysis"):
                        cur.execute("""
                            SELECT
                                t.title,
                                t.artist,
                                a.sentiment,
                                a.model_version,
                                a.analysis_date
                            FROM ai_analysis a
                            JOIN tracks t ON a.track_id = t.id
                            ORDER BY a.analysis_date DESC
                            LIMIT 5
                        """)
                        recent = cur.fetchall()

                        for track in recent:
                            artist = track["artist"]
                            title = track["title"]
                            sentiment = track["sentiment"] or "N/A"
                            model = track["model_version"] or "Unknown"
                            date = track["analysis_date"]
                            print(
                                f"  • {artist} - {title} | {sentiment} | {model} | {date}"
                            )

        except Exception as e:
            # TODO(CODE_REVIEW): Use specific exceptions
            # TODO(CODE_REVIEW): Add logging.error with exc_info
            print(f"❌ Ошибка при проверке статуса анализа: {e}")

    # TODO(CODE_REVIEW): Add type hints: (self, limit: int = 10) -> Optional[int]
    # TODO(CODE_REVIEW): Add comprehensive docstring with Args, Returns
    # TODO(CODE_REVIEW): Consider returning List[Dict] with track info instead of just first_id
    def find_unanalyzed(self, limit=10):
        """Поиск неанализированных записей в PostgreSQL"""
        print("🔍 ПОИСК НЕАНАЛИЗИРОВАННЫХ ЗАПИСЕЙ")
        print("=" * 50)

        if not self.conn:
            print("❌ Нет подключения к базе данных")
            return None

        if not self._table_exists("tracks"):
            print("❌ Таблица tracks не найдена")
            return None

        # TODO(CODE_REVIEW): Handle invalid limit values (negative, zero, too large)
        # TODO(CODE_REVIEW): Add max limit constant to prevent huge queries
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Полностью неанализированные записи
                # TODO(CODE_REVIEW): Extract query to constant or method
                # TODO(CODE_REVIEW): Huge code duplication - extract to helper method
                # TODO(CODE_REVIEW): NOT IN can be slow - use LEFT JOIN instead
                if self._table_exists("analysis_results"):
                    cur.execute(
                        """
                        SELECT t.id, t.artist, t.title
                        FROM tracks t
                        WHERE t.lyrics IS NOT NULL
                        AND t.lyrics != ''
                        AND t.id NOT IN (SELECT DISTINCT track_id FROM analysis_results)
                        ORDER BY t.id
                        LIMIT %s
                    """,
                        (limit,),
                    )
                elif self._table_exists("ai_analysis"):
                    cur.execute(
                        """
                        SELECT t.id, t.artist, t.title
                        FROM tracks t
                        WHERE t.lyrics IS NOT NULL
                        AND t.lyrics != ''
                        AND t.id NOT IN (SELECT DISTINCT track_id FROM ai_analysis)
                        ORDER BY t.id
                        LIMIT %s
                    """,
                        (limit,),
                    )
                else:
                    # Если нет таблицы анализов, все треки неанализированы
                    cur.execute(
                        """
                        SELECT id, artist, title
                        FROM tracks
                        WHERE lyrics IS NOT NULL
                        AND lyrics != ''
                        ORDER BY id
                        LIMIT %s
                    """,
                        (limit,),
                    )

                unanalyzed = cur.fetchall()

                print(f"📋 Первые {limit} полностью неанализированных записей:")
                if unanalyzed:
                    for i, track in enumerate(unanalyzed, 1):
                        track_id = track["id"]
                        artist = track["artist"]
                        title = track["title"]
                        print(f"  {i:2d}. ID: {track_id} | {artist} - {title}")

                    first_id = unanalyzed[0]["id"]
                    print(f"\n🎯 Первая неанализированная: ID {first_id}")
                    # TODO(CODE_REVIEW): Extract script name to constant
                    # TODO(CODE_REVIEW): This is business logic mixed with presentation
                    print("💡 Рекомендуемая команда для анализа:")
                    print(
                        f"   python scripts/mass_qwen_analysis.py --start-id {first_id}"
                    )

                    return first_id
                # TODO(CODE_REVIEW): Inconsistent return - else clause returns None, this prints
                print("  ✅ Все записи проанализированы!")
                return None

        except Exception as e:
            # TODO(CODE_REVIEW): Use specific exceptions
            # TODO(CODE_REVIEW): Add logging
            print(f"❌ Ошибка при поиске неанализированных записей: {e}")
            return None

    # TODO(CODE_REVIEW): Add type hint -> None
    # TODO(CODE_REVIEW): Add docstring
    # TODO(CODE_REVIEW): Code duplication with check_general_status() and check_analysis_status()
    # TODO(CODE_REVIEW): Consider combining similar queries to reduce DB roundtrips
    def quick_check(self):
        """Быстрая проверка основных метрик"""
        print("⚡ БЫСТРАЯ ПРОВЕРКА")
        print("=" * 30)

        if not self.conn:
            print("❌ Нет подключения к базе данных")
            return

        # TODO(CODE_REVIEW): Handle exceptions for better error reporting
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                tracks_with_lyrics = 0

                # Основные цифры
                if self._table_exists("tracks"):
                    # TODO(CODE_REVIEW): Combine these two queries into one
                    cur.execute("SELECT COUNT(*) FROM tracks")
                    total_tracks = cur.fetchone()[0]

                    cur.execute("""
                        SELECT COUNT(*) FROM tracks
                        WHERE lyrics IS NOT NULL AND lyrics != ''
                    """)
                    tracks_with_lyrics = cur.fetchone()[0]

                    print(
                        f"🎵 Треков: {total_tracks:,} (с текстами: {tracks_with_lyrics:,})"
                    )

                    # AI анализ
                    # TODO(CODE_REVIEW): Extract to helper method
                    analyzed = 0
                    if self._table_exists("analysis_results"):
                        cur.execute(
                            "SELECT COUNT(DISTINCT track_id) FROM analysis_results"
                        )
                        analyzed = cur.fetchone()[0]
                    elif self._table_exists("ai_analysis"):
                        cur.execute("SELECT COUNT(DISTINCT track_id) FROM ai_analysis")
                        analyzed = cur.fetchone()[0]

                    # TODO(CODE_REVIEW): Extract percentage calculation to utility
                    coverage = (
                        analyzed / tracks_with_lyrics * 100
                        if tracks_with_lyrics > 0
                        else 0
                    )
                    print(
                        f"🤖 Анализ: {analyzed:,}/{tracks_with_lyrics:,} ({coverage:.1f}%)"
                    )

                # Spotify данные
                if self._table_exists("spotify_tracks"):
                    cur.execute("SELECT COUNT(*) FROM spotify_tracks")
                    spotify = cur.fetchone()[0]
                    spotify_coverage = (
                        spotify / tracks_with_lyrics * 100
                        if tracks_with_lyrics > 0
                        else 0
                    )
                    print(f"🎵 Spotify: {spotify:,} ({spotify_coverage:.1f}%)")

                # Размер БД
                cur.execute(
                    "SELECT pg_size_pretty(pg_database_size(current_database()))"
                )
                db_size = cur.fetchone()[0]
                print(f"💾 Размер БД: {db_size}")

        except Exception as e:
            # TODO(CODE_REVIEW): Use specific exceptions
            # TODO(CODE_REVIEW): Add logging
            print(f"❌ Ошибка при быстрой проверке: {e}")

    # TODO(CODE_REVIEW): Add type hints: (self, table_name: str) -> bool
    # TODO(CODE_REVIEW): Add docstring with Args and Returns
    # TODO(CODE_REVIEW): Add caching decorator to avoid repeated checks
    # TODO(CODE_REVIEW): Consider using information_schema.tables directly in queries (WITH clause)
    def _table_exists(self, table_name):
        """Проверка существования таблицы в PostgreSQL"""
        if not self.conn:
            return False

        # TODO(CODE_REVIEW): Don't silently catch all exceptions - log them at least
        # TODO(CODE_REVIEW): Use specific exception types
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = %s
                    )
                """,
                    (table_name,),
                )
                return cur.fetchone()[0]
        except Exception:
            # TODO(CODE_REVIEW): Log the exception
            return False


# TODO(CODE_REVIEW): Add type hint -> int
# TODO(CODE_REVIEW): Add comprehensive docstring
# TODO(CODE_REVIEW): Consider using argparse.ArgumentParser.parse_known_args for extensibility
# TODO(CODE_REVIEW): Too much logic in main() - extract to separate functions
def main():
    """Главная функция с обработкой аргументов"""
    # TODO(CODE_REVIEW): Add --verbose flag for debug logging
    # TODO(CODE_REVIEW): Add --format flag for output format (json, yaml, text)
    # TODO(CODE_REVIEW): Add --output flag to save results to file
    parser = argparse.ArgumentParser(
        description="PostgreSQL Database Diagnostics Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                    # Полная диагностика
  %(prog)s --quick            # Быстрая проверка
  %(prog)s --schema           # Только схема таблиц
  %(prog)s --analysis         # Только статус AI анализа
  %(prog)s --unanalyzed       # Поиск неанализированных записей
  %(prog)s --unanalyzed -n 20 # Первые 20 неанализированных
        """,
    )

    parser.add_argument("--schema", action="store_true", help="Проверка схемы таблиц")
    parser.add_argument("--analysis", action="store_true", help="Статус AI анализа")
    parser.add_argument(
        "--unanalyzed", action="store_true", help="Поиск неанализированных записей"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Быстрая проверка основных метрик"
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=10,
        # TODO(CODE_REVIEW): Add validation for limit (must be positive, reasonable max)
        help="Количество неанализированных записей для показа (по умолчанию: 10)",
    )

    args = parser.parse_args()

    # Создаем экземпляр диагностики
    # TODO(CODE_REVIEW): Use context manager (with statement) for automatic cleanup
    diagnostics = PostgreSQLDiagnostics()

    if not diagnostics.connect():
        # TODO(CODE_REVIEW): Use sys.exit() instead of return for clarity
        return 1

    # TODO(CODE_REVIEW): Use try-finally to ensure close() is called even on errors
    # TODO(CODE_REVIEW): Better yet, implement context manager in class
    try:
        # Если нет специфических флагов, показываем полную диагностику
        # TODO(CODE_REVIEW): Extract each check to a separate function
        # TODO(CODE_REVIEW): Consider using command pattern for extensibility
        if not any([args.schema, args.analysis, args.unanalyzed, args.quick]):
            diagnostics.check_general_status()
            print("\n")
            diagnostics.check_analysis_status()
            print("\n")
            diagnostics.find_unanalyzed(args.limit)
        else:
            # Выполняем только запрошенные проверки
            # TODO(CODE_REVIEW): Use a list of checks and iterate instead of multiple ifs
            if args.quick:
                diagnostics.quick_check()

            if args.schema:
                diagnostics.check_schema()

            if args.analysis:
                if args.schema:
                    print("\n")
                diagnostics.check_analysis_status()

            if args.unanalyzed:
                if args.schema or args.analysis:
                    print("\n")
                diagnostics.find_unanalyzed(args.limit)

        print("\n✅ Диагностика завершена")

    except Exception as e:
        # TODO(CODE_REVIEW): Use specific exceptions
        # TODO(CODE_REVIEW): Add logging.exception() for full traceback
        # TODO(CODE_REVIEW): Add --debug flag to show full traceback
        print(f"❌ Ошибка при выполнении диагностики: {e}")
        return 1

    finally:
        # TODO(CODE_REVIEW): Log successful closure
        diagnostics.close()

    return 0


if __name__ == "__main__":
    # TODO(CODE_REVIEW): Add sys.exit() explicitly
    # TODO(CODE_REVIEW): Consider adding signal handlers for graceful shutdown
    # TODO(CODE_REVIEW): Add performance timing logging
    exit(main())


# ============================================================================
# SUMMARY OF CODE REVIEW ISSUES (FAANG/Google Standards)
# ============================================================================
#
# CRITICAL ISSUES (Must Fix):
# 1. SECURITY: Password printed in logs (line ~136)
# 2. SECURITY: SQL injection vulnerability with f-string (line ~370)
# 3. Missing type hints throughout entire file
# 4. No logging - using print statements
# 5. Methods too long (>100 lines) - violates SRP
#
# HIGH PRIORITY:
# 6. Massive code duplication (analysis_results vs ai_analysis checks)
# 7. No error handling granularity (catching Exception too broadly)
# 8. No unit tests
# 9. Missing docstrings in Google format
# 10. Hard-coded magic strings and numbers
# 11. No constants defined
# 12. Class doesn't implement context manager
# 13. No retry logic with exponential backoff
# 14. Inefficient SQL (NOT IN instead of LEFT JOIN, multiple queries instead of one)
#
# MEDIUM PRIORITY:
# 15. Mixing business logic with presentation
# 16. No structured data return (only printing)
# 17. sys.path manipulation at runtime
# 18. Emojis in production code
# 19. Russian comments/text in code
# 20. No input validation
# 21. No connection pooling
# 22. No timeout configuration
# 23. Silent exception swallowing (bare pass)
#
# LOW PRIORITY:
# 24. Missing copyright header
# 25. No --verbose, --debug, --format, --output flags
# 26. No performance metrics/timing
# 27. Inconsistent return values
# 28. Missing signal handlers
# 29. No caching for repeated operations
# 30. Could use dataclasses for structured data
#
# ARCHITECTURE SUGGESTIONS:
# - Split into multiple classes (ConnectionManager, SchemaAnalyzer, etc.)
# - Use strategy pattern for different analysis table types
# - Implement repository pattern for data access
# - Use dependency injection for configuration
# - Add abstract base classes for extensibility
# - Consider using SQLAlchemy for database abstraction
# - Add observability (structured logging, metrics, tracing)
#
# ============================================================================
