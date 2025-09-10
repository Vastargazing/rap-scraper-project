#!/usr/bin/env python3
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

import psycopg2
import psycopg2.extras
import argparse
import os
from pathlib import Path
from datetime import datetime
import sys
import json

# Добавляем корневую папку в path для доступа к src модулям
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

class PostgreSQLDiagnostics:
    """Класс для диагностики PostgreSQL базы данных"""
    
    def __init__(self):
        self.project_root = project_root
        self.conn = None
        
        # Загружаем конфигурацию БД
        try:
            from src.utils.config import get_db_config
            self.db_config = get_db_config()
        except ImportError:
            # Fallback конфигурация для Windows PostgreSQL
            self.db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'port': int(os.getenv('DB_PORT', '5432')),
                'database': os.getenv('DB_NAME', 'postgres'),  # Используем дефолтную БД
                'user': os.getenv('DB_USER', 'postgres'),
                'password': os.getenv('DB_PASSWORD', '')  # Пустой пароль для Windows
            }
    
    def connect(self):
        """Подключение к PostgreSQL базе данных"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            # Включаем автокоммит для диагностических запросов
            self.conn.autocommit = True
            print("✅ Подключение к PostgreSQL успешно!")
            return True
        except Exception as e:
            print(f"❌ Ошибка подключения к PostgreSQL: {e}")
            print(f"� Текущие настройки: {self.db_config}")
            
            # Предлагаем альтернативные варианты подключения
            print("\n�💡 Попробуйте альтернативные варианты:")
            
            # Вариант 1: пустой пароль
            if self.db_config['password']:
                print("1️⃣ Попытка подключения без пароля...")
                try:
                    alt_config = self.db_config.copy()
                    alt_config['password'] = ''
                    self.conn = psycopg2.connect(**alt_config)
                    self.conn.autocommit = True
                    self.db_config = alt_config
                    print("✅ Подключение без пароля успешно!")
                    return True
                except Exception:
                    pass
            
            # Вариант 2: база данных postgres
            if self.db_config['database'] != 'postgres':
                print("2️⃣ Попытка подключения к БД 'postgres'...")
                try:
                    alt_config = self.db_config.copy()
                    alt_config['database'] = 'postgres'
                    alt_config['password'] = ''
                    self.conn = psycopg2.connect(**alt_config)
                    self.conn.autocommit = True
                    self.db_config = alt_config
                    print("✅ Подключение к БД 'postgres' успешно!")
                    return True
                except Exception:
                    pass
            
            # Вариант 3: пользователь из .env
            env_user = os.getenv('POSTGRES_USERNAME')
            env_password = os.getenv('POSTGRES_PASSWORD')
            env_db = os.getenv('POSTGRES_DATABASE')
            
            if env_user and env_password and env_db:
                print("3️⃣ Попытка подключения с настройками из .env...")
                try:
                    alt_config = {
                        'host': self.db_config['host'],
                        'port': self.db_config['port'],
                        'database': env_db,
                        'user': env_user,
                        'password': env_password
                    }
                    self.conn = psycopg2.connect(**alt_config)
                    self.conn.autocommit = True
                    self.db_config = alt_config
                    print("✅ Подключение с настройками из .env успешно!")
                    return True
                except Exception as e2:
                    print(f"   ❌ Также не удалось: {e2}")
            
            print("\n🛠️  Для решения проблемы:")
            print("   1. Проверьте, что PostgreSQL запущен")
            print("   2. Убедитесь, что пользователь и пароль корректны")
            print("   3. Проверьте настройки в .env файле")
            print("   4. Возможно, нужно создать базу данных и пользователя")
            
            return False
    
    def close(self):
        """Закрытие соединения"""
        if self.conn:
            self.conn.close()
    
    def check_general_status(self):
        """Общая диагностика PostgreSQL базы данных"""
        print("🔍 ОБЩАЯ ДИАГНОСТИКА POSTGRESQL БАЗЫ ДАННЫХ")
        print("=" * 50)
        
        if not self.conn:
            print("❌ Нет подключения к базе данных")
            return
        
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Размер базы данных
                cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
                db_size = cur.fetchone()[0]
                print(f"📁 Размер БД: {db_size}")
                
                # Список таблиц
                cur.execute("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public' 
                    ORDER BY tablename
                """)
                tables = [row[0] for row in cur.fetchall()]
                print(f"📋 Таблицы в БД ({len(tables)}): {', '.join(tables)}")
                
                # Основная статистика
                print(f"\n📊 ОСНОВНАЯ СТАТИСТИКА:")
                
                # Треки
                if self._table_exists('tracks'):
                    cur.execute("SELECT COUNT(*) FROM tracks")
                    total_tracks = cur.fetchone()[0]
                    
                    cur.execute("""
                        SELECT COUNT(*) FROM tracks 
                        WHERE lyrics IS NOT NULL AND lyrics != ''
                    """)
                    tracks_with_lyrics = cur.fetchone()[0]
                    
                    print(f"🎵 Всего треков: {total_tracks:,}")
                    if total_tracks > 0:
                        print(f"🎵 С текстами: {tracks_with_lyrics:,} ({tracks_with_lyrics/total_tracks*100:.1f}%)")
                    
                    # Проверяем наличие AI анализов в разных возможных таблицах
                    analyzed_tracks = 0
                    if self._table_exists('analysis_results'):
                        cur.execute("SELECT COUNT(DISTINCT track_id) FROM analysis_results")
                        analyzed_tracks = cur.fetchone()[0]
                        cur.execute("SELECT COUNT(*) FROM analysis_results")
                        total_analyses = cur.fetchone()[0]
                        print(f"🤖 С AI анализом: {analyzed_tracks:,} треков ({analyzed_tracks/total_tracks*100:.1f}%)")
                        print(f"🤖 Всего анализов: {total_analyses:,}")
                    elif self._table_exists('ai_analysis'):
                        cur.execute("SELECT COUNT(DISTINCT track_id) FROM ai_analysis")
                        analyzed_tracks = cur.fetchone()[0]
                        if analyzed_tracks > 0:
                            print(f"🤖 С AI анализом: {analyzed_tracks:,} ({analyzed_tracks/total_tracks*100:.1f}%)")
                    else:
                        print("🤖 Таблица AI анализов не найдена")
                
                # Артисты (уникальные из треков)
                if self._table_exists('tracks'):
                    cur.execute("SELECT COUNT(DISTINCT artist) FROM tracks")
                    unique_artists = cur.fetchone()[0]
                    print(f"🎤 Уникальных артистов: {unique_artists:,}")
                elif self._table_exists('artists'):
                    cur.execute("SELECT COUNT(*) FROM artists")
                    total_artists = cur.fetchone()[0]
                    print(f"🎤 Всего артистов: {total_artists:,}")
                
                # Spotify данные
                if self._table_exists('spotify_tracks'):
                    cur.execute("SELECT COUNT(*) FROM spotify_tracks")
                    spotify_tracks = cur.fetchone()[0]
                    print(f"🎵 Spotify треков: {spotify_tracks:,}")
                
                if self._table_exists('spotify_artists'):
                    cur.execute("SELECT COUNT(*) FROM spotify_artists")
                    spotify_artists = cur.fetchone()[0]
                    print(f"🎤 Spotify артистов: {spotify_artists:,}")
                
                # Топ артистов
                if self._table_exists('tracks'):
                    print(f"\n🏆 ТОП-10 АРТИСТОВ ПО КОЛИЧЕСТВУ ТРЕКОВ:")
                    cur.execute("""
                        SELECT artist, COUNT(*) as count 
                        FROM tracks 
                        GROUP BY artist 
                        ORDER BY count DESC 
                        LIMIT 10
                    """)
                    top_artists = cur.fetchall()
                    
                    for i, (artist, count) in enumerate(top_artists, 1):
                        print(f"  {i:2d}. {artist}: {count:,} треков")
                
                # Последние добавленные
                if self._table_exists('tracks'):
                    print(f"\n📅 ПОСЛЕДНИЕ ДОБАВЛЕННЫЕ ТРЕКИ:")
                    cur.execute("""
                        SELECT title, artist, created_at
                        FROM tracks 
                        WHERE created_at IS NOT NULL 
                        ORDER BY created_at DESC 
                        LIMIT 5
                    """)
                    recent_tracks = cur.fetchall()
                    
                    for title, artist, date in recent_tracks:
                        print(f"  • {artist} - {title} ({date})")
        
        except Exception as e:
            print(f"❌ Ошибка при получении статистики: {e}")
    
    def check_schema(self):
        """Проверка схемы PostgreSQL таблиц"""
        print("🏗️ ПРОВЕРКА СХЕМЫ POSTGRESQL БАЗЫ ДАННЫХ")
        print("=" * 50)
        
        if not self.conn:
            print("❌ Нет подключения к базе данных")
            return
        
        # Основные таблицы для проверки
        important_tables = ['tracks', 'analysis_results', 'songs', 'ai_analysis', 'spotify_tracks', 'spotify_artists', 'spotify_audio_features']
        
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                for table in important_tables:
                    if self._table_exists(table):
                        print(f"\n📋 Таблица: {table}")
                        
                        # Получаем информацию о колонках
                        cur.execute("""
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
                        """, (table,))
                        
                        columns = cur.fetchall()
                        print(f"  Колонок: {len(columns)}")
                        
                        for col in columns:
                            col_name = col['column_name']
                            col_type = col['data_type']
                            is_nullable = col['is_nullable']
                            default_val = col['column_default']
                            max_length = col['character_maximum_length']
                            
                            # Формируем тип с длиной
                            if max_length and col_type == 'character varying':
                                col_type = f"varchar({max_length})"
                            
                            constraints = []
                            if is_nullable == 'NO':
                                constraints.append("NOT NULL")
                            if default_val:
                                constraints.append(f"DEFAULT {default_val}")
                            
                            constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                            print(f"    {col_name}: {col_type}{constraint_str}")
                        
                        # Индексы
                        cur.execute("""
                            SELECT indexname, indexdef
                            FROM pg_indexes
                            WHERE tablename = %s 
                            AND schemaname = 'public'
                        """, (table,))
                        
                        indexes = cur.fetchall()
                        if indexes:
                            print(f"  Индексы: {len(indexes)}")
                            for idx in indexes:
                                print(f"    {idx['indexname']}")
                        
                        # Количество записей
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cur.fetchone()[0]
                        print(f"  Записей: {count:,}")
                        
                        # Размер таблицы
                        cur.execute("""
                            SELECT pg_size_pretty(pg_total_relation_size(%s))
                        """, (table,))
                        size = cur.fetchone()[0]
                        print(f"  Размер: {size}")
                    else:
                        print(f"\n❌ Таблица {table} не найдена")
        
        except Exception as e:
            print(f"❌ Ошибка при проверке схемы: {e}")
    
    def check_analysis_status(self):
        """Проверка статуса AI анализа в PostgreSQL"""
        print("🤖 СТАТУС AI АНАЛИЗА")
        print("=" * 50)
        
        if not self.conn:
            print("❌ Нет подключения к базе данных")
            return
        
        if not self._table_exists('tracks'):
            print("❌ Таблица tracks не найдена")
            return
        
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Общая статистика
                cur.execute("""
                    SELECT COUNT(*) FROM tracks 
                    WHERE lyrics IS NOT NULL AND lyrics != ''
                """)
                total_tracks = cur.fetchone()[0]
                
                # Проверяем AI анализы в отдельной таблице
                analyzed_tracks = 0
                total_analyses = 0
                
                if self._table_exists('analysis_results'):
                    cur.execute("SELECT COUNT(DISTINCT track_id) FROM analysis_results")
                    analyzed_tracks = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM analysis_results") 
                    total_analyses = cur.fetchone()[0]
                elif self._table_exists('ai_analysis'):
                    cur.execute("SELECT COUNT(DISTINCT track_id) FROM ai_analysis")
                    analyzed_tracks = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(*) FROM ai_analysis")
                    total_analyses = cur.fetchone()[0]
                
                print(f"📊 Общая статистика:")
                print(f"  🎵 Треков с текстами: {total_tracks:,}")
                print(f"  🤖 Проанализированных треков: {analyzed_tracks:,}")
                print(f"  📊 Всего анализов: {total_analyses:,}")
                
                if total_tracks > 0:
                    coverage = analyzed_tracks / total_tracks * 100
                    print(f"  📈 Покрытие: {coverage:.1f}%")
                    print(f"  📋 Неанализированных: {total_tracks - analyzed_tracks:,}")
                
                # Статистика по моделям (из отдельной таблицы)
                if analyzed_tracks > 0:
                    print(f"\n🧠 Статистика по анализаторам:")
                    
                    if self._table_exists('analysis_results'):
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
                            analyzer = model_data['analyzer_type'] or 'Unknown'
                            count = model_data['count']
                            unique_count = model_data['unique_tracks']
                            percentage = count / total_analyses * 100 if total_analyses > 0 else 0
                            print(f"  • {analyzer}: {count:,} анализов ({unique_count:,} треков, {percentage:.1f}%)")
                            
                    elif self._table_exists('ai_analysis'):
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
                    
                    for model_data in models:
                        model = model_data['model_version'] or 'Unknown'
                        count = model_data['count']
                        percentage = count / analyzed_tracks * 100
                        print(f"  • {model}: {count:,} ({percentage:.1f}%)")
                
                # Временная статистика
                if analyzed_tracks > 0:
                    print(f"\n📅 Временная статистика:")
                    
                    if self._table_exists('analysis_results'):
                        cur.execute("""
                            SELECT 
                                MIN(created_at) as first_analysis,
                                MAX(created_at) as last_analysis
                            FROM analysis_results
                        """)
                        time_stats = cur.fetchone()
                        
                        if time_stats and time_stats['first_analysis']:
                            print(f"  🏁 Первый анализ: {time_stats['first_analysis']}")
                            print(f"  🏆 Последний анализ: {time_stats['last_analysis']}")
                    
                    elif self._table_exists('ai_analysis'):
                        cur.execute("""
                            SELECT 
                                MIN(analysis_date) as first_analysis,
                                MAX(analysis_date) as last_analysis
                            FROM ai_analysis
                        """)
                        time_stats = cur.fetchone()
                        
                        if time_stats and time_stats['first_analysis']:
                            print(f"  🏁 Первый анализ: {time_stats['first_analysis']}")
                            print(f"  🏆 Последний анализ: {time_stats['last_analysis']}")
                
                # Последние анализы
                if analyzed_tracks > 0:
                    print(f"\n🕐 Последние 5 анализов:")
                    
                    if self._table_exists('analysis_results'):
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
                            artist = track['artist']
                            title = track['title'] 
                            sentiment = track['sentiment'] or 'N/A'
                            analyzer = track['analyzer_type'] or 'Unknown'
                            date = track['created_at']
                            confidence = track['confidence'] or 0
                            print(f"  • {artist} - {title} | {sentiment} ({confidence:.1%}) | {analyzer} | {date}")
                    
                    elif self._table_exists('ai_analysis'):
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
                            artist = track['artist']
                            title = track['title'] 
                            sentiment = track['sentiment'] or 'N/A'
                            model = track['model_version'] or 'Unknown'
                            date = track['analysis_date']
                            print(f"  • {artist} - {title} | {sentiment} | {model} | {date}")
        
        except Exception as e:
            print(f"❌ Ошибка при проверке статуса анализа: {e}")
    
    def find_unanalyzed(self, limit=10):
        """Поиск неанализированных записей в PostgreSQL"""
        print("🔍 ПОИСК НЕАНАЛИЗИРОВАННЫХ ЗАПИСЕЙ")
        print("=" * 50)
        
        if not self.conn:
            print("❌ Нет подключения к базе данных")
            return None
        
        if not self._table_exists('tracks'):
            print("❌ Таблица tracks не найдена")
            return None
        
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                # Полностью неанализированные записи
                if self._table_exists('analysis_results'):
                    cur.execute("""
                        SELECT t.id, t.artist, t.title
                        FROM tracks t
                        WHERE t.lyrics IS NOT NULL 
                        AND t.lyrics != '' 
                        AND t.id NOT IN (SELECT DISTINCT track_id FROM analysis_results)
                        ORDER BY t.id
                        LIMIT %s
                    """, (limit,))
                elif self._table_exists('ai_analysis'):
                    cur.execute("""
                        SELECT t.id, t.artist, t.title
                        FROM tracks t
                        WHERE t.lyrics IS NOT NULL 
                        AND t.lyrics != '' 
                        AND t.id NOT IN (SELECT DISTINCT track_id FROM ai_analysis)
                        ORDER BY t.id
                        LIMIT %s
                    """, (limit,))
                else:
                    # Если нет таблицы анализов, все треки неанализированы
                    cur.execute("""
                        SELECT id, artist, title
                        FROM tracks 
                        WHERE lyrics IS NOT NULL 
                        AND lyrics != '' 
                        ORDER BY id
                        LIMIT %s
                    """, (limit,))
                
                unanalyzed = cur.fetchall()
                
                print(f"📋 Первые {limit} полностью неанализированных записей:")
                if unanalyzed:
                    for i, track in enumerate(unanalyzed, 1):
                        track_id = track['id']
                        artist = track['artist']
                        title = track['title']
                        print(f"  {i:2d}. ID: {track_id} | {artist} - {title}")
                    
                    first_id = unanalyzed[0]['id']
                    print(f"\n🎯 Первая неанализированная: ID {first_id}")
                    print(f"💡 Рекомендуемая команда для анализа:")
                    print(f"   python scripts/mass_qwen_analysis.py --start-id {first_id}")
                    
                    return first_id
                else:
                    print("  ✅ Все записи проанализированы!")
                    return None
        
        except Exception as e:
            print(f"❌ Ошибка при поиске неанализированных записей: {e}")
            return None
    
    def quick_check(self):
        """Быстрая проверка основных метрик"""
        print("⚡ БЫСТРАЯ ПРОВЕРКА")
        print("=" * 30)
        
        if not self.conn:
            print("❌ Нет подключения к базе данных")
            return
        
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                tracks_with_lyrics = 0
                
                # Основные цифры
                if self._table_exists('tracks'):
                    cur.execute("SELECT COUNT(*) FROM tracks")
                    total_tracks = cur.fetchone()[0]
                    
                    cur.execute("""
                        SELECT COUNT(*) FROM tracks 
                        WHERE lyrics IS NOT NULL AND lyrics != ''
                    """)
                    tracks_with_lyrics = cur.fetchone()[0]
                    
                    print(f"🎵 Треков: {total_tracks:,} (с текстами: {tracks_with_lyrics:,})")
                    
                    # AI анализ
                    analyzed = 0
                    if self._table_exists('analysis_results'):
                        cur.execute("SELECT COUNT(DISTINCT track_id) FROM analysis_results")
                        analyzed = cur.fetchone()[0]
                    elif self._table_exists('ai_analysis'):
                        cur.execute("SELECT COUNT(DISTINCT track_id) FROM ai_analysis")
                        analyzed = cur.fetchone()[0]
                    
                    coverage = analyzed / tracks_with_lyrics * 100 if tracks_with_lyrics > 0 else 0
                    print(f"🤖 Анализ: {analyzed:,}/{tracks_with_lyrics:,} ({coverage:.1f}%)")
                
                # Spotify данные
                if self._table_exists('spotify_tracks'):
                    cur.execute("SELECT COUNT(*) FROM spotify_tracks")
                    spotify = cur.fetchone()[0]
                    spotify_coverage = spotify / tracks_with_lyrics * 100 if tracks_with_lyrics > 0 else 0
                    print(f"🎵 Spotify: {spotify:,} ({spotify_coverage:.1f}%)")
                
                # Размер БД
                cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
                db_size = cur.fetchone()[0]
                print(f"💾 Размер БД: {db_size}")
        
        except Exception as e:
            print(f"❌ Ошибка при быстрой проверке: {e}")
    
    def _table_exists(self, table_name):
        """Проверка существования таблицы в PostgreSQL"""
        if not self.conn:
            return False
            
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    )
                """, (table_name,))
                return cur.fetchone()[0]
        except Exception:
            return False


def main():
    """Главная функция с обработкой аргументов"""
    parser = argparse.ArgumentParser(
        description='PostgreSQL Database Diagnostics Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                    # Полная диагностика
  %(prog)s --quick            # Быстрая проверка
  %(prog)s --schema           # Только схема таблиц
  %(prog)s --analysis         # Только статус AI анализа
  %(prog)s --unanalyzed       # Поиск неанализированных записей
  %(prog)s --unanalyzed -n 20 # Первые 20 неанализированных
        """
    )
    
    parser.add_argument('--schema', action='store_true', 
                       help='Проверка схемы таблиц')
    parser.add_argument('--analysis', action='store_true',
                       help='Статус AI анализа')
    parser.add_argument('--unanalyzed', action='store_true',
                       help='Поиск неанализированных записей')
    parser.add_argument('--quick', action='store_true',
                       help='Быстрая проверка основных метрик')
    parser.add_argument('-n', '--limit', type=int, default=10,
                       help='Количество неанализированных записей для показа (по умолчанию: 10)')
    
    args = parser.parse_args()
    
    # Создаем экземпляр диагностики
    diagnostics = PostgreSQLDiagnostics()
    
    if not diagnostics.connect():
        return 1
    
    try:
        # Если нет специфических флагов, показываем полную диагностику
        if not any([args.schema, args.analysis, args.unanalyzed, args.quick]):
            diagnostics.check_general_status()
            print("\n")
            diagnostics.check_analysis_status()
            print("\n")
            diagnostics.find_unanalyzed(args.limit)
        else:
            # Выполняем только запрошенные проверки
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
        
        print(f"\n✅ Диагностика завершена")
        
    except Exception as e:
        print(f"❌ Ошибка при выполнении диагностики: {e}")
        return 1
    
    finally:
        diagnostics.close()
    
    return 0


if __name__ == "__main__":
    exit(main())
