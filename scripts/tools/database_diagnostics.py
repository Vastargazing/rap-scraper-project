#!/usr/bin/env python3
"""
🔍 Unified Database Diagnostics Tool
Объединенный инструмент диагностики базы данных

ФУНКЦИИ:
- Общая диагностика базы данных (структура, размер, статистика)
- Проверка схемы таблиц и их структуры
- Анализ статуса AI-анализа и покрытия
- Поиск неанализированных записей
- Рекомендации для оптимизации

ИСПОЛЬЗОВАНИЕ:
python scripts/tools/database_diagnostics.py                # Полная диагностика
python scripts/tools/database_diagnostics.py --schema       # Только схема
python scripts/tools/database_diagnostics.py --analysis     # Только AI анализ
python scripts/tools/database_diagnostics.py --unanalyzed   # Неанализированные записи
python scripts/tools/database_diagnostics.py --quick        # Быстрая проверка

АВТОР: AI Assistant (объединение check_db.py, db_status.py, check_schema.py)
ДАТА: Сентябрь 2025
"""

import sqlite3
import argparse
import os
from pathlib import Path
from datetime import datetime
import sys

# Добавляем корневую папку в path для доступа к src модулям
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

class DatabaseDiagnostics:
    """Класс для диагностики базы данных"""
    
    def __init__(self):
        self.project_root = project_root
        self.db_path = self.project_root / 'data' / 'rap_lyrics.db'
        self.conn = None
    
    def connect(self):
        """Подключение к базе данных"""
        if not self.db_path.exists():
            print(f"❌ База данных не найдена: {self.db_path}")
            return False
        
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            return True
        except Exception as e:
            print(f"❌ Ошибка подключения к БД: {e}")
            return False
    
    def close(self):
        """Закрытие соединения"""
        if self.conn:
            self.conn.close()
    
    def check_general_status(self):
        """Общая диагностика базы данных (из check_db.py)"""
        print("🔍 ОБЩАЯ ДИАГНОСТИКА БАЗЫ ДАННЫХ")
        print("=" * 50)
        
        # Размер файла БД
        db_size = self.db_path.stat().st_size / (1024 * 1024)  # MB
        print(f"📁 Размер файла БД: {db_size:.2f} MB")
        
        # Список таблиц
        tables_query = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        tables = [row[0] for row in self.conn.execute(tables_query).fetchall()]
        print(f"📋 Таблицы в БД ({len(tables)}): {', '.join(tables)}")
        
        # Основная статистика
        print(f"\n📊 ОСНОВНАЯ СТАТИСТИКА:")
        
        # Песни
        total_songs = self.conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        songs_with_lyrics = self.conn.execute(
            "SELECT COUNT(*) FROM songs WHERE lyrics IS NOT NULL AND lyrics != ''"
        ).fetchone()[0]
        print(f"🎵 Всего песен: {total_songs:,}")
        print(f"🎵 С текстами: {songs_with_lyrics:,} ({songs_with_lyrics/total_songs*100:.1f}%)")
        
        # Артисты
        total_artists = self.conn.execute("SELECT COUNT(DISTINCT artist) FROM songs").fetchone()[0]
        print(f"🎤 Уникальных артистов: {total_artists:,}")
        
        # AI анализы
        if self._table_exists('ai_analysis'):
            total_analyses = self.conn.execute("SELECT COUNT(*) FROM ai_analysis").fetchone()[0]
            print(f"🤖 AI анализов: {total_analyses:,}")
        
        # Spotify данные
        if self._table_exists('spotify_tracks'):
            spotify_tracks = self.conn.execute("SELECT COUNT(*) FROM spotify_tracks").fetchone()[0]
            print(f"🎵 Spotify треков: {spotify_tracks:,}")
        
        if self._table_exists('spotify_artists'):
            spotify_artists = self.conn.execute("SELECT COUNT(*) FROM spotify_artists").fetchone()[0]
            print(f"🎤 Spotify артистов: {spotify_artists:,}")
        
        # Топ артистов
        print(f"\n🏆 ТОП-10 АРТИСТОВ ПО КОЛИЧЕСТВУ ПЕСЕН:")
        top_artists = self.conn.execute("""
            SELECT artist, COUNT(*) as count 
            FROM songs 
            GROUP BY artist 
            ORDER BY count DESC 
            LIMIT 10
        """).fetchall()
        
        for i, (artist, count) in enumerate(top_artists, 1):
            print(f"  {i:2d}. {artist}: {count:,} песен")
        
        # Последние добавленные
        print(f"\n📅 ПОСЛЕДНИЕ ДОБАВЛЕННЫЕ ПЕСНИ:")
        recent_songs = self.conn.execute("""
            SELECT artist, title, scraped_date 
            FROM songs 
            WHERE scraped_date IS NOT NULL 
            ORDER BY scraped_date DESC 
            LIMIT 5
        """).fetchall()
        
        for artist, title, date in recent_songs:
            print(f"  • {artist} - {title} ({date})")
    
    def check_schema(self):
        """Проверка схемы таблиц (из check_schema.py)"""
        print("🏗️ ПРОВЕРКА СХЕМЫ БАЗЫ ДАННЫХ")
        print("=" * 50)
        
        # Основные таблицы для проверки
        important_tables = ['songs', 'ai_analysis', 'spotify_tracks', 'spotify_artists', 'spotify_audio_features']
        
        for table in important_tables:
            if self._table_exists(table):
                print(f"\n📋 Таблица: {table}")
                
                # Получаем CREATE TABLE
                create_sql = self.conn.execute("""
                    SELECT sql FROM sqlite_master 
                    WHERE type='table' AND name=?
                """, (table,)).fetchone()
                
                if create_sql:
                    # Показываем структуру таблицы
                    columns = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
                    print(f"  Колонок: {len(columns)}")
                    for cid, name, dtype, notnull, default, pk in columns:
                        constraints = []
                        if pk:
                            constraints.append("PRIMARY KEY")
                        if notnull:
                            constraints.append("NOT NULL")
                        if default is not None:
                            constraints.append(f"DEFAULT {default}")
                        
                        constraint_str = f" ({', '.join(constraints)})" if constraints else ""
                        print(f"    {name}: {dtype}{constraint_str}")
                
                # Количество записей
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"  Записей: {count:,}")
            else:
                print(f"\n❌ Таблица {table} не найдена")
    
    def check_analysis_status(self):
        """Проверка статуса AI анализа (из db_status.py)"""
        print("🤖 СТАТУС AI АНАЛИЗА")
        print("=" * 50)
        
        if not self._table_exists('ai_analysis'):
            print("❌ Таблица ai_analysis не найдена")
            return
        
        # Общая статистика
        total_songs = self.conn.execute(
            "SELECT COUNT(*) FROM songs WHERE lyrics IS NOT NULL AND lyrics != ''"
        ).fetchone()[0]
        
        total_analyzed = self.conn.execute("SELECT COUNT(*) FROM ai_analysis").fetchone()[0]
        unique_analyzed = self.conn.execute(
            "SELECT COUNT(DISTINCT song_id) FROM ai_analysis"
        ).fetchone()[0]
        
        print(f"📊 Общая статистика:")
        print(f"  🎵 Песен с текстами: {total_songs:,}")
        print(f"  🤖 Всего анализов: {total_analyzed:,}")
        print(f"  🎯 Уникальных проанализированных: {unique_analyzed:,}")
        print(f"  📈 Покрытие: {unique_analyzed/total_songs*100:.1f}%")
        print(f"  📋 Неанализированных: {total_songs - unique_analyzed:,}")
        
        # Статистика по моделям
        print(f"\n🧠 Статистика по моделям:")
        models = self.conn.execute("""
            SELECT model_version, COUNT(*) as count
            FROM ai_analysis 
            GROUP BY model_version 
            ORDER BY count DESC
        """).fetchall()
        
        for model, count in models:
            percentage = count / total_analyzed * 100 if total_analyzed > 0 else 0
            print(f"  • {model}: {count:,} ({percentage:.1f}%)")
        
        # Временная статистика
        if total_analyzed > 0:
            print(f"\n📅 Временная статистика:")
            time_stats = self.conn.execute("""
                SELECT 
                    MIN(analysis_date) as first_analysis,
                    MAX(analysis_date) as last_analysis
                FROM ai_analysis
            """).fetchone()
            
            if time_stats[0]:
                print(f"  🏁 Первый анализ: {time_stats[0]}")
                print(f"  🏆 Последний анализ: {time_stats[1]}")
        
        # Последние анализы
        print(f"\n🕐 Последние 5 анализов:")
        recent = self.conn.execute("""
            SELECT s.artist, s.title, a.sentiment, a.model_version, a.analysis_date
            FROM ai_analysis a
            JOIN songs s ON a.song_id = s.id
            ORDER BY a.analysis_date DESC
            LIMIT 5
        """).fetchall()
        
        for artist, title, sentiment, model, date in recent:
            print(f"  • {artist} - {title} | {sentiment} | {model} | {date}")
    
    def find_unanalyzed(self, limit=10):
        """Поиск неанализированных записей (объединенная логика)"""
        print("🔍 ПОИСК НЕАНАЛИЗИРОВАННЫХ ЗАПИСЕЙ")
        print("=" * 50)
        
        if not self._table_exists('ai_analysis'):
            print("❌ Таблица ai_analysis не найдена")
            return None
        
        # Полностью неанализированные записи
        query = """
        SELECT s.id, s.artist, s.title
        FROM songs s
        WHERE s.lyrics IS NOT NULL 
        AND s.lyrics != '' 
        AND s.id NOT IN (SELECT DISTINCT song_id FROM ai_analysis)
        ORDER BY s.id
        LIMIT ?
        """
        
        unanalyzed = self.conn.execute(query, (limit,)).fetchall()
        
        print(f"📋 Первые {limit} полностью неанализированных записей:")
        if unanalyzed:
            for i, (song_id, artist, title) in enumerate(unanalyzed, 1):
                print(f"  {i:2d}. ID: {song_id} | {artist} - {title}")
            
            first_id = unanalyzed[0][0]
            print(f"\n🎯 Первая неанализированная: ID {first_id}")
            print(f"💡 Рекомендуемая команда:")
            print(f"   python scripts/mass_qwen_analysis.py --start-offset {first_id - 1}")
            
            return first_id
        else:
            print("  ✅ Все записи проанализированы!")
            return None
    
    def quick_check(self):
        """Быстрая проверка основных метрик"""
        print("⚡ БЫСТРАЯ ПРОВЕРКА")
        print("=" * 30)
        
        # Основные цифры
        total_songs = self.conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
        songs_with_lyrics = self.conn.execute(
            "SELECT COUNT(*) FROM songs WHERE lyrics IS NOT NULL AND lyrics != ''"
        ).fetchone()[0]
        
        print(f"🎵 Песен: {total_songs:,} (с текстами: {songs_with_lyrics:,})")
        
        if self._table_exists('ai_analysis'):
            analyzed = self.conn.execute("SELECT COUNT(DISTINCT song_id) FROM ai_analysis").fetchone()[0]
            coverage = analyzed / songs_with_lyrics * 100 if songs_with_lyrics > 0 else 0
            print(f"🤖 Анализ: {analyzed:,}/{songs_with_lyrics:,} ({coverage:.1f}%)")
        
        if self._table_exists('spotify_tracks'):
            spotify = self.conn.execute("SELECT COUNT(*) FROM spotify_tracks").fetchone()[0]
            spotify_coverage = spotify / songs_with_lyrics * 100 if songs_with_lyrics > 0 else 0
            print(f"🎵 Spotify: {spotify:,}/{songs_with_lyrics:,} ({spotify_coverage:.1f}%)")
        
        # Размер БД
        db_size = self.db_path.stat().st_size / (1024 * 1024)
        print(f"💾 Размер БД: {db_size:.1f} MB")
    
    def _table_exists(self, table_name):
        """Проверка существования таблицы"""
        result = self.conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """, (table_name,)).fetchone()
        return result is not None


def main():
    """Главная функция с обработкой аргументов"""
    parser = argparse.ArgumentParser(
        description='Unified Database Diagnostics Tool',
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
    diagnostics = DatabaseDiagnostics()
    
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
