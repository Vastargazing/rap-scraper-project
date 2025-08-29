"""
Утилита для проверки и обновления структуры базы данных.
"""

import sqlite3
import sys
from pathlib import Path

def check_database_structure():
    """Проверка текущей структуры БД"""
    db_path = "data/rap_lyrics.db"
    
    if not Path(db_path).exists():
        print(f"❌ База данных не найдена: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Получаем список таблиц
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"📋 Существующие таблицы в {db_path}:")
        for table in tables:
            print(f"   - {table}")
            
            # Показываем структуру каждой таблицы
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"     └─ {col[1]} ({col[2]})")
        
        # Проверяем, какие таблицы нужны для новой архитектуры
        required_tables = ['songs', 'enhanced_songs', 'artists', 'analysis_queue']
        missing_tables = [table for table in required_tables if table not in tables]
        
        if missing_tables:
            print(f"\n⚠️ Отсутствующие таблицы: {', '.join(missing_tables)}")
            return False
        else:
            print(f"\n✅ Все необходимые таблицы присутствуют")
            return True
            
    except Exception as e:
        print(f"❌ Ошибка при проверке БД: {e}")
        return False
    finally:
        conn.close()

def update_database_structure():
    """Обновление структуры БД до новой архитектуры"""
    db_path = "data/rap_lyrics.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🔧 Обновление структуры базы данных...")
        
        # Enhanced songs table для результатов анализа
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enhanced_songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id INTEGER NOT NULL,
                analyzer_type TEXT NOT NULL,
                analysis_data TEXT,  -- JSON data
                confidence REAL,
                processing_time REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (song_id) REFERENCES songs (id),
                UNIQUE(song_id, analyzer_type)
            )
        """)
        
        # Artists table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                genius_id INTEGER,
                url TEXT,
                metadata TEXT,  -- JSON data
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Analysis queue table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS analysis_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                song_id INTEGER NOT NULL,
                analyzer_type TEXT NOT NULL,
                priority INTEGER DEFAULT 5,
                status TEXT DEFAULT 'pending',
                attempts INTEGER DEFAULT 0,
                last_attempt TIMESTAMP,
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (song_id) REFERENCES songs (id)
            )
        """)
        
        # Создание индексов для лучшей производительности
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_songs_artist ON songs(artist)",
            "CREATE INDEX IF NOT EXISTS idx_songs_title ON songs(title)",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_songs_song_id ON enhanced_songs(song_id)",
            "CREATE INDEX IF NOT EXISTS idx_enhanced_songs_analyzer ON enhanced_songs(analyzer_type)",
            "CREATE INDEX IF NOT EXISTS idx_queue_status ON analysis_queue(status)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        print("✅ Структура базы данных успешно обновлена")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при обновлении БД: {e}")
        return False
    finally:
        conn.close()

def show_statistics():
    """Показать статистику по базе данных"""
    db_path = "data/rap_lyrics.db"
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("\n📊 Статистика базы данных:")
        
        # Общее количество песен
        cursor.execute("SELECT COUNT(*) FROM songs")
        total_songs = cursor.fetchone()[0]
        print(f"   Всего песен: {total_songs}")
        
        # Песни с текстами
        cursor.execute("SELECT COUNT(*) FROM songs WHERE lyrics IS NOT NULL AND lyrics != ''")
        songs_with_lyrics = cursor.fetchone()[0]
        print(f"   С текстами: {songs_with_lyrics}")
        
        # Проверяем новые таблицы
        try:
            cursor.execute("SELECT COUNT(*) FROM enhanced_songs")
            enhanced_count = cursor.fetchone()[0]
            print(f"   Проанализированных: {enhanced_count}")
            
            cursor.execute("SELECT analyzer_type, COUNT(*) FROM enhanced_songs GROUP BY analyzer_type")
            analysis_types = cursor.fetchall()
            if analysis_types:
                print("   По типам анализа:")
                for analyzer, count in analysis_types:
                    print(f"     - {analyzer}: {count}")
                    
        except sqlite3.OperationalError:
            print("   Новые таблицы еще не созданы")
        
    except Exception as e:
        print(f"❌ Ошибка получения статистики: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    print("🔍 Проверка структуры базы данных rap-scraper-project\n")
    
    # Проверяем текущую структуру
    if not check_database_structure():
        print("\n🔧 Требуется обновление структуры БД...")
        if update_database_structure():
            print("✅ База данных готова к работе с новой архитектурой")
        else:
            print("❌ Не удалось обновить базу данных")
            sys.exit(1)
    
    # Показываем статистику
    show_statistics()
    
    print("\n🎉 База данных готова к работе!")
