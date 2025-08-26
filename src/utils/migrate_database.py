#!/usr/bin/env python3
"""
Скрипт для миграции существующей базы данных к новой схеме с метаданными
"""
import sqlite3
import os
import shutil
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def migrate_database(source_db: str = "rap_lyrics.db", backup: bool = True):
    """Миграция базы данных к новой схеме"""
    
    if not os.path.exists(source_db):
        logger.error(f"❌ База данных {source_db} не найдена!")
        return False
    
    logger.info(f"🔄 Начинаем миграцию базы данных: {source_db}")
    
    # Создание резервной копии
    if backup:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"backup_{source_db}_{timestamp}"
        shutil.copy2(source_db, backup_name)
        logger.info(f"💾 Создана резервная копия: {backup_name}")
    
    try:
        conn = sqlite3.connect(source_db)
        cursor = conn.cursor()
        
        # Проверяем текущую схему
        cursor.execute("PRAGMA table_info(songs)")
        current_columns = {row[1]: row[2] for row in cursor.fetchall()}
        logger.info(f"📋 Текущие колонки: {list(current_columns.keys())}")
        
        # Новые колонки для добавления
        new_columns = [
            ("genre", "TEXT"),
            ("release_date", "TEXT"),
            ("album", "TEXT"),
            ("language", "TEXT"),
            ("explicit", "BOOLEAN"),
            ("song_art_url", "TEXT"),
            ("popularity_score", "INTEGER"),
            ("lyrics_quality_score", "REAL")
        ]
        
        # Добавляем новые колонки
        added_columns = []
        for col_name, col_type in new_columns:
            if col_name not in current_columns:
                try:
                    cursor.execute(f"ALTER TABLE songs ADD COLUMN {col_name} {col_type}")
                    added_columns.append(col_name)
                    logger.info(f"✅ Добавлена колонка: {col_name} ({col_type})")
                except sqlite3.OperationalError as e:
                    logger.warning(f"⚠️ Не удалось добавить колонку {col_name}: {e}")
        
        # Создаем новые индексы
        new_indexes = [
            "CREATE INDEX IF NOT EXISTS idx_genre ON songs(genre)",
            "CREATE INDEX IF NOT EXISTS idx_release_date ON songs(release_date)",
            "CREATE INDEX IF NOT EXISTS idx_quality ON songs(lyrics_quality_score)",
            "CREATE INDEX IF NOT EXISTS idx_word_count ON songs(word_count)"
        ]
        
        for index_sql in new_indexes:
            try:
                cursor.execute(index_sql)
                logger.info(f"✅ Создан индекс: {index_sql.split('idx_')[1].split(' ')[0]}")
            except sqlite3.OperationalError as e:
                logger.warning(f"⚠️ Индекс уже существует или ошибка: {e}")
        
        # Включаем WAL режим для лучшей производительности
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-2000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        
        logger.info("🚀 Включен WAL режим для оптимизации")
        
        # Обновляем lyrics_quality_score для существующих записей
        if 'lyrics_quality_score' in added_columns:
            logger.info("📊 Рассчитываем качество для существующих текстов...")
            
            cursor.execute("SELECT id, lyrics FROM songs WHERE lyrics_quality_score IS NULL LIMIT 1000")
            rows = cursor.fetchall()
            
            updated = 0
            for row_id, lyrics in rows:
                quality_score = calculate_lyrics_quality(lyrics)
                cursor.execute("UPDATE songs SET lyrics_quality_score = ? WHERE id = ?", (quality_score, row_id))
                updated += 1
                
                if updated % 100 == 0:
                    conn.commit()
                    logger.info(f"📈 Обновлено качество для {updated} песен...")
            
            conn.commit()
            logger.info(f"✅ Обновлено качество для {updated} песен")
        
        # Финальная статистика
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT artist) as artists,
                AVG(word_count) as avg_words,
                COUNT(CASE WHEN lyrics_quality_score IS NOT NULL THEN 1 END) as with_quality
            FROM songs
        """)
        stats = cursor.fetchone()
        
        conn.commit()
        conn.close()
        
        avg_words = stats[2] if stats[2] else 0.0
        logger.info(f"""
        ✅ МИГРАЦИЯ ЗАВЕРШЕНА!
        📊 Статистика:
           • Всего песен: {stats[0]}
           • Артистов: {stats[1]}
           • Среднее слов: {avg_words:.1f}
           • С оценкой качества: {stats[3]}
           • Добавлено колонок: {len(added_columns)}
        """)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка миграции: {e}")
        return False

def calculate_lyrics_quality(lyrics: str) -> float:
    """Расчет качества текста (копия из основного скрипта)"""
    if not lyrics:
        return 0.0
    
    score = 0.0
    words = lyrics.split()
    
    # Длина текста
    if len(words) > 50:
        score += 0.3
    if len(words) > 100:
        score += 0.2
    
    # Разнообразие слов
    unique_words = len(set(word.lower() for word in words))
    if len(words) > 0:
        diversity = unique_words / len(words)
        score += diversity * 0.3
    
    # Отсутствие инструментальных маркеров
    instrumental_markers = ["instrumental", "no lyrics", "без слов"]
    if not any(marker in lyrics.lower() for marker in instrumental_markers):
        score += 0.2
    
    return min(score, 1.0)

def main():
    print("🔄 Миграция базы данных rap_lyrics")
    print("=" * 50)
    
    db_file = "rap_lyrics.db"
    
    if not os.path.exists(db_file):
        print(f"❌ Файл {db_file} не найден!")
        return
    
    # Получаем размер файла
    file_size_mb = os.path.getsize(db_file) / (1024 * 1024)
    print(f"📁 База данных: {db_file} ({file_size_mb:.1f} МБ)")
    
    response = input("\n❓ Начать миграцию? Будет создана резервная копия (y/N): ")
    if response.lower() != 'y':
        print("❌ Отменено пользователем")
        return
    
    success = migrate_database(db_file)
    
    if success:
        print("\n✅ Миграция успешно завершена!")
        print("💡 Теперь можно использовать rap_scraper_optimized.py")
        print("📁 Резервная копия сохранена автоматически")
    else:
        print("❌ Ошибка при миграции")

if __name__ == "__main__":
    main()
