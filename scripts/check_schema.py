#!/usr/bin/env python3
"""
🔍 Проверка схемы базы данных
"""

import sqlite3
from pathlib import Path

def check_schema():
    """Проверяет схему ai_analysis таблицы"""
    
    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'rap_lyrics.db'
    conn = sqlite3.connect(str(db_path))
    
    # Получаем CREATE TABLE
    result = conn.execute("""
        SELECT sql FROM sqlite_master 
        WHERE type='table' AND name='ai_analysis'
    """).fetchone()
    
    if result:
        print("🗂️ CREATE TABLE для ai_analysis:")
        print(result[0])
    
    # Проверяем constraint
    print("\n🔒 Проверяем UNIQUE constraint:")
    try:
        # Попробуем вставить дублирующуюся запись
        conn.execute("INSERT INTO ai_analysis (song_id, model_version) VALUES (1, 'test')")
        conn.execute("INSERT INTO ai_analysis (song_id, model_version) VALUES (1, 'test2')")
        print("✅ Множественные записи для одной песни разрешены")
        conn.rollback()
    except sqlite3.IntegrityError as e:
        print(f"❌ UNIQUE constraint: {e}")
        print("🎯 Проблема: можно анализировать песню только одной моделью")
    
    conn.close()

def find_first_unanalyzed():
    """Находит первую реально неанализированную запись"""
    
    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'rap_lyrics.db'
    conn = sqlite3.connect(str(db_path))
    
    # Ищем записи, не проанализированные вообще
    query = """
    SELECT s.id, s.artist, s.title
    FROM songs s
    WHERE s.lyrics IS NOT NULL 
    AND s.lyrics != '' 
    AND s.id NOT IN (SELECT DISTINCT song_id FROM ai_analysis)
    ORDER BY s.id
    LIMIT 10
    """
    
    records = conn.execute(query).fetchall()
    
    print(f"\n📋 Первые 10 полностью неанализированных записей:")
    if records:
        for i, (song_id, artist, title) in enumerate(records, 1):
            print(f"  {i}. ID: {song_id} | {artist} - {title}")
        
        print(f"\n🎯 Первая полностью неанализированная: ID {records[0][0]}")
        print(f"💡 Используйте: --start-offset {records[0][0] - 1}")
    else:
        print("  ✅ Все записи проанализированы хотя бы одной моделью")
    
    conn.close()

if __name__ == "__main__":
    check_schema()
    find_first_unanalyzed()
