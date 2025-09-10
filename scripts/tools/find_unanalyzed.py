#!/usr/bin/env python3
"""
🔍 Поиск неанализированных записей в базе данных
"""

import sqlite3
import os
from pathlib import Path

def find_unanalyzed_records():
    """Находит первые неанализированные записи"""
    
    # Подключение к БД
    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'rap_lyrics.db'
    conn = sqlite3.connect(str(db_path))
    
    print("🔍 Поиск неанализированных записей...")
    
    # Находим первые 10 неанализированных записей
    query = """
    SELECT s.id, s.artist, s.title, 
           CASE WHEN a.song_id IS NOT NULL THEN 'Analyzed' ELSE 'Not analyzed' END as status
    FROM songs s
    LEFT JOIN ai_analysis a ON s.id = a.song_id AND a.model_version LIKE '%qwen%'
    WHERE s.lyrics IS NOT NULL 
    AND s.lyrics != '' 
    AND a.song_id IS NULL
    ORDER BY s.id
    LIMIT 10
    """
    
    records = conn.execute(query).fetchall()
    
    if records:
        print(f"\n📋 Первые 10 неанализированных записей:")
        for i, (song_id, artist, title, status) in enumerate(records, 1):
            print(f"  {i}. ID: {song_id} | {artist} - {title}")
        
        print(f"\n🎯 Первая неанализированная запись: ID {records[0][0]}")
        return records[0][0]
    else:
        print("✅ Все записи проанализированы!")
        return None
    
    conn.close()

def get_analysis_stats():
    """Получает статистику анализа"""
    
    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'rap_lyrics.db'
    conn = sqlite3.connect(str(db_path))
    
    # Общая статистика
    total_songs = conn.execute("SELECT COUNT(*) FROM songs WHERE lyrics IS NOT NULL AND lyrics != ''").fetchone()[0]
    total_analyzed = conn.execute("SELECT COUNT(DISTINCT song_id) FROM ai_analysis").fetchone()[0]
    qwen_analyzed = conn.execute("SELECT COUNT(*) FROM ai_analysis WHERE model_version LIKE '%qwen%'").fetchone()[0]
    
    print(f"\n📊 Статистика базы данных:")
    print(f"  📚 Всего песен с текстами: {total_songs:,}")
    print(f"  ✅ Проанализированные (любой моделью): {total_analyzed:,}")
    print(f"  🤖 Проанализированные Qwen: {qwen_analyzed:,}")
    print(f"  ⏳ Неанализированные: {total_songs - total_analyzed:,}")
    
    # Модели
    models = conn.execute("""
        SELECT model_version, COUNT(*) 
        FROM ai_analysis 
        GROUP BY model_version 
        ORDER BY COUNT(*) DESC
    """).fetchall()
    
    print(f"\n🤖 По моделям:")
    for model, count in models:
        print(f"  {model}: {count:,}")
    
    conn.close()

if __name__ == "__main__":
    get_analysis_stats()
    first_unanalyzed_id = find_unanalyzed_records()
    
    if first_unanalyzed_id:
        print(f"\n💡 Рекомендация: Используйте --start-offset {first_unanalyzed_id - 1} для быстрого начала")
