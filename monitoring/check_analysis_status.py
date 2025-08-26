#!/usr/bin/env python3
"""
Быстрая проверка статуса анализов в базе данных
"""

import sqlite3
import os
from datetime import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.utils.config import DB_PATH

def check_analysis_status(db_path=None):
    """Проверяем статус анализов в базе данных"""
    
    if db_path is None:
        db_path = str(DB_PATH)
    
    if not os.path.exists(db_path):
        print("❌ База данных не найдена!")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Общая статистика
        cursor.execute("SELECT COUNT(*) FROM songs WHERE lyrics IS NOT NULL")
        total_songs = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM ai_analysis")
        total_analyses = cursor.fetchone()[0]
        
        print(f"📊 Общая статистика:")
        print(f"   🎵 Всего песен с текстами: {total_songs}")
        print(f"   🤖 Всего анализов: {total_analyses}")
        
        # Статистика по моделям
        cursor.execute("""
            SELECT model_version, COUNT(*) 
            FROM ai_analysis 
            GROUP BY model_version 
            ORDER BY COUNT(*) DESC
        """)
        
        print(f"\n📈 Анализы по моделям:")
        for model, count in cursor.fetchall():
            print(f"   {model}: {count} анализов")
        
        # Последние анализы
        cursor.execute("""
            SELECT a.analysis_date, a.model_version, s.artist, s.title
            FROM ai_analysis a
            JOIN songs s ON a.song_id = s.id
            ORDER BY a.id DESC
            LIMIT 5
        """)
        
        print(f"\n🕐 Последние анализы:")
        for date, model, artist, title in cursor.fetchall():
            date_str = date[:16] if date else "Unknown"
            print(f"   {date_str} | {model} | {artist} - {title}")
        
        # Проверяем специально Gemma анализы
        cursor.execute("""
            SELECT COUNT(*) FROM ai_analysis 
            WHERE model_version LIKE 'gemma%'
        """)
        gemma_count = cursor.fetchone()[0]
        
        print(f"\n🔥 Gemma анализы: {gemma_count}")
        
        if gemma_count > 0:
            cursor.execute("""
                SELECT model_version, COUNT(*) 
                FROM ai_analysis 
                WHERE model_version LIKE 'gemma%'
                GROUP BY model_version
            """)
            for model, count in cursor.fetchall():
                print(f"   {model}: {count}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    print("🔍 Проверка статуса анализов...")
    check_analysis_status()
