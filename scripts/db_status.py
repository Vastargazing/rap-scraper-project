#!/usr/bin/env python3
"""
📊 Быстрая проверка статуса анализа базы данных

ОПИСАНИЕ РАБОТЫ:
Этот скрипт предназначен для быстрого анализа состояния AI-анализа в базе данных.
Показывает статистику по проанализированным и неанализированным записям,
распределение по моделям анализа и рекомендации для продолжения анализа.

ОСНОВНЫЕ ФУНКЦИИ:
- Подсчет общего количества песен с текстами
- Статистика проанализированных/неанализированных записей
- Распределение анализов по моделям (Qwen, Gemma и др.)
- Список первых неанализированных записей
- Рекомендации команд для запуска массового анализа

ОТЛИЧИЯ ОТ check_db.py:
- Фокус на статусе AI-анализа (не на общей структуре базы)
- Показывает неанализированные записи и команды для запуска
- Более детальная статистика по моделям анализа
- Автономный (не зависит от других модулей проекта)
- Красивый вывод с эмодзи и подсказками

ИСПОЛЬЗОВАНИЕ:
python scripts/db_status.py

АВТОР: AI Assistant
ДАТА: Август 2025
"""

import sqlite3
from pathlib import Path

def show_analysis_status():
    """
    Основная функция отображения статуса анализа.
    
    Выполняет анализ состояния AI-анализа:
    - Подсчитывает общее количество песен с текстами
    - Определяет количество проанализированных и неанализированных записей
    - Показывает распределение по моделям анализа
    - Находит первые неанализированные записи
    - Дает рекомендации по запуску анализа
    """
    
    project_root = Path(__file__).parent.parent
    db_path = project_root / 'data' / 'rap_lyrics.db'
    conn = sqlite3.connect(str(db_path))
    
    print("📊 СТАТУС АНАЛИЗА БАЗЫ ДАННЫХ")
    print("=" * 50)
    
    # Общая статистика
    total_songs = conn.execute("SELECT COUNT(*) FROM songs WHERE lyrics IS NOT NULL AND lyrics != ''").fetchone()[0]
    total_analyzed = conn.execute("SELECT COUNT(DISTINCT song_id) FROM ai_analysis").fetchone()[0]
    unanalyzed = total_songs - total_analyzed
    
    print(f"📚 Всего песен с текстами: {total_songs:,}")
    print(f"✅ Проанализировано: {total_analyzed:,} ({(total_analyzed/total_songs)*100:.1f}%)")
    print(f"⏳ Неанализированных: {unanalyzed:,} ({(unanalyzed/total_songs)*100:.1f}%)")
    
    # По моделям
    models = conn.execute("""
        SELECT model_version, COUNT(*) 
        FROM ai_analysis 
        GROUP BY model_version 
        ORDER BY COUNT(*) DESC
    """).fetchall()
    
    print(f"\n🤖 Анализ по моделям:")
    for model, count in models:
        percent = (count / total_analyzed) * 100 if total_analyzed > 0 else 0
        print(f"  {model}: {count:,} ({percent:.1f}%)")
    
    # Первые неанализированные
    unanalyzed_records = conn.execute("""
        SELECT s.id, s.artist, s.title
        FROM songs s
        WHERE s.lyrics IS NOT NULL 
        AND s.lyrics != '' 
        AND s.id NOT IN (SELECT DISTINCT song_id FROM ai_analysis)
        ORDER BY s.id
        LIMIT 5
    """).fetchall()
    
    if unanalyzed_records:
        print(f"\n📋 Первые неанализированные записи:")
        for song_id, artist, title in unanalyzed_records:
            print(f"  ID: {song_id} | {artist} - {title}")
        
        first_id = unanalyzed_records[0][0]
        print(f"\n💡 Для анализа используйте:")
        print(f"python scripts/qwen_mass_analysis.py --start-from {first_id}")
        print(f"python scripts/qwen_mass_analysis.py --start-from {first_id} --max-records 100")
    else:
        print(f"\n✅ Все записи проанализированы!")
    
    conn.close()

if __name__ == "__main__":
    # Точка входа для запуска скрипта из командной строки
    show_analysis_status()
