#!/usr/bin/env python3
"""
🔍 Модуль проверки состояния базы данных

ОПИСАНИЕ РАБОТЫ:
Этот модуль содержит функции для комплексной проверки состояния базы данных rap_lyrics.db.
Проверяет структуру, содержимое и статистику всех таблиц.

ОСНОВНЫЕ ФУНКЦИИ:
1. check_database():
   - Проверка существования файла базы данных
   - Анализ структуры таблиц (songs, ai_analysis)
   - Статистика по песням: общее количество, уникальные артисты
   - Информация о размере файла базы данных
   - Топ-5 артистов по количеству песен
   - Последние добавленные песни
   - Статистика AI-анализов (если таблица существует)
   - Последние проведенные анализы

ОТЛИЧИЯ ОТ db_status.py:
- Более широкая проверка всей базы (не только анализа)
- Не фокусируется на неанализированных записях
- Не дает рекомендаций по запуску анализа
- Использует конфигурацию проекта (DB_PATH)
- Более детальная информация о структуре базы

ЗАВИСИМОСТИ:
- sqlite3 для работы с базой данных
- src.utils.config.DB_PATH для пути к файлу базы

ИСПОЛЬЗОВАНИЕ:
from src.utils.check_db import check_database, main
check_database()  # Основная функция
main()           # Точка входа для CLI

АВТОР: AI Assistant
ДАТА: Август 2025
"""

import sqlite3
import os
from .config import DB_PATH

def check_database():
    """
    Основная функция проверки состояния базы данных.
    
    Выполняет комплексную диагностику:
    - Проверяет существование файла БД
    - Анализирует структуру таблиц
    - Собирает статистику по песням и артистам
    - Показывает информацию о AI-анализах
    """
    db_file = str(DB_PATH)
    
    if not os.path.exists(db_file):
        print(f"❌ Файл {db_file} не найден!")
        return
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Проверяем таблицы
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📋 Таблицы в базе: {tables}")
        
        if 'songs' in tables:
            # Общая статистика
            cursor.execute("SELECT COUNT(*) FROM songs")
            total_songs = cursor.fetchone()[0]
            print(f"🎵 Всего песен: {total_songs}")
            
            # Количество артистов
            cursor.execute("SELECT COUNT(DISTINCT artist) FROM songs")
            unique_artists = cursor.fetchone()[0]
            print(f"👤 Уникальных артистов: {unique_artists}")
            
            # Размер базы
            file_size = os.path.getsize(db_file) / (1024 * 1024)  # В МБ
            print(f"💾 Размер файла: {file_size:.2f} МБ")
            
            # Топ 5 артистов по количеству песен
            cursor.execute("""
                SELECT artist, COUNT(*) as song_count 
                FROM songs 
                GROUP BY artist 
                ORDER BY song_count DESC 
                LIMIT 5
            """)
            top_artists = cursor.fetchall()
            print(f"\n🏆 Топ 5 артистов:")
            for artist, count in top_artists:
                print(f"  • {artist}: {count} песен")
            
            # Последние добавленные
            cursor.execute("""
                SELECT artist, title, scraped_date 
                FROM songs 
                ORDER BY id DESC 
                LIMIT 3
            """)
            recent = cursor.fetchall()
            print(f"\n🕒 Последние добавленные:")
            for artist, title, date in recent:
                print(f"  • {artist} - {title} ({date})")
        
        # Проверяем AI анализ если таблица существует
        if 'ai_analysis' in tables:
            cursor.execute("SELECT COUNT(*) FROM ai_analysis")
            analyzed_count = cursor.fetchone()[0]
            print(f"\n🤖 AI анализов: {analyzed_count}")
            
            if analyzed_count > 0:
                # Последние анализы
                cursor.execute("""
                    SELECT s.artist, s.title, a.genre, a.overall_quality, a.analysis_date
                    FROM ai_analysis a
                    JOIN songs s ON a.song_id = s.id
                    ORDER BY a.id DESC
                    LIMIT 3
                """)
                analyses = cursor.fetchall()
                print(f"🔍 Последние анализы:")
                for artist, title, genre, quality, date in analyses:
                    print(f"  • {artist} - {title} | {genre} | {quality} ({date[:10]})")
        else:
            print(f"\n⚠️ Таблица ai_analysis не найдена")
        
        conn.close()
        print(f"\n✅ База данных проверена успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка при проверке базы: {e}")

def main():
    """Точка входа для запуска из командной строки."""
    check_database()

if __name__ == "__main__":
    # Точка входа для запуска модуля напрямую
    main()
