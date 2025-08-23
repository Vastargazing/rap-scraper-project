#!/usr/bin/env python3
"""
Быстрый тест анализатора
"""

import sqlite3
import json
from datetime import datetime

def test_analysis_results():
    """Проверяем результаты анализа в БД"""
    
    print("🧪 Тестирование результатов AI анализа")
    print("=" * 50)
    
    try:
        conn = sqlite3.connect("rap_lyrics.db")
        conn.row_factory = sqlite3.Row
        
        # Проверяем наличие анализов
        cursor = conn.execute("SELECT COUNT(*) as count FROM ai_analysis")
        total_analyses = cursor.fetchone()['count']
        
        print(f"📊 Всего анализов в БД: {total_analyses}")
        
        if total_analyses == 0:
            print("❌ Анализы не найдены. Запустите multi_model_analyzer.py")
            return
        
        # Показываем последние анализы
        cursor = conn.execute("""
            SELECT 
                s.artist,
                s.title,
                a.genre,
                a.mood,
                a.energy_level,
                a.structure,
                a.complexity_level,
                a.overall_quality,
                a.authenticity_score,
                a.lyrical_creativity,
                a.commercial_appeal,
                a.uniqueness,
                a.ai_likelihood,
                a.model_version,
                a.analysis_date
            FROM ai_analysis a
            JOIN songs s ON a.song_id = s.id
            ORDER BY a.id DESC
            LIMIT 5
        """)
        
        analyses = cursor.fetchall()
        
        print(f"\n🎯 Последние {len(analyses)} анализов:")
        print("-" * 80)
        
        for analysis in analyses:
            print(f"🎵 {analysis['artist']} - {analysis['title']}")
            print(f"   📁 Жанр: {analysis['genre']}")
            print(f"   😊 Настроение: {analysis['mood']} | Энергия: {analysis['energy_level']}")
            print(f"   🏗️  Структура: {analysis['structure']} | Сложность: {analysis['complexity_level']}")
            print(f"   ⭐ Качество: {analysis['overall_quality']}")
            
            print(f"   📈 Метрики:")
            print(f"      Аутентичность: {analysis['authenticity_score']:.2f}")
            print(f"      Креативность: {analysis['lyrical_creativity']:.2f}")
            print(f"      Коммерческий потенциал: {analysis['commercial_appeal']:.2f}")
            print(f"      Уникальность: {analysis['uniqueness']:.2f}")
            print(f"      Вероятность ИИ: {analysis['ai_likelihood']:.2f}")
            
            print(f"   🤖 Модель: {analysis['model_version']}")
            print(f"   📅 Дата: {analysis['analysis_date'][:10]}")
            print("-" * 80)
        
        # Статистика по жанрам
        cursor = conn.execute("""
            SELECT genre, COUNT(*) as count
            FROM ai_analysis
            GROUP BY genre
            ORDER BY count DESC
        """)
        
        genres = cursor.fetchall()
        print(f"\n📊 Распределение по жанрам:")
        for genre in genres:
            print(f"   {genre['genre']}: {genre['count']} песен")
        
        # Средние оценки
        cursor = conn.execute("""
            SELECT 
                AVG(authenticity_score) as avg_auth,
                AVG(lyrical_creativity) as avg_creat,
                AVG(commercial_appeal) as avg_comm,
                AVG(uniqueness) as avg_uniq,
                AVG(ai_likelihood) as avg_ai
            FROM ai_analysis
        """)
        
        avg_scores = cursor.fetchone()
        print(f"\n📈 Средние оценки:")
        print(f"   Аутентичность: {avg_scores['avg_auth']:.3f}")
        print(f"   Креативность: {avg_scores['avg_creat']:.3f}")
        print(f"   Коммерческий потенциал: {avg_scores['avg_comm']:.3f}")
        print(f"   Уникальность: {avg_scores['avg_uniq']:.3f}")
        print(f"   Вероятность ИИ: {avg_scores['avg_ai']:.3f}")
        
        conn.close()
        print(f"\n✅ Тест завершен успешно!")
        
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")

def check_database_schema():
    """Проверяем схему базы данных"""
    
    print("\n🔍 Проверка схемы базы данных:")
    
    try:
        conn = sqlite3.connect("rap_lyrics.db")
        cursor = conn.cursor()
        
        # Получаем информацию о таблице ai_analysis
        cursor.execute("PRAGMA table_info(ai_analysis)")
        columns = cursor.fetchall()
        
        if columns:
            print("📋 Столбцы таблицы ai_analysis:")
            for col in columns:
                print(f"   {col[1]} ({col[2]})")
        else:
            print("❌ Таблица ai_analysis не найдена")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Ошибка при проверке схемы: {e}")

if __name__ == "__main__":
    test_analysis_results()
    check_database_schema()
