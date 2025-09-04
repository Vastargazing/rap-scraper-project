#!/usr/bin/env python3
"""
🎯 Массовый анализ базы данных с Qwen анализатором
"""

import sys
import os
import asyncio
import time
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Добавляем корневую папку в path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.core.app import create_app
from src.interfaces.analyzer_interface import AnalyzerFactory
from src.interfaces.database_interface import create_database_manager

async def mass_analyze_database(batch_size: int = 100, max_records: int = None):
    """
    Массовый анализ всей базы данных с Qwen анализатором
    
    Args:
        batch_size: Размер батча для обработки
        max_records: Максимальное количество записей (None = все)
    """
    print("🎵 Rap Scraper - Mass Database Analysis with Qwen")
    print("=" * 60)
    
    # Инициализация приложения
    app = create_app()
    
    # Получение анализатора
    analyzer = AnalyzerFactory.create("qwen")
    if not analyzer.available:
        print("❌ Qwen анализатор недоступен!")
        return
    
    print(f"✅ Qwen анализатор готов: {analyzer.model_name}")
    
    # Подключение к базе данных
    import sqlite3
    db_path = os.path.join(project_root, 'data', 'rap_lyrics.db')
    db_connection = sqlite3.connect(db_path)
    
    # Получение всех песен без анализа Qwen (оптимизированный запрос)
    print("📊 Подсчет записей для анализа...")
    
    # Сначала подсчитываем количество неанализированных записей
    count_query = """
    SELECT COUNT(*) 
    FROM songs 
    WHERE lyrics IS NOT NULL 
    AND lyrics != '' 
    AND id NOT IN (
        SELECT DISTINCT song_id 
        FROM ai_analysis 
        WHERE model_version LIKE '%qwen%'
    )
    """
    
    total_unanalyzed = db_connection.execute(count_query).fetchone()[0]
    print(f"📈 Найдено {total_unanalyzed} неанализированных записей")
    
    if total_unanalyzed == 0:
        print("✅ Все записи уже проанализированы!")
        db_connection.close()
        return
    
    # Теперь получаем записи с OFFSET для пропуска уже проанализированных
    # Используем более эффективный подход - выбираем из неанализированных напрямую
    query = """
    SELECT s.id, s.artist, s.title, s.lyrics 
    FROM songs s
    LEFT JOIN ai_analysis a ON s.id = a.song_id AND a.model_version LIKE '%qwen%'
    WHERE s.lyrics IS NOT NULL 
    AND s.lyrics != '' 
    AND a.song_id IS NULL
    ORDER BY s.id
    """
    
    if max_records:
        actual_limit = min(max_records, total_unanalyzed)
        query += f" LIMIT {actual_limit}"
        print(f"🎯 Ограничиваем анализ до {actual_limit} записей")
    else:
        actual_limit = total_unanalyzed
    
    cursor = db_connection.execute(query)
    records = cursor.fetchall()
    total_records = len(records)
    
    print(f"📈 Загружено {total_records} записей для анализа")
    
    if total_records == 0:
        print("✅ Все записи уже проанализированы!")
        return
    
    # Подтверждение
    print(f"\n⚠️  ВНИМАНИЕ: Будет проанализировано {total_records} записей")
    print(f"⏱️  Примерное время: {(total_records * 15) // 60} минут")
    print(f"🌟 Бесплатная модель Qwen через Novita AI - без затрат!")
    
    confirm = input("\nПродолжить? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Анализ отменен")
        return
    
    # Массовый анализ
    start_time = time.time()
    processed = 0
    errors = 0
    
    print(f"\n🚀 Начинаем массовый анализ...")
    print(f"📦 Размер батча: {batch_size}")
    
    for i in range(0, total_records, batch_size):
        batch = records[i:i+batch_size]
        batch_start = time.time()
        
        print(f"\n📦 Батч {i//batch_size + 1}/{(total_records + batch_size - 1)//batch_size}")
        print(f"📊 Записи {i+1}-{min(i+batch_size, total_records)} из {total_records}")
        
        for record in batch:
            track_id, artist, title, lyrics = record
            
            try:
                print(f"🎵 Анализируем: {artist} - {title}")
                
                # Выполняем анализ
                result = analyzer.analyze_song(artist, title, lyrics)
                
                # Извлекаем данные из результата анализа
                raw_output = result.raw_output
                
                # Подготавливаем данные для вставки в ai_analysis таблицу
                genre = raw_output.get('genre_analysis', {}).get('primary_genre', 'unknown')
                subgenre = raw_output.get('genre_analysis', {}).get('subgenre', 'unknown')
                mood = raw_output.get('mood_analysis', {}).get('primary_mood', 'neutral')
                energy_level = raw_output.get('mood_analysis', {}).get('energy_level', 'medium')
                explicit_content = raw_output.get('content_analysis', {}).get('explicit_content', False)
                structure = raw_output.get('technical_analysis', {}).get('structure', 'unknown')
                rhyme_scheme = raw_output.get('technical_analysis', {}).get('rhyme_scheme', 'simple')
                complexity_level = raw_output.get('technical_analysis', {}).get('complexity_level', 'intermediate')
                main_themes = str(raw_output.get('content_analysis', {}).get('main_themes', []))
                emotional_tone = raw_output.get('mood_analysis', {}).get('valence', 'neutral')
                storytelling_type = raw_output.get('content_analysis', {}).get('narrative_style', 'unknown')
                wordplay_quality = raw_output.get('technical_analysis', {}).get('wordplay_quality', 'basic')
                authenticity_score = raw_output.get('quality_metrics', {}).get('authenticity', 0.5)
                lyrical_creativity = raw_output.get('quality_metrics', {}).get('lyrical_creativity', 0.5)
                commercial_appeal = raw_output.get('quality_metrics', {}).get('commercial_appeal', 0.5)
                uniqueness = raw_output.get('quality_metrics', {}).get('originality', 0.5)
                overall_quality = raw_output.get('quality_metrics', {}).get('overall_quality', 0.5)
                ai_likelihood = raw_output.get('quality_metrics', {}).get('ai_generated_likelihood', 0.5)
                
                # Вставляем в таблицу ai_analysis с обработкой дублирования
                try:
                    db_connection.execute("""
                        INSERT INTO ai_analysis 
                        (song_id, genre, subgenre, mood, year_estimate, energy_level, explicit_content, 
                         structure, rhyme_scheme, complexity_level, main_themes, emotional_tone, 
                         storytelling_type, wordplay_quality, authenticity_score, lyrical_creativity, 
                         commercial_appeal, uniqueness, overall_quality, ai_likelihood, analysis_date, model_version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        track_id, genre[:50], subgenre[:50], mood[:50], '2020s', energy_level[:20], 
                        explicit_content, structure[:50], rhyme_scheme[:30], complexity_level[:30], 
                        main_themes[:200], emotional_tone[:30], storytelling_type[:50], wordplay_quality[:30],
                        authenticity_score, lyrical_creativity, commercial_appeal, uniqueness, 
                        overall_quality, ai_likelihood, datetime.now().isoformat(), 'qwen-3-4b-fp8'
                    ))
                    
                    db_connection.commit()
                    processed += 1
                    
                    print(f"✅ Успешно: {result.sentiment} (confidence: {result.confidence:.2f})")
                    
                except sqlite3.IntegrityError as e:
                    if "UNIQUE constraint failed" in str(e):
                        print(f"⚠️ Запись уже существует, пропускаем")
                        continue
                    else:
                        raise
                
            except Exception as e:
                errors += 1
                print(f"❌ Ошибка: {e}")
                continue
        
        batch_time = time.time() - batch_start
        elapsed_time = time.time() - start_time
        avg_time_per_record = elapsed_time / max(processed, 1)
        remaining_time = (total_records - processed) * avg_time_per_record
        
        print(f"\n📊 Статистика батча:")
        print(f"  ⏱️  Время батча: {batch_time:.1f}с")
        print(f"  ✅ Обработано: {processed}/{total_records}")
        print(f"  ❌ Ошибки: {errors}")
        print(f"  🕐 Оставшееся время: {remaining_time/60:.1f} мин")
        print(f"  📈 Прогресс: {processed/total_records*100:.1f}%")
        
        # Пауза между батчами для снижения нагрузки на API
        await asyncio.sleep(2)
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 АНАЛИЗ ЗАВЕРШЕН!")
    print(f"=" * 50)
    print(f"✅ Успешно обработано: {processed} записей")
    print(f"❌ Ошибок: {errors}")
    print(f"⏱️  Общее время: {total_time/60:.1f} минут")
    print(f"📊 Средняя скорость: {processed/(total_time/60):.1f} записей/мин")
    print(f"💾 Результаты сохранены в базу данных")
    
    # Закрываем соединение с базой данных
    db_connection.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Массовый анализ базы данных с Qwen")
    parser.add_argument("--batch-size", type=int, default=50, help="Размер батча")
    parser.add_argument("--max-records", type=int, help="Максимальное количество записей")
    parser.add_argument("--test", action="store_true", help="Тестовый режим (10 записей)")
    
    args = parser.parse_args()
    
    if args.test:
        args.max_records = 10
        args.batch_size = 5
    
    asyncio.run(mass_analyze_database(args.batch_size, args.max_records))
