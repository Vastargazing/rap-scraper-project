#!/usr/bin/env python3
"""
🎯 Массовый анализ базы данных с Qwen анализатором
Единственный скрипт для анализа всей базы данных
"""

import sys
import os
import time
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Добавляем путь к проекту
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from analyzers.qwen_analyzer import QwenAnalyzer

def analyze_database(max_records: int = None, start_from_id: int = None):
    """
    Анализирует всю базу данных с Qwen анализатором.
    
    Args:
        max_records: Максимальное количество записей для анализа
        start_from_id: ID записи, с которой начать анализ
    """
    
    print("🎵 Qwen Mass Database Analysis")
    print("=" * 40)
    
    # Создание анализатора
    analyzer = QwenAnalyzer()
    
    if not analyzer.available:
        print("❌ Qwen анализатор недоступен!")
        print("💡 Проверьте NOVITA_API_KEY в .env файле")
        return
    
    print(f"✅ Qwen анализатор готов: {analyzer.model_name}")
    
    # Подключение к базе данных
    db_path = project_root / 'data' / 'rap_lyrics.db'
    db_connection = sqlite3.connect(str(db_path))
    
    # Подсчет статистики
    print("📊 Анализ базы данных...")
    
    total_songs = db_connection.execute(
        "SELECT COUNT(*) FROM songs WHERE lyrics IS NOT NULL AND lyrics != ''"
    ).fetchone()[0]
    
    already_analyzed_total = db_connection.execute(
        "SELECT COUNT(DISTINCT song_id) FROM ai_analysis"
    ).fetchone()[0]
    
    already_analyzed_qwen = db_connection.execute(
        "SELECT COUNT(*) FROM ai_analysis WHERE model_version LIKE '%qwen%'"
    ).fetchone()[0]
    
    # Находим записи, которые НЕ анализировались ВООБЩЕ (из-за UNIQUE constraint)
    query = """
    SELECT s.id, s.artist, s.title, s.lyrics
    FROM songs s
    WHERE s.lyrics IS NOT NULL 
    AND s.lyrics != '' 
    AND s.id NOT IN (
        SELECT DISTINCT song_id FROM ai_analysis
    )
    """
    
    # Добавляем условие для начала с определенного ID
    if start_from_id:
        query += f" AND s.id >= {start_from_id}"
    
    query += " ORDER BY s.id"
    
    if max_records:
        query += f" LIMIT {max_records}"
    
    cursor = db_connection.execute(query)
    records = cursor.fetchall()
    total_to_analyze = len(records)
    
    print(f"📈 Статистика:")
    print(f"  📚 Всего песен: {total_songs:,}")
    print(f"  ✅ Проанализировано всего: {already_analyzed_total:,}")
    print(f"  🤖 Проанализировано Qwen: {already_analyzed_qwen:,}")
    print(f"  🎯 Найдено для анализа: {total_to_analyze:,}")
    print(f"  📊 Неанализированных: {total_songs - already_analyzed_total:,}")
    
    if total_to_analyze == 0:
        print("✅ Все доступные записи уже проанализированы!")
        print("\n💡 ИНФОРМАЦИЯ:")
        print("База данных имеет ограничение UNIQUE(song_id), поэтому")
        print("одну песню может анализировать только одна модель.")
        print("Qwen может анализировать только неанализированные записи.")
        db_connection.close()
        return
    
    # Показываем диапазон ID
    if records:
        first_id = records[0][0]
        last_id = records[-1][0]
        print(f"  🔢 Диапазон ID: {first_id} - {last_id}")
    
    # Подтверждение
    print(f"\n⚠️  АНАЛИЗ:")
    print(f"🎯 Записей к анализу: {total_to_analyze:,}")
    print(f"⏱️  Примерное время: {(total_to_analyze * 25) // 3600} часов {((total_to_analyze * 25) % 3600) // 60} минут")
    print(f"🌟 Бесплатная модель Qwen через Novita AI")
    
    confirm = input("\nПродолжить анализ? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Анализ отменен")
        db_connection.close()
        return
    
    # Начинаем анализ
    print(f"\n🚀 Начинаем массовый анализ...")
    print(f"🕐 Начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    processed = 0
    errors = 0
    start_time = time.time()
    last_checkpoint = time.time()
    
    for i, (track_id, artist, title, lyrics) in enumerate(records, 1):
        try:
            print(f"\n📊 [{i:,}/{total_to_analyze:,}] ID:{track_id} | {artist} - {title}")
            
            # Выполняем анализ
            result = analyzer.analyze_song(artist, title, lyrics)
            
            # Извлекаем данные из результата анализа
            raw_output = result.raw_output
            
            # Подготавливаем данные для вставки
            genre = str(raw_output.get('genre_analysis', {}).get('primary_genre', 'unknown'))[:50]
            subgenre = str(raw_output.get('genre_analysis', {}).get('subgenre', 'unknown'))[:50]
            mood = str(raw_output.get('mood_analysis', {}).get('primary_mood', 'neutral'))[:50]
            energy_level = str(raw_output.get('mood_analysis', {}).get('energy_level', 'medium'))[:20]
            explicit_content = bool(raw_output.get('content_analysis', {}).get('explicit_content', False))
            structure = str(raw_output.get('technical_analysis', {}).get('structure', 'unknown'))[:50]
            rhyme_scheme = str(raw_output.get('technical_analysis', {}).get('rhyme_scheme', 'simple'))[:30]
            complexity_level = str(raw_output.get('technical_analysis', {}).get('complexity_level', 'intermediate'))[:30]
            main_themes = str(raw_output.get('content_analysis', {}).get('main_themes', []))[:200]
            emotional_tone = str(raw_output.get('mood_analysis', {}).get('valence', 'neutral'))[:30]
            storytelling_type = str(raw_output.get('content_analysis', {}).get('narrative_style', 'unknown'))[:50]
            wordplay_quality = str(raw_output.get('technical_analysis', {}).get('wordplay_quality', 'basic'))[:30]
            
            # Метрики качества
            authenticity_score = float(raw_output.get('quality_metrics', {}).get('authenticity', 0.5))
            lyrical_creativity = float(raw_output.get('quality_metrics', {}).get('lyrical_creativity', 0.5))
            commercial_appeal = float(raw_output.get('quality_metrics', {}).get('commercial_appeal', 0.5))
            uniqueness = float(raw_output.get('quality_metrics', {}).get('originality', 0.5))
            overall_quality = str(raw_output.get('quality_metrics', {}).get('overall_quality', 0.5))[:20]
            ai_likelihood = float(raw_output.get('quality_metrics', {}).get('ai_generated_likelihood', 0.5))
            
            # Вставляем в таблицу
            try:
                db_connection.execute("""
                    INSERT INTO ai_analysis 
                    (song_id, genre, subgenre, mood, year_estimate, energy_level, explicit_content, 
                     structure, rhyme_scheme, complexity_level, main_themes, emotional_tone, 
                     storytelling_type, wordplay_quality, authenticity_score, lyrical_creativity, 
                     commercial_appeal, uniqueness, overall_quality, ai_likelihood, analysis_date, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    track_id, genre, subgenre, mood, '2020s', energy_level, 
                    explicit_content, structure, rhyme_scheme, complexity_level, 
                    main_themes, emotional_tone, storytelling_type, wordplay_quality,
                    authenticity_score, lyrical_creativity, commercial_appeal, uniqueness, 
                    overall_quality, ai_likelihood, datetime.now().isoformat(), 'qwen-3-4b-fp8'
                ))
                
                db_connection.commit()
                processed += 1
                
                print(f"✅ Успешно! (confidence: {result.confidence:.2f})")
                
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    print(f"⚠️ Запись уже существует, пропускаем")
                    continue
                else:
                    raise
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            errors += 1
            continue
        
        # Статистика каждые 50 записей или каждые 10 минут
        current_time = time.time()
        if i % 50 == 0 or (current_time - last_checkpoint) > 600:  # 10 минут
            elapsed = current_time - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = total_to_analyze - i
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_hours = eta_seconds // 3600
            eta_minutes = (eta_seconds % 3600) // 60
            
            progress_percent = (i / total_to_analyze) * 100
            
            print(f"\n📊 СТАТИСТИКА ПРОГРЕССА:")
            print(f"  📈 Прогресс: {i:,}/{total_to_analyze:,} ({progress_percent:.1f}%)")
            print(f"  ✅ Успешно: {processed:,}")
            print(f"  ❌ Ошибки: {errors}")
            print(f"  ⏱️  Скорость: {rate*3600:.1f} записей/час")
            print(f"  🕐 Осталось: ~{int(eta_hours)}ч {int(eta_minutes)}м")
            print(f"  🕐 Время работы: {elapsed//3600:.0f}ч {(elapsed%3600)//60:.0f}м")
            
            last_checkpoint = current_time
    
    # Финальная статистика
    total_time = time.time() - start_time
    hours = total_time // 3600
    minutes = (total_time % 3600) // 60
    
    print(f"\n🎉 АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 50)
    print(f"🕐 Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Успешно обработано: {processed:,} записей")
    print(f"❌ Ошибок: {errors}")
    print(f"⏱️  Общее время: {int(hours)}ч {int(minutes)}м")
    print(f"📊 Средняя скорость: {processed/(total_time/3600):.1f} записей/час")
    print(f"💾 Результаты сохранены в базу данных")
    
    # Обновленная статистика
    final_analyzed_total = db_connection.execute(
        "SELECT COUNT(DISTINCT song_id) FROM ai_analysis"
    ).fetchone()[0]
    
    final_analyzed_qwen = db_connection.execute(
        "SELECT COUNT(*) FROM ai_analysis WHERE model_version LIKE '%qwen%'"
    ).fetchone()[0]
    
    print(f"\n📈 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"📚 Всего песен в базе: {total_songs:,}")
    print(f"✅ Проанализировано всего: {final_analyzed_total:,}")
    print(f"🤖 Проанализировано Qwen: {final_analyzed_qwen:,}")
    print(f"📊 Покрытие общее: {(final_analyzed_total/total_songs)*100:.1f}%")
    print(f"🎯 Покрытие Qwen: {(final_analyzed_qwen/total_songs)*100:.1f}%")
    
    db_connection.close()

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Qwen Mass Database Analysis')
    parser.add_argument('--max-records', type=int, help='Максимальное количество записей для анализа')
    parser.add_argument('--start-from', type=int, help='ID записи, с которой начать анализ')
    parser.add_argument('--continue', action='store_true', help='Продолжить с последней проанализированной записи')
    
    args = parser.parse_args()
    
    start_from_id = args.start_from
    
    # Если указан флаг --continue, находим последнюю проанализированную запись
    if getattr(args, 'continue'):
        try:
            db_path = project_root / 'data' / 'rap_lyrics.db'
            db_connection = sqlite3.connect(str(db_path))
            
            last_analyzed = db_connection.execute("""
                SELECT MAX(song_id) FROM ai_analysis WHERE model_version LIKE '%qwen%'
            """).fetchone()[0]
            
            if last_analyzed:
                start_from_id = last_analyzed + 1
                print(f"🔄 Продолжаем анализ с ID: {start_from_id}")
            else:
                print("📍 Начинаем анализ с начала базы данных")
            
            db_connection.close()
            
        except Exception as e:
            print(f"⚠️ Ошибка при поиске последней записи: {e}")
    
    # Запускаем анализ
    analyze_database(args.max_records, start_from_id)

if __name__ == "__main__":
    main()
