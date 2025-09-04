#!/usr/bin/env python3
"""
🎯 Универсальный анализатор с выбором модели
Поддерживает Qwen, Gemma, Ollama и другие анализаторы
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

def get_available_analyzers():
    """Получает список доступных анализаторов"""
    try:
        from interfaces.analyzer_interface import AnalyzerFactory
        
        # Импортируем анализаторы для регистрации
        try:
            import analyzers.qwen_analyzer
            import analyzers.gemma_analyzer  
            import analyzers.ollama_analyzer
            import analyzers.algorithmic_analyzer
        except ImportError:
            pass
        
        return AnalyzerFactory.list_available()
    except Exception as e:
        print(f"❌ Ошибка загрузки анализаторов: {e}")
        return []

def test_analyzer(analyzer_name: str):
    """Тестирует конкретный анализатор"""
    try:
        from interfaces.analyzer_interface import AnalyzerFactory
        
        # Импортируем анализаторы для регистрации
        try:
            import analyzers.qwen_analyzer
            import analyzers.gemma_analyzer  
            import analyzers.ollama_analyzer
            import analyzers.algorithmic_analyzer
        except ImportError:
            pass
        
        print(f"🧪 Тестируем анализатор: {analyzer_name}")
        
        # Создаем анализатор
        analyzer = AnalyzerFactory.create(analyzer_name)
        
        # Проверяем доступность
        if hasattr(analyzer, 'available') and not analyzer.available:
            print(f"❌ Анализатор {analyzer_name} недоступен")
            return False
        
        # Тестовые данные
        test_artist = "Test Artist"
        test_title = "Test Song"
        test_lyrics = "This is a test rap song with simple lyrics for testing"
        
        print(f"📝 Анализируем тестовую песню...")
        
        # Выполняем анализ
        result = analyzer.analyze_song(test_artist, test_title, test_lyrics)
        
        print(f"✅ Анализ успешен!")
        print(f"   Тип: {result.analysis_type}")
        print(f"   Уверенность: {result.confidence:.2f}")
        print(f"   Время: {result.processing_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования {analyzer_name}: {e}")
        return False

def mass_analyze_with_model(model_name: str, max_records: int = None):
    """Массовый анализ с выбранной моделью"""
    
    print(f"🎵 Массовый анализ с моделью: {model_name}")
    print("=" * 50)
    
    try:
        from interfaces.analyzer_interface import AnalyzerFactory
        
        # Создаем анализатор
        analyzer = AnalyzerFactory.create(model_name)
        
        # Проверяем доступность
        if hasattr(analyzer, 'available') and not analyzer.available:
            print(f"❌ Анализатор {model_name} недоступен!")
            return
        
        print(f"✅ Анализатор {model_name} готов")
        
    except Exception as e:
        print(f"❌ Ошибка создания анализатора {model_name}: {e}")
        return
    
    # Подключение к базе данных
    db_path = project_root / 'data' / 'rap_lyrics.db'
    db_connection = sqlite3.connect(str(db_path))
    
    # Проверяем, какие записи можно анализировать
    model_version_pattern = f"%{model_name}%"
    
    if model_name == 'qwen':
        model_version_pattern = "%qwen%"
    elif model_name == 'gemma':
        model_version_pattern = "%gemma%"
    elif model_name == 'ollama':
        model_version_pattern = "%ollama%"
    
    # Ищем неанализированные записи для этой модели
    query = """
    SELECT s.id, s.artist, s.title, s.lyrics
    FROM songs s
    WHERE s.lyrics IS NOT NULL 
    AND s.lyrics != '' 
    AND s.id NOT IN (
        SELECT DISTINCT song_id 
        FROM ai_analysis 
        WHERE model_version LIKE ?
    )
    ORDER BY s.id
    """
    
    if max_records:
        query += f" LIMIT {max_records}"
    
    cursor = db_connection.execute(query, (model_version_pattern,))
    records = cursor.fetchall()
    total_records = len(records)
    
    print(f"📈 Найдено {total_records} записей для анализа моделью {model_name}")
    
    if total_records == 0:
        print(f"✅ Все доступные записи уже проанализированы моделью {model_name}!")
        db_connection.close()
        return
    
    # Подтверждение
    print(f"\n⚠️  Будет проанализировано {total_records} записей")
    print(f"🤖 Модель: {model_name}")
    print(f"⏱️  Примерное время: {(total_records * 15) // 60} минут")
    
    confirm = input("\nПродолжить? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Анализ отменен")
        db_connection.close()
        return
    
    # Анализ
    print(f"\n🚀 Начинаем анализ с {model_name}...")
    
    processed = 0
    errors = 0
    start_time = time.time()
    
    # Определяем версию модели для базы данных
    db_model_version = f"{model_name}-analysis"
    if model_name == 'qwen':
        db_model_version = 'qwen-3-4b-fp8'
    elif model_name == 'gemma':
        db_model_version = 'gemma-3-27b-it'
    
    for i, (track_id, artist, title, lyrics) in enumerate(records, 1):
        try:
            print(f"\n📊 {i}/{total_records}: {artist} - {title}")
            
            # Выполняем анализ
            result = analyzer.analyze_song(artist, title, lyrics)
            
            # Извлекаем данные
            raw_output = result.raw_output
            
            # Подготавливаем данные для вставки (универсально)
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
            
            # Вставляем в таблицу (только если записи нет)
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
                    overall_quality, ai_likelihood, datetime.now().isoformat(), db_model_version
                ))
                
                db_connection.commit()
                processed += 1
                
                print(f"✅ Успешно! (confidence: {result.confidence:.2f})")
                
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" in str(e):
                    print(f"⚠️ Запись уже существует")
                    continue
                else:
                    raise
                
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            errors += 1
            continue
        
        # Статистика каждые 10 записей
        if i % 10 == 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = total_records - i
            eta = remaining / rate / 60 if rate > 0 else 0
            
            print(f"\n📊 Прогресс: {i}/{total_records} ({i/total_records*100:.1f}%)")
            print(f"⏱️  Скорость: {rate*60:.1f} записей/мин")
            print(f"🕐 Осталось: ~{eta:.1f} мин")
    
    # Финальная статистика
    total_time = time.time() - start_time
    
    print(f"\n🎉 АНАЛИЗ ЗАВЕРШЕН!")
    print("=" * 30)
    print(f"🤖 Модель: {model_name}")
    print(f"✅ Успешно: {processed}")
    print(f"❌ Ошибки: {errors}")
    print(f"⏱️  Время: {total_time/60:.1f} мин")
    print(f"📊 Скорость: {processed/(total_time/60):.1f} записей/мин")
    
    db_connection.close()

def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Universal Model Analyzer')
    parser.add_argument('--model', help='Название модели (qwen, gemma, ollama, etc.)')
    parser.add_argument('--max-records', type=int, help='Максимальное количество записей')
    parser.add_argument('--test', action='store_true', help='Только тестирование модели')
    parser.add_argument('--list', action='store_true', help='Показать доступные модели')
    
    args = parser.parse_args()
    
    if args.list:
        print("🤖 Доступные анализаторы:")
        analyzers = get_available_analyzers()
        for analyzer in analyzers:
            print(f"  📊 {analyzer}")
        return
    
    if args.test:
        if not args.model:
            print("❌ Для тестирования нужно указать модель: --model <название>")
            sys.exit(1)
        success = test_analyzer(args.model)
        sys.exit(0 if success else 1)
    
    if not args.model:
        print("❌ Нужно указать модель: --model <название>")
        sys.exit(1)
    
    # Проверяем модель
    available_analyzers = get_available_analyzers()
    if args.model not in available_analyzers:
        print(f"❌ Модель '{args.model}' не найдена!")
        print(f"Доступные модели: {', '.join(available_analyzers)}")
        sys.exit(1)
    
    # Запускаем анализ
    mass_analyze_with_model(args.model, args.max_records)

if __name__ == "__main__":
    main()
