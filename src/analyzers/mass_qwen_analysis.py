#!/usr/bin/env python3
"""
#!/usr/bin/env python3
🤖 Массовый AI-анализ базы данных с помощью Qwen

НАЗНАЧЕНИЕ:
- Пакетный анализ всех треков в базе с использованием Qwen AI
- Сбор статистики, обновление результатов анализа

ИСПОЛЬЗОВАНИЕ:
python scripts/mass_qwen_analysis.py                 # Полный анализ всей базы
python scripts/mass_qwen_analysis.py --batch 200     # Анализ с указанным размером батча

ЗАВИСИМОСТИ:
- Python 3.8+
- src/core/app.py, src/interfaces/analyzer_interface.py
- PostgreSQL база данных (rap_lyrics)
- Novita AI/Qwen API ключи

РЕЗУЛЬТАТ:
- Обновленные записи анализа в базе данных
- Статистика по качеству и покрытию
- Логирование прогресса и ошибок

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""

#!/usr/bin/env python3
"""
🤖 Массовый AI-анализ базы данных с помощью Qwen (PostgreSQL версия)

НАЗНАЧЕНИЕ:
- Пакетный анализ всех треков в PostgreSQL базе с использованием Qwen AI
- Сбор статистики, обновление результатов анализа

ИСПОЛЬЗОВАНИЕ:
python scripts/mass_qwen_analysis_postgres.py                 # Полный анализ всей базы
python scripts/mass_qwen_analysis_postgres.py --batch 200     # Анализ с указанным размером батча

ЗАВИСИМОСТИ:
- Python 3.8+
- src/core/app.py, src/interfaces/analyzer_interface.py
- PostgreSQL база данных (rap_lyrics)
- Novita AI/Qwen API ключи

РЕЗУЛЬТАТ:
- Обновленные записи анализа в PostgreSQL базе данных
- Статистика по качеству и покрытию
- Логирование прогресса и ошибок

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""

import sys
import os
import asyncio
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Добавляем корневую папку в path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.core.app import create_app
from src.interfaces.analyzer_interface import AnalyzerFactory
from src.database.postgres_adapter import PostgreSQLManager

# Импортируем анализаторы для их регистрации
from src.analyzers.qwen_analyzer import QwenAnalyzer

async def mass_analyze_database(batch_size: int = 100, max_records: Optional[int] = None):
    """
    Массовый анализ всей базы данных с Qwen анализатором
    
    Args:
        batch_size: Размер батча для обработки
        max_records: Максимальное количество записей (None = все)
    """
    print("🎵 Rap Scraper - Mass Database Analysis with Qwen (PostgreSQL)")
    print("=" * 70)
    
    # Инициализация приложения
    app = create_app()
    
    # Получение анализатора
    analyzer = AnalyzerFactory.create("qwen")
    if not analyzer.available:
        print("❌ Qwen анализатор недоступен!")
        return
    
    print(f"✅ Qwen анализатор готов: {analyzer.model_name}")
    
    # Подключение к PostgreSQL базе данных
    db_manager = PostgreSQLManager()
    await db_manager.initialize()
    
    try:
        async with db_manager.get_connection() as conn:
            # Подсчитываем количество неанализированных записей
            count_query = """
            SELECT COUNT(*) 
            FROM tracks 
            WHERE lyrics IS NOT NULL 
            AND lyrics != '' 
            AND id NOT IN (
                SELECT DISTINCT track_id 
                FROM analysis_results 
                WHERE analyzer_type LIKE '%qwen%'
            )
            """
            
            total_unanalyzed = await conn.fetchval(count_query)
            print(f"📈 Найдено {total_unanalyzed} неанализированных записей")
            
            if total_unanalyzed == 0:
                print("✅ Все записи уже проанализированы!")
                return
            
            # Получение записей для анализа
            query = """
            SELECT t.id, t.artist, t.title, t.lyrics 
            FROM tracks t
            WHERE t.lyrics IS NOT NULL 
            AND t.lyrics != '' 
            AND t.id NOT IN (
                SELECT DISTINCT track_id 
                FROM analysis_results 
                WHERE analyzer_type LIKE '%qwen%'
            )
            ORDER BY t.id
            """
            
            if max_records:
                actual_limit = min(max_records, total_unanalyzed)
                query += f" LIMIT {actual_limit}"
                print(f"🎯 Ограничиваем анализ до {actual_limit} записей")
            else:
                actual_limit = total_unanalyzed
            
            records = await conn.fetch(query)
            total_records = len(records)
            
            print(f"📈 Загружено {total_records} записей для анализа")
            
            if total_records == 0:
                print("✅ Все записи уже проанализированы!")
                return
            
            # Информация о предстоящем анализе
            print(f"\n🎯 Будет проанализировано {total_records} записей")
            print(f"⏱️  Примерное время: {(total_records * 15) // 60} минут")
            print(f"🌟 Бесплатная модель Qwen через Novita AI - без затрат!")
            
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
                    track_id = record['id']
                    artist = record['artist']
                    title = record['title']
                    lyrics = record['lyrics']
                    
                    try:
                        print(f"🎵 Анализируем: {artist} - {title}")
                        
                        # Анализ текста (используем правильный метод - синхронный)
                        result = analyzer.analyze_song(artist, title, lyrics)
                        
                        # Проверяем, что анализ прошел успешно
                        if result is None:
                            print(f"❌ Анализ не удался - получен None")
                            errors += 1
                            continue
                        
                        # Подготовка данных для сохранения в PostgreSQL
                        analysis_data = {
                            'track_id': track_id,
                            'analyzer_type': 'qwen-3-4b-fp8',
                            'sentiment': result.metadata.get('mood', {}).get('overall', 'neutral'),
                            'confidence': result.confidence or 0.5,
                            'complexity_score': 3.0,  # Промежуточная сложность
                            'themes': result.metadata.get('themes', []),
                            'analysis_data': result.raw_output or {},
                            'processing_time_ms': int((result.processing_time or 0) * 1000),
                            'model_version': 'qwen-3-4b-fp8'
                        }
                        
                        # Сохранение в PostgreSQL
                        analysis_id = await db_manager.save_analysis_result(analysis_data)
                        
                        if analysis_id:
                            processed += 1
                            sentiment = analysis_data['sentiment']
                            print(f"✅ Успешно: {sentiment} (ID: {analysis_id})")
                        else:
                            print(f"❌ Ошибка сохранения в базу")
                            errors += 1
                        
                        # Пауза между запросами
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"❌ Ошибка обработки записи {track_id}: {e}")
                        errors += 1
                        continue
                
                # Статистика батча
                batch_time = time.time() - batch_start
                print(f"⏱️  Батч завершен за {batch_time:.1f}с")
                
                # Пауза между батчами
                if i + batch_size < total_records:
                    print("⏸️  Пауза между батчами...")
                    time.sleep(2)
            
            # Финальная статистика
            total_time = time.time() - start_time
            success_rate = (processed / total_records) * 100 if total_records > 0 else 0
            
            print(f"\n🏆 АНАЛИЗ ЗАВЕРШЕН!")
            print(f"=" * 50)
            print(f"✅ Успешно проанализировано: {processed}")
            print(f"❌ Ошибок: {errors}")
            print(f"📊 Всего записей: {total_records}")
            print(f"🎯 Успешность: {success_rate:.1f}%")
            print(f"⏱️  Общее время: {total_time//60:.0f}м {total_time%60:.0f}с")
            print(f"⚡ Скорость: {processed/total_time*60:.1f} анализов/мин")
            
            # Обновленная статистика базы
            final_count = await conn.fetchval(count_query)
            print(f"📈 Осталось неанализированных: {final_count}")
            
    finally:
        await db_manager.close()

async def main():
    """Главная функция с парсингом аргументов"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Массовый анализ базы данных с Qwen")
    parser.add_argument('--batch', type=int, default=100, help='Размер батча (default: 100)')
    parser.add_argument('--max', type=int, help='Максимальное количество записей для анализа')
    parser.add_argument('--test', action='store_true', help='Тестовый режим (только 10 записей)')
    
    args = parser.parse_args()
    
    if args.test:
        print("🧪 ТЕСТОВЫЙ РЕЖИМ: анализируем только 10 записей")
        await mass_analyze_database(batch_size=5, max_records=10)
    else:
        await mass_analyze_database(batch_size=args.batch, max_records=args.max)

if __name__ == "__main__":
    asyncio.run(main())
