"""
Тестовый скрипт для проверки интеграции LangChain + Gemini
Тестируем на 5 песнях из базы данных
"""
import sqlite3
import json
import time
import logging
from datetime import datetime
from pathlib import Path

from langchain_analyzer import GeminiLyricsAnalyzer
from models import AnalysisResult, BatchAnalysisStats

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('langchain_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_test_songs(db_path: str = "rap_lyrics.db", limit: int = 5):
    """Получаем тестовые песни из базы данных"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Берем песни с хорошим количеством слов (100-1000 слов)
    cursor = conn.execute("""
        SELECT artist, title, lyrics, url, genius_id, scraped_date, word_count
        FROM songs 
        WHERE word_count BETWEEN 100 AND 1000
        ORDER BY RANDOM()
        LIMIT ?
    """, (limit,))
    
    songs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    logger.info(f"Loaded {len(songs)} test songs from database")
    return songs

def test_langchain_integration():
    """Основной тест интеграции"""
    logger.info("Starting LangChain + Gemini integration test")
    
    # Проверяем наличие API ключа
    try:
        analyzer = GeminiLyricsAnalyzer()
        logger.info("✅ Gemini analyzer initialized successfully")
    except ValueError as e:
        logger.error(f"❌ Failed to initialize analyzer: {e}")
        logger.error("Please add your GOOGLE_API_KEY to .env file")
        logger.error("Get it from: https://makersuite.google.com/app/apikey")
        return
    
    # Получаем тестовые песни
    test_songs = get_test_songs(limit=3)  # Начнем с 3 песен для теста
    
    if not test_songs:
        logger.error("❌ No test songs found in database")
        return
    
    # Статистика
    stats = BatchAnalysisStats(
        total_processed=0,
        successful=0,
        failed=0,
        average_processing_time=0.0,
        start_time=datetime.now().isoformat(),
        end_time=""
    )
    
    results = []
    processing_times = []
    
    # Обрабатываем каждую песню
    for i, song in enumerate(test_songs, 1):
        logger.info(f"\\n=== Processing song {i}/{len(test_songs)} ===")
        logger.info(f"Artist: {song['artist']}")
        logger.info(f"Title: {song['title']}")
        logger.info(f"Word count: {song['word_count']}")
        
        start_time = time.time()
        
        try:
            # Анализируем песню
            enhanced_data = analyzer.analyze_song_complete(song)
            processing_time = time.time() - start_time
            
            result = AnalysisResult(
                success=True,
                song_data=enhanced_data,
                processing_time=processing_time
            )
            
            stats.successful += 1
            processing_times.append(processing_time)
            
            logger.info(f"✅ Successfully analyzed in {processing_time:.1f}s")
            logger.info(f"   Genre: {enhanced_data.ai_metadata.genre}")
            logger.info(f"   Mood: {enhanced_data.ai_metadata.mood}")
            logger.info(f"   Quality: {enhanced_data.quality_metrics.overall_quality}")
            logger.info(f"   Authenticity: {enhanced_data.quality_metrics.authenticity_score:.2f}")
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Failed to analyze: {e}")
            
            result = AnalysisResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
            
            stats.failed += 1
            stats.errors.append(f"{song['artist']} - {song['title']}: {str(e)}")
        
        results.append(result)
        stats.total_processed += 1
        
        # Небольшая пауза между запросами
        if i < len(test_songs):
            logger.info("Waiting 5 seconds before next song...")
            time.sleep(5)
    
    # Финализируем статистику
    stats.end_time = datetime.now().isoformat()
    if processing_times:
        stats.average_processing_time = sum(processing_times) / len(processing_times)
    
    # Сохраняем результаты
    save_test_results(results, stats)
    
    # Выводим итоговую статистику
    print_final_stats(stats, analyzer)

def save_test_results(results, stats):
    """Сохраняем результаты тестирования"""
    
    # Создаем директорию для результатов
    results_dir = Path("langchain_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Сохраняем успешные анализы в JSONL
    successful_results = [r for r in results if r.success]
    if successful_results:
        with open(results_dir / f"enhanced_songs_{timestamp}.jsonl", "w", encoding="utf-8") as f:
            for result in successful_results:
                f.write(result.song_data.model_dump_json() + "\\n")
    
    # Сохраняем статистику
    with open(results_dir / f"test_stats_{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(stats.model_dump(), f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_dir}")

def print_final_stats(stats, analyzer):
    """Выводим финальную статистику"""
    print("\\n" + "="*50)
    print("🎵 LANGCHAIN + GEMINI TEST RESULTS 🎵")
    print("="*50)
    print(f"Total processed: {stats.total_processed}")
    print(f"✅ Successful: {stats.successful}")
    print(f"❌ Failed: {stats.failed}")
    print(f"Success rate: {(stats.successful/stats.total_processed*100):.1f}%")
    print(f"Average processing time: {stats.average_processing_time:.1f}s")
    
    analyzer_stats = analyzer.get_stats()
    print(f"Total API requests: {analyzer_stats['total_requests']}")
    print(f"Cost: {analyzer_stats['estimated_cost']}")
    
    if stats.errors:
        print("\\n❌ Errors:")
        for error in stats.errors:
            print(f"   {error}")
    
    print("\\n🎉 Test completed! Check langchain_results/ for detailed output")

if __name__ == "__main__":
    test_langchain_integration()
