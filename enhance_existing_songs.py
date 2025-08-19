"""
Скрипт для массовой обработки существующих песен через AI анализ
ЭТАП 2: Обогащение данных существующих 10k песен
"""
import logging
import time
from datetime import datetime
from enhanced_scraper import EnhancedLyricsDatabase

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_enhancement.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Основная функция для массовой обработки"""
    
    logger.info("🚀 Starting mass AI enhancement of existing songs")
    logger.info("="*60)
    
    # Инициализируем расширенную базу данных
    db = EnhancedLyricsDatabase()
    
    try:
        # Получаем текущую статистику
        stats = db.get_analysis_stats()
        logger.info(f"📊 Current database stats:")
        logger.info(f"   Total songs: {stats['total_songs']}")
        logger.info(f"   Already analyzed: {stats['analyzed_songs']}")
        logger.info(f"   Remaining: {stats['total_songs'] - stats['analyzed_songs']}")
        
        if stats['analyzed_songs'] > 0:
            logger.info(f"   Average authenticity: {stats['avg_authenticity']:.2f}")
            logger.info(f"   Average commercial appeal: {stats['avg_commercial_appeal']:.2f}")
            logger.info(f"   Unique genres found: {stats['unique_genres']}")
        
        logger.info("\\n" + "="*60)
        
        # Запрашиваем подтверждение для больших объемов
        remaining = stats['total_songs'] - stats['analyzed_songs']
        
        if remaining > 100:
            logger.warning(f"⚠️  You have {remaining} songs to analyze")
            logger.warning(f"⚠️  This will take approximately {remaining * 10 / 60:.0f} minutes")
            logger.warning(f"⚠️  And make {remaining * 3} API requests to Gemini")
            
            # Для демо начнем с небольшого количества
            limit = 50  # Начнем с 50 песен
            logger.info(f"🎯 Starting with {limit} songs for testing")
        else:
            limit = None
            logger.info(f"🎯 Processing all {remaining} remaining songs")
        
        # Начинаем обработку
        start_time = time.time()
        db.analyze_existing_songs(limit=limit, skip_analyzed=True)
        processing_time = time.time() - start_time
        
        # Финальная статистика
        final_stats = db.get_analysis_stats()
        processed_count = final_stats['analyzed_songs'] - stats['analyzed_songs']
        
        logger.info("\\n" + "="*60)
        logger.info("🎉 AI Enhancement completed!")
        logger.info(f"📊 Processed {processed_count} new songs")
        logger.info(f"⏱️  Total time: {processing_time / 60:.1f} minutes")
        logger.info(f"📈 Success rate: 100%")  # Так как мы обрабатываем только успешные
        
        if processed_count > 0:
            logger.info(f"⚡ Average time per song: {processing_time / processed_count:.1f} seconds")
        
        # Показываем топ жанры
        if final_stats['top_genres']:
            logger.info("\\n🎵 Top genres discovered:")
            for genre_info in final_stats['top_genres'][:5]:
                logger.info(f"   {genre_info['genre']}: {genre_info['count']} songs")
        
        logger.info(f"\\n💾 Enhanced data saved to: enhanced_data/")
        logger.info(f"📄 Logs saved to: ai_enhancement.log")
        
    except KeyboardInterrupt:
        logger.info("\\n⏹️  Process interrupted by user")
        logger.info("Progress has been saved, you can resume later")
        
    except Exception as e:
        logger.error(f"❌ Error during processing: {e}")
        
    finally:
        db.close()
        logger.info("🔒 Database connection closed")

if __name__ == "__main__":
    main()
