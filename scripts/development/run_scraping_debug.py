#!/usr/bin/env python3
"""
Улучшенный скрипт запуска скрапинга с детальной отладкой загрузки артистов.
"""

import sys
import os
import logging
import json
from pathlib import Path

# Добавляем корневую папку в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scrapers.rap_scraper_optimized import OptimizedGeniusScraper, load_artist_list
from src.utils.config import GENIUS_TOKEN, LOG_FORMAT, LOG_FILE, DATA_DIR

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def debug_artist_loading():
    """Детальная отладка загрузки списка артистов"""
    logger.info("🔍 Отладка загрузки списка артистов:")
    
    # Проверяем каждый возможный файл
    remaining_file = DATA_DIR / "remaining_artists.json"
    full_file = DATA_DIR / "rap_artists.json"
    test_file = DATA_DIR / "test_artists.json"
    
    logger.info(f"📂 DATA_DIR: {DATA_DIR}")
    logger.info(f"📂 Проверяем файлы:")
    
    files_to_check = [
        ("remaining_artists.json", remaining_file),
        ("rap_artists.json", full_file),
        ("test_artists.json", test_file)
    ]
    
    for name, file_path in files_to_check:
        exists = file_path.exists()
        logger.info(f"  • {name}: {'✅ существует' if exists else '❌ не найден'}")
        
        if exists:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    artists = json.load(f)
                    logger.info(f"    Содержит {len(artists)} артистов")
                    if len(artists) > 0:
                        logger.info(f"    Первые 3: {artists[:3]}")
                        if len(artists) > 3:
                            logger.info(f"    Последние 3: {artists[-3:]}")
            except Exception as e:
                logger.error(f"    ❌ Ошибка чтения: {e}")
    
    # Загружаем через стандартную функцию
    logger.info("\n🎯 Загружаем через load_artist_list():")
    try:
        artists = load_artist_list()
        logger.info(f"✅ Загружено {len(artists)} артистов")
        logger.info(f"Первые 5 артистов: {artists[:5]}")
        return artists
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки: {e}")
        return []

def main():
    """Главная функция с детальной отладкой"""
    logger.info("🚀 Запуск улучшенного скрапера")
    
    if not GENIUS_TOKEN:
        logger.error("❌ GENIUS_TOKEN не найден в .env!")
        return
    
    # Отладка загрузки артистов
    artists = debug_artist_loading()
    
    if not artists:
        logger.error("❌ Список артистов пуст!")
        return
    
    # Спрашиваем у пользователя подтверждение
    logger.info(f"\n📋 Готов к скрапингу {len(artists)} артистов")
    logger.info("Нажмите Enter для продолжения или Ctrl+C для отмены...")
    
    try:
        input()
    except KeyboardInterrupt:
        logger.info("❌ Отменено пользователем")
        return
    
    # Создаем скрапер
    MEMORY_LIMIT_MB = 3072
    scraper = OptimizedGeniusScraper(GENIUS_TOKEN, None, MEMORY_LIMIT_MB)
    
    try:
        SONGS_PER_ARTIST = 500
        
        logger.info(f"🎯 Загружено {len(artists)} артистов")
        logger.info(f"🎵 Цель: ~{len(artists) * SONGS_PER_ARTIST} песен")
        logger.info(f"💾 Лимит памяти: {MEMORY_LIMIT_MB}MB")
        logger.info("🛑 Для остановки: Ctrl+C")
        
        scraper.run_scraping_session(artists, SONGS_PER_ARTIST)
        
    except KeyboardInterrupt:
        logger.info("⌨️ Остановлено пользователем")
    except Exception as e:
        logger.error(f"💥 Ошибка: {e}")
    finally:
        scraper.close()
        logger.info("🏁 Программа завершена")

if __name__ == "__main__":
    main()
