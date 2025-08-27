#!/usr/bin/env python3
"""
Тестовый скрипт для проверки исправлений основного скрапера.
Тестирует на небольшом списке артистов.
"""

import sys
import os
import logging
import json
from pathlib import Path

# Добавляем корневую папку в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scrapers.rap_scraper_optimized import OptimizedGeniusScraper
from src.utils.config import GENIUS_TOKEN, LOG_FORMAT, LOG_FILE, DATA_DIR

def main():
    """Тестирование исправленного скрапера"""
    
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
    
    if not GENIUS_TOKEN:
        logger.error("❌ GENIUS_TOKEN не найден!")
        return
    
    # Небольшой тестовый список артистов
    test_artists = ["Drake", "Eminem", "J. Cole"]  # Известные артисты для теста
    
    logger.info("🧪 ТЕСТИРОВАНИЕ ИСПРАВЛЕННОГО СКРАПЕРА")
    logger.info(f"🎤 Тестовые артисты: {', '.join(test_artists)}")
    
    # Создаем скрапер
    try:
        scraper = OptimizedGeniusScraper(GENIUS_TOKEN, None, 2048)
        logger.info("✅ Скрапер создан")
        
        # Показываем начальную статистику
        initial_stats = scraper.db.get_stats()
        logger.info(f"📊 Начальная БД: {initial_stats['total_songs']} песен")
        
        # Тестируем каждого артиста
        total_added = 0
        for i, artist in enumerate(test_artists, 1):
            if scraper.shutdown_requested:
                break
                
            logger.info(f"\n{'='*50}")
            logger.info(f"🎤 ТЕСТ {i}/{len(test_artists)}: {artist}")
            logger.info(f"{'='*50}")
            
            try:
                added = scraper.scrape_artist_songs(artist, max_songs=10)  # Только 10 песен для теста
                total_added += added
                logger.info(f"✅ {artist}: добавлено {added} песен")
                
                # Показываем статистику после каждого артиста
                current_stats = scraper.db.get_stats()
                logger.info(f"📊 Сейчас в БД: {current_stats['total_songs']} песен")
                
            except Exception as e:
                logger.error(f"❌ Ошибка с артистом {artist}: {e}")
        
        # Финальная статистика
        final_stats = scraper.db.get_stats()
        logger.info(f"\n{'='*50}")
        logger.info(f"🏁 ТЕСТ ЗАВЕРШЕН")
        logger.info(f"➕ Добавлено за тест: {total_added} песен")
        logger.info(f"📊 Всего в БД: {final_stats['total_songs']} песен")
        logger.info(f"👤 Артистов в БД: {final_stats['unique_artists']}")
        logger.info(f"⭐ Среднее качество: {final_stats['avg_quality']}")
        
        # Показываем последние добавленные песни
        recent_songs = scraper.db.get_recent_songs(5)
        if recent_songs:
            logger.info(f"\n🎶 Последние добавленные песни:")
            for song in recent_songs:
                logger.info(f"  • {song['artist']} - {song['title']} ({song['word_count']} слов)")
        
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
    finally:
        try:
            scraper.close()
            logger.info("🔒 Скрапер закрыт")
        except:
            pass
    
    logger.info("🏁 Тест завершен!")

if __name__ == "__main__":
    main()
