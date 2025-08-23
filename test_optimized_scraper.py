#!/usr/bin/env python3
"""
Тест оптимизированного скрапера на небольшом количестве данных
"""

import os
import sys
sys.path.append('.')

from rap_scraper_optimized import OptimizedGeniusScraper
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("GENIUS_TOKEN")

def test_optimized_scraper():
    """Тест оптимизированного скрапера на 1-2 артистах"""
    
    if not TOKEN:
        print("❌ Genius API token не найден!")
        return
    
    print("🧪 Тестирование оптимизированного скрапера")
    print("=" * 60)
    
    # Создаем тестовый скрапер с лимитом памяти 1GB
    scraper = OptimizedGeniusScraper(TOKEN, "test_optimized.db", memory_limit_mb=1024)
    
    try:
        # Тестируем на небольшом списке артистов
        test_artists = ["Logic", "Vince Staples"]  # Возьмем 2 артистов для теста
        songs_per_artist = 5  # Только 5 песен каждого
        
        print(f"🎯 Тест на {len(test_artists)} артистах, {songs_per_artist} песен каждый")
        print("📊 Мониторинг включен...")
        
        scraper.run_scraping_session(test_artists, songs_per_artist)
        
        print("\n✅ Тест завершен!")
        
        # Показать финальную статистику
        stats = scraper.db.get_stats()
        print(f"""
        📈 Результаты теста:
        • Добавлено песен: {scraper.session_stats['added']}
        • Пропущено: {scraper.session_stats['skipped']}
        • Ошибок: {scraper.session_stats['errors']}
        • Очисток памяти: {scraper.session_stats['gc_runs']}
        • Среднее качество: {stats['avg_quality']}
        """)
        
    except Exception as e:
        print(f"❌ Ошибка в тесте: {e}")
    finally:
        scraper.close()

if __name__ == "__main__":
    test_optimized_scraper()
