#!/usr/bin/env python3
"""
Улучшенный скрипт запуска скрапинга с дополнительными проверками и настройками.
Исправляет проблемы с прокси и сетевыми подключениями.
"""

import json
import logging
import sys
from pathlib import Path

# Добавляем корневую папку в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scrapers.rap_scraper_optimized import OptimizedGeniusScraper, load_artist_list
from src.utils.config import DATA_DIR, GENIUS_TOKEN, LOG_FILE, LOG_FORMAT


def setup_logging():
    """Настройка расширенного логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def check_environment():
    """Проверка окружения и конфигурации"""
    logger = logging.getLogger(__name__)

    if not GENIUS_TOKEN:
        logger.error("❌ GENIUS_TOKEN не найден в .env файле!")
        logger.error("💡 Создайте .env файл в корне проекта с токеном:")
        logger.error("   GENIUS_TOKEN=your_token_here")
        return False

    # Проверяем доступность API
    try:
        import requests

        response = requests.get("https://api.genius.com/", timeout=10)
        if response.status_code == 401:  # Unauthorized но API доступно
            logger.info("✅ API Genius доступно")
        else:
            logger.warning(f"⚠️ Необычный ответ API: {response.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Проблема доступа к API: {e}")
        logger.info("🔄 Будем пробовать обходные пути...")

    # Проверяем папки
    DATA_DIR.mkdir(exist_ok=True)
    logger.info(f"📁 Рабочая папка: {DATA_DIR}")

    return True


def save_remaining_artists(artists: list, processed_count: int):
    """Сохранение оставшихся артистов для продолжения"""
    remaining = artists[processed_count:]
    if remaining:
        remaining_file = DATA_DIR / "remaining_artists.json"
        with open(remaining_file, "w", encoding="utf-8") as f:
            json.dump(remaining, f, indent=2, ensure_ascii=False)
        logging.info(
            f"💾 Сохранено {len(remaining)} оставшихся артистов в {remaining_file}"
        )


def main():
    """Главная функция с улучшенной обработкой ошибок"""
    logger = setup_logging()

    # Проверка окружения
    if not check_environment():
        return 1

    logger.info("🚀 Запуск улучшенного скрапера...")
    logger.info(f"📝 Логи записываются в: {LOG_FILE}")

    # Настройки
    MEMORY_LIMIT_MB = 4096  # Увеличиваем лимит памяти
    SONGS_PER_ARTIST = 100  # Уменьшаем для стабильности

    # Загрузка списка артистов
    try:
        artists = load_artist_list()
        logger.info(f"🎯 Загружено {len(artists)} артистов")
        logger.info(f"🎵 Цель: ~{len(artists) * SONGS_PER_ARTIST} песен")
        logger.info(f"💾 Лимит памяти: {MEMORY_LIMIT_MB}MB")

        # Показываем первых несколько артистов
        logger.info("🎤 Первые артисты в списке:")
        for i, artist in enumerate(artists[:5], 1):
            logger.info(f"  {i}. {artist}")
        if len(artists) > 5:
            logger.info(f"  ... и еще {len(artists) - 5} артистов")

    except Exception as e:
        logger.error(f"❌ Ошибка загрузки списка артистов: {e}")
        return 1

    # Создание скрапера
    try:
        scraper = OptimizedGeniusScraper(GENIUS_TOKEN, None, MEMORY_LIMIT_MB)
        logger.info("✅ Скрапер инициализирован")

        # Показываем текущую статистику БД
        initial_stats = scraper.db.get_stats()
        logger.info(
            f"📚 Текущая БД: {initial_stats['total_songs']} песен от {initial_stats['unique_artists']} артистов"
        )

    except Exception as e:
        logger.error(f"❌ Ошибка создания скрапера: {e}")
        return 1

    # Запуск скрапинга с обработкой ошибок
    processed_count = 0
    try:
        logger.info("\n" + "=" * 60)
        logger.info("🎵 НАЧИНАЕМ СКРАПИНГ")
        logger.info("=" * 60)
        logger.info("🛑 Для остановки: Ctrl+C")

        scraper.run_scraping_session(artists, SONGS_PER_ARTIST)

    except KeyboardInterrupt:
        logger.info("\n⌨️ Получен сигнал остановки (Ctrl+C)")
        logger.info("💾 Сохраняем прогресс...")

        # Сохраняем оставшихся артистов
        save_remaining_artists(artists, processed_count)

    except MemoryError:
        logger.error("🚨 Критическая нехватка памяти!")
        logger.error("💡 Попробуйте уменьшить SONGS_PER_ARTIST или MEMORY_LIMIT_MB")
        save_remaining_artists(artists, processed_count)
        return 1

    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        logger.error("💾 Сохраняем прогресс...")
        save_remaining_artists(artists, processed_count)
        return 1

    finally:
        try:
            scraper.close()
            logger.info("🔒 Скрапер закрыт")
        except:
            pass

    logger.info("🏁 Скрапинг завершен!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        print(f"\n❌ Программа завершилась с ошибкой (код: {exit_code})")
        print("💡 Проверьте логи для деталей")
    sys.exit(exit_code)
