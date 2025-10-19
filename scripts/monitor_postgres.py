#!/usr/bin/env python3
"""
#!/usr/bin/env python3
🐘 Мониторинг состояния PostgreSQL базы данных Rap Scraper

НАЗНАЧЕНИЕ:
- Мониторинг количества песен, артистов и качества данных в PostgreSQL базе
- Быстрая диагностика прогресса массового скрапинга и анализа

ИСПОЛЬЗОВАНИЕ:
python scripts/monitor_postgres.py

ЗАВИСИМОСТИ:
- Python 3.8+
- src/utils/postgres_db.py
- PostgreSQL база данных (rap_lyrics)

РЕЗУЛЬТАТ:
- Консольный вывод с текущей статистикой базы
- Последние добавленные песни
- Быстрая проверка состояния БД

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""

import sys
import time
from pathlib import Path

# Добавляем корневую папку в path
sys.path.append(str(Path(__file__).parent.parent))

import logging

from src.utils.postgres_db import PostgreSQLManager

logging.basicConfig(level=logging.ERROR)  # Только ошибки для чистого вывода


def monitor_postgres():
    """Мониторинг состояния PostgreSQL базы"""
    try:
        db = PostgreSQLManager()

        print("🐘 PostgreSQL Database Monitor")
        print("=" * 50)

        while True:
            stats = db.get_stats()
            recent = db.get_recent_songs(3)

            print(
                f"\r📊 Всего песен: {stats['total_songs']} | "
                f"Артистов: {stats['unique_artists']} | "
                f"Ср.слов: {stats['avg_words']} | "
                f"Качество: {stats['avg_quality']:.3f}",
                end="",
                flush=True,
            )

            if recent:
                print(f"\n🎵 Последние: {recent[0]['artist']} - {recent[0]['title']}")

            time.sleep(5)  # Обновление каждые 5 секунд

    except KeyboardInterrupt:
        print("\n👋 Мониторинг остановлен")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        try:
            db.close()
        except:
            pass


if __name__ == "__main__":
    monitor_postgres()
