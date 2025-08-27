#!/usr/bin/env python3
"""
Быстрый запуск основного скрапера с правильной загрузкой remaining_artists.json
"""

import sys
import os
from pathlib import Path

# Добавляем корневую папку в путь
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scrapers.rap_scraper_optimized import main

if __name__ == "__main__":
    print("🎯 Запуск основного скрапера с remaining_artists.json")
    print("✅ Скрипт автоматически загрузит оставшихся артистов")
    print("🛑 Для остановки: Ctrl+C")
    print("")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⌨️ Остановлено пользователем")
    except Exception as e:
        print(f"\n💥 Ошибка: {e}")
    finally:
        print("🏁 Программа завершена")
