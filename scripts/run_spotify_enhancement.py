#!/usr/bin/env python3
"""
#!/usr/bin/env python3
🎵 Точка входа для обогащения данных Spotify

НАЗНАЧЕНИЕ:
- Запуск процесса обогащения треков и артистов метаданными Spotify
- Автоматическое обновление базы данных

ИСПОЛЬЗОВАНИЕ:
python scripts/run_spotify_enhancement.py           # Запуск полного обогащения

ЗАВИСИМОСТИ:
- Python 3.8+
- src/enhancers/spotify_enhancer.py
- data/rap_lyrics.db
- Spotify API ключи

РЕЗУЛЬТАТ:
- Обновленные записи в базе данных с метаданными Spotify
- Логирование прогресса и ошибок

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.enhancers.spotify_enhancer import main

if __name__ == "__main__":
    main()


