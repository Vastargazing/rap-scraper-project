#!/usr/bin/env python3
"""
Создание красивого CLI showcase для GitHub профиля (без matplotlib)
"""

import os
import sqlite3


def get_database_stats():
    """Получение статистики из базы данных"""
    db_path = "data/rap_lyrics.db"

    if not os.path.exists(db_path):
        print("❌ Database not found")
        return None

    conn = sqlite3.connect(db_path)
    stats = {}

    try:
        # Основная статистика
        cursor = conn.execute("SELECT COUNT(*) FROM tracks")
        stats["total_tracks"] = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(DISTINCT artist) FROM tracks")
        stats["total_artists"] = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM ai_analysis")
        stats["ai_analyses"] = cursor.fetchone()[0]

        # Топ артисты
        cursor = conn.execute("""
            SELECT artist, COUNT(*) as track_count 
            FROM tracks 
            GROUP BY artist 
            ORDER BY track_count DESC 
            LIMIT 10
        """)
        stats["top_artists"] = cursor.fetchall()

        # Размер базы
        cursor = conn.execute(
            "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
        )
        size_bytes = cursor.fetchone()[0]
        stats["db_size_mb"] = size_bytes / (1024 * 1024)

        # Жанры из AI анализа
        cursor = conn.execute("""
            SELECT genre, COUNT(*) as count
            FROM ai_analysis 
            WHERE genre IS NOT NULL AND genre != ''
            GROUP BY genre
            ORDER BY count DESC
            LIMIT 8
        """)
        stats["genres"] = cursor.fetchall()

    except Exception as e:
        print(f"Error getting stats: {e}")
        return None
    finally:
        conn.close()

    return stats


def create_ascii_chart(data, title, max_width=50):
    """Создание ASCII графика"""
    print(f"\n📊 {title}")
    print("─" * (max_width + 20))

    if not data:
        print("   No data available")
        return

    max_value = max([item[1] for item in data])

    for name, value in data:
        # Обрезаем длинные имена
        display_name = name[:15] + "..." if len(name) > 15 else name
        display_name = display_name.ljust(18)

        # Создаем бар
        bar_length = int((value / max_value) * max_width)
        bar = "█" * bar_length + "░" * (max_width - bar_length)

        # Форматируем значение
        value_str = f"{value:,}" if value >= 1000 else str(value)

        print(f"   {display_name} │{bar}│ {value_str}")


def create_beautiful_cli_showcase():
    """Создание красивого CLI showcase"""

    # Заголовок
    print("\n" + "═" * 85)
    print(
        "🎤                 RAP SCRAPER PROJECT - ML PIPELINE DASHBOARD                  🎤"
    )
    print("═" * 85)
    print("🚀 Production-ready ML system for hip-hop culture analysis")
    print("🎯 Built by a creative artist turned AI Engineer")

    # Получаем статистику
    stats = get_database_stats()

    if not stats:
        print("❌ Could not load database statistics")
        return

    # Основные метрики в красивом формате
    print(f"""
╭─────────────────── 📊 PROJECT METRICS ────────────────────╮
│                                                           │
│  🎵  Total Tracks:     {stats["total_tracks"]:>8,}                      │
│  👤  Artists:          {stats["total_artists"]:>8,}                      │  
│  🤖  AI Analyses:      {stats["ai_analyses"]:>8,}                      │
│  💾  Database Size:    {stats["db_size_mb"]:>8.1f} MB                  │
│  🔥  Success Rate:     {(stats["ai_analyses"] / stats["total_tracks"] * 100):>8.1f}%                   │
│                                                           │
╰───────────────────────────────────────────────────────────╯""")

    # Топ артисты
    create_ascii_chart(stats["top_artists"][:8], "TOP ARTISTS BY TRACK COUNT")

    # Жанры
    if stats["genres"]:
        create_ascii_chart(stats["genres"], "GENRE DISTRIBUTION (AI ANALYZED)")

    # Технический стек
    print("""
╭─────────────────── 🛠️  TECH STACK ────────────────────╮
│                                                      │
│  🐍 Python 3.13      📊 Pydantic Models             │
│  🕷️  Genius API       🎵 Spotify Web API             │
│  🤖 Gemma 27B        🔄 Async Processing             │
│  💾 SQLite DB        ⚡ CLI Interface                │
│                                                      │
╰──────────────────────────────────────────────────────╯""")

    # Pipeline статус
    print("""
╭─────────────────── ⚙️  ML PIPELINE STATUS ────────────────────╮
│                                                             │
│  ✅ Data Scraping      ████████████████████████████ 100%    │
│  🎵 Spotify Enrichment ███████████████████████████░  95%    │
│  🤖 AI Analysis        ████████░░░░░░░░░░░░░░░░░░░░  30%    │
│  🔧 Feature Engineering ███░░░░░░░░░░░░░░░░░░░░░░░░░  10%    │
│                                                             │
╰─────────────────────────────────────────────────────────────╯""")

    # Текущие задачи
    print("""
╭─────────────────── 🎯 CURRENT GOALS ────────────────────╮
│                                                        │
│  ✅ 52K+ Tracks Collected                              │
│  🔄 Migrating to GPT-4o (content filtering issues)    │
│  📈 Scaling to 100K+ tracks with monitoring           │
│  🤖 Conditional lyrics generation model               │
│  🚀 Production deployment pipeline                    │
│                                                        │
╰────────────────────────────────────────────────────────╯""")

    # CLI команды
    print("""
╭─────────────────── 🚀 AVAILABLE COMMANDS ────────────────────╮
│                                                             │
│  📊 python scripts/rap_scraper_cli.py status               │
│  🕷️  python scripts/rap_scraper_cli.py scraping             │
│  🎵 python scripts/rap_scraper_cli.py spotify --continue   │
│  🤖 python scripts/rap_scraper_cli.py analysis --analyzer  │
│  📈 python scripts/rap_scraper_cli.py monitoring           │
│                                                             │
╰─────────────────────────────────────────────────────────────╯""")

    # Подпись
    print("\n" + "─" * 85)
    print("🎨 Creative Background + AI Engineering = Cultural Intelligence at Scale")
    print("🤝 Open to collaborations and learning opportunities!")
    print("─" * 85 + "\n")


if __name__ == "__main__":
    create_beautiful_cli_showcase()
