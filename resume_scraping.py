#!/usr/bin/env python3
"""
Скрипт для возобновления скрапинга с места остановки
"""
import sqlite3
import json
import os

def get_processed_artists():
    """Получает список уже обработанных артистов из базы"""
    if not os.path.exists("rap_lyrics.db"):
        return []
    
    conn = sqlite3.connect("rap_lyrics.db")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT artist FROM songs")
    processed = [row[0] for row in cursor.fetchall()]
    conn.close()
    return processed

def get_remaining_artists():
    """Возвращает список артистов, которые еще не обработаны"""
    # Загружаем полный список
    with open("rap_artists.json", "r", encoding="utf-8") as f:
        all_artists = json.load(f)
    
    # Получаем обработанных
    processed = get_processed_artists()
    
    # Находим необработанных
    remaining = [artist for artist in all_artists if artist not in processed]
    
    print(f"📊 Статистика:")
    print(f"  Всего артистов: {len(all_artists)}")
    print(f"  Обработано: {len(processed)}")
    print(f"  Осталось: {len(remaining)}")
    
    if processed:
        print(f"\n✅ Уже обработаны:")
        for artist in processed:
            print(f"  • {artist}")
    
    if remaining:
        print(f"\n⏳ Ожидают обработки:")
        for i, artist in enumerate(remaining[:10], 1):  # Показываем первых 10
            print(f"  {i}. {artist}")
        if len(remaining) > 10:
            print(f"  ... и еще {len(remaining) - 10}")
    
    return remaining

def save_remaining_artists(filename="remaining_artists.json"):
    """Сохраняет список необработанных артистов в файл"""
    remaining = get_remaining_artists()
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(remaining, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Список необработанных артистов сохранен в {filename}")
    return filename

if __name__ == "__main__":
    save_remaining_artists()
