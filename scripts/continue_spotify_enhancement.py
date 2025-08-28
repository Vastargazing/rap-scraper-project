#!/usr/bin/env python3
"""Продолжение Spotify enhancement для оставшихся артистов и треков. 
находит артистов в БД и обогащает оставшихся (и затем берет выборку 
до 50 треков для трек-обогащения), он более осторожный / incremental."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import sqlite3
import time
from src.enhancers.spotify_enhancer import SpotifyEnhancer
from src.utils.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET

def main():
    """Продолжение обогащения базы данных."""
    print("🔄 Продолжение Spotify Enhancement")
    
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("❌ Нужны SPOTIFY_CLIENT_ID и SPOTIFY_CLIENT_SECRET в .env")
        return
    
    enhancer = SpotifyEnhancer()
    
    # Получаем список всех артистов
    all_artists = enhancer.get_db_artists()
    print(f"👤 Всего артистов в базе: {len(all_artists)}")
    
    # Получаем уже обогащенных артистов
    conn = sqlite3.connect(enhancer.db_path)
    cursor = conn.execute("SELECT artist_name FROM spotify_artists")
    enriched_artists = {row[0] for row in cursor.fetchall()}
    print(f"✅ Уже обогащено артистов: {len(enriched_artists)}")
    
    # Находим оставшихся
    remaining_artists = [artist for artist in all_artists if artist not in enriched_artists]
    print(f"🔄 Осталось обогатить: {len(remaining_artists)} артистов")
    
    if remaining_artists:
        print("🎤 Обогащаем оставшихся артистов...")
        enriched_count = 0
        
        for i, artist_name in enumerate(remaining_artists, 1):
            print(f"🎤 {i}/{len(remaining_artists)}: {artist_name}")
            
            result = enhancer.enhance_artist(artist_name)
            if result.success and result.artist_data:
                enhancer.save_artist_to_db(artist_name, result.artist_data)
                enriched_count += 1
                print(f"✅ {artist_name} обогащен")
            else:
                print(f"⚠️ {artist_name}: {result.error_message or 'Не найден'}")
            
            time.sleep(0.2)  # Пауза между запросами
        
        print(f"🎯 Обогащено {enriched_count} новых артистов")
    else:
        print("✅ Все артисты уже обогащены!")
    
    # Переходим к обогащению треков
    print("\n🎵 Начинаем обогащение треков...")
    
    # Получаем треки для обогащения (начинаем с первых 50)
    cursor = conn.execute("""
        SELECT DISTINCT s.artist, s.title 
        FROM songs s 
        LEFT JOIN spotify_tracks st ON s.id = st.song_id
        WHERE st.id IS NULL
        LIMIT 50
    """)
    tracks_to_enrich = cursor.fetchall()
    conn.close()
    
    print(f"🎵 Найдено {len(tracks_to_enrich)} треков для обогащения")
    
    if tracks_to_enrich:
        enriched_tracks = 0
        for i, (artist, title) in enumerate(tracks_to_enrich, 1):
            print(f"🎵 {i}/{len(tracks_to_enrich)}: {artist} - {title}")
            
            # Обогащаем без audio features чтобы избежать 403 ошибок
            result = enhancer.enhance_track(title, artist, get_audio_features=False)
            if result.success:
                enriched_tracks += 1
                print(f"✅ Трек обогащен")
            else:
                print(f"⚠️ Ошибка: {result.error_message or 'Не найден'}")
            
            time.sleep(0.3)  # Пауза между запросами треков
        
        print(f"🎯 Обогащено {enriched_tracks} из {len(tracks_to_enrich)} треков")
    
    # Показываем финальную статистику
    stats = enhancer.get_stats()
    print(f"\n📊 Финальная статистика: {stats}")

if __name__ == "__main__":
    main()
