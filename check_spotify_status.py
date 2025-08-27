#!/usr/bin/env python3
"""
Быстрая проверка статуса Spotify обогащения
"""
import sqlite3

def check_spotify_status():
    try:
        conn = sqlite3.connect('data/rap_lyrics.db')
        
        print("🎵 SPOTIFY STATUS CHECK")
        print("=" * 50)
        
        # Проверяем Spotify таблицы
        spotify_tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'spotify%'"
        ).fetchall()
        
        print(f"📋 Spotify таблицы: {[t[0] for t in spotify_tables]}")
        
        # Основная статистика
        total_songs = conn.execute('SELECT COUNT(*) FROM songs').fetchone()[0]
        total_artists = conn.execute('SELECT COUNT(DISTINCT artist) FROM songs').fetchone()[0]
        
        print(f"\n📊 БАЗА ДАННЫХ:")
        print(f"🎵 Всего песен: {total_songs:,}")
        print(f"👤 Уникальных артистов: {total_artists}")
        
        # Проверяем каждую Spotify таблицу
        if 'spotify_artists' in [t[0] for t in spotify_tables]:
            spotify_artists_count = conn.execute('SELECT COUNT(*) FROM spotify_artists').fetchone()[0]
            coverage = (spotify_artists_count / total_artists * 100) if total_artists > 0 else 0
            print(f"\n🎤 SPOTIFY ARTISTS:")
            print(f"✅ Обогащенных артистов: {spotify_artists_count}/{total_artists} ({coverage:.1f}%)")
            
            # Последние обогащенные артисты
            recent = conn.execute(
                'SELECT genius_name, followers, popularity FROM spotify_artists ORDER BY rowid DESC LIMIT 3'
            ).fetchall()
            print("🕒 Последние:")
            for name, followers, popularity in recent:
                followers_str = f"{followers:,}" if followers else "N/A"
                print(f"  • {name}: {followers_str} followers, popularity {popularity}")
        
        if 'spotify_tracks' in [t[0] for t in spotify_tables]:
            spotify_tracks_count = conn.execute('SELECT COUNT(*) FROM spotify_tracks').fetchone()[0]
            track_coverage = (spotify_tracks_count / total_songs * 100) if total_songs > 0 else 0
            print(f"\n🎼 SPOTIFY TRACKS:")
            print(f"✅ Обогащенных треков: {spotify_tracks_count:,}/{total_songs:,} ({track_coverage:.1f}%)")
        
        if 'spotify_audio_features' in [t[0] for t in spotify_tables]:
            audio_features_count = conn.execute('SELECT COUNT(*) FROM spotify_audio_features').fetchone()[0]
            print(f"\n🎚️ AUDIO FEATURES:")
            print(f"✅ Треков с аудио фичами: {audio_features_count:,}")
        
        conn.close()
        
        # Оценка статуса
        print(f"\n🎯 ИТОГ:")
        if spotify_artists_count >= total_artists * 0.99:
            print("✅ Spotify обогащение артистов ЗАВЕРШЕНО!")
        elif spotify_artists_count >= total_artists * 0.8:
            print("🔄 Spotify обогащение артистов почти завершено")
        else:
            print("🔄 Spotify обогащение артистов в процессе")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    check_spotify_status()
