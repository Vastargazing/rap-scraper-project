#!/usr/bin/env python3
"""
🎵 Настройка и тестирование интеграции с Spotify API

НАЗНАЧЕНИЕ:
- Проверка и настройка Spotify API credentials
- Тестирование доступа к Spotify API
- Диагностика интеграции для обогащения треков

ИСПОЛЬЗОВАНИЕ:
python src/utils/setup_spotify.py

ЗАВИСИМОСТИ:
- Python 3.8+
- python-dotenv
- src/enhancers/spotify_enhancer.py
- .env файл с SPOTIFY_CLIENT_ID и SPOTIFY_CLIENT_SECRET

РЕЗУЛЬТАТ:
- Проверка и вывод статуса credentials
- Тест доступа к API
- Диагностика ошибок интеграции

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""
import os
import json
from dotenv import load_dotenv
from spotify_enhancer import SpotifyEnhancer

# Загружаем переменные окружения
load_dotenv()

def check_credentials():
    """Проверка существующих credentials"""
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if client_id and client_secret and client_id != 'your_client_id_here':
        print("✅ Найдены Spotify credentials в .env файле")
        return client_id, client_secret
    else:
        print("⚠️ Spotify credentials не найдены или не заполнены в .env файле")
        return None, None

def test_api_connection(enhancer: SpotifyEnhancer):
    """Тестирование подключения к API"""
    print("\n🔍 Тестирование подключения к Spotify API...")
    
    # Тест получения токена
    if not enhancer.get_access_token():
        print("❌ Не удалось получить access token")
        return False
    
    print("✅ Access token получен")
    
    # Тест поиска популярного артиста
    print("🎯 Тестирование поиска артиста...")
    result = enhancer.enhance_artist("Drake")
    
    if result.success:
        artist = result.artist_data
        print(f"✅ Артист найден: {artist.name}")
        print(f"   📊 Популярность: {artist.popularity}")
        print(f"   👥 Подписчики: {artist.followers:,}")
        print(f"   🎵 Жанры: {', '.join(artist.genres[:3])}")
        print(f"   ⏱️ Время обработки: {result.processing_time:.2f}с")
        return True
    else:
        print(f"❌ Ошибка поиска: {result.error_message}")
        return False

def test_track_search(enhancer: SpotifyEnhancer):
    """Тестирование поиска треков"""
    print("\n🎵 Тестирование поиска треков...")
    
    result = enhancer.enhance_track("Hotline Bling", "Drake", get_audio_features=True)
    
    if result.success:
        track = result.track_data
        print(f"✅ Трек найден: {track.name}")
        print(f"   📊 Популярность: {track.popularity}")
        print(f"   💿 Альбом: {track.album_name}")
        print(f"   📅 Релиз: {track.release_date}")
        
        if track.audio_features:
            af = track.audio_features
            print(f"   🎶 Аудио-характеристики:")
            print(f"      • Танцевальность: {af.danceability:.2f}")
            print(f"      • Энергичность: {af.energy:.2f}")
            print(f"      • Темп: {af.tempo:.0f} BPM")
            print(f"      • Валентность: {af.valence:.2f}")
        
        return True
    else:
        print(f"❌ Ошибка поиска трека: {result.error_message}")
        return False

def show_database_preview(enhancer: SpotifyEnhancer):
    """Показать превью артистов из базы"""
    print("\n📋 Превью артистов из базы данных:")
    
    artists = enhancer.get_db_artists()
    print(f"Всего уникальных артистов: {len(artists)}")
    
    # Показываем первых 10
    print("\nПервые 10 артистов:")
    for i, artist in enumerate(artists[:10], 1):
        print(f"  {i}. {artist}")
    
    if len(artists) > 10:
        print(f"  ... и еще {len(artists) - 10} артистов")
    
    return artists

def main():
    print("🚀 Spotify API Integration Setup")
    print("=" * 50)
    
    # Проверяем существующие credentials
    client_id, client_secret = check_credentials()
    
    if not client_id or not client_secret:
        print("\n📝 Инструкция по настройке:")
        print("1. Откройте файл .env")
        print("2. Замените 'your_client_id_here' на ваш Client ID")
        print("3. Замените 'your_client_secret_here' на ваш Client Secret")
        print("4. Сохраните файл и перезапустите скрипт")
        print("\n🔗 Получить credentials: https://developer.spotify.com/dashboard/applications")
        return
    
    # Создаем enhancer
    enhancer = SpotifyEnhancer(client_id, client_secret)
    
    # Создаем таблицы
    print("\n📊 Создание таблиц для Spotify данных...")
    enhancer.create_spotify_tables()
    
    # Тестируем API
    if not test_api_connection(enhancer):
        print("\n❌ Тестирование API провалилось. Проверьте credentials.")
        return
    
    # Тестируем поиск треков
    test_track_search(enhancer)
    
    # Показываем базу данных
    artists = show_database_preview(enhancer)
    
    # Показываем статистику
    print("\n📈 Текущая статистика:")
    stats = enhancer.get_stats()
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    print(f"\n✅ Spotify API готов к использованию!")
    print(f"📊 API вызовов использовано: {enhancer.api_calls_count}")
    
    # Предлагаем запустить обогащение
    print("\n🚀 Готовы начать обогащение базы?")
    print("Используйте: python bulk_spotify_enhancement.py")

if __name__ == "__main__":
    main()
