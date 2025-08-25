#!/usr/bin/env python3
"""
Массовое обогащение базы данных метаданными из Spotify API
"""
import os
import sys
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from spotify_enhancer import SpotifyEnhancer

# Загружаем переменные окружения
load_dotenv()

class BulkSpotifyEnhancement:
    """Класс для массового обогащения базы"""
    
    def __init__(self, enhancer: SpotifyEnhancer):
        self.enhancer = enhancer
        self.stats = {
            'artists_processed': 0,
            'artists_success': 0,
            'artists_failed': 0,
            'tracks_processed': 0,
            'tracks_success': 0,
            'tracks_failed': 0,
            'total_api_calls': 0,
            'start_time': None,
            'errors': []
        }
    
    def enhance_all_artists(self, limit: int = None):
        """Обогащение всех артистов из базы"""
        print("🎤 Начинаем обогащение артистов...")
        
        artists = self.enhancer.get_db_artists()
        if limit:
            artists = artists[:limit]
            print(f"📊 Ограничиваемся первыми {limit} артистами")
        
        print(f"📋 Всего артистов для обработки: {len(artists)}")
        
        self.stats['start_time'] = datetime.now()
        
        for i, artist in enumerate(artists, 1):
            print(f"\n[{i}/{len(artists)}] Обрабатываем: {artist}")
            
            result = self.enhancer.enhance_artist(artist)
            self.stats['artists_processed'] += 1
            self.stats['total_api_calls'] += result.api_calls_used
            
            if result.success:
                self.stats['artists_success'] += 1
                
                # Сохраняем в базу
                self.enhancer.save_artist_to_db(artist, result.artist_data)
                
                print(f"  ✅ Успешно: {result.artist_data.name}")
                print(f"     📊 Популярность: {result.artist_data.popularity}")
                print(f"     👥 Подписчики: {result.artist_data.followers:,}")
                print(f"     🎵 Жанры: {', '.join(result.artist_data.genres[:3])}")
                
            else:
                self.stats['artists_failed'] += 1
                self.stats['errors'].append(f"Артист '{artist}': {result.error_message}")
                print(f"  ❌ Ошибка: {result.error_message}")
            
            print(f"     ⏱️ Время: {result.processing_time:.2f}с, API вызовов: {result.api_calls_used}")
            
            # Каждые 10 артистов показываем прогресс
            if i % 10 == 0:
                self._show_progress()
            
            # Небольшая пауза для вежливости к API
            time.sleep(0.1)
        
        self._show_final_stats()
    
    def enhance_all_tracks(self):
        """Полное обогащение всех треков с автоматической остановкой при лимитах"""
        print("🎵 МАССОВОЕ ОБОГАЩЕНИЕ ВСЕХ ТРЕКОВ...")
        print("⚡ Скрипт остановится автоматически при достижении лимитов API")
        
        import sqlite3
        conn = sqlite3.connect(self.enhancer.db_path)
        cursor = conn.cursor()
        
        # Получаем все треки, которых нет в spotify_tracks
        cursor.execute("""
            SELECT s.id, s.title, s.artist 
            FROM songs s
            LEFT JOIN spotify_tracks st ON s.id = st.song_id
            WHERE st.song_id IS NULL
            ORDER BY s.id
        """)
        
        tracks = cursor.fetchall()
        conn.close()
        
        print(f"📋 Найдено {len(tracks)} треков для обработки")
        print(f"🚀 Начинаем массовую обработку...")
        
        try:
            for i, (song_id, title, artist) in enumerate(tracks, 1):
                if i % 100 == 0:  # Показываем прогресс каждые 100 треков
                    print(f"\n🔄 ПРОГРЕСС: {i}/{len(tracks)} треков обработано")
                    print(f"✅ Успешно: {self.stats['tracks_success']}")
                    print(f"❌ Ошибок: {self.stats['tracks_failed']}")
                    print(f"🌐 API вызовов: {self.stats['total_api_calls']}")
                
                result = self.enhancer.enhance_track(title, artist, get_audio_features=False)
                self.stats['tracks_processed'] += 1
                self.stats['total_api_calls'] += result.api_calls_used
                
                if result.success:
                    self.stats['tracks_success'] += 1
                    track = result.track_data
                    self._save_track_to_db(song_id, track)
                    if i % 10 == 0:  # Компактный вывод каждые 10 треков
                        print(f"[{i}] ✅ {artist} - {title}")
                else:
                    self.stats['tracks_failed'] += 1
                    if result.error_message and "rate limit" in result.error_message.lower():
                        print(f"\n🛑 ЛИМИТ API ДОСТИГНУТ!")
                        print(f"📊 Обработано {i} из {len(tracks)} треков")
                        break
                    elif i % 10 == 0:  # Показываем ошибки реже
                        print(f"[{i}] ❌ {artist} - {title}")
                
                # Короткая пауза между запросами
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Обработка остановлена пользователем")
            print(f"📊 Обработано {self.stats['tracks_processed']} треков")
        
        print(f"\n🎯 МАССОВАЯ ОБРАБОТКА ЗАВЕРШЕНА!")
        self._show_final_stats()

    def enhance_sample_tracks(self, sample_size: int = 50):
        """Обогащение образца треков для тестирования"""
        print(f"🎵 Обогащение образца из {sample_size} треков...")
        
        import sqlite3
        conn = sqlite3.connect(self.enhancer.db_path)
        cursor = conn.cursor()
        
        # Получаем случайную выборку треков
        cursor.execute("""
            SELECT id, title, artist 
            FROM songs 
            ORDER BY RANDOM() 
            LIMIT ?
        """, (sample_size,))
        
        tracks = cursor.fetchall()
        conn.close()
        
        print(f"📋 Получено {len(tracks)} треков для обработки")
        
        for i, (song_id, title, artist) in enumerate(tracks, 1):
            print(f"\n[{i}/{len(tracks)}] {artist} - {title}")
            
            result = self.enhancer.enhance_track(title, artist, get_audio_features=False)
            self.stats['tracks_processed'] += 1
            self.stats['total_api_calls'] += result.api_calls_used
            
            if result.success:
                self.stats['tracks_success'] += 1
                track = result.track_data
                
                # Сохраняем трек в базу
                self._save_track_to_db(song_id, track)
                
                print(f"  ✅ Найден и сохранен: {track.name}")
                print(f"     📊 Популярность: {track.popularity}")
                print(f"     💿 Альбом: {track.album_name}")
                print(f"     📅 Релиз: {track.release_date}")
            else:
                self.stats['tracks_failed'] += 1
                print(f"  ❌ {result.error_message}")
            
            time.sleep(0.2)  # Больше паузы для треков (больше API вызовов)
    
    def _save_track_to_db(self, song_id: int, track):
        """Сохранение трека в базу данных"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.enhancer.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO spotify_tracks 
                (song_id, spotify_id, artist_spotify_id, album_name, 
                 release_date, duration_ms, popularity, explicit, 
                 spotify_url, preview_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                song_id,
                track.spotify_id,
                track.artist_id,
                track.album_name,
                track.release_date,
                track.duration_ms,
                track.popularity,
                track.explicit,
                track.spotify_url,
                track.preview_url
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"     ⚠️ Ошибка сохранения в БД: {e}")
    
    def _show_progress(self):
        """Показать текущий прогресс"""
        if self.stats['start_time']:
            elapsed = datetime.now() - self.stats['start_time']
            print(f"\n📊 ПРОГРЕСС:")
            print(f"   ⏱️ Время работы: {elapsed}")
            print(f"   🎤 Артисты: {self.stats['artists_success']}/{self.stats['artists_processed']}")
            print(f"   🎵 Треки: {self.stats['tracks_success']}/{self.stats['tracks_processed']}")
            print(f"   📡 API вызовов: {self.stats['total_api_calls']}")
    
    def _show_final_stats(self):
        """Показать финальную статистику"""
        print("\n" + "="*50)
        print("📈 ФИНАЛЬНАЯ СТАТИСТИКА")
        print("="*50)
        
        if self.stats['start_time']:
            elapsed = datetime.now() - self.stats['start_time']
            print(f"⏱️ Общее время: {elapsed}")
        
        print(f"🎤 Артисты:")
        print(f"   • Обработано: {self.stats['artists_processed']}")
        print(f"   • Успешно: {self.stats['artists_success']}")
        print(f"   • Ошибки: {self.stats['artists_failed']}")
        
        if self.stats['tracks_processed'] > 0:
            print(f"🎵 Треки:")
            print(f"   • Обработано: {self.stats['tracks_processed']}")
            print(f"   • Успешно: {self.stats['tracks_success']}")
            print(f"   • Ошибки: {self.stats['tracks_failed']}")
        
        print(f"📡 Всего API вызовов: {self.stats['total_api_calls']}")
        
        if self.stats['errors']:
            print(f"\n❌ Ошибки ({len(self.stats['errors'])}):")
            for error in self.stats['errors'][-5:]:  # Показываем последние 5
                print(f"   • {error}")
        
        # Обновленная статистика базы
        db_stats = self.enhancer.get_stats()
        print(f"\n📊 СТАТИСТИКА БАЗЫ:")
        for key, value in db_stats.items():
            print(f"   • {key}: {value}")

def main():
    print("🚀 Bulk Spotify Enhancement")
    print("=" * 50)
    
    # Проверяем credentials
    client_id = os.getenv('SPOTIFY_CLIENT_ID')
    client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
    
    if not client_id or not client_secret or client_id == 'your_client_id_here':
        print("❌ Spotify credentials не найдены или не заполнены в .env файле!")
        print("Сначала настройте credentials в .env и запустите: python setup_spotify.py")
        return
    
    # Создаем enhancer
    enhancer = SpotifyEnhancer(client_id, client_secret)
    bulk_enhancer = BulkSpotifyEnhancement(enhancer)
    
    # Меню выбора
    print("\nВыберите режим:")
    print("1. Тест на первых 10 артистах")
    print("2. Обогащение всех артистов")
    print("3. Обогащение образца треков (50 штук)")
    print("4. 🚀 ОБОГАЩЕНИЕ ВСЕХ ТРЕКОВ (с автостопом)")
    print("5. Показать текущую статистику")
    
    choice = input("\nВаш выбор (1-5): ").strip()
    
    if choice == "1":
        print("\n🧪 ТЕСТОВЫЙ РЕЖИМ")
        bulk_enhancer.enhance_all_artists(limit=10)
    
    elif choice == "2":
        confirm = input("\n⚠️ Это обработает ВСЕ 259 артистов. Продолжить? (y/N): ")
        if confirm.lower() == 'y':
            print("\n🚀 ПОЛНОЕ ОБОГАЩЕНИЕ")
            bulk_enhancer.enhance_all_artists()
        else:
            print("Отменено")
    
    elif choice == "3":
        print("\n🎵 ТЕСТ ТРЕКОВ")
        bulk_enhancer.enhance_sample_tracks()
    
    elif choice == "4":
        confirm = input("\n⚠️ Это обработает ВСЕ треки (~48K). API остановит при лимите. Продолжить? (y/N): ")
        if confirm.lower() == 'y':
            print("\n🚀 МАССОВОЕ ОБОГАЩЕНИЕ ТРЕКОВ")
            bulk_enhancer.enhance_all_tracks()
        else:
            print("Отменено")
    
    elif choice == "5":
        print("\n📊 ТЕКУЩАЯ СТАТИСТИКА")
        stats = enhancer.get_stats()
        for key, value in stats.items():
            print(f"  • {key}: {value}")
    
    else:
        print("❌ Неверный выбор")

if __name__ == "__main__":
    main()
