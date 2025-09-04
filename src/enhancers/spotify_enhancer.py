#!/usr/bin/env python3
"""
Spotify API Integration для обогащения рэп-базы метаданными
"""
import os
import time
import sqlite3
import requests
import base64
import json
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging
from ..utils.config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
from ..models.models import SpotifyArtist, SpotifyTrack, SpotifyAudioFeatures, SpotifyEnrichmentResult

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/spotify_enhancement.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SpotifyEnhancer:
    """Класс для обогащения базы данных метаданными из Spotify API"""
    
    def __init__(self, client_id: str = None, client_secret: str = None, db_path: str = None):
        from ..utils.config import DB_PATH
        self.client_id = client_id or SPOTIFY_CLIENT_ID
        self.client_secret = client_secret or SPOTIFY_CLIENT_SECRET  
        self.db_path = db_path or str(DB_PATH)
        self.access_token = None
        self.token_expires_at = None
        self.api_calls_count = 0
        self.base_url = "https://api.spotify.com/v1"
        
        # Rate limiting
        self.requests_per_second = 10  # Консервативный лимит
        self.last_request_time = 0
        
        print("✅ SpotifyEnhancer инициализирован")  # Используем print вместо logger для русского текста
    
    def get_access_token(self) -> bool:
        """Получение access token через Client Credentials Flow"""
        try:
            # Проверяем, не истек ли токен
            if self.access_token and self.token_expires_at:
                if datetime.now() < self.token_expires_at:
                    return True
            
            # Кодируем credentials
            credentials = f"{self.client_id}:{self.client_secret}"
            credentials_b64 = base64.b64encode(credentials.encode()).decode()
            
            headers = {
                "Authorization": f"Basic {credentials_b64}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            data = {"grant_type": "client_credentials"}
            
            response = requests.post(
                "https://accounts.spotify.com/api/token",
                headers=headers,
                data=data,
                timeout=10
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data["access_token"]
                expires_in = token_data["expires_in"]
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)  # -60 сек для буфера
                logger.info("Access token obtained successfully")
                return True
            else:
                logger.error(f"Ошибка получения токена: {response.status_code}, {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Исключение при получении токена: {e}")
            return False
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Универсальный метод для запросов к Spotify API с rate limiting"""
        if not self.get_access_token():
            return None
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = requests.get(url, headers=headers, params=params, timeout=10)
            self.last_request_time = time.time()
            self.api_calls_count += 1
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limit exceeded
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limit exceeded, ждем {retry_after} секунд")
                time.sleep(retry_after)
                return self._make_request(endpoint, params)  # Retry
            elif response.status_code == 401:
                # Token expired
                self.access_token = None
                if self.get_access_token():
                    return self._make_request(endpoint, params)  # Retry with new token
                return None
            else:
                logger.error(f"API ошибка: {response.status_code}, {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Исключение при запросе к API: {e}")
            return None
    
    def search_artist(self, artist_name: str) -> Optional[SpotifyArtist]:
        """Поиск артиста по имени"""
        params = {
            "q": artist_name,
            "type": "artist",
            "limit": 1
        }
        
        data = self._make_request("search", params)
        if not data or "artists" not in data:
            return None
        
        artists = data["artists"]["items"]
        if not artists:
            logger.warning(f"Артист '{artist_name}' не найден в Spotify")
            return None
        
        artist_data = artists[0]
        
        return SpotifyArtist(
            spotify_id=artist_data["id"],
            name=artist_data["name"],
            genres=artist_data.get("genres", []),
            popularity=artist_data.get("popularity", 0),
            followers=artist_data.get("followers", {}).get("total", 0),
            image_url=artist_data.get("images", [{}])[0].get("url") if artist_data.get("images") else None,
            spotify_url=artist_data["external_urls"]["spotify"]
        )
    
    def search_track(self, track_name: str, artist_name: str, artist_id: str = None) -> Optional[SpotifyTrack]:
        """Поиск трека по названию и артисту"""
        query = f"track:\"{track_name}\" artist:\"{artist_name}\""
        params = {
            "q": query,
            "type": "track",
            "limit": 1
        }
        
        data = self._make_request("search", params)
        if not data or "tracks" not in data:
            return None
        
        tracks = data["tracks"]["items"]
        if not tracks:
            return None
        
        track_data = tracks[0]
        
        return SpotifyTrack(
            spotify_id=track_data["id"],
            name=track_data["name"],
            artist_id=track_data["artists"][0]["id"],
            album_name=track_data.get("album", {}).get("name"),
            release_date=track_data.get("album", {}).get("release_date"),
            duration_ms=track_data.get("duration_ms"),
            popularity=track_data.get("popularity", 0),
            explicit=track_data.get("explicit", False),
            spotify_url=track_data["external_urls"]["spotify"],
            preview_url=track_data.get("preview_url")
        )
    
    def get_audio_features(self, track_id: str) -> Optional[SpotifyAudioFeatures]:
        """Получение аудио-характеристик трека"""
        data = self._make_request(f"audio-features/{track_id}")
        if not data:
            return None
        
        return SpotifyAudioFeatures(
            danceability=data.get("danceability", 0.0),
            energy=data.get("energy", 0.0),
            valence=data.get("valence", 0.0),
            tempo=data.get("tempo", 0.0),
            acousticness=data.get("acousticness", 0.0),
            instrumentalness=data.get("instrumentalness", 0.0),
            speechiness=data.get("speechiness", 0.0),
            liveness=data.get("liveness", 0.0),
            loudness=data.get("loudness", 0.0)
        )
    
    def enhance_artist(self, artist_name: str) -> SpotifyEnrichmentResult:
        """Обогащение информации об артисте"""
        start_time = time.time()
        api_calls_start = self.api_calls_count
        
        try:
            artist_data = self.search_artist(artist_name)
            if not artist_data:
                return SpotifyEnrichmentResult(
                    success=False,
                    error_message=f"Артист '{artist_name}' не найден",
                    processing_time=time.time() - start_time,
                    api_calls_used=self.api_calls_count - api_calls_start
                )
            
            return SpotifyEnrichmentResult(
                success=True,
                artist_data=artist_data,
                processing_time=time.time() - start_time,
                api_calls_used=self.api_calls_count - api_calls_start
            )
            
        except Exception as e:
            return SpotifyEnrichmentResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
                api_calls_used=self.api_calls_count - api_calls_start
            )
    
    def enhance_track(self, track_name: str, artist_name: str, get_audio_features: bool = True) -> SpotifyEnrichmentResult:
        """Обогащение информации о треке"""
        start_time = time.time()
        api_calls_start = self.api_calls_count
        
        try:
            track_data = self.search_track(track_name, artist_name)
            if not track_data:
                return SpotifyEnrichmentResult(
                    success=False,
                    error_message=f"Трек '{track_name}' от '{artist_name}' не найден",
                    processing_time=time.time() - start_time,
                    api_calls_used=self.api_calls_count - api_calls_start
                )
            
            # Получаем аудио-характеристики если нужно
            if get_audio_features:
                audio_features = self.get_audio_features(track_data.spotify_id)
                track_data.audio_features = audio_features
            
            return SpotifyEnrichmentResult(
                success=True,
                track_data=track_data,
                processing_time=time.time() - start_time,
                api_calls_used=self.api_calls_count - api_calls_start
            )
            
        except Exception as e:
            return SpotifyEnrichmentResult(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
                api_calls_used=self.api_calls_count - api_calls_start
            )
    
    def get_db_artists(self) -> List[str]:
        """Получение списка уникальных артистов из базы"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT DISTINCT artist FROM songs ORDER BY artist")
            artists = [row[0] for row in cursor.fetchall()]
            
            conn.close()
            return artists
            
        except Exception as e:
            logger.error(f"Ошибка при получении артистов из базы: {e}")
            return []
    
    def create_spotify_tables(self):
        """Создание таблиц для хранения Spotify данных"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Таблица артистов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS spotify_artists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    artist_name TEXT NOT NULL,
                    spotify_id TEXT UNIQUE,
                    genres TEXT,  -- JSON список жанров
                    popularity INTEGER,
                    followers INTEGER,
                    image_url TEXT,
                    spotify_url TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(artist_name)
                )
            """)
            
            # Таблица треков
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS spotify_tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_id INTEGER,  -- Связь с таблицей songs
                    spotify_id TEXT UNIQUE,
                    artist_spotify_id TEXT,
                    album_name TEXT,
                    release_date TEXT,
                    duration_ms INTEGER,
                    popularity INTEGER,
                    explicit BOOLEAN,
                    spotify_url TEXT,
                    preview_url TEXT,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (song_id) REFERENCES songs (id)
                )
            """)
            
            # Таблица аудио-характеристик
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS spotify_audio_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_spotify_id TEXT UNIQUE,
                    danceability REAL,
                    energy REAL,
                    valence REAL,
                    tempo REAL,
                    acousticness REAL,
                    instrumentalness REAL,
                    speechiness REAL,
                    liveness REAL,
                    loudness REAL,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (track_spotify_id) REFERENCES spotify_tracks (spotify_id)
                )
            """)
            
            conn.commit()
            conn.close()
            print("✅ Таблицы Spotify созданы успешно")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
    
    def save_artist_to_db(self, artist_name: str, spotify_artist: SpotifyArtist):
        """Сохранение данных артиста в базу"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO spotify_artists 
                (artist_name, spotify_id, genres, popularity, followers, image_url, spotify_url)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                artist_name,
                spotify_artist.spotify_id,
                json.dumps(spotify_artist.genres),
                spotify_artist.popularity,
                spotify_artist.followers,
                spotify_artist.image_url,
                spotify_artist.spotify_url
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Ошибка при сохранении артиста {artist_name}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики обогащения"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Статистика артистов
            cursor.execute("SELECT COUNT(*) FROM spotify_artists")
            artists_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM spotify_tracks") 
            tracks_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM spotify_audio_features")
            audio_features_count = cursor.fetchone()[0]
            
            # Общее количество в основной базе
            cursor.execute("SELECT COUNT(DISTINCT artist) FROM songs")
            total_artists = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM songs")
            total_songs = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "spotify_artists": artists_count,
                "spotify_tracks": tracks_count,
                "spotify_audio_features": audio_features_count,
                "total_artists": total_artists,
                "total_songs": total_songs,
                "artists_coverage": f"{artists_count}/{total_artists} ({artists_count/total_artists*100:.1f}%)" if total_artists > 0 else "0%",
                "tracks_coverage": f"{tracks_count}/{total_songs} ({tracks_count/total_songs*100:.1f}%)" if total_songs > 0 else "0%",
                "api_calls_used": self.api_calls_count
            }
            
        except Exception as e:
            logger.error(f"Ошибка при получении статистики: {e}")
            return {}

def main():
    """Entry point для spotify enhancement."""
    print("🎵 Spotify Enhancer для рэп-базы")
    
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        print("❌ Для использования нужны SPOTIFY_CLIENT_ID и SPOTIFY_CLIENT_SECRET в .env")
        return
    
    enhancer = SpotifyEnhancer()
    print("✅ SpotifyEnhancer инициализирован")
    
    # Создаем таблицы если их нет
    enhancer.create_spotify_tables()
    
    # Получаем список артистов из базы
    artists = enhancer.get_db_artists()
    print(f"👤 Найдено {len(artists)} артистов для обогащения")
    
    enriched_count = 0
    for i, artist_name in enumerate(artists[:10], 1):  # Ограничиваем первыми 10 для теста
        print(f"🎤 Обрабатываем {i}/{min(10, len(artists))}: {artist_name}")
        
        result = enhancer.enhance_artist(artist_name)
        if result.success and result.artist_data:
            enhancer.save_artist_to_db(artist_name, result.artist_data)
            enriched_count += 1
            print(f"✅ {artist_name} обогащен")
        else:
            print(f"⚠️ {artist_name}: {result.error_message or 'Unknown error'}")
        
        # Небольшая пауза между запросами
        import time
        time.sleep(0.1)
    
    print(f"✅ Обогащено {enriched_count} из {min(10, len(artists))} артистов")
    
    # Показываем статистику
    stats = enhancer.get_stats()
    print(f"📊 Итоговая статистика: {stats}")

if __name__ == "__main__":
    main()
