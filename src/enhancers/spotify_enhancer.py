#!/usr/bin/env python3
"""
🎵 Pure PostgreSQL Spotify Enhancer

НАЗНАЧЕНИЕ:
- Интеграция с Spotify API для обогащения треков
- Получение метаданных и audio features
- Сохранение в PostgreSQL (НИКАКОГО SQLite!)
- Async/await архитектура

ИСПОЛЬЗОВАНИЕ:
from src.enhancers.spotify_enhancer import SpotifyEnhancer

ЗАВИСИМОСТИ:
- PostgreSQL через database/postgres_adapter.py
- spotipy для Spotify Web API
- SPOTIFY_CLIENT_ID/SPOTIFY_CLIENT_SECRET в env

РЕЗУЛЬТАТ:
- Обогащенные треки в spotify_data таблице
- Логи в logs/spotify_enhancement.log

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Настройка путей
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.postgres_adapter import PostgreSQLManager

# Простое логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/spotify_enhancement.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('spotify_enhancer')

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False
    logger.warning("⚠️ Spotipy не установлен: pip install spotipy")


class SpotifyEnhancer:
    """
    Pure PostgreSQL Spotify Enhancer
    
    🔥 ОСОБЕННОСТИ:
    - 100% PostgreSQL, никакого SQLite
    - Async/await операции
    - Spotify Web API интеграция
    - Rate limiting и кэширование
    """
    
    def __init__(self):
        # PostgreSQL менеджер
        self.db_manager = PostgreSQLManager()
        
        # Spotify API
        self.spotify = None
        if SPOTIPY_AVAILABLE:
            self.spotify = self._setup_spotify_client()
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        self.max_requests_per_minute = 100
        
        # Кэширование поиска
        self.search_cache = {}
        
        logger.info("✅ Pure PostgreSQL Spotify Enhancer готов к работе")
    
    def _setup_spotify_client(self):
        """Настройка Spotify API клиента"""
        try:
            client_id = os.getenv('SPOTIFY_CLIENT_ID')
            client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
            
            if not client_id or not client_secret:
                logger.warning("⚠️ Spotify credentials не найдены в окружении")
                return None
            
            auth_manager = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
            
            spotify = spotipy.Spotify(auth_manager=auth_manager)
            
            # Тест подключения
            spotify.search(q='test', type='track', limit=1)
            logger.info("✅ Spotify API подключен")
            
            return spotify
            
        except Exception as e:
            logger.error(f"❌ Ошибка Spotify API: {e}")
            return None
    
    async def initialize(self):
        """Инициализация PostgreSQL соединения"""
        try:
            await self.db_manager.initialize()
            logger.info("✅ PostgreSQL инициализирован")
        except Exception as e:
            logger.error(f"❌ Ошибка PostgreSQL: {e}")
            raise
    
    def _apply_rate_limiting(self):
        """Rate limiting для Spotify API"""
        current_time = time.time()
        
        if self.request_count >= self.max_requests_per_minute:
            if current_time - self.last_request_time < 60:
                sleep_time = 60 - (current_time - self.last_request_time)
                logger.info(f"⏰ Rate limit: ждем {sleep_time:.1f}с")
                time.sleep(sleep_time)
                self.request_count = 0
        
        if current_time - self.last_request_time >= 60:
            self.request_count = 0
            self.last_request_time = current_time
        
        self.request_count += 1
    
    async def search_track(self, artist: str, title: str) -> Optional[Dict[str, Any]]:
        """Поиск трека в Spotify"""
        if not self.spotify:
            return None
        
        try:
            # Проверяем кэш
            cache_key = f"{artist.lower()}||{title.lower()}"
            if cache_key in self.search_cache:
                return self.search_cache[cache_key]
            
            # Rate limiting
            self._apply_rate_limiting()
            
            # Поиск
            query = f'artist:"{artist}" track:"{title}"'
            results = self.spotify.search(q=query, type='track', limit=5)
            
            if not results['tracks']['items']:
                # Упрощенный поиск
                results = self.spotify.search(q=f"{artist} {title}", type='track', limit=5)
            
            if results['tracks']['items']:
                best_match = self._find_best_match(results['tracks']['items'], artist, title)
                self.search_cache[cache_key] = best_match
                return best_match
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска {artist} - {title}: {e}")
            return None
    
    def _find_best_match(self, tracks: List[Dict], target_artist: str, target_title: str) -> Dict[str, Any]:
        """Находит лучший матч из результатов поиска"""
        def similarity(s1: str, s2: str) -> float:
            s1, s2 = s1.lower().strip(), s2.lower().strip()
            if s1 == s2:
                return 1.0
            if s1 in s2 or s2 in s1:
                return 0.8
            
            words1, words2 = set(s1.split()), set(s2.split())
            common = len(words1.intersection(words2))
            total = len(words1.union(words2))
            return common / total if total > 0 else 0.0
        
        best_score = 0.0
        best_track = tracks[0]
        
        for track in tracks:
            artist_name = track['artists'][0]['name'] if track['artists'] else ''
            track_name = track['name']
            
            artist_sim = similarity(target_artist, artist_name)
            title_sim = similarity(target_title, track_name)
            
            score = (artist_sim * 0.4) + (title_sim * 0.6)
            
            if score > best_score:
                best_score = score
                best_track = track
        
        return best_track
    
    async def get_audio_features(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Получение audio features"""
        if not self.spotify:
            return None
        
        try:
            self._apply_rate_limiting()
            features = self.spotify.audio_features([track_id])
            return features[0] if features and features[0] else None
        except Exception as e:
            logger.error(f"❌ Ошибка audio features {track_id}: {e}")
            return None
    
    async def enhance_song(self, song_id: int, artist: str, title: str) -> bool:
        """Обогащение одной песни"""
        try:
            # Проверяем, есть ли уже данные
            if await self._has_spotify_data(song_id):
                return True
            
            # Поиск трека
            track_data = await self.search_track(artist, title)
            if not track_data:
                return False
            
            # Audio features
            audio_features = await self.get_audio_features(track_data['id'])
            
            # Сохранение в PostgreSQL
            return await self._save_spotify_data(song_id, track_data, audio_features)
            
        except Exception as e:
            logger.error(f"❌ Ошибка обогащения песни {song_id}: {e}")
            return False
    
    async def _has_spotify_data(self, song_id: int) -> bool:
        """Проверка наличия Spotify данных"""
        try:
            async with self.db_manager.get_connection() as conn:
                query = "SELECT 1 FROM spotify_data WHERE song_id = $1 LIMIT 1"
                result = await conn.fetch(query, song_id)
                return len(result) > 0
        except Exception:
            return False
    
    async def _save_spotify_data(self, song_id: int, track_data: Dict, audio_features: Optional[Dict]) -> bool:
        """Сохранение в PostgreSQL"""
        try:
            query = """
                INSERT INTO spotify_data (
                    song_id, track_id, album_name, album_id, release_date,
                    popularity, preview_url, external_urls, audio_features
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (song_id) DO UPDATE SET
                    track_id = EXCLUDED.track_id,
                    album_name = EXCLUDED.album_name,
                    album_id = EXCLUDED.album_id,
                    release_date = EXCLUDED.release_date,
                    popularity = EXCLUDED.popularity,
                    preview_url = EXCLUDED.preview_url,
                    external_urls = EXCLUDED.external_urls,
                    audio_features = EXCLUDED.audio_features,
                    created_at = CURRENT_TIMESTAMP
            """
            
            album = track_data.get('album', {})
            
            async with self.db_manager.get_connection() as conn:
                await conn.execute(
                    query,
                    song_id,
                    track_data['id'],
                    album.get('name'),
                    album.get('id'),
                    album.get('release_date'),
                    track_data.get('popularity', 0),
                    track_data.get('preview_url'),
                    json.dumps(track_data.get('external_urls', {})),
                    json.dumps(audio_features) if audio_features else None
                )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения в PostgreSQL: {e}")
            return False
    
    async def get_songs_for_enhancement(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Получение песен для обогащения"""
        try:
            query = """
                SELECT s.id, s.artist, s.title
                FROM songs s
                LEFT JOIN spotify_data sd ON s.id = sd.song_id
                WHERE sd.song_id IS NULL
                ORDER BY s.id
                LIMIT $1
            """
            
            async with self.db_manager.get_connection() as conn:
                result = await conn.fetch(query, limit)
                return [dict(row) for row in result] if result else []
        except Exception as e:
            logger.error(f"❌ Ошибка получения песен: {e}")
            return []
    
    async def bulk_enhance(self, batch_size: int = 50, max_songs: Optional[int] = None) -> Dict[str, int]:
        """Массовое обогащение песен"""
        logger.info("🚀 Начинаем массовое Spotify обогащение")
        
        stats = {'processed': 0, 'enhanced': 0, 'failed': 0, 'skipped': 0}
        
        if not self.spotify:
            logger.error("❌ Spotify API недоступен")
            return stats
        
        try:
            songs = await self.get_songs_for_enhancement(max_songs or 10000)
            
            if not songs:
                logger.info("✅ Все песни уже обогащены")
                return stats
            
            logger.info(f"📊 Найдено {len(songs)} песен для обогащения")
            
            for i in range(0, len(songs), batch_size):
                batch = songs[i:i + batch_size]
                
                logger.info(f"📦 Батч {i//batch_size + 1}/{(len(songs)-1)//batch_size + 1}")
                
                for song in batch:
                    success = await self.enhance_song(song['id'], song['artist'], song['title'])
                    
                    stats['processed'] += 1
                    if success:
                        stats['enhanced'] += 1
                    else:
                        stats['failed'] += 1
                    
                    if stats['processed'] % 25 == 0:
                        logger.info(f"📈 Прогресс: {stats}")
                
                # Пауза между батчами
                if i + batch_size < len(songs):
                    await asyncio.sleep(1)
            
            logger.info(f"✅ Обогащение завершено: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Ошибка массового обогащения: {e}")
            return stats
    
    async def get_enhancement_stats(self) -> Dict[str, Any]:
        """Статистика обогащения"""
        try:
            query = """
                SELECT 
                    (SELECT COUNT(*) FROM songs) as total_songs,
                    (SELECT COUNT(*) FROM spotify_data) as enhanced_songs,
                    (SELECT COUNT(*) FROM spotify_data WHERE audio_features IS NOT NULL) as with_features,
                    (SELECT AVG(popularity) FROM spotify_data WHERE popularity > 0) as avg_popularity
            """
            
            async with self.db_manager.get_connection() as conn:
                result = await conn.fetch(query)
                
                if result:
                    stats = dict(result[0])
                    if stats['total_songs'] > 0:
                        stats['enhancement_percentage'] = round(
                            stats['enhanced_songs'] / stats['total_songs'] * 100, 2
                        )
                    return stats
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения статистики: {e}")
            return {}
    
    async def close(self):
        """Закрытие соединений"""
        try:
            if self.db_manager:
                await self.db_manager.close()
            logger.info("✅ Spotify Enhancer закрыт")
        except Exception as e:
            logger.error(f"❌ Ошибка закрытия: {e}")


# CLI интерфейс
async def main():
    """CLI для Spotify Enhancer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PostgreSQL Spotify Enhancer')
    parser.add_argument('--enhance', action='store_true', help='Запустить обогащение')
    parser.add_argument('--stats', action='store_true', help='Показать статистику')
    parser.add_argument('--limit', type=int, default=1000, help='Лимит песен')
    parser.add_argument('--batch-size', type=int, default=50, help='Размер батча')
    
    args = parser.parse_args()
    
    enhancer = SpotifyEnhancer()
    
    try:
        await enhancer.initialize()
        
        if args.stats:
            stats = await enhancer.get_enhancement_stats()
            print("📊 Статистика Spotify обогащения:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        if args.enhance:
            stats = await enhancer.bulk_enhance(
                batch_size=args.batch_size,
                max_songs=args.limit
            )
            print(f"✅ Результат обогащения: {stats}")
            
    finally:
        await enhancer.close()


if __name__ == "__main__":
    asyncio.run(main())
