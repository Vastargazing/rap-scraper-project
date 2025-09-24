#!/usr/bin/env python3
"""
#!/usr/bin/env python3
🐘 Менеджер PostgreSQL базы данных для проекта Rap Scraper

НАЗНАЧЕНИЕ:
- Управление подключением и транзакциями с PostgreSQL
- Создание таблиц, индексов, добавление песен, статистика
- Абстракция для всех операций с базой

ИСПОЛЬЗОВАНИЕ:
from src.utils.postgres_db import PostgreSQLManager
db = PostgreSQLManager(); db.add_song(...)

ЗАВИСИМОСТИ:
- Python 3.8+
- psycopg2
- config.yaml
- PostgreSQL база данных (rap_lyrics)

РЕЗУЛЬТАТ:
- Надежное сохранение песен и метаданных
- Статистика, мониторинг, поддержка миграций

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""

import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Optional
import os

# Добавляем корневую папку в path для импорта конфигурации
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class PostgreSQLManager:
    """Менеджер PostgreSQL базы данных"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.conn = None
        self.cursor = None
        self.config = self._load_config(config_path)
        self._connect()
        self._create_tables()
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Загружает конфигурацию из config.yaml"""

        if config_path is None:
            config_path = PROJECT_ROOT / "config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config['database']
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            # Fallback конфигурация
            return {
                "host": "localhost",
                "port": 5432,
                "name": "rap_lyrics",
                "username": "rap_user",
                "password": "securepassword123"
            }
    
    def _connect(self):
        """Подключение к PostgreSQL"""
        try:
            self.conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['name'],
                user=self.config['username'],
                password=self.config['password'],
                cursor_factory=RealDictCursor
            )
            self.conn.autocommit = False
            self.cursor = self.conn.cursor()
            logger.info(f"✅ Подключение к PostgreSQL: {self.config['host']}:{self.config['port']}/{self.config['name']}")
        except psycopg2.Error as e:
            logger.error(f"❌ Ошибка подключения к PostgreSQL: {e}")
            # Попробуем создать базу данных если её нет
            if "does not exist" in str(e):
                self._create_database()
            else:
                raise e
    
    def _create_database(self):
        """Создание базы данных если её не существует"""
        try:
            # Подключаемся к системной базе postgres для создания новой БД
            temp_conn = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                database='postgres',  # Системная БД
                user=self.config['username'],
                password=self.config['password']
            )
            temp_conn.autocommit = True
            temp_cursor = temp_conn.cursor()
            
            # Создаем базу данных
            temp_cursor.execute(f"CREATE DATABASE {self.config['name']}")
            logger.info(f"✅ Создана база данных: {self.config['name']}")
            
            temp_cursor.close()
            temp_conn.close()
            
            # Теперь подключаемся к созданной БД
            self._connect()
            
        except psycopg2.Error as e:
            logger.error(f"❌ Ошибка создания базы данных: {e}")
            raise e
    
    def _create_tables(self):
        """Создание таблиц в базе данных"""
        try:
            # Таблица для песен
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS songs (
                    id SERIAL PRIMARY KEY,
                    artist TEXT NOT NULL,
                    title TEXT NOT NULL,
                    lyrics TEXT NOT NULL,
                    url TEXT UNIQUE NOT NULL,
                    genius_id INTEGER UNIQUE,
                    scraped_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    word_count INTEGER,
                    -- Новые поля для метаданных
                    genre TEXT,
                    release_date TEXT,
                    album TEXT,
                    language TEXT DEFAULT 'en',
                    explicit BOOLEAN DEFAULT FALSE,
                    song_art_url TEXT,
                    popularity_score INTEGER DEFAULT 0,
                    lyrics_quality_score REAL DEFAULT 0.0,
                    -- Уникальность по артисту и названию
                    UNIQUE(artist, title)
                )
            """)
            
            # Создание индексов для быстрого поиска
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_songs_artist ON songs(artist)",
                "CREATE INDEX IF NOT EXISTS idx_songs_url ON songs(url)", 
                "CREATE INDEX IF NOT EXISTS idx_songs_genius_id ON songs(genius_id)",
                "CREATE INDEX IF NOT EXISTS idx_songs_genre ON songs(genre)",
                "CREATE INDEX IF NOT EXISTS idx_songs_release_date ON songs(release_date)",
                "CREATE INDEX IF NOT EXISTS idx_songs_quality ON songs(lyrics_quality_score)",
                "CREATE INDEX IF NOT EXISTS idx_songs_word_count ON songs(word_count)",
                "CREATE INDEX IF NOT EXISTS idx_songs_scraped_date ON songs(scraped_date)"
            ]
            
            for index_sql in indexes:
                self.cursor.execute(index_sql)
            
            self.conn.commit()
            logger.info("✅ Таблицы и индексы созданы")
            
        except psycopg2.Error as e:
            logger.error(f"❌ Ошибка создания таблиц: {e}")
            self.conn.rollback()
            raise e
    
    def add_song(self, artist: str, title: str, lyrics: str, url: str, 
                 genius_id: int = None, metadata: Dict = None) -> bool:
        """Добавление песни с метаданными"""
        try:
            word_count = len(lyrics.split()) if lyrics else 0
            
            # Расчет качества текста (простая метрика)
            lyrics_quality = self._calculate_lyrics_quality(lyrics)
            
            # Подготовка метаданных
            if metadata is None:
                metadata = {}
            
            self.cursor.execute(
                """INSERT INTO tracks (
                    artist, title, lyrics, url, genius_id, word_count,
                    genre, release_date, album, language, explicit,
                    song_art_url, popularity_score, lyrics_quality_score
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    artist, title, lyrics, url, genius_id, word_count,
                    metadata.get('genre'),
                    metadata.get('release_date'),
                    metadata.get('album'),
                    metadata.get('language', 'en'),
                    metadata.get('explicit', False),
                    metadata.get('song_art_url'),
                    metadata.get('popularity_score', 0),
                    lyrics_quality
                )
            )
            
            self.conn.commit()
            return True
            
        except psycopg2.IntegrityError as e:
            self.conn.rollback()
            if "duplicate key" in str(e).lower():
                logger.debug(f"Duplicate: {artist} - {title}")
                return False
            else:
                logger.error(f"Integrity error: {e}")
                raise e
        except psycopg2.Error as e:
            self.conn.rollback()
            logger.error(f"Database error adding song: {e}")
            return False
    
    def _calculate_lyrics_quality(self, lyrics: str) -> float:
        """Простая метрика качества текста песни"""
        if not lyrics:
            return 0.0
        
        score = 0.0
        words = lyrics.split()
        
        # Длина текста
        if len(words) > 50:
            score += 0.3
        if len(words) > 100:
            score += 0.2
        
        # Разнообразие слов
        unique_words = len(set(word.lower() for word in words))
        if len(words) > 0:
            diversity = unique_words / len(words)
            score += diversity * 0.3
        
        # Отсутствие инструментальных маркеров
        instrumental_markers = ["instrumental", "no lyrics", "без слов"]
        if not any(marker in lyrics.lower() for marker in instrumental_markers):
            score += 0.2
        
        return min(score, 1.0)
    
    def song_exists(self, url: str = None, genius_id: int = None) -> bool:
        """Проверка существования песни"""
        try:
            if url:
                self.cursor.execute("SELECT 1 FROM tracks WHERE url = %s", (url,))
            elif genius_id:
                self.cursor.execute("SELECT 1 FROM tracks WHERE genius_id = %s", (genius_id,))
            else:
                return False
            return self.cursor.fetchone() is not None
        except psycopg2.Error as e:
            logger.error(f"Error checking song existence: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Получение статистики базы данных"""
        try:
            self.cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(DISTINCT artist) as artists,
                    AVG(word_count) as avg_words,
                    AVG(lyrics_quality_score) as avg_quality,
                    COUNT(CASE WHEN genre IS NOT NULL THEN 1 END) as with_genre
                FROM tracks
            """)
            result = self.cursor.fetchone()
            return {
                "total_songs": result["total"],
                "unique_artists": result["artists"],
                "avg_words": round(result["avg_words"] or 0, 1),
                "avg_quality": round(result["avg_quality"] or 0, 3),
                "with_metadata": result["with_genre"]
            }
        except psycopg2.Error as e:
            logger.error(f"Error getting stats: {e}")
            return {"total_songs": 0, "unique_artists": 0, "avg_words": 0, "avg_quality": 0, "with_metadata": 0}
    
    def get_recent_songs(self, limit: int = 5) -> List[Dict]:
        """Получение последних добавленных песен"""
        try:
            self.cursor.execute("""
                SELECT artist, title, word_count, lyrics_quality_score, genre, scraped_date 
                FROM tracks 
                ORDER BY id DESC 
                LIMIT %s
            """, (limit,))
            return [dict(row) for row in self.cursor.fetchall()]
        except psycopg2.Error as e:
            logger.error(f"Error getting recent songs: {e}")
            return []
    
    def get_artist_count(self, artist: str) -> int:
        """Получение количества песен артиста"""
        try:
            self.cursor.execute("SELECT COUNT(*) as count FROM tracks WHERE artist = %s", (artist,))
            result = self.cursor.fetchone()
            return result["count"]
        except psycopg2.Error as e:
            logger.error(f"Error getting artist count: {e}")
            return 0
    
    def close(self):
        """Закрытие соединения"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            logger.info("🔒 Соединение с PostgreSQL закрыто")
        except psycopg2.Error as e:
            logger.error(f"Error closing connection: {e}")

def test_connection():
    """Тестирование подключения к PostgreSQL"""
    try:
        db = PostgreSQLManager()
        stats = db.get_stats()
        print(f"✅ PostgreSQL подключение успешно!")
        print(f"📊 Статистика: {stats}")
        db.close()
        return True
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

if __name__ == "__main__":
    test_connection()
