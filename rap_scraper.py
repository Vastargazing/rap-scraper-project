import lyricsgenius
import sqlite3
import time
import random
import logging
import re
import signal
import sys
from datetime import datetime
import json
import os
from typing import List, Optional
from dotenv import load_dotenv

# Загрузка переменных окружения из .env
load_dotenv()
TOKEN = os.getenv("GENIUS_TOKEN")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not TOKEN:
    logger.error("Токен Genius API не найден в .env!")
    exit(1)

class LyricsDatabase:
    def __init__(self, db_name="lyrics.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.create_table()
        logger.info(f"База данных {db_name} инициализирована")
        self.batch_count = 0
        self.batch_size = 20  # Коммит каждые 20 записей

    def create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artist TEXT NOT NULL,
                title TEXT NOT NULL,
                lyrics TEXT NOT NULL,
                url TEXT UNIQUE NOT NULL,
                genius_id INTEGER UNIQUE,
                scraped_date TEXT DEFAULT CURRENT_TIMESTAMP,
                word_count INTEGER,
                UNIQUE(artist, title)
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_artist ON songs(artist)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON songs(url)")
        self.conn.commit()

    def add_song(self, artist: str, title: str, lyrics: str, url: str, genius_id: int = None) -> bool:
        try:
            word_count = len(lyrics.split()) if lyrics else 0
            self.conn.execute(
                """INSERT INTO songs (artist, title, lyrics, url, genius_id, word_count) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (artist, title, lyrics, url, genius_id, word_count)
            )
            self.batch_count += 1
            if self.batch_count >= self.batch_size:
                self.conn.commit()
                self.batch_count = 0
            return True
        except sqlite3.IntegrityError as e:
            logger.debug(f"Дубликат: {artist} - {title}")
            return False

    def song_exists(self, url: str = None, genius_id: int = None) -> bool:
        if url:
            cur = self.conn.execute("SELECT 1 FROM songs WHERE url=?", (url,))
        elif genius_id:
            cur = self.conn.execute("SELECT 1 FROM songs WHERE genius_id=?", (genius_id,))
        else:
            return False
        return cur.fetchone() is not None

    def get_stats(self) -> dict:
        self.conn.commit()  # Обновляем статистику
        cur = self.conn.execute("SELECT COUNT(*) as total, COUNT(DISTINCT artist) as artists FROM songs")
        result = cur.fetchone()
        return {"total_songs": result["total"], "unique_artists": result["artists"]}

    def get_recent_songs(self, limit: int = 5) -> List[dict]:
        self.conn.commit()
        cur = self.conn.execute("""
            SELECT artist, title, word_count, scraped_date 
            FROM songs 
            ORDER BY id DESC 
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cur.fetchall()]

    def close(self):
        self.conn.commit()  # Финальный коммит
        self.conn.close()

class SafeGeniusScraper:
    def __init__(self, token: str, db_name: str = "lyrics.db"):
        self.genius = lyricsgenius.Genius(
            token,
            timeout=15,
            retries=3,
            remove_section_headers=True,
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)", "(Instrumental)"]
        )
        self.db = LyricsDatabase(db_name)
        self.session_stats = {"processed": 0, "added": 0, "skipped": 0, "errors": 0}
        self.min_delay = 2.0
        self.max_delay = 5.0
        self.error_delay = 10.0
        self.max_retries = 3
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"\nПолучен сигнал {signum}. Завершение работы...")
        self.shutdown_requested = True

    def safe_delay(self, is_error: bool = False):
        delay = self.error_delay if is_error else random.uniform(self.min_delay, self.max_delay)
        intervals = int(delay)
        remainder = delay - intervals
        for _ in range(intervals):
            if self.shutdown_requested:
                return
            time.sleep(1)
        if remainder > 0 and not self.shutdown_requested:
            time.sleep(remainder)
        logger.debug(f"Пауза: {delay:.1f}с")

    def clean_lyrics(self, lyrics: str) -> str:
        if not lyrics:
            return ""
        lyrics = re.sub(r"(?i)(Embed|Submitted by [^\n]*|Written by [^\n]*|You might also like).*$", "", lyrics, flags=re.DOTALL)
        lyrics = re.sub(r"\n{2,}", "\n", lyrics.strip())
        return lyrics

    def _is_valid_lyrics(self, lyrics: str) -> bool:
        if not lyrics:
            return False
        lyrics = lyrics.strip()
        if len(lyrics) < 100 or len(lyrics.split()) < 20:
            return False
        instrumental_markers = ["instrumental", "no lyrics", "без слов", "music only", "beat only"]
        return not any(marker in lyrics.lower() for marker in instrumental_markers)

    def scrape_artist_songs(self, artist_name: str, max_songs: int = 100) -> int:
        added_count = 0
        retry_count = 0
        
        while retry_count < self.max_retries and not self.shutdown_requested:
            try:
                logger.info(f"Поиск артиста: {artist_name} (попытка {retry_count + 1})")
                artist = self.genius.search_artist(artist_name, max_songs=max_songs, sort="popularity", get_full_info=False)
                
                if not artist or not artist.songs:
                    logger.warning(f"Артист {artist_name} не найден")
                    return 0

                logger.info(f"Найдено {len(artist.songs)} песен для {artist_name}")

                for i, song in enumerate(artist.songs):
                    if self.shutdown_requested:
                        logger.info(f"Останавливаем на песне {i+1}/{len(artist.songs)} для {artist_name}")
                        break
                        
                    try:
                        if self.db.song_exists(url=song.url):
                            logger.debug(f"Пропуск (дубликат): {song.title}")
                            self.session_stats["skipped"] += 1
                            continue

                        lyrics = self.clean_lyrics(song.lyrics)
                        if not self._is_valid_lyrics(lyrics):
                            logger.debug(f"Пропуск (невалидный текст): {song.title}")
                            self.session_stats["skipped"] += 1
                            continue

                        if self.db.add_song(artist_name, song.title, lyrics, song.url, song.id if hasattr(song, 'id') else None):
                            added_count += 1
                            self.session_stats["added"] += 1
                            logger.info(f"✅ Добавлено: {artist_name} - {song.title} ({len(lyrics.split())} слов)")
                            if self.session_stats["added"] % 5 == 0:
                                current_stats = self.db.get_stats()
                                logger.info(f"📊 Промежуточная статистика: {current_stats['total_songs']} песен в базе")
                        else:
                            self.session_stats["skipped"] += 1

                        self.session_stats["processed"] += 1
                        if (i + 1) % 10 == 0:
                            logger.info(f"Обработано {i + 1}/{len(artist.songs)} песен для {artist_name}")
                        self.safe_delay()

                    except lyricsgenius.exceptions.Timeout as e:
                        logger.error(f"Таймаут для {song.title}: {e}")
                        self.session_stats["errors"] += 1
                        self.safe_delay(is_error=True)
                    except Exception as e:
                        logger.error(f"Ошибка с песней {song.title}: {e}")
                        self.session_stats["errors"] += 1
                        self.safe_delay(is_error=True)

                break

            except lyricsgenius.exceptions.RateLimitExceeded as e:
                logger.error(f"429 Too Many Requests для {artist_name}: {e}")
                self.safe_delay(is_error=True)
                if not self.shutdown_requested:
                    time.sleep(60)
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(f"Максимум попыток достигнут для {artist_name}")
                    self.session_stats["errors"] += 1
                    break
            except Exception as e:
                retry_count += 1
                logger.error(f"Ошибка с артистом {artist_name} (попытка {retry_count}): {e}")
                if retry_count >= self.max_retries:
                    logger.error(f"Максимум попыток достигнут для {artist_name}")
                    self.session_stats["errors"] += 1
                    break
                self.safe_delay(is_error=True)

        return added_count

    def show_current_results(self):
        stats = self.db.get_stats()
        recent_songs = self.db.get_recent_songs(5)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 ТЕКУЩИЕ РЕЗУЛЬТАТЫ:")
        logger.info(f"Всего песен в базе: {stats['total_songs']}")
        logger.info(f"Уникальных артистов: {stats['unique_artists']}")
        logger.info(f"За эту сессию добавлено: {self.session_stats['added']}")
        
        if recent_songs:
            logger.info(f"\n🎵 Последние добавленные песни:")
            for song in recent_songs:
                logger.info(f"  • {song['artist']} - {song['title']} ({song['word_count']} слов)")
        
        logger.info(f"{'='*60}\n")

    def run_scraping_session(self, artists: List[str], songs_per_artist: int = 100):
        logger.info(f"Начало сессии скрапинга: {len(artists)} артистов, {songs_per_artist} песен каждого")
        start_time = datetime.now()
        
        initial_stats = self.db.get_stats()
        logger.info(f"В базе уже есть: {initial_stats['total_songs']} песен")
        
        try:
            for i, artist_name in enumerate(artists, 1):
                if self.shutdown_requested:
                    logger.info("Получен запрос на остановку")
                    break
                    
                logger.info(f"\n{'='*50}")
                logger.info(f"Артист {i}/{len(artists)}: {artist_name}")
                
                added = self.scrape_artist_songs(artist_name, songs_per_artist)
                logger.info(f"Добавлено песен для {artist_name}: {added}")
                
                stats = self.db.get_stats()
                logger.info(f"Всего в базе: {stats['total_songs']} песен от {stats['unique_artists']} артистов")
                
                if i < len(artists) and not self.shutdown_requested:
                    artist_delay = random.uniform(5, 10)
                    logger.info(f"Пауза между артистами: {artist_delay:.1f}с")
                    for _ in range(int(artist_delay)):
                        if self.shutdown_requested:
                            break
                        time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Получено прерывание от пользователя (Ctrl+C)")
            self.shutdown_requested = True
        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
        finally:
            self.db.conn.commit()  # Принудительный коммит перед финальной статистикой
            self.show_current_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            final_stats = self.db.get_stats()
            
            logger.info(f"\n{'='*50}")
            logger.info(f"🏁 СЕССИЯ ЗАВЕРШЕНА")
            logger.info(f"Время выполнения: {duration}")
            logger.info(f"Обработано: {self.session_stats['processed']}")
            logger.info(f"Добавлено: {self.session_stats['added']}")
            logger.info(f"Пропущено: {self.session_stats['skipped']}")
            logger.info(f"Ошибок: {self.session_stats['errors']}")
            logger.info(f"Всего в базе: {final_stats['total_songs']} песен")
            
            self.close()

    def close(self):
        logger.info("Закрытие соединения с базой данных...")
        self.db.close()

def load_artist_list(filename: str = "rap_artists.json") -> List[str]:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        artists = [
            "Kendrick Lamar", "J. Cole", "Drake", "Eminem", "Kanye West",
            "Travis Scott", "Lil Wayne", "Jay-Z", "Nas", "Tupac",
            "The Notorious B.I.G.", "Lil Baby", "Future", "21 Savage", "Post Malone",
            "Tyler, The Creator", "A$AP Rocky", "Mac Miller", "Childish Gambino", "Logic",
            "Big Sean", "Chance the Rapper", "Wiz Khalifa", "Meek Mill", "2 Chainz",
            "Pusha T", "Danny Brown", "Earl Sweatshirt", "Vince Staples", "JID",
            "Denzel Curry", "Joey Bada$$", "Capital STEEZ", "MF DOOM", "Madlib",
            "ScHoolboy Q", "Ab-Soul", "Jay Rock", "SiR", "Reason",
            "YG", "Nipsey Hussle", "The Game", "Ice Cube", "Eazy-E",
            "Dr. Dre", "Snoop Dogg", "Warren G", "Nate Dogg", "Xzibit"
        ]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(artists, f, indent=2, ensure_ascii=False)
        return artists

def main():
    if not TOKEN:
        logger.error("Токен Genius API не найден в .env!")
        exit(1)
        
    scraper = SafeGeniusScraper(TOKEN, "rap_lyrics.db")
    
    try:
        artists = load_artist_list()
        SONGS_PER_ARTIST = 100  # Увеличено для цели 10,000 треков
        
        logger.info(f"Загружено {len(artists)} артистов")
        logger.info(f"Цель: ~{len(artists) * SONGS_PER_ARTIST} песен")
        logger.info("Используйте Ctrl+C для безопасной остановки")
        
        scraper.run_scraping_session(artists, SONGS_PER_ARTIST)
        
    except Exception as e:
        logger.error(f"Ошибка в main: {e}")
    finally:
        logger.info("Программа завершена")

if __name__ == "__main__":
    main()