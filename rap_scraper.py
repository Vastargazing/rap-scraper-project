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
from typing import List, Optional, Dict, Any, Generator
from dotenv import load_dotenv
import psutil
import gc

# Загрузка переменных окружения из .env
load_dotenv()
TOKEN = os.getenv("GENIUS_TOKEN")

# Настройка логирования с правильной кодировкой
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraping.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not TOKEN:
    logger.error("Genius API token not found in .env!")
    exit(1)

class MemoryMonitor:
    """Класс для мониторинга использования памяти и автоматической очистки"""
    
    def __init__(self, memory_limit_mb: float = 1000.0, warning_threshold: float = 0.8):
        self.process = psutil.Process()
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.warning_threshold = warning_threshold
        self.last_check = time.time()
        self.check_interval = 30  # Проверка каждые 30 секунд
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Получить текущее использование памяти"""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent()
        }
    
    def check_memory(self, force_check: bool = False) -> bool:
        """Проверить память и вернуть True если нужна очистка"""
        current_time = time.time()
        if not force_check and current_time - self.last_check < self.check_interval:
            return False
            
        self.last_check = current_time
        usage = self.get_memory_usage()
        
        # Логируем каждые 5 минут
        if int(current_time) % 300 < self.check_interval:
            logger.info(f"Memory usage: {usage['rss_mb']:.1f}MB ({usage['percent']:.1f}%)")
        
        # Проверяем лимиты
        if usage['rss_mb'] > self.memory_limit_bytes / 1024 / 1024:
            logger.warning(f"Memory limit exceeded: {usage['rss_mb']:.1f}MB > {self.memory_limit_bytes/1024/1024:.1f}MB")
            return True
            
        if usage['percent'] > self.warning_threshold * 100:
            logger.warning(f"Memory warning: {usage['percent']:.1f}% > {self.warning_threshold*100:.1f}%")
            return True
            
        return False
    
    def force_cleanup(self):
        """Принудительная очистка памяти"""
        logger.info("Forcing garbage collection...")
        before = self.get_memory_usage()
        gc.collect()
        after = self.get_memory_usage()
        freed = before['rss_mb'] - after['rss_mb']
        logger.info(f"Freed {freed:.1f}MB memory ({before['rss_mb']:.1f} -> {after['rss_mb']:.1f}MB)")

class LyricsDatabase:
    def __init__(self, db_name="lyrics.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Включаем WAL mode для лучшей производительности
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=10000")
        
        self.create_table()
        logger.info(f"База данных {db_name} инициализирована (WAL mode)")
        self.batch_count = 0
        self.batch_size = 100  # Увеличил batch size

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
                -- Новые поля для метаданных
                release_date TEXT,
                genre TEXT,
                album TEXT,
                primary_artist_id INTEGER,
                featured_artists TEXT, -- JSON array
                producer_artists TEXT, -- JSON array
                language TEXT,
                UNIQUE(artist, title)
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_artist ON songs(artist)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_url ON songs(url)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_genre ON songs(genre)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_release_date ON songs(release_date)")
        self.conn.commit()

    def add_song(self, artist: str, title: str, lyrics: str, url: str, 
                 genius_id: int = None, metadata: Dict[str, Any] = None) -> bool:
        try:
            word_count = len(lyrics.split()) if lyrics else 0
            
            # Извлекаем метаданные
            if metadata is None:
                metadata = {}
                
            release_date = metadata.get('release_date')
            genre = metadata.get('genre')
            album = metadata.get('album')
            primary_artist_id = metadata.get('primary_artist_id')
            featured_artists = json.dumps(metadata.get('featured_artists', [])) if metadata.get('featured_artists') else None
            producer_artists = json.dumps(metadata.get('producer_artists', [])) if metadata.get('producer_artists') else None
            language = metadata.get('language', 'en')
            
            self.conn.execute(
                """INSERT INTO songs (artist, title, lyrics, url, genius_id, word_count,
                   release_date, genre, album, primary_artist_id, featured_artists, 
                   producer_artists, language) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (artist, title, lyrics, url, genius_id, word_count,
                 release_date, genre, album, primary_artist_id, 
                 featured_artists, producer_artists, language)
            )
            self.batch_count += 1
            if self.batch_count >= self.batch_size:
                self.conn.commit()
                self.batch_count = 0
            return True
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                logger.warning(f"База заблокирована, повторная попытка через 2 сек: {artist} - {title}")
                time.sleep(2)
                try:
                    self.conn.execute(
                        """INSERT INTO songs (artist, title, lyrics, url, genius_id, word_count,
                           release_date, genre, album, primary_artist_id, featured_artists, 
                           producer_artists, language) 
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (artist, title, lyrics, url, genius_id, word_count,
                         release_date, genre, album, primary_artist_id, 
                         featured_artists, producer_artists, language)
                    )
                    return True
                except Exception:
                    logger.error(f"Повторная попытка не удалась: {artist} - {title}")
                    return False
            else:
                raise e
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed" in str(e):
                logger.debug(f"Duplicate: {artist} - {title}")
            else:
                logger.warning(f"Integrity error for {artist} - {title}: {e}")
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
        self.memory_monitor = MemoryMonitor(memory_limit_mb=1500.0)  # 1.5GB лимит
        self.session_stats = {
            "processed": 0, "added": 0, "skipped": 0, "errors": 0,
            "memory_cleanups": 0, "songs_per_minute": 0.0
        }
        self.session_start_time = time.time()
        self.min_delay = 3.0  # Увеличил с 2.0
        self.max_delay = 7.0  # Увеличил с 5.0
        self.error_delay = 15.0  # Увеличил с 10.0
        self.max_retries = 3
        self.shutdown_requested = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        # Дополнительная обработка для Windows
        if sys.platform == "win32":
            try:
                signal.signal(signal.SIGBREAK, self._signal_handler)
            except AttributeError:
                pass  # SIGBREAK может отсутствовать

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
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                self.shutdown_requested = True
                return
        if remainder > 0 and not self.shutdown_requested:
            try:
                time.sleep(remainder)
            except KeyboardInterrupt:
                self.shutdown_requested = True
                return
        logger.debug(f"Пауза: {delay:.1f}с")

    def clean_lyrics(self, lyrics: str) -> str:
        if not lyrics:
            return ""
        
        # Удаляем информацию о контрибьюторах (например, "81 Contributors")
        lyrics = re.sub(r"^\d+\s+Contributors.*?Lyrics", "", lyrics, flags=re.MULTILINE | re.DOTALL)
        
        # Удаляем блок с переводами (например, "TranslationsEnglishEspañol") 
        lyrics = re.sub(r"Translations[A-Za-z]+", "", lyrics, flags=re.MULTILINE)
        
        # Удаляем информацию о исполнителе и описание песни в начале
        # (обычно идет после "Lyrics" и до первой строки песни)
        lyrics = re.sub(r"Lyrics[A-Z].*?Read More\s*", "", lyrics, flags=re.DOTALL)
        
        # Удаляем стандартные блоки от Genius
        lyrics = re.sub(r"(?i)(Embed|Submitted by [^\n]*|Written by [^\n]*|You might also like).*$", "", lyrics, flags=re.DOTALL)
        
        # Удаляем ссылки и URL
        lyrics = re.sub(r"https?://[^\s]+", "", lyrics)
        
        # Удаляем блоки в квадратных скобках (обычно это описания или переходы)
        lyrics = re.sub(r"\[.*?\]", "", lyrics)
        
        # Удаляем множественные переносы строк
        lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)
        lyrics = re.sub(r"\n{2,}", "\n", lyrics.strip())
        
        # Удаляем пустые строки в начале и конце
        return lyrics.strip()

    def _is_valid_lyrics(self, lyrics: str) -> bool:
        if not lyrics:
            return False
        lyrics = lyrics.strip()
        if len(lyrics) < 100 or len(lyrics.split()) < 20:
            return False
        instrumental_markers = ["instrumental", "no lyrics", "без слов", "music only", "beat only"]
        return not any(marker in lyrics.lower() for marker in instrumental_markers)
    
    def _extract_metadata(self, song) -> Dict[str, Any]:
        """Извлечь метаданные из объекта песни"""
        metadata = {}
        
        try:
            # Дата релиза
            if hasattr(song, 'release_date_for_display'):
                metadata['release_date'] = song.release_date_for_display
            elif hasattr(song, 'release_date_components'):
                if song.release_date_components:
                    year = song.release_date_components.get('year')
                    month = song.release_date_components.get('month')
                    day = song.release_date_components.get('day')
                    if year:
                        date_str = str(year)
                        if month:
                            date_str += f"-{month:02d}"
                            if day:
                                date_str += f"-{day:02d}"
                        metadata['release_date'] = date_str
            
            # Альбом
            if hasattr(song, 'album') and song.album:
                if hasattr(song.album, 'name'):
                    metadata['album'] = song.album.name
                elif hasattr(song.album, 'full_title'):
                    metadata['album'] = song.album.full_title
            
            # ID основного артиста
            if hasattr(song, 'primary_artist') and song.primary_artist:
                if hasattr(song.primary_artist, 'id'):
                    metadata['primary_artist_id'] = song.primary_artist.id
            
            # Featured артисты
            if hasattr(song, 'featured_artists') and song.featured_artists:
                featured = []
                for artist in song.featured_artists:
                    if hasattr(artist, 'name'):
                        featured.append(artist.name)
                if featured:
                    metadata['featured_artists'] = featured
            
            # Продюсеры
            if hasattr(song, 'producer_artists') and song.producer_artists:
                producers = []
                for artist in song.producer_artists:
                    if hasattr(artist, 'name'):
                        producers.append(artist.name)
                if producers:
                    metadata['producer_artists'] = producers
            
            # Попытка определить жанр (примерный)
            if hasattr(song, 'primary_artist') and song.primary_artist:
                artist_name = getattr(song.primary_artist, 'name', '').lower()
                # Простая эвристика для жанров
                if any(term in artist_name for term in ['lil', 'young', 'future', 'travis']):
                    metadata['genre'] = 'trap'
                elif any(term in artist_name for term in ['eminem', 'nas', 'jay-z', 'biggie']):
                    metadata['genre'] = 'hip-hop'
                else:
                    metadata['genre'] = 'rap'
            
        except Exception as e:
            logger.debug(f"Error extracting metadata: {e}")
        
        return metadata

    def _get_songs_generator(self, artist_name: str, max_songs: int = 500) -> Generator:
        """Генератор для получения песен по одной, экономя память"""
        try:
            # Сначала получаем базовую информацию об артисте
            search_results = self.genius.search_artist(artist_name, max_songs=1, get_full_info=False)
            if not search_results:
                logger.warning(f"Artist {artist_name} not found")
                return
            
            artist_id = search_results.id
            logger.info(f"Found artist {artist_name} (ID: {artist_id})")
            
            # Получаем список песен через API по частям
            per_page = 50  # Загружаем по 50 песен за раз
            page = 1
            total_fetched = 0
            
            while total_fetched < max_songs:
                try:
                    songs_response = self.genius.artist_songs(
                        artist_id, 
                        sort='popularity', 
                        per_page=min(per_page, max_songs - total_fetched),
                        page=page
                    )
                    
                    if not songs_response or 'songs' not in songs_response:
                        break
                    
                    songs = songs_response['songs']
                    if not songs:
                        break
                    
                    for song_data in songs:
                        if self.shutdown_requested:
                            return
                        
                        try:
                            # Получаем полную информацию о песне
                            song = self.genius.song(song_data['id'])
                            if song:
                                yield song
                                total_fetched += 1
                                
                                # Принудительно удаляем объект для экономии памяти
                                del song
                                
                                if total_fetched >= max_songs:
                                    break
                        except Exception as e:
                            logger.debug(f"Error fetching song {song_data.get('title', 'unknown')}: {e}")
                            continue
                    
                    page += 1
                    
                    # Проверяем память после каждой страницы
                    if self.memory_monitor.check_memory():
                        self.memory_monitor.force_cleanup()
                        self.session_stats["memory_cleanups"] += 1
                    
                except Exception as e:
                    logger.error(f"Error fetching page {page} for {artist_name}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Error in songs generator for {artist_name}: {e}")

    def scrape_artist_songs(self, artist_name: str, max_songs: int = 500) -> int:
        added_count = 0
        retry_count = 0
        
        logger.info(f"Starting processing artist: {artist_name}")
        
        while retry_count < self.max_retries and not self.shutdown_requested:
            try:
                # Используем генератор для экономии памяти
                song_count = 0
                for song in self._get_songs_generator(artist_name, max_songs):
                    if self.shutdown_requested:
                        logger.info(f"Stopping at song {song_count+1} for {artist_name}")
                        break
                        
                    try:
                        song_count += 1
                        
                        if self.db.song_exists(url=song.url):
                            logger.debug(f"Skip (duplicate): {song.title}")
                            self.session_stats["skipped"] += 1
                            continue

                        lyrics = self.clean_lyrics(song.lyrics)
                        if not self._is_valid_lyrics(lyrics):
                            logger.debug(f"Skip (invalid lyrics): {song.title}")
                            self.session_stats["skipped"] += 1
                            continue

                        # Извлекаем метаданные
                        metadata = self._extract_metadata(song)
                        
                        if self.db.add_song(
                            artist_name, 
                            song.title, 
                            lyrics, 
                            song.url, 
                            song.id if hasattr(song, 'id') else None,
                            metadata
                        ):
                            added_count += 1
                            self.session_stats["added"] += 1
                            word_count = len(lyrics.split())
                            
                            # Добавляем информацию о жанре в лог
                            genre_info = f" [{metadata.get('genre', 'unknown')}]" if metadata.get('genre') else ""
                            logger.info(f"Added: {artist_name} - {song.title}{genre_info} ({word_count} words)")
                            
                            if self.session_stats["added"] % 10 == 0:
                                current_stats = self.db.get_stats()
                                # Вычисляем скорость
                                elapsed = time.time() - self.session_start_time
                                songs_per_minute = (self.session_stats["added"] / elapsed) * 60 if elapsed > 0 else 0
                                self.session_stats["songs_per_minute"] = songs_per_minute
                                
                                logger.info(f"Stats: {current_stats['total_songs']} songs in database "
                                          f"({songs_per_minute:.1f} songs/min)")
                        else:
                            self.session_stats["skipped"] += 1

                        self.session_stats["processed"] += 1
                        
                        if song_count % 10 == 0:
                            logger.info(f"Processed {song_count} songs for {artist_name}")
                        
                        # Проверка памяти каждые 5 песен
                        if song_count % 5 == 0:
                            if self.memory_monitor.check_memory():
                                logger.info("Memory cleanup triggered during scraping")
                                self.memory_monitor.force_cleanup()
                                self.session_stats["memory_cleanups"] += 1
                        
                        # Check for shutdown before pause
                        if self.shutdown_requested:
                            break
                            
                        self.safe_delay()

                    except Exception as song_e:
                        if "timeout" in str(song_e).lower():
                            logger.error(f"Timeout for {song.title}: {song_e}")
                            self.session_stats["errors"] += 1
                            self.safe_delay(is_error=True)
                        else:
                            logger.error(f"Error with song {song.title}: {song_e}")
                            self.session_stats["errors"] += 1
                            self.safe_delay(is_error=True)
                    finally:
                        # Принудительно удаляем объект песни
                        if 'song' in locals():
                            del song

                logger.info(f"Processed {song_count} total songs for {artist_name}")
                break

            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    logger.error(f"Rate Limit for {artist_name}: {e}")
                    logger.info(f"Waiting 60 seconds before retry...")
                    self.safe_delay(is_error=True)
                    if not self.shutdown_requested:
                        time.sleep(60)
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        logger.error(f"Max retries reached for {artist_name}")
                        self.session_stats["errors"] += 1
                        break
                else:
                    retry_count += 1
                    logger.error(f"Error with artist {artist_name} (attempt {retry_count}): {e}")
                    logger.error(f"Error type: {type(e).__name__}")
                    if retry_count >= self.max_retries:
                        logger.error(f"Max retries reached for {artist_name}")
                        self.session_stats["errors"] += 1
                        break
                    logger.info(f"Pause before retry...")
                    self.safe_delay(is_error=True)

        logger.info(f"Completed processing {artist_name}: added {added_count} songs")
        return added_count

    def show_current_results(self):
        stats = self.db.get_stats()
        recent_songs = self.db.get_recent_songs(5)
        memory_usage = self.memory_monitor.get_memory_usage()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"CURRENT RESULTS:")
        logger.info(f"Total songs in database: {stats['total_songs']}")
        logger.info(f"Unique artists: {stats['unique_artists']}")
        logger.info(f"Added this session: {self.session_stats['added']}")
        logger.info(f"Processing speed: {self.session_stats['songs_per_minute']:.1f} songs/min")
        logger.info(f"Memory usage: {memory_usage['rss_mb']:.1f}MB ({memory_usage['percent']:.1f}%)")
        logger.info(f"Memory cleanups: {self.session_stats['memory_cleanups']}")
        
        if recent_songs:
            logger.info(f"\nRecent added songs:")
            for song in recent_songs:
                logger.info(f"  - {song['artist']} - {song['title']} ({song['word_count']} words)")
        
        logger.info(f"{'='*60}\n")

    def run_scraping_session(self, artists: List[str], songs_per_artist: int = 500):
        logger.info(f"Starting scraping session: {len(artists)} artists, {songs_per_artist} songs each")
        start_time = datetime.now()
        
        initial_stats = self.db.get_stats()
        logger.info(f"Already in database: {initial_stats['total_songs']} songs")
        
        try:
            for i, artist_name in enumerate(artists, 1):
                if self.shutdown_requested:
                    logger.info("Shutdown requested")
                    break
                    
                logger.info(f"\n{'='*50}")
                logger.info(f"Artist {i}/{len(artists)}: {artist_name}")
                
                added = self.scrape_artist_songs(artist_name, songs_per_artist)
                logger.info(f"Added songs for {artist_name}: {added}")
                
                stats = self.db.get_stats()
                logger.info(f"Total in database: {stats['total_songs']} songs from {stats['unique_artists']} artists")
                
                if i < len(artists) and not self.shutdown_requested:
                    artist_delay = random.uniform(5, 10)
                    logger.info(f"Pause between artists: {artist_delay:.1f}s")
                    intervals = int(artist_delay)
                    for _ in range(intervals):
                        if self.shutdown_requested:
                            break
                        try:
                            time.sleep(1)
                        except KeyboardInterrupt:
                            self.shutdown_requested = True
                            break

        except KeyboardInterrupt:
            logger.info("User interruption received (Ctrl+C)")
            self.shutdown_requested = True
        except Exception as e:
            logger.error(f"Critical error: {e}")
        finally:
            self.db.conn.commit()  # Принудительный коммит перед финальной статистикой
            self.show_current_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            final_stats = self.db.get_stats()
            
            logger.info(f"\n{'='*50}")
            logger.info(f"SESSION COMPLETED")
            logger.info(f"Execution time: {duration}")
            logger.info(f"Processed: {self.session_stats['processed']}")
            logger.info(f"Added: {self.session_stats['added']}")
            logger.info(f"Skipped: {self.session_stats['skipped']}")
            logger.info(f"Errors: {self.session_stats['errors']}")
            logger.info(f"Memory cleanups: {self.session_stats['memory_cleanups']}")
            logger.info(f"Average speed: {self.session_stats['songs_per_minute']:.1f} songs/min")
            logger.info(f"Total in database: {final_stats['total_songs']} songs")
            logger.info(f"Final memory usage: {self.memory_monitor.get_memory_usage()['rss_mb']:.1f}MB")
            
            self.close()

    def close(self):
        logger.info("Closing database connection...")
        self.db.close()

def load_artist_list(filename: str = "rap_artists.json") -> List[str]:
    # Приоритет: сначала проверяем файл с оставшимися артистами
    remaining_file = "remaining_artists.json"
    if os.path.exists(remaining_file):
        logger.info(f"Loading remaining artists from {remaining_file}")
        with open(remaining_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Если файла с оставшимися нет, загружаем основной список
    if os.path.exists(filename):
        logger.info(f"Loading full artist list from {filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        logger.info("Using built-in artist list")
        artists = [
            "J. Cole", "Drake", "Eminem", "Kanye West",
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
        logger.error("Genius API token not found in .env!")
        exit(1)
        
    scraper = SafeGeniusScraper(TOKEN, "rap_lyrics.db")
    
    try:
        artists = load_artist_list()
        SONGS_PER_ARTIST = 500  # Увеличено до 500 песен на артиста
        
        logger.info(f"Loaded {len(artists)} artists")
        logger.info(f"Target: ~{len(artists) * SONGS_PER_ARTIST} songs")
        logger.info("To stop use: Get-Process python | Stop-Process -Force")
        
        scraper.run_scraping_session(artists, SONGS_PER_ARTIST)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        logger.info("Program completed")

if __name__ == "__main__":
    main()