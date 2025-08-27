import lyricsgenius
from requests.exceptions import ProxyError, RequestException
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
import gc
import psutil
from typing import List, Optional, Dict, Generator, Tuple
from ..utils.config import GENIUS_TOKEN, DB_PATH, LOG_FORMAT, LOG_FILE, DATA_DIR

# Проверка токена
TOKEN = GENIUS_TOKEN

# Настройка логирования с правильной кодировкой
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not TOKEN:
    logger.error("Genius API token not found in .env!")
    exit(1)

class ResourceMonitor:
    """Мониторинг системных ресурсов"""
    
    def __init__(self, memory_limit_mb: int = 2048):
        self.process = psutil.Process()
        self.memory_limit_mb = memory_limit_mb
        self.start_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> float:
        """Возвращает использование памяти в МБ"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self) -> float:
        """Возвращает использование CPU в %"""
        return self.process.cpu_percent(interval=1)
    
    def check_memory_limit(self) -> bool:
        """Проверяет превышение лимита памяти"""
        current_memory = self.get_memory_usage()
        return current_memory > self.memory_limit_mb
    
    def log_resources(self):
        """Логирует текущее использование ресурсов"""
        memory_mb = self.get_memory_usage()
        cpu_percent = self.get_cpu_usage()
        logger.info(f"💾 Memory: {memory_mb:.1f}MB | 🖥️ CPU: {cpu_percent:.1f}%")
        
        if memory_mb > self.memory_limit_mb * 0.8:  # 80% от лимита
            logger.warning(f"⚠️ High memory usage: {memory_mb:.1f}MB (limit: {self.memory_limit_mb}MB)")
    
    def force_garbage_collection(self):
        """Принудительная очистка памяти"""
        collected = gc.collect()
        logger.debug(f"🗑️ Garbage collection: freed {collected} objects")

class EnhancedLyricsDatabase:
    """Улучшенная база данных с метаданными"""
    
    def __init__(self, db_name=None):
        if db_name is None:
            db_name = str(DB_PATH)
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Оптимизация SQLite
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-2000")  # 2MB cache
        self.conn.execute("PRAGMA temp_store=MEMORY")
        
        self.create_table()
        logger.info(f"База данных {db_name} инициализирована с WAL режимом")
        self.batch_count = 0
        self.batch_size = 1000  # Увеличен до 1000 для лучшей производительности

    def create_table(self):
        """Создание таблицы с расширенными метаданными"""
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
                genre TEXT,
                release_date TEXT,
                album TEXT,
                language TEXT,
                explicit BOOLEAN,
                song_art_url TEXT,
                popularity_score INTEGER,
                lyrics_quality_score REAL,
                UNIQUE(artist, title)
            )
        """)
        
        # Создание индексов для быстрого поиска
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_artist ON songs(artist)",
            "CREATE INDEX IF NOT EXISTS idx_url ON songs(url)",
            "CREATE INDEX IF NOT EXISTS idx_genius_id ON songs(genius_id)",
            "CREATE INDEX IF NOT EXISTS idx_genre ON songs(genre)",
            "CREATE INDEX IF NOT EXISTS idx_release_date ON songs(release_date)",
            "CREATE INDEX IF NOT EXISTS idx_quality ON songs(lyrics_quality_score)",
            "CREATE INDEX IF NOT EXISTS idx_word_count ON songs(word_count)"
        ]
        
        for index_sql in indexes:
            self.conn.execute(index_sql)
        
        self.conn.commit()

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
            
            self.conn.execute(
                """INSERT INTO songs (
                    artist, title, lyrics, url, genius_id, word_count,
                    genre, release_date, album, language, explicit,
                    song_art_url, popularity_score, lyrics_quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    artist, title, lyrics, url, genius_id, word_count,
                    metadata.get('genre'),
                    metadata.get('release_date'),
                    metadata.get('album'),
                    metadata.get('language'),
                    metadata.get('explicit'),
                    metadata.get('song_art_url'),
                    metadata.get('popularity_score'),
                    lyrics_quality
                )
            )
            
            self.batch_count += 1
            if self.batch_count >= self.batch_size:
                self.conn.commit()
                self.batch_count = 0
                logger.debug(f"💾 Batch commit: {self.batch_size} records")
            
            return True
            
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                logger.warning(f"База заблокирована, повторная попытка: {artist} - {title}")
                time.sleep(2)
                try:
                    # Повторная попытка
                    return self.add_song(artist, title, lyrics, url, genius_id, metadata)
                except Exception:
                    logger.error(f"Повторная попытка не удалась: {artist} - {title}")
                    return False
            else:
                raise e
        except sqlite3.IntegrityError:
            logger.debug(f"Duplicate: {artist} - {title}")
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
        if url:
            cur = self.conn.execute("SELECT 1 FROM songs WHERE url=?", (url,))
        elif genius_id:
            cur = self.conn.execute("SELECT 1 FROM songs WHERE genius_id=?", (genius_id,))
        else:
            return False
        return cur.fetchone() is not None

    def get_stats(self) -> dict:
        self.conn.commit()
        cur = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT artist) as artists,
                AVG(word_count) as avg_words,
                AVG(lyrics_quality_score) as avg_quality,
                COUNT(CASE WHEN genre IS NOT NULL THEN 1 END) as with_genre
            FROM songs
        """)
        result = cur.fetchone()
        return {
            "total_songs": result["total"],
            "unique_artists": result["artists"],
            "avg_words": round(result["avg_words"] or 0, 1),
            "avg_quality": round(result["avg_quality"] or 0, 3),
            "with_metadata": result["with_genre"]
        }

    def get_recent_songs(self, limit: int = 5) -> List[dict]:
        self.conn.commit()
        cur = self.conn.execute("""
            SELECT artist, title, word_count, lyrics_quality_score, genre, scraped_date 
            FROM songs 
            ORDER BY id DESC 
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cur.fetchall()]

    def close(self):
        self.conn.commit()
        self.conn.close()

class OptimizedGeniusScraper:
    """Оптимизированный скрапер с мониторингом ресурсов"""
    
    def __init__(self, token: str, db_name: str = None, memory_limit_mb: int = 2048):
        if db_name is None:
            db_name = str(DB_PATH)
        
        # Убираем проблемные прокси переменные перед созданием клиента
        self._clear_proxy_env()
        
        self.genius = lyricsgenius.Genius(
            token,
            timeout=30,  # Увеличили timeout
            retries=2,   # Уменьшили retries для быстрого переключения
            remove_section_headers=True,
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)", "(Instrumental)", "(Skit)"]
        )
        
        self.db = EnhancedLyricsDatabase(db_name)
        self.monitor = ResourceMonitor(memory_limit_mb)
        self.session_stats = {
            "processed": 0, "added": 0, "skipped": 0, "errors": 0,
            "memory_warnings": 0, "gc_runs": 0
        }
        
        self.min_delay = 2.0
        self.max_delay = 5.0
        self.error_delay = 15.0
        self.max_retries = 3
        self.shutdown_requested = False
        
        # Счетчики для мониторинга
        self.songs_since_gc = 0
        self.gc_interval = 50  # Принудительная очистка каждые 50 песен
        
        # Сохраняем удаленные прокси переменные
        self.cleared_proxies = {}
        
        self._setup_signal_handlers()
        
    def _clear_proxy_env(self):
        """Убираем все проблемные прокси переменные"""
        proxy_vars = [
            'HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy',
            'FTP_PROXY', 'ftp_proxy', 'ALL_PROXY', 'all_proxy',
            'NO_PROXY', 'no_proxy'
        ]
        
        self.cleared_proxies = {}
        for var in proxy_vars:
            if var in os.environ:
                self.cleared_proxies[var] = os.environ.pop(var)
                logger.debug(f"🚫 Убрали прокси переменную: {var}")
                
    def _restore_proxy_env(self):
        """Восстанавливаем прокси переменные если нужно"""
        for var, value in self.cleared_proxies.items():
            os.environ[var] = value
        
    def _setup_signal_handlers(self):
        """Настройка обработчиков сигналов"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if sys.platform == "win32":
            try:
                signal.signal(signal.SIGBREAK, self._signal_handler)
            except AttributeError:
                pass

    def _signal_handler(self, signum, frame):
        logger.info(f"\nПолучен сигнал {signum}. Завершение работы...")
        self.shutdown_requested = True

    def safe_delay(self, is_error: bool = False):
        """Безопасная пауза с проверкой прерывания"""
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

    def clean_lyrics(self, lyrics: str) -> str:
        """Очистка текста песни"""
        if not lyrics:
            return ""
        
        # Удаляем информацию о контрибьюторах
        lyrics = re.sub(r"^\d+\s+Contributors.*?Lyrics", "", lyrics, flags=re.MULTILINE | re.DOTALL)
        
        # Удаляем блоки переводов
        lyrics = re.sub(r"Translations[A-Za-z]+", "", lyrics, flags=re.MULTILINE)
        
        # Удаляем метаинформацию Genius
        lyrics = re.sub(r"Lyrics[A-Z].*?Read More\s*", "", lyrics, flags=re.DOTALL)
        lyrics = re.sub(r"(?i)(Embed|Submitted by [^\n]*|Written by [^\n]*|You might also like).*$", "", lyrics, flags=re.DOTALL)
        
        # Удаляем ссылки и блоки в скобках
        lyrics = re.sub(r"https?://[^\s]+", "", lyrics)
        lyrics = re.sub(r"\[.*?\]", "", lyrics)
        
        # Нормализация переносов строк
        lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)
        lyrics = re.sub(r"\n{2,}", "\n", lyrics.strip())
        
        return lyrics.strip()

    def _is_valid_lyrics(self, lyrics: str) -> bool:
        """Проверка качества текста"""
        if not lyrics:
            return False
            
        lyrics = lyrics.strip()
        if len(lyrics) < 100 or len(lyrics.split()) < 20:
            return False
            
        # Проверка на инструментальные треки
        instrumental_markers = [
            "instrumental", "no lyrics", "без слов", "music only", 
            "beat only", "outro", "intro", "skit"
        ]
        return not any(marker in lyrics.lower() for marker in instrumental_markers)

    def extract_metadata(self, song) -> Dict:
        """Извлечение метаданных из объекта песни"""
        metadata = {}
        
        try:
            # Получаем доступные атрибуты
            if hasattr(song, 'album') and song.album:
                metadata['album'] = song.album.get('name') if isinstance(song.album, dict) else str(song.album)
            
            if hasattr(song, 'release_date_for_display'):
                metadata['release_date'] = song.release_date_for_display
                
            if hasattr(song, 'song_art_image_url'):
                metadata['song_art_url'] = song.song_art_image_url
                
            if hasattr(song, 'stats') and song.stats:
                if 'pageviews' in song.stats:
                    metadata['popularity_score'] = song.stats['pageviews']
            
            # Попытка определить язык (простая эвристика)
            if hasattr(song, 'language'):
                metadata['language'] = song.language
            else:
                metadata['language'] = 'en'  # По умолчанию английский
                
        except Exception as e:
            logger.debug(f"Ошибка извлечения метаданных: {e}")
            
        return metadata

    def get_songs_generator(self, artist_name: str, max_songs: int = 500) -> Generator[Tuple[any, int], None, None]:
        """Генератор для пошаговой загрузки песен (экономия памяти) с улучшенной обработкой ошибок"""
        try:
            logger.info(f"🎵 Поиск артиста: {artist_name}")
            artist = None
            
            # Попытка 1: Обычный поиск
            try:
                artist = self.genius.search_artist(
                    artist_name, 
                    max_songs=min(max_songs, 50),  # Ограничиваем первую попытку
                    sort="popularity", 
                    get_full_info=False
                )
                if artist and artist.songs:
                    logger.info(f"✅ Найден артист: {artist.name} с {len(artist.songs)} песнями")
                
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"❌ Ошибка при поиске артиста {artist_name}: {e}")
                
                # Проверяем на прокси/сетевые ошибки
                if any(keyword in error_msg for keyword in ['proxy', 'connection', 'timeout', 'retries exceeded']):
                    logger.warning("🔄 Детектирована сетевая проблема, пробуем альтернативы...")
                    
                    # Попытка 2: Изменяем настройки сессии
                    try:
                        logger.info("🔧 Попытка 2: Изменяем User-Agent и headers...")
                        self.genius._session.headers.update({
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            'Accept': 'application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                            'Accept-Language': 'en-US,en;q=0.5',
                            'Connection': 'keep-alive'
                        })
                        
                        artist = self.genius.search_artist(
                            artist_name, 
                            max_songs=min(max_songs, 20),  # Еще меньше песен
                            sort="popularity"
                        )
                        if artist and artist.songs:
                            logger.info(f"✅ Попытка 2 успешна: {len(artist.songs)} песен")
                            
                    except Exception as e2:
                        logger.warning(f"⚠️ Попытка 2 не удалась: {e2}")
                        
                        # Попытка 3: Минимальный поиск
                        try:
                            logger.info("🔧 Попытка 3: Минимальный поиск...")
                            artist = self.genius.search_artist(
                                artist_name, 
                                max_songs=5,
                                sort="popularity"
                            )
                            if artist and artist.songs:
                                logger.info(f"✅ Попытка 3 успешна: {len(artist.songs)} песен")
                        except Exception as e3:
                            logger.error(f"❌ Все попытки исчерпаны: {e3}")
                            return
                else:
                    # Не сетевая ошибка, просто логируем и завершаем
                    logger.error(f"❌ Неизвестная ошибка поиска: {e}")
                    return
            
            # Проверяем результат
            if not artist or not hasattr(artist, 'songs') or not artist.songs:
                logger.warning(f"❌ Артист {artist_name} не найден или нет песен")
                return
            
            total_songs = len(artist.songs)
            logger.info(f"📀 Найдено {total_songs} песен для {artist_name}")
            
            # Показываем первые несколько песен для подтверждения
            logger.info("🎵 Первые найденные песни:")
            for i, song in enumerate(artist.songs[:5], 1):
                logger.info(f"  {i}. {song.title}")
            
            # Возвращаем песни по одной
            for i, song in enumerate(artist.songs):
                if self.shutdown_requested:
                    logger.info(f"🛑 Остановка на песне {i+1}/{total_songs}")
                    break
                    
                yield song, i + 1
                
                # Принудительная очистка памяти каждые N песен
                self.songs_since_gc += 1
                if self.songs_since_gc >= self.gc_interval:
                    self.monitor.force_garbage_collection()
                    self.session_stats["gc_runs"] += 1
                    self.songs_since_gc = 0
                    
                # Удаляем обработанную песню из памяти
                del song
                
        except Exception as e:
            logger.error(f"❌ Критическая ошибка получения списка песен для {artist_name}: {e}")
            return

    def scrape_artist_songs(self, artist_name: str, max_songs: int = 500) -> int:
        """Скрапинг песен артиста с оптимизацией памяти"""
        added_count = 0
        retry_count = 0
        
        logger.info(f"🎤 Начинаем обработку артиста: {artist_name}")
        
        while retry_count < self.max_retries and not self.shutdown_requested:
            try:
                # Мониторинг ресурсов перед началом
                self.monitor.log_resources()
                
                # Проверка лимита памяти
                if self.monitor.check_memory_limit():
                    logger.warning("⚠️ Превышен лимит памяти! Принудительная очистка...")
                    self.monitor.force_garbage_collection()
                    self.session_stats["memory_warnings"] += 1
                    
                    if self.monitor.check_memory_limit():
                        logger.error("🚨 Критичное использование памяти! Пропускаем артиста.")
                        return 0
                
                # Получаем генератор песен
                songs_generator = self.get_songs_generator(artist_name, max_songs)
                processed_count = 0
                
                for song, song_number in songs_generator:
                    if self.shutdown_requested:
                        break
                        
                    try:
                        # Проверка на дубликат
                        if self.db.song_exists(url=song.url):
                            logger.debug(f"⏩ Пропускаем дубликат: {song.title}")
                            self.session_stats["skipped"] += 1
                            continue

                        # Очистка текста
                        lyrics = self.clean_lyrics(song.lyrics)
                        if not self._is_valid_lyrics(lyrics):
                            logger.debug(f"⏩ Некачественный текст: {song.title}")
                            self.session_stats["skipped"] += 1
                            continue

                        # Извлечение метаданных
                        metadata = self.extract_metadata(song)

                        # Сохранение в БД
                        if self.db.add_song(
                            artist_name, song.title, lyrics, song.url, 
                            getattr(song, 'id', None), metadata
                        ):
                            added_count += 1
                            self.session_stats["added"] += 1
                            word_count = len(lyrics.split())
                            quality = self.db._calculate_lyrics_quality(lyrics)
                            
                            logger.info(
                                f"✅ Добавлено: {artist_name} - {song.title} "
                                f"({word_count} слов, качество: {quality:.2f})"
                            )
                            
                            # Статистика каждые 10 песен
                            if self.session_stats["added"] % 10 == 0:
                                current_stats = self.db.get_stats()
                                logger.info(f"📊 В БД: {current_stats['total_songs']} песен")
                                
                        else:
                            self.session_stats["skipped"] += 1

                        self.session_stats["processed"] += 1
                        processed_count += 1
                        
                        # Прогресс каждые 25 песен
                        if processed_count % 25 == 0:
                            logger.info(f"📈 Обработано {processed_count} песен для {artist_name}")
                            self.monitor.log_resources()

                        # Пауза между песнями
                        if not self.shutdown_requested:
                            self.safe_delay()

                    except Exception as e:
                        if "timeout" in str(e).lower():
                            logger.error(f"⏰ Timeout для {song.title}: {e}")
                            self.session_stats["errors"] += 1
                            self.safe_delay(is_error=True)
                        elif any(keyword in str(e).lower() for keyword in ['proxy', 'connection', 'retries exceeded']):
                            logger.error(f"🌐 Сетевая ошибка для {song.title}: {e}")
                            self.session_stats["errors"] += 1
                            # Увеличиваем паузу при сетевых ошибках
                            logger.info("⏳ Дополнительная пауза при сетевой ошибке...")
                            time.sleep(10)
                        else:
                            logger.error(f"❌ Ошибка с песней {song.title}: {e}")
                            self.session_stats["errors"] += 1
                            self.safe_delay(is_error=True)

                break  # Успешно завершили артиста

            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    logger.error(f"🚫 Rate Limit для {artist_name}: {e}")
                    logger.info("⏳ Ожидание 60 секунд...")
                    time.sleep(60)
                    retry_count += 1
                else:
                    retry_count += 1
                    logger.error(f"❌ Ошибка с артистом {artist_name} (попытка {retry_count}): {e}")
                    
                    if retry_count >= self.max_retries:
                        logger.error(f"🔄 Превышено количество попыток для {artist_name}")
                        self.session_stats["errors"] += 1
                        break
                        
                    self.safe_delay(is_error=True)

        logger.info(f"✅ Завершена обработка {artist_name}: добавлено {added_count} песен")
        return added_count

    def show_current_results(self):
        """Показ текущих результатов с расширенной статистикой"""
        stats = self.db.get_stats()
        recent_songs = self.db.get_recent_songs(5)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 ТЕКУЩИЕ РЕЗУЛЬТАТЫ:")
        logger.info(f"🎵 Всего песен в БД: {stats['total_songs']}")
        logger.info(f"👤 Уникальных артистов: {stats['unique_artists']}")
        logger.info(f"📝 Среднее слов в песне: {stats['avg_words']}")
        logger.info(f"⭐ Среднее качество: {stats['avg_quality']}")
        logger.info(f"🏷️ С метаданными: {stats['with_metadata']}")
        logger.info(f"➕ Добавлено за сессию: {self.session_stats['added']}")
        
        # Статистика ресурсов
        memory_mb = self.monitor.get_memory_usage()
        logger.info(f"💾 Использование памяти: {memory_mb:.1f}MB")
        logger.info(f"🗑️ Очисток памяти: {self.session_stats['gc_runs']}")
        
        if recent_songs:
            logger.info(f"\n🎶 Последние добавленные песни:")
            for song in recent_songs:
                genre_info = f" [{song['genre']}]" if song['genre'] else ""
                quality_info = f" (Q:{song['lyrics_quality_score']:.2f})" if song['lyrics_quality_score'] else ""
                logger.info(f"  • {song['artist']} - {song['title']}{genre_info} "
                          f"({song['word_count']} слов){quality_info}")
        
        logger.info(f"{'='*70}\n")

    def run_scraping_session(self, artists: List[str], songs_per_artist: int = 500):
        """Запуск сессии скрапинга с мониторингом"""
        logger.info(f"🚀 Начинаем сессию скрапинга: {len(artists)} артистов, {songs_per_artist} песен каждый")
        start_time = datetime.now()
        
        initial_stats = self.db.get_stats()
        logger.info(f"📚 Уже в базе: {initial_stats['total_songs']} песен")
        
        try:
            for i, artist_name in enumerate(artists, 1):
                if self.shutdown_requested:
                    logger.info("🛑 Получен запрос на остановку")
                    break
                    
                logger.info(f"\n{'='*60}")
                logger.info(f"🎤 Артист {i}/{len(artists)}: {artist_name}")
                
                # Мониторинг перед каждым артистом
                self.monitor.log_resources()
                
                added = self.scrape_artist_songs(artist_name, songs_per_artist)
                logger.info(f"✅ Добавлено песен для {artist_name}: {added}")
                
                stats = self.db.get_stats()
                logger.info(f"📊 Всего в БД: {stats['total_songs']} песен от {stats['unique_artists']} артистов")
                
                # Пауза между артистами
                if i < len(artists) and not self.shutdown_requested:
                    artist_delay = random.uniform(10, 20)
                    logger.info(f"⏳ Пауза между артистами: {artist_delay:.1f}с")
                    
                    intervals = int(artist_delay)
                    for _ in range(intervals):
                        if self.shutdown_requested:
                            break
                        time.sleep(1)

        except KeyboardInterrupt:
            logger.info("⌨️ Прерывание пользователем (Ctrl+C)")
            self.shutdown_requested = True
        except MemoryError:
            logger.error("🚨 Критическая нехватка памяти!")
            self.shutdown_requested = True
        except Exception as e:
            logger.error(f"💥 Критическая ошибка: {e}")
        finally:
            self.db.conn.commit()
            self.show_current_results()
            
            end_time = datetime.now()
            duration = end_time - start_time
            final_stats = self.db.get_stats()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🏁 СЕССИЯ ЗАВЕРШЕНА")
            logger.info(f"⏱️ Время выполнения: {duration}")
            logger.info(f"📈 Обработано: {self.session_stats['processed']}")
            logger.info(f"✅ Добавлено: {self.session_stats['added']}")
            logger.info(f"⏩ Пропущено: {self.session_stats['skipped']}")
            logger.info(f"❌ Ошибок: {self.session_stats['errors']}")
            logger.info(f"⚠️ Предупреждений памяти: {self.session_stats['memory_warnings']}")
            logger.info(f"🗑️ Очисток памяти: {self.session_stats['gc_runs']}")
            logger.info(f"📚 Всего в БД: {final_stats['total_songs']} песен")
            logger.info(f"⭐ Среднее качество: {final_stats['avg_quality']}")
            
            self.close()

    def close(self):
        logger.info("🔒 Закрытие соединения с БД...")
        self.db.close()
        # Восстанавливаем прокси настройки если они были
        self._restore_proxy_env()

def load_artist_list(filename: str = "rap_artists.json") -> List[str]:
    """Загрузка списка артистов с приоритетом remaining_artists.json"""
    remaining_file = DATA_DIR / "remaining_artists.json"
    if remaining_file.exists():
        logger.info(f"📂 Загружаем оставшихся артистов из {remaining_file}")
        with open(remaining_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    full_file = DATA_DIR / filename
    if full_file.exists():
        logger.info(f"📂 Загружаем полный список артистов из {full_file}")
        with open(full_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        logger.info("📂 Используем встроенный список артистов")
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
        full_file = DATA_DIR / filename
        with open(full_file, 'w', encoding='utf-8') as f:
            json.dump(artists, f, indent=2, ensure_ascii=False)
        return artists

def main():
    if not TOKEN:
        logger.error("❌ Genius API token не найден в .env!")
        exit(1)
        
    # Настройки для оптимизированного скрапера
    MEMORY_LIMIT_MB = 3072  # 3GB лимит памяти
    scraper = OptimizedGeniusScraper(TOKEN, None, MEMORY_LIMIT_MB)
    
    try:
        artists = load_artist_list()
        SONGS_PER_ARTIST = 500
        
        logger.info(f"🎯 Загружено {len(artists)} артистов")
        logger.info(f"🎵 Цель: ~{len(artists) * SONGS_PER_ARTIST} песен")
        logger.info(f"💾 Лимит памяти: {MEMORY_LIMIT_MB}MB")
        logger.info("🛑 Для остановки: Get-Process python | Stop-Process -Force")
        
        scraper.run_scraping_session(artists, SONGS_PER_ARTIST)
        
    except Exception as e:
        logger.error(f"💥 Ошибка в main: {e}")
    finally:
        logger.info("🏁 Программа завершена")

if __name__ == "__main__":
    main()
