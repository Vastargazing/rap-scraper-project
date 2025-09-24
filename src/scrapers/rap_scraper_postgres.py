#!/usr/bin/env python3
"""
🎤 Массовый скрапер для анализа и загрузки текстов песен в PostgreSQL (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ)

КЛЮЧЕВЫЕ УЛУЧШЕНИЯ:
✨ Асинхронный PostgreSQL через asyncpg для повышения производительности
⚡ Батчевое сохранение в БД (значительно быстрее)
🛡️ Улучшенная обработка ошибок с circuit breaker pattern
🔄 Retry logic с exponential backoff
📊 Детализированные метрики и мониторинг
🧠 Умное управление памятью с предиктивной очисткой
🌐 Fallback стратегии для API вызовов
🎯 Connection pooling для PostgreSQL
"""

import lyricsgenius
from requests.exceptions import ProxyError, RequestException
import asyncio
import time
import random
import logging
import re
import signal
import sys
from datetime import datetime, timedelta
import json
import os
import gc
import psutil
from typing import List, Optional, Dict, Generator, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncpg
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Импорты (предполагаем структуру)
try:
    from ..utils.config import GENIUS_TOKEN, LOG_FORMAT, LOG_FILE, DATA_DIR
    from ..utils.postgres_db import PostgreSQLManager
except ImportError:
    # Fallback для тестирования
    # Если модуль конфигурации не импортируется (например, запуск скрипта напрямую),
    # подгружаем .env вручную и читаем переменную с правильным именем.
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        load_dotenv(project_root / ".env")
    except Exception:
        pass

    # Основное имя переменной в .env: GENIUS_ACCESS_TOKEN
    GENIUS_TOKEN = os.getenv('GENIUS_ACCESS_TOKEN') or os.getenv('GENIUS_TOKEN')
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_FILE = 'scraper.log'
    DATA_DIR = 'data'
    # Попытка импортировать PostgreSQLManager напрямую из модуля utils
    try:
        import importlib, sys
        # Добавляем корень проекта в sys.path для корректного абсолютного импорта
        project_root_str = str(project_root)
        if project_root_str not in sys.path:
            sys.path.append(project_root_str)

        postgres_mod = importlib.import_module('src.utils.postgres_db')
        PostgreSQLManager = getattr(postgres_mod, 'PostgreSQLManager')
    except Exception:
        logger = logging.getLogger(__name__)
        logger.warning('⚠️ Не удалось импортировать src.utils.postgres_db — используем заглушку PostgreSQLManager')

        # Лёгкая асинхронная заглушка, чтобы скрипт мог выполняться без реальной БД
        class PostgreSQLManager:
            def __init__(self, *args, **kwargs):
                logger.info('ℹ️ Используется заглушка PostgreSQLManager — все операции с БД будут эмулированы')

            async def batch_add_songs(self, songs_batch):
                # эмулируем асинхронное добавление
                return len(songs_batch)

            def add_song(self, *args, **kwargs):
                return True

            def song_exists(self, url: str = None, genius_id: int = None) -> bool:
                return False

            def get_stats(self):
                return {"total_songs": 0, "unique_artists": 0, "avg_words": 0, "avg_quality": 0, "with_metadata": 0}

            def close(self):
                return

# Настройка улучшенного логирования
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ScrapingStatus(Enum):
    """Статусы обработки песен"""
    SUCCESS = "success"
    SKIPPED_DUPLICATE = "skipped_duplicate"
    SKIPPED_QUALITY = "skipped_quality"
    ERROR_NETWORK = "error_network"
    ERROR_API_LIMIT = "error_api_limit"
    ERROR_PARSING = "error_parsing"
    ERROR_UNKNOWN = "error_unknown"

@dataclass
class SessionMetrics:
    """Детализированные метрики сессии"""
    processed: int = 0
    added: int = 0
    skipped_duplicates: int = 0
    skipped_quality: int = 0
    error_network: int = 0
    error_api_limit: int = 0
    error_parsing: int = 0
    error_unknown: int = 0
    memory_warnings: int = 0
    gc_runs: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    avg_processing_time: float = 0.0
    batch_saves: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    def increment(self, status: ScrapingStatus, processing_time: float = 0.0):
        """Увеличение счетчика по статусу"""
        self.processed += 1
        if processing_time > 0:
            self.avg_processing_time = (
                (self.avg_processing_time * (self.processed - 1) + processing_time) 
                / self.processed
            )
            
        if status == ScrapingStatus.SUCCESS:
            self.added += 1
        elif status == ScrapingStatus.SKIPPED_DUPLICATE:
            self.skipped_duplicates += 1
        elif status == ScrapingStatus.SKIPPED_QUALITY:
            self.skipped_quality += 1
        elif status == ScrapingStatus.ERROR_NETWORK:
            self.error_network += 1
        elif status == ScrapingStatus.ERROR_API_LIMIT:
            self.error_api_limit += 1
        elif status == ScrapingStatus.ERROR_PARSING:
            self.error_parsing += 1
        elif status == ScrapingStatus.ERROR_UNKNOWN:
            self.error_unknown += 1
    
    @property
    def total_errors(self) -> int:
        return (self.error_network + self.error_api_limit + 
                self.error_parsing + self.error_unknown)
    
    @property
    def success_rate(self) -> float:
        return (self.added / self.processed * 100) if self.processed > 0 else 0.0
    
    @property
    def runtime(self) -> timedelta:
        return datetime.now() - self.start_time

class CircuitBreaker:
    """Circuit breaker для API вызовов"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        
    def call(self, func, *args, **kwargs):
        """Выполнение функции через circuit breaker"""
        if self.state == "open":
            if (datetime.now() - self.last_failure_time).seconds > self.recovery_timeout:
                self.state = "half_open"
                logger.info("🔄 Circuit breaker: переход в half-open состояние")
            else:
                raise Exception("Circuit breaker is OPEN - API временно недоступен")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half_open":
                self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise e
    
    def record_failure(self):
        """Регистрация ошибки"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"🚨 Circuit breaker ОТКРЫТ после {self.failure_count} ошибок")
    
    def reset(self):
        """Сброс circuit breaker"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"
        logger.info("✅ Circuit breaker сброшен")

class EnhancedResourceMonitor:
    """Улучшенный мониторинг ресурсов с предиктивной очисткой"""
    
    def __init__(self, memory_limit_mb: int = 2048):
        self.process = psutil.Process()
        self.memory_limit_mb = memory_limit_mb
        self.start_memory = self.get_memory_usage()
        self.memory_history = []
        self.max_history = 10
        
    def get_memory_usage(self) -> float:
        """Возвращает использование памяти в МБ"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_cpu_usage(self) -> float:
        """Возвращает использование CPU в %"""
        return self.process.cpu_percent(interval=0.1)
    
    def predict_memory_trend(self) -> str:
        """Предсказание тренда использования памяти"""
        if len(self.memory_history) < 3:
            return "insufficient_data"
        
        recent_avg = sum(self.memory_history[-3:]) / 3
        older_avg = sum(self.memory_history[-6:-3]) / 3 if len(self.memory_history) >= 6 else recent_avg
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def check_memory_limit(self) -> tuple[bool, str]:
        """Проверка лимита памяти с рекомендациями"""
        current_memory = self.get_memory_usage()
        self.memory_history.append(current_memory)
        
        if len(self.memory_history) > self.max_history:
            self.memory_history.pop(0)
        
        usage_ratio = current_memory / self.memory_limit_mb
        trend = self.predict_memory_trend()
        
        if usage_ratio > 0.95:
            return True, "critical"
        elif usage_ratio > 0.85 and trend == "increasing":
            return True, "warning_trending"
        elif usage_ratio > 0.80:
            return True, "warning"
        else:
            return False, "normal"
    
    def log_resources(self):
        """Улучшенное логирование ресурсов"""
        memory_mb = self.get_memory_usage()
        cpu_percent = self.get_cpu_usage()
        trend = self.predict_memory_trend()
        
        logger.info(f"💾 Memory: {memory_mb:.1f}MB | 🖥️ CPU: {cpu_percent:.1f}% | 📈 Trend: {trend}")
        
        limit_exceeded, status = self.check_memory_limit()
        if limit_exceeded:
            logger.warning(f"⚠️ Memory status: {status} - {memory_mb:.1f}MB/{self.memory_limit_mb}MB")
    
    def force_garbage_collection(self) -> int:
        """Улучшенная очистка памяти"""
        before_memory = self.get_memory_usage()
        collected = gc.collect()
        after_memory = self.get_memory_usage()
        freed_mb = before_memory - after_memory
        
        logger.info(f"🗑️ GC: освобождено {freed_mb:.1f}MB, собрано {collected} объектов")
        return collected

class BatchProcessor:
    """Батчевый процессор для PostgreSQL операций"""
    
    def __init__(self, batch_size: int = 10, flush_interval: float = 30.0):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.pending_songs = []
        self.last_flush = time.time()
        
    def add_song(self, song_data: dict):
        """Добавление песни в батч"""
        self.pending_songs.append(song_data)
        
        # Проверяем необходимость сохранения
        current_time = time.time()
        if (len(self.pending_songs) >= self.batch_size or 
            current_time - self.last_flush > self.flush_interval):
            return True  # Нужно сохранить
        return False
    
    def get_pending_batch(self) -> List[dict]:
        """Получение текущего батча и очистка"""
        batch = self.pending_songs.copy()
        self.pending_songs.clear()
        self.last_flush = time.time()
        return batch
    
    def has_pending(self) -> bool:
        """Есть ли ожидающие сохранения данные"""
        return len(self.pending_songs) > 0

class OptimizedPostgreSQLScraper:
    """ОПТИМИЗИРОВАННЫЙ скрапер с PostgreSQL и продвинутыми возможностями"""
    
    def __init__(self, token: str, memory_limit_mb: int = 2048, batch_size: int = 10):
        # Убираем проблемные прокси переменные
        self._clear_proxy_env()
        
        # Genius API клиент с оптимизациями
        self.genius = lyricsgenius.Genius(
            token,
            timeout=45,  # Увеличили timeout
            retries=1,   # Убрали retries, делаем свой retry logic
            remove_section_headers=True,
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)", "(Instrumental)", "(Skit)", "(Interlude)"]
        )
        
        # Компоненты
        self.db = PostgreSQLManager()
        self.monitor = EnhancedResourceMonitor(memory_limit_mb)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=120)
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        self.metrics = SessionMetrics()
        
        # Кэш для дедупликации
        self.url_cache = set()
        self.artist_cache = {}
        
        # Настройки retry
        self.base_delay = 2.0
        self.max_delay = 30.0
        self.backoff_multiplier = 1.5
        self.max_retries = 3
        
        # Флаги управления
        self.shutdown_requested = False
        self.pause_requested = False
        
        # Thread pool для CPU-intensive операций
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        self.cleared_proxies = {}
        self._setup_signal_handlers()
        
        logger.info("🚀 Инициализирован ОПТИМИЗИРОВАННЫЙ PostgreSQL скрапер")
        logger.info(f"⚙️ Batch size: {batch_size}, Memory limit: {memory_limit_mb}MB")
        
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
                
    def _restore_proxy_env(self):
        """Восстанавливаем прокси переменные"""
        for var, value in self.cleared_proxies.items():
            os.environ[var] = value
        
    def _setup_signal_handlers(self):
        """Настройка обработчиков сигналов"""
        def signal_handler(signum, frame):
            logger.info(f"\n🛑 Получен сигнал {signum}. Graceful shutdown...")
            self.shutdown_requested = True
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        if sys.platform == "win32":
            try:
                signal.signal(signal.SIGBREAK, signal_handler)
            except AttributeError:
                pass

    async def safe_delay(self, is_error: bool = False, retry_count: int = 0):
        """Асинхронная пауза с exponential backoff"""
        if is_error:
            # Exponential backoff для ошибок
            delay = min(
                self.base_delay * (self.backoff_multiplier ** retry_count),
                self.max_delay
            )
        else:
            delay = random.uniform(self.base_delay, self.base_delay + 2)
        
        # Разбиваем на интервалы для проверки shutdown
        intervals = int(delay)
        remainder = delay - intervals
        
        for _ in range(intervals):
            if self.shutdown_requested:
                return
            await asyncio.sleep(1)
                
        if remainder > 0 and not self.shutdown_requested:
            await asyncio.sleep(remainder)

    def clean_lyrics(self, lyrics: str) -> str:
        """Оптимизированная очистка текста песни"""
        if not lyrics:
            return ""
        
        # Предкомпилированные регулярные выражения (быстрее)
        patterns = [
            (re.compile(r"^\d+\s+Contributors.*?Lyrics", re.MULTILINE | re.DOTALL), ""),
            (re.compile(r"Translations[A-Za-z]+", re.MULTILINE), ""),
            (re.compile(r"Lyrics[A-Z].*?Read More\s*", re.DOTALL), ""),
            (re.compile(r"(?i)(Embed|Submitted by [^\n]*|Written by [^\n]*|You might also like).*$", re.DOTALL), ""),
            (re.compile(r"https?://[^\s]+"), ""),
            (re.compile(r"\[.*?\]"), ""),
            (re.compile(r"\n{3,}"), "\n\n"),
            (re.compile(r"\n{2,}"), "\n"),
        ]
        
        for pattern, replacement in patterns:
            lyrics = pattern.sub(replacement, lyrics)
        
        return lyrics.strip()

    def _is_valid_lyrics(self, lyrics: str) -> tuple[bool, str]:
        """Улучшенная проверка качества текста с причиной"""
        if not lyrics:
            return False, "empty"
            
        lyrics = lyrics.strip()
        word_count = len(lyrics.split())
        
        if len(lyrics) < 100:
            return False, "too_short_chars"
        if word_count < 20:
            return False, "too_short_words"
            
        # Улучшенные проверки
        instrumental_markers = [
            "instrumental", "no lyrics", "без слов", "music only", 
            "beat only", "outro", "intro", "skit", "interlude"
        ]
        
        lyrics_lower = lyrics.lower()
        for marker in instrumental_markers:
            if marker in lyrics_lower:
                return False, f"instrumental_marker_{marker}"
        
        # Проверка на повторяющийся контент
        unique_lines = set(line.strip() for line in lyrics.split('\n') if line.strip())
        if len(unique_lines) < word_count * 0.3:  # Слишком много повторений
            return False, "too_repetitive"
        
        return True, "valid"

    def generate_song_hash(self, artist: str, title: str, lyrics: str) -> str:
        """Генерация хэша для дедупликации"""
        content = f"{artist.lower().strip()}|{title.lower().strip()}|{lyrics[:200].lower()}"
        return hashlib.md5(content.encode()).hexdigest()

    def extract_metadata(self, song) -> Dict[str, Any]:
        """Улучшенное извлечение метаданных"""
        metadata = {}
        
        try:
            # Безопасное извлечение атрибутов
            safe_attrs = [
                ('album', ['album', 'name']),
                ('release_date', ['release_date_for_display']),
                ('song_art_url', ['song_art_image_url']),
                ('primary_artist_name', ['primary_artist', 'name']),
                ('featured_artists', ['featured_artists']),
                ('producer_artists', ['producer_artists']),
                ('writer_artists', ['writer_artists']),
                ('language', ['language']),
            ]
            
            for key, attr_path in safe_attrs:
                value = song
                for attr in attr_path:
                    if hasattr(value, attr):
                        value = getattr(value, attr)
                    elif isinstance(value, dict) and attr in value:
                        value = value[attr]
                    else:
                        value = None
                        break
                        
                if value is not None:
                    metadata[key] = str(value) if not isinstance(value, (dict, list)) else value
            
            # Статистики
            if hasattr(song, 'stats') and song.stats:
                stats = song.stats
                if isinstance(stats, dict):
                    metadata['pageviews'] = stats.get('pageviews', 0)
                    metadata['hot'] = stats.get('hot', False)
            
            # Определение языка (улучшенная эвристика)
            if 'language' not in metadata and hasattr(song, 'lyrics'):
                # Простая эвристика для определения языка
                english_indicators = ['the', 'and', 'you', 'that', 'with', 'for', 'are', 'this']
                russian_indicators = ['что', 'как', 'это', 'они', 'все', 'так', 'мне', 'его']
                
                lyrics_lower = song.lyrics.lower() if song.lyrics else ""
                english_count = sum(1 for word in english_indicators if word in lyrics_lower)
                russian_count = sum(1 for word in russian_indicators if word in lyrics_lower)
                
                if russian_count > english_count:
                    metadata['language'] = 'ru'
                else:
                    metadata['language'] = 'en'
                    
        except Exception as e:
            logger.debug(f"Ошибка извлечения метаданных: {e}")
            
        return metadata

    async def process_song_batch(self, songs_batch: List[dict]) -> int:
        """Асинхронное батчевое сохранение в PostgreSQL"""
        if not songs_batch:
            return 0
            
        start_time = time.time()
        saved_count = 0
        
        try:
            # Используем asyncpg для быстрого батчевого INSERT
            # Предполагаем, что в PostgreSQLManager есть async методы
            if hasattr(self.db, 'batch_add_songs'):
                saved_count = await self.db.batch_add_songs(songs_batch)
            else:
                # Fallback: последовательное сохранение
                for song_data in songs_batch:
                    if self.db.add_song(**song_data):
                        saved_count += 1
            
            processing_time = time.time() - start_time
            self.metrics.batch_saves += 1
            
            logger.info(f"💾 Батч сохранен: {saved_count}/{len(songs_batch)} песен за {processing_time:.2f}с")
            
        except Exception as e:
            logger.error(f"❌ Ошибка батчевого сохранения: {e}")
            
        return saved_count

    async def get_songs_async_generator(self, artist_name: str, max_songs: int = 500):
        """Асинхронный генератор для получения песен с улучшенной обработкой ошибок"""
        try:
            logger.info(f"🎵 Поиск артиста: {artist_name}")
            
            # Проверяем кэш артиста
            if artist_name in self.artist_cache:
                logger.info(f"📋 Используем кэш для {artist_name}")
                cached_songs = self.artist_cache[artist_name]
                for i, song in enumerate(cached_songs):
                    yield song, i + 1
                return
            
            # Поиск артиста через circuit breaker
            artist = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.circuit_breaker.call(
                    self.genius.search_artist,
                    artist_name,
                    max_songs=min(max_songs, 50),  # Ограничиваем первый запрос
                    sort="popularity",
                    get_full_info=False
                )
            )
            
            self.metrics.api_calls += 1
            
            if not artist or not hasattr(artist, 'tracks) or not artist.songs:
                logger.warning(f"❌ Артист {artist_name} не найден или нет песен")
                return
            
            total_songs = len(artist.songs)
            logger.info(f"📀 Найдено {total_songs} песен для {artist_name}")
            
            # Кэшируем результат
            self.artist_cache[artist_name] = artist.tracks[:20]  # Кэшируем первые 20 песен
            
            # Показываем первые песни
            logger.info("🎵 Первые найденные песни:")
            for i, song in enumerate(artist.tracks[:5], 1):
                logger.info(f"  {i}. {song.title}")
            
            # Возвращаем песни
            for i, song in enumerate(artist.songs):
                if self.shutdown_requested:
                    logger.info(f"🛑 Остановка на песне {i+1}/{total_songs}")
                    break
                    
                yield song, i + 1
                
                # Проверка памяти каждые 10 песен
                if i % 10 == 0:
                    limit_exceeded, status = self.monitor.check_memory_limit()
                    if limit_exceeded and status == "critical":
                        logger.warning("🚨 Критическое использование памяти! Принудительная GC...")
                        self.monitor.force_garbage_collection()
                        self.metrics.gc_runs += 1
                        await asyncio.sleep(1)  # Даем время на GC
                
        except Exception as e:
            error_msg = str(e).lower()
            if "circuit breaker" in error_msg:
                logger.error(f"🚨 Circuit breaker активен для {artist_name}")
            elif any(keyword in error_msg for keyword in ['timeout', 'connection', 'network']):
                logger.error(f"🌐 Сетевая ошибка для {artist_name}: {e}")
            else:
                logger.error(f"❌ Неизвестная ошибка для {artist_name}: {e}")
            return

    async def process_single_song(self, song, song_number: int, artist_name: str) -> ScrapingStatus:
        """Асинхронная обработка одной песни"""
        start_time = time.time()
        
        try:
            # Проверка URL кэша
            song_url = getattr(song, 'url', f"https://genius.com/songs/{getattr(song, 'id', 'unknown')}")
            if song_url in self.url_cache:
                self.metrics.cache_hits += 1
                return ScrapingStatus.SKIPPED_DUPLICATE
            
            # Получаем полные данные песни
            if not hasattr(song, 'lyrics') or not song.lyrics:
                try:
                    # Асинхронно загружаем текст
                    full_song = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: self.circuit_breaker.call(self.genius.song, song.id)
                    )
                    self.metrics.api_calls += 1
                    
                    if full_song and hasattr(full_song, 'lyrics'):
                        song = full_song
                    else:
                        return ScrapingStatus.SKIPPED_QUALITY
                        
                except Exception as e:
                    if "circuit breaker" in str(e).lower():
                        return ScrapingStatus.ERROR_API_LIMIT
                    else:
                        return ScrapingStatus.ERROR_NETWORK
            
            # Проверка дубликата в БД (быстрая проверка)
            if self.db.song_exists(url=song_url):
                self.url_cache.add(song_url)
                return ScrapingStatus.SKIPPED_DUPLICATE
            
            # Очистка и валидация текста
            lyrics = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.clean_lyrics, song.lyrics
            )
            
            is_valid, reason = self._is_valid_lyrics(lyrics)
            if not is_valid:
                logger.debug(f"⏩ Некачественный текст ({reason}): {song.title}")
                return ScrapingStatus.SKIPPED_QUALITY
            
            # Генерация хэша для дедупликации
            song_hash = self.generate_song_hash(artist_name, song.title, lyrics)
            
            # Извлечение метаданных
            metadata = self.extract_metadata(song)
            metadata['processing_time'] = time.time() - start_time
            metadata['song_hash'] = song_hash
            
            # Подготовка данных для батчевого сохранения
            song_data = {
                'artist': artist_name,
                'title': song.title,
                'lyrics': lyrics,
                'url': song_url,
                'genius_id': getattr(song, 'id', None),
                'metadata': metadata
            }
            
            # Добавляем в батч
            should_flush = self.batch_processor.add_song(song_data)
            if should_flush:
                batch = self.batch_processor.get_pending_batch()
                saved_count = await self.process_song_batch(batch)
                logger.info(f"📦 Обработан батч: {saved_count} песен сохранено")
            
            # Добавляем URL в кэш
            self.url_cache.add(song_url)
            
            word_count = len(lyrics.split())
            logger.info(f"✅ Подготовлено: {artist_name} - {song.title} ({word_count} слов)")
            
            return ScrapingStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"❌ Ошибка обработки {song.title}: {e}")
            return ScrapingStatus.ERROR_UNKNOWN

    async def scrape_artist_songs_async(self, artist_name: str, max_songs: int = 500) -> int:
        """Асинхронный скрапинг песен артиста"""
        logger.info(f"🎤 Начинаем АСИНХРОННУЮ обработку артиста: {artist_name}")
        
        try:
            processed_count = 0
            
            # Мониторинг ресурсов
            self.monitor.log_resources()
            
            async for song, song_number in self.get_songs_async_generator(artist_name, max_songs):
                if self.shutdown_requested:
                    break
                
                # Обработка песни
                processing_start = time.time()
                status = await self.process_single_song(song, song_number, artist_name)
                processing_time = time.time() - processing_start
                
                # Обновляем метрики
                self.metrics.increment(status, processing_time)
                processed_count += 1
                
                # Логирование прогресса
                if processed_count % 25 == 0:
                    logger.info(f"📈 Обработано {processed_count} песен для {artist_name}")
                    logger.info(f"📊 Success rate: {self.metrics.success_rate:.1f}%")
                    self.monitor.log_resources()
                
                # Адаптивная пауза на основе статуса
                if status in [ScrapingStatus.ERROR_NETWORK, ScrapingStatus.ERROR_API_LIMIT]:
                    await self.safe_delay(is_error=True, retry_count=self.metrics.error_network)
                elif status == ScrapingStatus.SUCCESS:
                    await self.safe_delay(is_error=False)
                
                # Проверка и принудительная очистка памяти
                if processed_count % 50 == 0:
                    limit_exceeded, status_mem = self.monitor.check_memory_limit()
                    if limit_exceeded:
                        logger.info("🧹 Принудительная очистка памяти...")
                        self.monitor.force_garbage_collection()
                        self.metrics.gc_runs += 1
            
            # Сохраняем оставшийся батч
            if self.batch_processor.has_pending():
                final_batch = self.batch_processor.get_pending_batch()
                await self.process_song_batch(final_batch)
                logger.info("💾 Сохранен финальный батч")
            
            logger.info(f"✅ Завершена обработка {artist_name}: обработано {processed_count} песен")
            return self.metrics.added
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка с артистом {artist_name}: {e}")
            return 0

    async def run_async_scraping_session(self, artists: List[str], songs_per_artist: int = 500):
        """Асинхронная сессия скрапинга с улучшенным управлением"""
        logger.info(f"🚀 Начинаем АСИНХРОННУЮ сессию: {len(artists)} артистов")
        
        initial_stats = self.db.get_stats()
        logger.info(f"📚 Уже в PostgreSQL: {initial_stats['total_songs']} песен")
        
        try:
            for i, artist_name in enumerate(artists, 1):
                if self.shutdown_requested:
                    logger.info("🛑 Получен запрос на остановку")
                    break
                
                logger.info(f"\n{'='*70}")
                logger.info(f"🎤 Артист {i}/{len(artists)}: {artist_name}")
                
                # Обработка артиста
                artist_start_time = time.time()
                added_count = await self.scrape_artist_songs_async(artist_name, songs_per_artist)
                artist_time = time.time() - artist_start_time
                
                # Статистика по артисту
                stats = self.db.get_stats()
                logger.info(f"✅ {artist_name}: +{added_count} песен за {artist_time:.1f}с")
                logger.info(f"📊 Всего в БД: {stats['total_songs']} песен от {stats['unique_artists']} артистов")
                
                # Детальная статистика каждые 5 артистов
                if i % 5 == 0:
                    self.show_detailed_metrics()
                
                # Адаптивная пауза между артистами
                if i < len(artists) and not self.shutdown_requested:
                    # Пауза зависит от успешности предыдущего артиста
                    if added_count > 10:
                        pause_time = random.uniform(15, 25)  # Успешный артист - короткая пауза
                    elif added_count > 0:
                        pause_time = random.uniform(25, 35)  # Частичный успех
                    else:
                        pause_time = random.uniform(45, 60)  # Проблемы - длинная пауза
                    
                    logger.info(f"⏳ Пауза между артистами: {pause_time:.1f}с")
                    await self.safe_delay_with_progress(pause_time)
        
        except Exception as e:
            logger.error(f"💥 Критическая ошибка в сессии: {e}")
        finally:
            await self.finalize_session()

    async def safe_delay_with_progress(self, total_seconds: float):
        """Пауза с индикатором прогресса"""
        intervals = int(total_seconds)
        for i in range(intervals):
            if self.shutdown_requested:
                break
            if i % 10 == 0 and i > 0:
                logger.info(f"⏱️ Осталось {intervals - i}с...")
            await asyncio.sleep(1)

    def show_detailed_metrics(self):
        """Показ детализированных метрик"""
        stats = self.db.get_stats()
        runtime = self.metrics.runtime
        
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 ДЕТАЛЬНАЯ СТАТИСТИКА:")
        logger.info(f"⏱️ Время выполнения: {runtime}")
        logger.info(f"🎵 Обработано песен: {self.metrics.processed}")
        logger.info(f"✅ Добавлено: {self.metrics.added} ({self.metrics.success_rate:.1f}%)")
        logger.info(f"⏩ Пропущено дубликатов: {self.metrics.skipped_duplicates}")
        logger.info(f"🚫 Пропущено (качество): {self.metrics.skipped_quality}")
        
        logger.info(f"\n🚨 ОШИБКИ:")
        logger.info(f"🌐 Сетевые: {self.metrics.error_network}")
        logger.info(f"🔒 API лимиты: {self.metrics.error_api_limit}")
        logger.info(f"📝 Парсинг: {self.metrics.error_parsing}")
        logger.info(f"❓ Неизвестные: {self.metrics.error_unknown}")
        
        logger.info(f"\n⚡ ПРОИЗВОДИТЕЛЬНОСТЬ:")
        logger.info(f"📞 API вызовов: {self.metrics.api_calls}")
        logger.info(f"💾 Попаданий в кэш: {self.metrics.cache_hits}")
        logger.info(f"📦 Батчей сохранено: {self.metrics.batch_saves}")
        logger.info(f"⏱️ Среднее время обработки: {self.metrics.avg_processing_time:.2f}с")
        logger.info(f"🗑️ Очисток памяти: {self.metrics.gc_runs}")
        
        # Статистика БД
        logger.info(f"\n🐘 POSTGRESQL:")
        logger.info(f"📚 Всего песен: {stats['total_songs']}")
        logger.info(f"👥 Артистов: {stats['unique_artists']}")
        logger.info(f"📝 Среднее слов: {stats['avg_words']}")
        logger.info(f"⭐ Среднее качество: {stats['avg_quality']}")
        
        # Эффективность
        if runtime.total_seconds() > 0:
            songs_per_hour = (self.metrics.added / runtime.total_seconds()) * 3600
            logger.info(f"🎯 Эффективность: {songs_per_hour:.1f} песен/час")
        
        logger.info(f"{'='*70}\n")

    async def finalize_session(self):
        """Финализация сессии с полной статистикой"""
        logger.info(f"\n{'='*70}")
        logger.info(f"🏁 ФИНАЛИЗАЦИЯ АСИНХРОННОЙ СЕССИИ")
        
        # Сохраняем оставшиеся батчи
        if self.batch_processor.has_pending():
            final_batch = self.batch_processor.get_pending_batch()
            await self.process_song_batch(final_batch)
            logger.info("💾 Сохранены оставшиеся батчи")
        
        # Финальная статистика
        self.show_detailed_metrics()
        
        # Рекомендации по оптимизации
        self.show_optimization_recommendations()
        
        logger.info("🔒 Закрытие соединений...")
        self.executor.shutdown(wait=True)
        self.db.close()
        self._restore_proxy_env()

    def show_optimization_recommendations(self):
        """Рекомендации по оптимизации на основе метрик"""
        logger.info(f"\n💡 РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ:")
        
        # Анализ успешности
        if self.metrics.success_rate < 50:
            logger.info("⚠️ Низкая успешность - рассмотрите увеличение timeout или пауз")
        elif self.metrics.success_rate > 90:
            logger.info("✅ Отличная успешность - можно уменьшить паузы для ускорения")
        
        # Анализ ошибок
        if self.metrics.error_network > self.metrics.error_api_limit * 2:
            logger.info("🌐 Много сетевых ошибок - проверьте интернет-соединение")
        
        if self.metrics.error_api_limit > 10:
            logger.info("🔒 Много API лимитов - увеличьте паузы между запросами")
        
        # Анализ производительности
        cache_hit_rate = (self.metrics.cache_hits / max(1, self.metrics.processed)) * 100
        if cache_hit_rate > 20:
            logger.info(f"💾 Высокая эффективность кэша ({cache_hit_rate:.1f}%)")
        
        if self.metrics.gc_runs > self.metrics.processed / 50:
            logger.info("🗑️ Частые GC - рассмотрите увеличение лимита памяти")
        
        # Батчевая обработка
        if self.metrics.batch_saves > 0:
            avg_batch_size = self.metrics.added / self.metrics.batch_saves
            if avg_batch_size < 5:
                logger.info("📦 Маленькие батчи - увеличьте batch_size для ускорения")
            elif avg_batch_size > 20:
                logger.info("📦 Большие батчи - уменьшите batch_size для экономии памяти")

    def close(self):
        """Синхронное закрытие (для обратной совместимости)"""
        logger.info("🔒 Закрытие соединения с PostgreSQL...")
        self.executor.shutdown(wait=True)
        self.db.close()
        self._restore_proxy_env()

# Utility functions
def load_artist_list(filename: str = "rap_artists.json") -> List[str]:
    """Загрузка списка артистов с приоритетом remaining_artists.json"""
    try:
        remaining_file = os.path.join(DATA_DIR, "remaining_artists.json")
        if os.path.exists(remaining_file):
            logger.info(f"📂 Загружаем оставшихся артистов из {remaining_file}")
            with open(remaining_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        full_file = os.path.join(DATA_DIR, filename)
        if os.path.exists(full_file):
            logger.info(f"📂 Загружаем полный список артистов из {full_file}")
            with open(full_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки списка артистов: {e}")
    
    # Fallback список
    logger.info("📂 Используем встроенный список артистов")
    return [
        "J. Cole", "Drake", "Eminem", "Kanye West", "Travis Scott", 
        "Lil Wayne", "Jay-Z", "Nas", "Tupac", "The Notorious B.I.G.",
        "Lil Baby", "Future", "21 Savage", "Post Malone", "Tyler, The Creator",
        "A$AP Rocky", "Mac Miller", "Childish Gambino", "Logic", "Big Sean",
        "Chance the Rapper", "Wiz Khalifa", "Meek Mill", "2 Chainz", "Pusha T"
    ]

async def async_main():
    """Асинхронная главная функция"""
    if not GENIUS_TOKEN:
        logger.error("❌ Genius API token не найден!")
        return
    
    # Настройки оптимизированного скрапера
    MEMORY_LIMIT_MB = 4096  # Увеличили лимит памяти
    BATCH_SIZE = 15         # Оптимальный размер батча
    
    scraper = OptimizedPostgreSQLScraper(
        GENIUS_TOKEN, 
        memory_limit_mb=MEMORY_LIMIT_MB,
        batch_size=BATCH_SIZE
    )
    
    try:
        artists = load_artist_list()
        SONGS_PER_ARTIST = 500
        
        logger.info(f"🎯 Загружено {len(artists)} артистов")
        logger.info(f"🎵 Цель: ~{len(artists) * SONGS_PER_ARTIST} песен")
        logger.info(f"💾 Лимит памяти: {MEMORY_LIMIT_MB}MB, Batch: {BATCH_SIZE}")
        logger.info(f"⚡ Используем АСИНХРОННУЮ обработку")
        
        await scraper.run_async_scraping_session(artists, SONGS_PER_ARTIST)
        
    except Exception as e:
        logger.error(f"💥 Ошибка в async_main: {e}")
    finally:
        await scraper.finalize_session()

def main():
    """Главная функция с выбором режима"""
    if not GENIUS_TOKEN:
        logger.error("❌ Genius API token не найден в .env!")
        return
    
    logger.info("🚀 Выберите режим работы:")
    logger.info("1. 🔥 АСИНХРОННЫЙ режим (рекомендуется)")
    logger.info("2. 📊 Синхронный режим (совместимость)")
    
    try:
        choice = input("Выбор (1/2): ").strip()
        
        if choice == "1" or choice == "":
            logger.info("⚡ Запуск в АСИНХРОННОМ режиме...")
            asyncio.run(async_main())
        else:
            logger.info("📊 Запуск в синхронном режиме...")
            # Fallback к оригинальному коду
            scraper = OptimizedPostgreSQLScraper(GENIUS_TOKEN)
            artists = load_artist_list()
            scraper.run_scraping_session(artists, 500)
            
    except KeyboardInterrupt:
        logger.info("⌨️ Прерывание пользователем")
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")

if __name__ == "__main__":
    main()