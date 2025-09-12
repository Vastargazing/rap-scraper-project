# 🚀 ПРОДВИНУТЫЕ УЛУЧШЕНИЯ ДЛЯ POSTGRESQL СКРАПЕРА
"""
Архитектурный подход:
1. Основной скрапер (ваш текущий rap_scraper_postgres.py)

Стабильная, проверенная версия
Работает без дополнительных зависимостей (Redis, Prometheus)
Быстро запускается и легко отлаживается
Подходит для повседневного использования

2. Продвинутая версия (новый файл ultra_rap_scraper_postgres.py)

Все улучшения: Redis, Prometheus, async pool
Требует дополнительную инфраструктуру
Подходит для production или больших объемов данных
Более сложная в настройке
"""

# Базовые импорты
import logging
import os
import sys
from typing import List

# Загрузка переменных окружения из .env файла
try:
    from dotenv import load_dotenv
    # Загружаем .env из корневой папки проекта
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
    load_dotenv(env_path)
    print(f"📁 Загружен .env файл: {env_path}")
except ImportError:
    print("⚠️ python-dotenv не установлен, загружаем переменные из системы")

# Добавляем путь для импорта базового скрапера
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from .rap_scraper_postgres import OptimizedPostgreSQLScraper, ScrapingStatus
except ImportError:
    # Fallback для прямого запуска
    from rap_scraper_postgres import OptimizedPostgreSQLScraper, ScrapingStatus

logger = logging.getLogger(__name__)

# Получаем токен из переменной окружения (пробуем разные названия)
GENIUS_TOKEN = (os.getenv('GENIUS_TOKEN') or 
                os.getenv('GENIUS_ACCESS_TOKEN') or 
                os.getenv('GENIUS_API_TOKEN') or '')

if GENIUS_TOKEN:
    print(f"✅ GENIUS_TOKEN найден: {GENIUS_TOKEN[:20]}...")
else:
    print("❌ GENIUS_TOKEN не найден в переменных окружения")

# 1. REDIS КЭШИРОВАНИЕ (добавить в requirements.txt: redis)
import redis
from typing import Optional
import pickle
import json

class RedisCache:
    """Redis кэш для артистов и метаданных"""
    
    def __init__(self, host='localhost', port=6379, db=0, ttl=3600):
        try:
            self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=False)
            self.redis.ping()  # Проверка соединения
            self.ttl = ttl
            self.enabled = True
            logger.info("✅ Redis кэш активирован")
        except Exception as e:
            logger.warning(f"⚠️ Redis недоступен, используем локальный кэш: {e}")
            self.enabled = False
            self.local_cache = {}
    
    def get_artist_songs(self, artist_name: str) -> Optional[List]:
        """Получение песен артиста из кэша"""
        if not self.enabled:
            return self.local_cache.get(f"artist:{artist_name}")
        
        try:
            key = f"artist_songs:{artist_name.lower()}"
            cached = self.redis.get(key)
            if cached:
                return pickle.loads(cached)
        except Exception as e:
            logger.debug(f"Redis get error: {e}")
        return None
    
    def cache_artist_songs(self, artist_name: str, songs: List, ttl_override=None):
        """Кэширование песен артиста"""
        ttl = ttl_override or self.ttl
        
        if not self.enabled:
            self.local_cache[f"artist:{artist_name}"] = songs
            return
        
        try:
            key = f"artist_songs:{artist_name.lower()}"
            self.redis.setex(key, ttl, pickle.dumps(songs))
            logger.debug(f"📋 Кэширован артист {artist_name}")
        except Exception as e:
            logger.debug(f"Redis cache error: {e}")
    
    def is_song_processed(self, song_hash: str) -> bool:
        """Проверка обработки песни по хэшу"""
        if not self.enabled:
            return song_hash in self.local_cache.get('processed_songs', set())
        
        try:
            return bool(self.redis.sismember('processed_songs', song_hash))
        except:
            return False
    
    def mark_song_processed(self, song_hash: str):
        """Отметка песни как обработанной"""
        if not self.enabled:
            if 'processed_songs' not in self.local_cache:
                self.local_cache['processed_songs'] = set()
            self.local_cache['processed_songs'].add(song_hash)
            return
        
        try:
            self.redis.sadd('processed_songs', song_hash)
            # Устанавливаем TTL для set (опционально)
            self.redis.expire('processed_songs', self.ttl * 24)  # 24 часа
        except Exception as e:
            logger.debug(f"Redis mark error: {e}")

# 2. PROMETHEUS МЕТРИКИ (добавить в requirements.txt: prometheus-client)
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import threading

class PrometheusMetrics:
    """Prometheus метрики для мониторинга"""
    
    def __init__(self, port=8090):
        # Счетчики
        self.songs_processed = Counter('scraper_songs_processed_total', 'Total processed songs')
        self.songs_added = Counter('scraper_songs_added_total', 'Total added songs')
        self.songs_skipped = Counter('scraper_songs_skipped_total', 'Total skipped songs', ['reason'])
        self.api_calls = Counter('scraper_api_calls_total', 'Total API calls', ['endpoint'])
        self.errors = Counter('scraper_errors_total', 'Total errors', ['type'])
        
        # Гистограммы времени
        self.processing_time = Histogram('scraper_processing_duration_seconds', 'Song processing time')
        self.batch_save_time = Histogram('scraper_batch_save_duration_seconds', 'Batch save time')
        self.api_response_time = Histogram('scraper_api_response_duration_seconds', 'API response time')
        
        # Gauge для текущих значений
        self.memory_usage = Gauge('scraper_memory_usage_mb', 'Current memory usage in MB')
        self.cpu_usage = Gauge('scraper_cpu_usage_percent', 'Current CPU usage')
        self.queue_size = Gauge('scraper_batch_queue_size', 'Current batch queue size')
        self.circuit_breaker_state = Gauge('scraper_circuit_breaker_open', 'Circuit breaker state (1=open, 0=closed)')
        
        # Запуск HTTP сервера в отдельном потоке
        try:
            start_http_server(port)
            logger.info(f"📊 Prometheus метрики доступны на порту {port}")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось запустить Prometheus сервер: {e}")
    
    def record_song_processed(self):
        self.songs_processed.inc()
    
    def record_song_added(self):
        self.songs_added.inc()
    
    def record_song_skipped(self, reason: str):
        self.songs_skipped.labels(reason=reason).inc()
    
    def record_api_call(self, endpoint: str, duration: float):
        self.api_calls.labels(endpoint=endpoint).inc()
        self.api_response_time.observe(duration)
    
    def record_error(self, error_type: str):
        self.errors.labels(type=error_type).inc()
    
    def update_memory_usage(self, mb: float):
        self.memory_usage.set(mb)
    
    def update_cpu_usage(self, percent: float):
        self.cpu_usage.set(percent)
    
    def update_queue_size(self, size: int):
        self.queue_size.set(size)
    
    def update_circuit_breaker_state(self, is_open: bool):
        self.circuit_breaker_state.set(1 if is_open else 0)

# 3. АСИНХРОННЫЙ POSTGRESQL CONNECTION POOL
import asyncpg
import asyncio
from contextlib import asynccontextmanager

class AsyncPostgreSQLManager:
    """Асинхронный менеджер PostgreSQL с connection pool"""
    
    def __init__(self, connection_string: str, min_size=5, max_size=20):
        self.connection_string = connection_string
        self.min_size = min_size
        self.max_size = max_size
        self.pool = None
        
    async def initialize(self):
        """Инициализация connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=self.min_size,
                max_size=self.max_size,
                command_timeout=60
            )
            logger.info(f"🏊 PostgreSQL pool создан: {self.min_size}-{self.max_size} соединений")
        except Exception as e:
            logger.error(f"❌ Ошибка создания PostgreSQL pool: {e}")
            raise
    
    @asynccontextmanager
    async def get_connection(self):
        """Асинхронный context manager для получения соединения"""
        if not self.pool:
            await self.initialize()
        
        async with self.pool.acquire() as conn:
            yield conn
    
    async def batch_add_songs_optimized(self, songs_batch: List[dict]) -> int:
        """Оптимизированное батчевое добавление песен"""
        if not songs_batch:
            return 0
        
        async with self.get_connection() as conn:
            # Подготавливаем данные для COPY
            rows = []
            for song in songs_batch:
                rows.append((
                    song['artist'],
                    song['title'],
                    song['lyrics'],
                    song['url'],
                    song.get('genius_id'),
                    json.dumps(song.get('metadata', {}))
                ))
            
            # Используем COPY для максимальной производительности
            try:
                result = await conn.copy_records_to_table(
                    'tracks',
                    records=rows,
                    columns=['artist', 'title', 'lyrics', 'url', 'genius_id', 'metadata_json'],
                    timeout=30
                )
                return len(rows)
            except Exception as e:
                logger.error(f"❌ Ошибка COPY: {e}")
                # Fallback к обычному INSERT
                return await self._fallback_batch_insert(conn, songs_batch)
    
    async def _fallback_batch_insert(self, conn, songs_batch: List[dict]) -> int:
        """Fallback метод с обычными INSERT"""
        saved_count = 0
        async with conn.transaction():
            for song in songs_batch:
                try:
                    await conn.execute("""
                        INSERT INTO tracks (artist, title, lyrics, url, genius_id, metadata_json)
                        VALUES ($1, $2, $3, $4, $5, $6)
                        ON CONFLICT (url) DO NOTHING
                    """, song['artist'], song['title'], song['lyrics'], 
                        song['url'], song.get('genius_id'), 
                        json.dumps(song.get('metadata', {})))
                    saved_count += 1
                except Exception as e:
                    logger.debug(f"Ошибка вставки песни {song['title']}: {e}")
        return saved_count
    
    async def song_exists_batch(self, urls: List[str]) -> set:
        """Батчевая проверка существования песен по URL"""
        if not urls:
            return set()
        
        async with self.get_connection() as conn:
            result = await conn.fetch(
                "SELECT url FROM tracks WHERE url = ANY($1)",
                urls
            )
            return {row['url'] for row in result}
    
    async def get_stats_async(self) -> dict:
        """Асинхронное получение статистики"""
        async with self.get_connection() as conn:
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_songs,
                    COUNT(DISTINCT artist) as unique_artists,
                    AVG(array_length(string_to_array(lyrics, ' '), 1)) as avg_words,
                    COUNT(CASE WHEN metadata_json IS NOT NULL THEN 1 END) as with_metadata
                FROM tracks
            """)
            return dict(stats) if stats else {}
    
    async def close(self):
        """Закрытие connection pool"""
        if self.pool:
            await self.pool.close()

# 4. INTELLIGENT RATE LIMITER
from collections import deque
import time

class IntelligentRateLimiter:
    """Умный rate limiter с адаптивными паузами"""
    
    def __init__(self, base_requests_per_minute=30):
        self.base_rpm = base_requests_per_minute
        self.current_rpm = base_requests_per_minute
        self.request_times = deque()
        self.recent_errors = deque()
        self.success_streak = 0
        self.last_adjustment = time.time()
        
    def can_make_request(self) -> bool:
        """Проверка возможности выполнения запроса"""
        now = time.time()
        
        # Очистка старых запросов (старше минуты)
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
        
        return len(self.request_times) < self.current_rpm
    
    def record_request(self, success: bool = True):
        """Запись выполненного запроса"""
        now = time.time()
        self.request_times.append(now)
        
        if success:
            self.success_streak += 1
            self.recent_errors = deque()  # Сброс ошибок при успехе
        else:
            self.recent_errors.append(now)
            self.success_streak = 0
        
        self._adjust_rate_limit()
    
    def _adjust_rate_limit(self):
        """Адаптивная корректировка лимита скорости"""
        now = time.time()
        
        # Корректируем не чаще раза в 30 секунд
        if now - self.last_adjustment < 30:
            return
        
        # Очистка старых ошибок
        while self.recent_errors and now - self.recent_errors[0] > 300:  # 5 минут
            self.recent_errors.popleft()
        
        error_rate = len(self.recent_errors) / 5  # ошибок в минуту за последние 5 минут
        
        if error_rate > 2:  # Много ошибок - замедляемся
            self.current_rpm = max(10, self.current_rpm * 0.7)
            logger.info(f"🐌 Rate limit снижен до {self.current_rpm:.0f} req/min")
        elif self.success_streak > 50 and error_rate < 0.5:  # Мало ошибок - ускоряемся
            self.current_rpm = min(self.base_rpm * 1.5, self.current_rpm * 1.2)
            logger.info(f"🚀 Rate limit увеличен до {self.current_rpm:.0f} req/min")
        
        self.last_adjustment = now
    
    async def wait_if_needed(self):
        """Асинхронная пауза при необходимости"""
        while not self.can_make_request():
            await asyncio.sleep(1)

# 5. SMART RETRY DECORATOR
from functools import wraps
import random

def smart_retry(max_retries=3, base_delay=1.0, backoff_factor=2.0, jitter=True):
    """Умный retry декоратор с jitter и категоризацией ошибок"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # Определяем тип ошибки
                    if any(keyword in error_msg for keyword in ['rate limit', '429', 'too many requests']):
                        delay = base_delay * (backoff_factor ** attempt) + random.uniform(10, 30)
                        logger.warning(f"🔒 Rate limit, пауза {delay:.1f}с")
                    elif any(keyword in error_msg for keyword in ['timeout', 'connection', 'network']):
                        delay = base_delay * (backoff_factor ** attempt)
                        if jitter:
                            delay += random.uniform(0, delay * 0.1)
                        logger.warning(f"🌐 Сетевая ошибка, пауза {delay:.1f}с")
                    elif '404' in error_msg or 'not found' in error_msg:
                        # Не ретраим 404 ошибки
                        logger.debug(f"📭 404 ошибка, пропускаем: {e}")
                        raise e
                    else:
                        delay = base_delay * (backoff_factor ** attempt)
                        logger.warning(f"❓ Неизвестная ошибка, пауза {delay:.1f}с: {e}")
                    
                    if attempt < max_retries:
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"❌ Исчерпаны попытки для {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator

# 6. ENHANCED BATCH PROCESSOR С ПРИОРИТЕТАМИ
from queue import PriorityQueue
import threading
from dataclasses import dataclass, field
from typing import Any

@dataclass
class PriorityTask:
    priority: int
    data: Any = field(compare=False)
    created_at: float = field(default_factory=time.time, compare=False)

class PriorityBatchProcessor:
    """Батчевый процессор с приоритетами и умной группировкой"""
    
    def __init__(self, batch_size=15, flush_interval=30.0, max_queue_size=1000):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_queue_size = max_queue_size
        
        self.priority_queue = PriorityQueue(maxsize=max_queue_size)
        self.last_flush = time.time()
        self.stats = {
            'high_priority': 0,
            'normal_priority': 0,
            'low_priority': 0,
            'batches_flushed': 0
        }
        
    def add_song(self, song_data: dict, priority: int = 5) -> bool:
        """Добавление песни с приоритетом (1=высокий, 10=низкий)"""
        try:
            task = PriorityTask(priority=priority, data=song_data)
            self.priority_queue.put_nowait(task)
            
            # Статистика по приоритетам
            if priority <= 3:
                self.stats['high_priority'] += 1
            elif priority <= 7:
                self.stats['normal_priority'] += 1
            else:
                self.stats['low_priority'] += 1
            
            return self._should_flush()
        except:
            logger.warning("⚠️ Очередь переполнена, принудительный flush")
            return True
    
    def _should_flush(self) -> bool:
        """Проверка необходимости сброса батча"""
        current_time = time.time()
        queue_size = self.priority_queue.qsize()
        
        return (queue_size >= self.batch_size or 
                current_time - self.last_flush > self.flush_interval or
                queue_size > self.max_queue_size * 0.8)
    
    def get_priority_batch(self) -> List[dict]:
        """Получение батча с учетом приоритетов"""
        batch = []
        
        # Собираем батч, начиная с высокого приоритета
        while len(batch) < self.batch_size and not self.priority_queue.empty():
            try:
                task = self.priority_queue.get_nowait()
                batch.append(task.data)
            except:
                break
        
        if batch:
            self.last_flush = time.time()
            self.stats['batches_flushed'] += 1
            
        return batch
    
    def get_queue_stats(self) -> dict:
        """Статистика очереди"""
        return {
            **self.stats,
            'current_queue_size': self.priority_queue.qsize(),
            'queue_utilization': f"{(self.priority_queue.qsize() / self.max_queue_size) * 100:.1f}%"
        }

# 7. ИНТЕГРАЦИЯ ВСЕХ УЛУЧШЕНИЙ В ОСНОВНОЙ КЛАСС
class UltraOptimizedScraper(OptimizedPostgreSQLScraper):
    """Максимально оптимизированный скрапер со всеми улучшениями"""
    
    def __init__(self, token: str, memory_limit_mb: int = 4096, batch_size: int = 15, 
                 redis_host='localhost', enable_prometheus=True):
        
        # Инициализация базового класса
        super().__init__(token, memory_limit_mb, batch_size)
        
        # Дополнительные компоненты
        self.redis_cache = RedisCache(host=redis_host)
        self.rate_limiter = IntelligentRateLimiter(base_requests_per_minute=45)
        self.priority_processor = PriorityBatchProcessor(batch_size=batch_size)
        
        # Замена обычного batch_processor на priority
        self.batch_processor = self.priority_processor
        
        # Prometheus метрики
        if enable_prometheus:
            try:
                self.prometheus = PrometheusMetrics()
            except Exception as e:
                logger.warning(f"⚠️ Prometheus недоступен: {e}")
                self.prometheus = None
        else:
            self.prometheus = None
        
        # Асинхронная БД
        try:
            from ..utils.config import get_db_config
            db_config = get_db_config()
            connection_string = (f"postgresql://{db_config['user']}:{db_config['password']}"
                               f"@{db_config['host']}:{db_config['port']}/{db_config['database']}")
            self.async_db = AsyncPostgreSQLManager(connection_string)
        except Exception as e:
            logger.warning(f"⚠️ Async DB недоступна: {e}")
            self.async_db = None
        
        logger.info("🚀 UltraOptimizedScraper инициализирован со всеми улучшениями")
    
    @smart_retry(max_retries=3, base_delay=2.0)
    async def enhanced_get_artist_songs(self, artist_name: str, max_songs: int = 500):
        """Улучшенный поиск песен с кэшированием и rate limiting"""
        
        # Проверяем Redis кэш
        cached_songs = self.redis_cache.get_artist_songs(artist_name)
        if cached_songs:
            logger.info(f"📋 Найдено в Redis кэше: {len(cached_songs)} песен для {artist_name}")
            self.metrics.cache_hits += len(cached_songs)
            for i, song in enumerate(cached_songs):
                yield song, i + 1
            return
        
        # Rate limiting
        await self.rate_limiter.wait_if_needed()
        
        # API запрос с метриками
        api_start = time.time()
        try:
            artist = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.genius.search_artist(
                    artist_name, 
                    max_songs=min(max_songs, 100),
                    sort="popularity"
                )
            )
            
            api_duration = time.time() - api_start
            self.rate_limiter.record_request(success=True)
            
            if self.prometheus:
                self.prometheus.record_api_call('search_artist', api_duration)
            
        except Exception as e:
            self.rate_limiter.record_request(success=False)
            if self.prometheus:
                self.prometheus.record_error('api_search')
            raise e
        
        if not artist or not artist.songs:
            return
        
        # Кэшируем результат в Redis
        self.redis_cache.cache_artist_songs(artist_name, artist.songs[:50])
        
        logger.info(f"📀 Найдено {len(artist.songs)} песен для {artist_name}")
        
        for i, song in enumerate(artist.songs):
            if self.shutdown_requested:
                break
            yield song, i + 1
    
    async def ultra_process_song(self, song, song_number: int, artist_name: str) -> ScrapingStatus:
        """Ультра-оптимизированная обработка песни"""
        
        if self.prometheus:
            self.prometheus.record_song_processed()
        
        # Генерируем хэш рано для проверки Redis
        song_hash = self.generate_song_hash(artist_name, song.title, 
                                           getattr(song, 'lyrics', '')[:200])
        
        # Проверка Redis кэша обработанных песен
        if self.redis_cache.is_song_processed(song_hash):
            if self.prometheus:
                self.prometheus.record_song_skipped('redis_cache')
            return ScrapingStatus.SKIPPED_DUPLICATE
        
        # Остальная логика обработки...
        status = await super().process_single_song(song, song_number, artist_name)
        
        # Отмечаем в Redis как обработанную
        if status == ScrapingStatus.SUCCESS:
            self.redis_cache.mark_song_processed(song_hash)
        
        # Prometheus метрики
        if self.prometheus:
            if status == ScrapingStatus.SUCCESS:
                self.prometheus.record_song_added()
            else:
                reason = status.value
                self.prometheus.record_song_skipped(reason)
        
        return status
    
    async def run_ultra_session(self, artists: List[str], songs_per_artist: int = 500):
        """Запуск ультра-оптимизированной сессии"""
        logger.info("🚀 Запуск ULTRA-ОПТИМИЗИРОВАННОЙ сессии скрапинга")
        
        # Инициализация async БД
        if self.async_db:
            await self.async_db.initialize()
        
        try:
            await self.run_async_scraping_session(artists, songs_per_artist)
        finally:
            if self.async_db:
                await self.async_db.close()
            
            # Финальная статистика
            if self.prometheus:
                logger.info("📊 Prometheus метрики обновлены")
            
            queue_stats = self.priority_processor.get_queue_stats()
            logger.info(f"📦 Статистика очереди: {queue_stats}")

# ПРИМЕР ИСПОЛЬЗОВАНИЯ УЛУЧШЕНИЙ

async def run_ultra_scraper():
    """Пример запуска ультра-оптимизированного скрапера"""
    
    # Проверяем наличие токена
    if not GENIUS_TOKEN:
        print("❌ GENIUS_TOKEN не установлен!")
        print("📝 Для работы ultra-скрапера нужен токен от Genius API")
        print("🔧 Установите: set GENIUS_TOKEN=your_token")
        print()
        print("🧪 Запускаем ДЕМО-ТЕСТ компонентов без токена...")
        
        # Тестируем компоненты без создания скрапера
        await test_ultra_components()
        return
    
    scraper = UltraOptimizedScraper(
        token=GENIUS_TOKEN,
        memory_limit_mb=6144,  # 6GB
        batch_size=20,         # Больший батч
        redis_host='localhost',
        enable_prometheus=True
    )
    
    artists = [
        "Kendrick Lamar", "J. Cole", "Drake", "Travis Scott",
        "Future", "21 Savage", "Tyler, The Creator"
    ]
    
    await scraper.run_ultra_session(artists, songs_per_artist=300)

async def test_ultra_components():
    """Тестирование компонентов ultra-скрапера без токена"""
    print("=" * 60)
    print("🧪 ТЕСТИРОВАНИЕ КОМПОНЕНТОВ ULTRA-СКРАПЕРА")
    print("=" * 60)
    
    # 1. Тест Redis Cache
    print("🔍 Тестируем Redis Cache...")
    try:
        cache = RedisCache()
        print(f"   ✅ Redis Cache инициализирован (enabled: {cache.enabled})")
        
        # Тест кэширования
        test_songs = [{'title': 'Test Song', 'artist': 'Test Artist'}]
        cache.cache_artist_songs('Test Artist', test_songs)
        cached = cache.get_artist_songs('Test Artist')
        print(f"   ✅ Кэширование работает: {len(cached) if cached else 0} песен")
    except Exception as e:
        print(f"   ❌ Redis Cache ошибка: {e}")
    
    # 2. Тест Rate Limiter
    print("🔍 Тестируем Rate Limiter...")
    try:
        limiter = IntelligentRateLimiter(base_requests_per_minute=60)
        can_request = limiter.can_make_request()
        print(f"   ✅ Rate Limiter: можно делать запрос = {can_request}")
        
        # Симуляция запросов
        for i in range(5):
            limiter.record_request(success=True)
        print(f"   ✅ Записано 5 успешных запросов, RPM = {limiter.current_rpm}")
    except Exception as e:
        print(f"   ❌ Rate Limiter ошибка: {e}")
    
    # 3. Тест Priority Processor
    print("🔍 Тестируем Priority Processor...")
    try:
        processor = PriorityBatchProcessor(batch_size=5)
        
        # Добавляем задачи с разными приоритетами
        test_tasks = [
            ({'artist': 'Eminem', 'song': 'Lose Yourself'}, 1),
            ({'artist': 'Unknown', 'song': 'Random Song'}, 8),
            ({'artist': 'Kendrick Lamar', 'song': 'HUMBLE'}, 2),
        ]
        
        for task_data, priority in test_tasks:
            processor.add_song(task_data, priority)
        
        batch = processor.get_priority_batch()
        print(f"   ✅ Priority Processor: получен батч из {len(batch)} задач")
        
        # Показываем порядок по приоритету
        for i, task in enumerate(batch, 1):
            print(f"      {i}. {task['artist']} - {task['song']}")
        
        stats = processor.get_queue_stats()
        print(f"   ✅ Статистика: {stats}")
        
    except Exception as e:
        print(f"   ❌ Priority Processor ошибка: {e}")
    
    # 4. Тест Prometheus Metrics
    print("🔍 Тестируем Prometheus Metrics...")
    try:
        metrics = PrometheusMetrics(port=8091)  # Другой порт на случай занятости
        print("   ✅ Prometheus Metrics инициализированы")
        
        # Тест метрик
        metrics.record_song_processed()
        metrics.record_song_added()
        metrics.record_song_skipped('test_reason')
        metrics.update_memory_usage(1024.5)
        metrics.update_cpu_usage(45.2)
        
        print("   ✅ Тестовые метрики записаны")
        print("   📊 Метрики доступны на http://localhost:8091")
        
    except Exception as e:
        print(f"   ❌ Prometheus Metrics ошибка: {e}")
    
    # 5. Тест Smart Retry
    print("🔍 Тестируем Smart Retry...")
    try:
        @smart_retry(max_retries=2, base_delay=0.1)
        async def test_function():
            import random
            if random.random() < 0.7:  # 70% вероятность ошибки
                raise Exception("Test error for retry")
            return "Success!"
        
        try:
            result = await test_function()
            print(f"   ✅ Smart Retry успешно: {result}")
        except:
            print("   ⚠️ Smart Retry: все попытки исчерпаны (это нормально для теста)")
            
    except Exception as e:
        print(f"   ❌ Smart Retry ошибка: {e}")
    
    print("=" * 60)
    print("🎉 ТЕСТИРОВАНИЕ КОМПОНЕНТОВ ЗАВЕРШЕНО")
    print("💡 Для полного запуска установите GENIUS_TOKEN")
    print("📋 Запустите Redis: docker run -d -p 6379:6379 redis")
    print("=" * 60)

# DOCKER COMPOSE ДОПОЛНЕНИЯ
"""
# Добавить в docker-compose.yml:

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
"""

# PROMETHEUS КОНФИГУРАЦИЯ
"""
# monitoring/prometheus.yml:
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rap-scraper'
    static_configs:
      - targets: ['localhost:8090']
    scrape_interval: 10s
"""

if __name__ == "__main__":
    import asyncio
    
    print("🚀 Продвинутые улучшения для PostgreSQL скрапера готовы!")
    print("📋 Добавлено в requirements.txt: redis prometheus-client")
    print("🐳 Используйте docker-compose для Redis и Prometheus")
    print("⚡ Запускаем ultra-скрапер...")
    print()
    
    # Запускаем ultra-скрапер
    asyncio.run(run_ultra_scraper())
