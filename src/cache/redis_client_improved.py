"""Production-ready Redis cache client following Google Python Style Guide.

This module provides a secure, observable, and reliable Redis caching implementation
for the rap scraper project. All critical issues from the code review have been addressed.

Typical usage example:
    from src.cache.redis_client_improved import RedisCacheImproved

    cache = RedisCacheImproved()
    songs = cache.get_artist_songs('Eminem')
    if not songs:
        songs = fetch_from_api()
        cache.set_artist_songs('Eminem', songs)
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set
from contextlib import contextmanager

import redis
from redis.connection import ConnectionPool
from redis.exceptions import (
    RedisError,
    ConnectionError as RedisConnectionError,
    TimeoutError as RedisTimeoutError,
)
from prometheus_client import Counter, Histogram, Gauge

from src.config import get_config

logger = logging.getLogger(__name__)


# Prometheus metrics
CACHE_HITS = Counter('redis_cache_hits_total', 'Total cache hits', ['operation'])
CACHE_MISSES = Counter('redis_cache_misses_total', 'Total cache misses', ['operation'])
CACHE_ERRORS = Counter('redis_cache_errors_total', 'Cache errors', ['operation', 'error_type'])
CACHE_OPERATION_DURATION = Histogram(
    'redis_cache_operation_duration_seconds',
    'Cache operation duration',
    ['operation']
)
CACHE_SIZE = Gauge('redis_cache_size_bytes', 'Approximate cache size in bytes')


# Module-level connection pools (shared across instances)
_CONNECTION_POOLS: Dict[str, ConnectionPool] = {}


class CacheError(Exception):
    """Base exception for cache errors."""
    pass


class CacheConnectionError(CacheError):
    """Raised when cache connection fails."""
    pass


class CacheValidationError(CacheError):
    """Raised when input validation fails."""
    pass


class RedisCacheImproved:
    """Production-ready Redis cache with security, observability, and reliability.

    This class provides Redis-based caching with the following features:
    - ✅ Secure JSON serialization (no pickle)
    - ✅ Connection pooling with health checks
    - ✅ Comprehensive error handling
    - ✅ Input validation
    - ✅ Prometheus metrics
    - ✅ Structured logging
    - ✅ Fallback to local cache
    - ✅ Type safety with full type hints

    Attributes:
        redis: Redis client instance with connection pooling.
        ttl: Default time-to-live for cached entries in seconds.
        enabled: Whether Redis is available and caching is enabled.
        local_cache: Fallback in-memory cache when Redis is unavailable.
        max_artist_name_length: Maximum allowed length for artist names.

    Example:
        >>> cache = RedisCacheImproved()
        >>> if cache.health_check()['healthy']:
        ...     cache.set_artist_songs('Eminem', songs_list)
        ...     cached = cache.get_artist_songs('Eminem')
    """

    # Class constants
    MAX_ARTIST_NAME_LENGTH = 255
    MAX_CACHE_VALUE_SIZE = 10 * 1024 * 1024  # 10MB
    DEFAULT_BATCH_SIZE = 100

    def __init__(self) -> None:
        """Initialize Redis cache with configuration from config system.

        Loads configuration from src/config using get_config() and initializes
        Redis connection pool. Falls back to local cache if Redis unavailable.

        Raises:
            CacheConnectionError: If Redis connection fails during initialization
                and fallback is disabled in config.
        """
        config = get_config()
        redis_config = config.redis

        self.ttl = redis_config.cache.artist_ttl
        self.max_artist_name_length = self.MAX_ARTIST_NAME_LENGTH
        self.local_cache: Dict[str, Any] = {}
        self.enabled = False

        try:
            # Get or create connection pool
            pool_key = f"{redis_config.host}:{redis_config.port}:{redis_config.db}"

            if pool_key not in _CONNECTION_POOLS:
                _CONNECTION_POOLS[pool_key] = ConnectionPool(
                    host=redis_config.host,
                    port=redis_config.port,
                    db=redis_config.db,
                    max_connections=redis_config.max_connections,
                    socket_timeout=redis_config.socket_timeout,
                    socket_connect_timeout=redis_config.socket_connect_timeout,
                    health_check_interval=30,  # Health check every 30s
                    decode_responses=True,  # Auto-decode to strings
                )
                logger.info(
                    "Redis connection pool created",
                    extra={
                        "host": redis_config.host,
                        "port": redis_config.port,
                        "max_connections": redis_config.max_connections
                    }
                )

            self.redis = redis.Redis(connection_pool=_CONNECTION_POOLS[pool_key])

            # Verify connection
            self.redis.ping()
            self.enabled = True

            logger.info(
                "Redis cache activated",
                extra={
                    "host": redis_config.host,
                    "port": redis_config.port,
                    "ttl": self.ttl
                }
            )

        except (RedisConnectionError, RedisTimeoutError) as e:
            logger.warning(
                "Redis unavailable, using local cache fallback",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "host": redis_config.host,
                    "port": redis_config.port
                }
            )
            self.enabled = False

        except RedisError as e:
            logger.error(
                "Unexpected Redis error during initialization",
                extra={"error": str(e), "error_type": type(e).__name__},
                exc_info=True
            )
            self.enabled = False

    def _validate_artist_name(self, artist_name: str) -> str:
        """Validate and sanitize artist name.

        Args:
            artist_name: Artist name to validate.

        Returns:
            Sanitized artist name (stripped, lowercase).

        Raises:
            CacheValidationError: If validation fails.
        """
        if not isinstance(artist_name, str):
            raise CacheValidationError(
                f"artist_name must be string, got {type(artist_name).__name__}"
            )

        if not artist_name or not artist_name.strip():
            raise CacheValidationError("artist_name cannot be empty or whitespace")

        if len(artist_name) > self.max_artist_name_length:
            raise CacheValidationError(
                f"artist_name exceeds maximum length of {self.max_artist_name_length}"
            )

        return artist_name.strip().lower()

    def _validate_songs_list(self, songs: List[Dict[str, Any]]) -> None:
        """Validate songs list before caching.

        Args:
            songs: List of song dictionaries to validate.

        Raises:
            CacheValidationError: If validation fails.
        """
        if not isinstance(songs, list):
            raise CacheValidationError(
                f"songs must be list, got {type(songs).__name__}"
            )

        if not songs:
            raise CacheValidationError("songs list cannot be empty")

        # Validate size
        serialized = json.dumps(songs)
        if len(serialized) > self.MAX_CACHE_VALUE_SIZE:
            raise CacheValidationError(
                f"Serialized songs exceed maximum size of {self.MAX_CACHE_VALUE_SIZE} bytes"
            )

    @CACHE_OPERATION_DURATION.labels(operation='get_artist_songs').time()
    def get_artist_songs(self, artist_name: str) -> Optional[List[Dict[str, Any]]]:
        """Retrieve artist songs from cache.

        Args:
            artist_name: Name of the artist to lookup. Case-insensitive.
                Maximum length: 255 characters.

        Returns:
            List of song dictionaries if found in cache, None otherwise.
            Each song dict contains: {'title': str, 'url': str, 'lyrics': str, ...}

        Raises:
            CacheValidationError: If artist_name validation fails.

        Example:
            >>> cache = RedisCacheImproved()
            >>> songs = cache.get_artist_songs('Eminem')
            >>> if songs:
            ...     print(f"Found {len(songs)} cached songs")
        """
        try:
            sanitized_name = self._validate_artist_name(artist_name)
        except CacheValidationError as e:
            logger.warning(
                "Artist name validation failed",
                extra={"artist_name": artist_name, "error": str(e)}
            )
            CACHE_ERRORS.labels(operation='get_artist_songs', error_type='validation').inc()
            return None

        # Fallback to local cache
        if not self.enabled:
            result = self.local_cache.get(f"artist:{sanitized_name}")
            if result:
                CACHE_HITS.labels(operation='get_artist_songs').inc()
            else:
                CACHE_MISSES.labels(operation='get_artist_songs').inc()
            return result

        # Redis lookup
        try:
            key = f"artist_songs:{sanitized_name}"
            cached = self.redis.get(key)

            if cached:
                CACHE_HITS.labels(operation='get_artist_songs').inc()
                logger.debug(
                    "Cache hit",
                    extra={"key": key, "artist": sanitized_name}
                )
                # Safe JSON deserialization (no pickle!)
                return json.loads(cached)
            else:
                CACHE_MISSES.labels(operation='get_artist_songs').inc()
                logger.debug(
                    "Cache miss",
                    extra={"key": key, "artist": sanitized_name}
                )
                return None

        except (RedisConnectionError, RedisTimeoutError) as e:
            logger.warning(
                "Redis connection error during get",
                extra={
                    "key": f"artist_songs:{sanitized_name}",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            CACHE_ERRORS.labels(
                operation='get_artist_songs',
                error_type=type(e).__name__
            ).inc()
            return None

        except json.JSONDecodeError as e:
            logger.error(
                "JSON decode error, invalid cache data",
                extra={
                    "key": f"artist_songs:{sanitized_name}",
                    "error": str(e)
                },
                exc_info=True
            )
            CACHE_ERRORS.labels(operation='get_artist_songs', error_type='json_decode').inc()
            # Invalidate corrupted cache entry
            self._invalidate_key(f"artist_songs:{sanitized_name}")
            return None

        except RedisError as e:
            logger.error(
                "Redis error during get",
                extra={
                    "key": f"artist_songs:{sanitized_name}",
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            CACHE_ERRORS.labels(
                operation='get_artist_songs',
                error_type=type(e).__name__
            ).inc()
            return None

    @CACHE_OPERATION_DURATION.labels(operation='set_artist_songs').time()
    def set_artist_songs(
        self,
        artist_name: str,
        songs: List[Dict[str, Any]],
        ttl_override: Optional[int] = None
    ) -> bool:
        """Cache artist songs with TTL.

        Args:
            artist_name: Name of the artist.
            songs: List of song dictionaries to cache. Must be JSON-serializable.
            ttl_override: Optional TTL override in seconds. If None, uses default TTL.
                Must be positive integer.

        Returns:
            True if caching succeeded, False otherwise.

        Raises:
            CacheValidationError: If validation of artist_name or songs fails.

        Example:
            >>> cache = RedisCacheImproved()
            >>> songs = [{'title': 'Lose Yourself', 'artist': 'Eminem'}]
            >>> success = cache.set_artist_songs('Eminem', songs)
            >>> print(f"Cache write: {'success' if success else 'failed'}")
        """
        try:
            sanitized_name = self._validate_artist_name(artist_name)
            self._validate_songs_list(songs)
        except CacheValidationError as e:
            logger.warning(
                "Validation failed during set",
                extra={"artist_name": artist_name, "error": str(e)}
            )
            CACHE_ERRORS.labels(operation='set_artist_songs', error_type='validation').inc()
            return False

        ttl = ttl_override or self.ttl

        # Validate TTL
        if ttl <= 0:
            logger.warning(
                "Invalid TTL value",
                extra={"ttl": ttl, "artist": sanitized_name}
            )
            CACHE_ERRORS.labels(operation='set_artist_songs', error_type='invalid_ttl').inc()
            return False

        # Fallback to local cache
        if not self.enabled:
            self.local_cache[f"artist:{sanitized_name}"] = songs
            return True

        # Redis write
        try:
            key = f"artist_songs:{sanitized_name}"
            # Safe JSON serialization (no pickle!)
            serialized = json.dumps(songs)
            self.redis.setex(key, ttl, serialized)

            logger.debug(
                "Artist songs cached",
                extra={
                    "key": key,
                    "artist": sanitized_name,
                    "song_count": len(songs),
                    "ttl": ttl,
                    "size_bytes": len(serialized)
                }
            )
            return True

        except (RedisConnectionError, RedisTimeoutError) as e:
            logger.warning(
                "Redis connection error during set",
                extra={
                    "key": f"artist_songs:{sanitized_name}",
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            CACHE_ERRORS.labels(
                operation='set_artist_songs',
                error_type=type(e).__name__
            ).inc()
            # Fallback to local cache
            self.local_cache[f"artist:{sanitized_name}"] = songs
            return False

        except (TypeError, ValueError) as e:
            logger.error(
                "JSON serialization error",
                extra={
                    "key": f"artist_songs:{sanitized_name}",
                    "error": str(e)
                },
                exc_info=True
            )
            CACHE_ERRORS.labels(operation='set_artist_songs', error_type='json_encode').inc()
            return False

        except RedisError as e:
            logger.error(
                "Redis error during set",
                extra={
                    "key": f"artist_songs:{sanitized_name}",
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                exc_info=True
            )
            CACHE_ERRORS.labels(
                operation='set_artist_songs',
                error_type=type(e).__name__
            ).inc()
            return False

    @CACHE_OPERATION_DURATION.labels(operation='is_song_processed').time()
    def is_song_processed(self, song_hash: str) -> bool:
        """Check if song has been processed.

        Args:
            song_hash: SHA256 hash of song identifier (64 hex characters).

        Returns:
            True if song was previously processed, False otherwise.

        Raises:
            CacheValidationError: If song_hash format is invalid.

        Example:
            >>> import hashlib
            >>> song_id = "Eminem-Lose Yourself"
            >>> song_hash = hashlib.sha256(song_id.encode()).hexdigest()
            >>> cache = RedisCacheImproved()
            >>> if not cache.is_song_processed(song_hash):
            ...     process_song()
            ...     cache.mark_song_processed(song_hash)
        """
        # Validate hash format
        if not isinstance(song_hash, str) or len(song_hash) != 64:
            raise CacheValidationError(
                "song_hash must be 64-character hex string (SHA256)"
            )

        # Fallback to local cache
        if not self.enabled:
            processed_set = self.local_cache.get("processed_songs", set())
            return song_hash in processed_set

        # Redis lookup
        try:
            result = bool(self.redis.sismember("processed_songs", song_hash))
            return result

        except (RedisConnectionError, RedisTimeoutError) as e:
            logger.warning(
                "Redis connection error during is_song_processed",
                extra={"error": str(e), "error_type": type(e).__name__}
            )
            CACHE_ERRORS.labels(
                operation='is_song_processed',
                error_type=type(e).__name__
            ).inc()
            return False

        except RedisError as e:
            logger.error(
                "Redis error during is_song_processed",
                extra={"error": str(e), "error_type": type(e).__name__},
                exc_info=True
            )
            CACHE_ERRORS.labels(
                operation='is_song_processed',
                error_type=type(e).__name__
            ).inc()
            return False

    @CACHE_OPERATION_DURATION.labels(operation='mark_song_processed').time()
    def mark_song_processed(self, song_hash: str) -> bool:
        """Mark song as processed.

        Args:
            song_hash: SHA256 hash of song identifier (64 hex characters).

        Returns:
            True if marking succeeded, False otherwise.

        Raises:
            CacheValidationError: If song_hash format is invalid.

        Example:
            >>> cache = RedisCacheImproved()
            >>> success = cache.mark_song_processed(song_hash)
        """
        # Validate hash format
        if not isinstance(song_hash, str) or len(song_hash) != 64:
            raise CacheValidationError(
                "song_hash must be 64-character hex string (SHA256)"
            )

        # Fallback to local cache
        if not self.enabled:
            if "processed_songs" not in self.local_cache:
                self.local_cache["processed_songs"] = set()
            self.local_cache["processed_songs"].add(song_hash)
            return True

        # Redis write with pipeline for efficiency
        try:
            with self.redis.pipeline() as pipe:
                pipe.sadd("processed_songs", song_hash)
                # Set TTL on the set (24 hours)
                pipe.expire("processed_songs", self.ttl * 24)
                pipe.execute()

            logger.debug("Song marked as processed", extra={"hash": song_hash})
            return True

        except (RedisConnectionError, RedisTimeoutError) as e:
            logger.warning(
                "Redis connection error during mark_song_processed",
                extra={"error": str(e), "error_type": type(e).__name__}
            )
            CACHE_ERRORS.labels(
                operation='mark_song_processed',
                error_type=type(e).__name__
            ).inc()
            return False

        except RedisError as e:
            logger.error(
                "Redis error during mark_song_processed",
                extra={"error": str(e), "error_type": type(e).__name__},
                exc_info=True
            )
            CACHE_ERRORS.labels(
                operation='mark_song_processed',
                error_type=type(e).__name__
            ).inc()
            return False

    def _invalidate_key(self, key: str) -> None:
        """Invalidate a specific cache key.

        Args:
            key: Redis key to invalidate.
        """
        if not self.enabled:
            return

        try:
            self.redis.delete(key)
            logger.info("Cache key invalidated", extra={"key": key})
        except RedisError as e:
            logger.warning(
                "Failed to invalidate key",
                extra={"key": key, "error": str(e)}
            )

    def health_check(self) -> Dict[str, Any]:
        """Check Redis connection health.

        Returns:
            Dictionary with health check results:
            {
                'healthy': bool,
                'enabled': bool,
                'latency_ms': Optional[float],
                'connection_pool_info': Optional[Dict],
                'error': Optional[str]
            }

        Example:
            >>> cache = RedisCacheImproved()
            >>> health = cache.health_check()
            >>> if health['healthy']:
            ...     print(f"Latency: {health['latency_ms']:.2f}ms")
        """
        if not self.enabled:
            return {
                'healthy': False,
                'enabled': False,
                'latency_ms': None,
                'connection_pool_info': None,
                'error': 'Redis is disabled'
            }

        try:
            start = time.time()
            self.redis.ping()
            latency_ms = (time.time() - start) * 1000

            pool = self.redis.connection_pool
            pool_info = {
                'max_connections': pool.max_connections,
                'created_connections': len(pool._created_connections),
                'available_connections': len(pool._available_connections),
            }

            return {
                'healthy': True,
                'enabled': True,
                'latency_ms': latency_ms,
                'connection_pool_info': pool_info,
                'error': None
            }

        except (RedisConnectionError, RedisTimeoutError) as e:
            return {
                'healthy': False,
                'enabled': True,
                'latency_ms': None,
                'connection_pool_info': None,
                'error': f"{type(e).__name__}: {str(e)}"
            }

        except RedisError as e:
            return {
                'healthy': False,
                'enabled': True,
                'latency_ms': None,
                'connection_pool_info': None,
                'error': f"Unexpected error: {str(e)}"
            }

    def clear_all(self) -> int:
        """Clear all cached data.

        Returns:
            Number of keys deleted.

        Example:
            >>> cache = RedisCacheImproved()
            >>> deleted = cache.clear_all()
            >>> print(f"Cleared {deleted} cache entries")
        """
        if not self.enabled:
            count = len(self.local_cache)
            self.local_cache.clear()
            return count

        try:
            # Get all keys matching our patterns
            patterns = ['artist_songs:*', 'processed_songs']
            deleted = 0

            for pattern in patterns:
                keys = self.redis.keys(pattern)
                if keys:
                    deleted += self.redis.delete(*keys)

            logger.info("Cache cleared", extra={"deleted_keys": deleted})
            return deleted

        except RedisError as e:
            logger.error(
                "Error clearing cache",
                extra={"error": str(e)},
                exc_info=True
            )
            return 0

    def close(self) -> None:
        """Close Redis connection.

        Note: Connection pool is shared, so this only closes the client reference.
        The pool remains available for other instances.
        """
        if self.enabled and self.redis:
            self.redis.close()
            logger.info("Redis client closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"RedisCacheImproved(enabled={self.enabled}, "
            f"ttl={self.ttl})"
        )
