"""
🎯 PostgreSQL Database Adapter
Concurrent-safe database operations for rap lyrics analysis

НАЗНАЧЕНИЕ: Replace SQLite with PostgreSQL for concurrent access
РЕЗУЛЬТАТ: Concurrent-safe database operations with connection pooling
"""

import asyncio
import logging
import os
from typing import List, Dict, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from contextlib import asynccontextmanager
import json
from datetime import datetime, date

try:
    import asyncpg
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    asyncpg = None
    psycopg2 = None

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """PostgreSQL connection configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "rap_lyrics"
    username: str = "rap_user"
    password: str = "securepassword123"
    max_connections: int = 20
    min_connections: int = 5
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Create config from environment variables"""
        return cls(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=int(os.getenv('POSTGRES_PORT', '5432')),
            database=os.getenv('POSTGRES_DATABASE', 'rap_lyrics'),
            username=os.getenv('POSTGRES_USERNAME', 'rap_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'securepassword123'),
            max_connections=int(os.getenv('POSTGRES_MAX_CONNECTIONS', '20')),
            min_connections=int(os.getenv('POSTGRES_MIN_CONNECTIONS', '5'))
        )
    
    @property
    def sync_url(self) -> str:
        """Synchronous connection URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_url(self) -> str:
        """Asynchronous connection URL"""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

class PostgreSQLManager:
    """PostgreSQL database manager with concurrent access support"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        if not POSTGRES_AVAILABLE:
            raise ImportError("PostgreSQL dependencies not installed. Run: pip install psycopg2-binary asyncpg")
        
        self.config = config or DatabaseConfig.from_env()
        self.connection_pool: Optional[asyncpg.Pool] = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize connection pool"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.config.async_url,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=30
            )
            self._initialized = True
            logger.info(f"✅ PostgreSQL pool initialized: {self.config.max_connections} connections")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize PostgreSQL pool: {e}")
            return False
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get async connection from pool"""
        if not self._initialized:
            await self.initialize()
        
        if not self.connection_pool:
            raise RuntimeError("Connection pool not initialized")
        
        async with self.connection_pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logger.error(f"Database connection error: {e}")
                raise
    
    def get_sync_connection(self):
        """Get synchronous connection"""
        return psycopg2.connect(
            self.config.sync_url,
            cursor_factory=RealDictCursor
        )
    
    async def execute_schema(self, schema_path: str) -> bool:
        """Execute SQL schema file"""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            async with self.get_connection() as conn:
                await conn.execute(schema_sql)
                logger.info(f"✅ Schema executed successfully: {schema_path}")
                return True
        except Exception as e:
            logger.error(f"❌ Schema execution failed: {e}")
            return False
    
    async def insert_track(self, track_data: Dict[str, Any]) -> Optional[int]:
        """Insert single track with SQLite schema mapping"""
        query = """
            INSERT INTO tracks (
                title, artist, lyrics, url, genius_id, scraped_date,
                word_count, genre, release_date, album, language,
                explicit, song_art_url, popularity_score, lyrics_quality_score
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
            ) RETURNING id
        """
        
        try:
            async with self.get_connection() as conn:
                track_id = await conn.fetchval(
                    query,
                    track_data.get('title', ''),
                    track_data.get('artist', ''),
                    track_data.get('lyrics'),
                    track_data.get('url'),
                    track_data.get('genius_id'),
                    track_data.get('scraped_date'),
                    track_data.get('word_count'),
                    track_data.get('genre'),
                    self._parse_date(track_data.get('release_date')),
                    track_data.get('album'),
                    track_data.get('language'),
                    track_data.get('explicit'),
                    track_data.get('song_art_url'),
                    track_data.get('popularity_score'),
                    track_data.get('lyrics_quality_score')
                )
                return track_id
        except Exception as e:
            logger.error(f"Error inserting track: {e}")
            return None
    
    async def batch_insert_tracks(self, tracks: List[Dict[str, Any]], batch_size: int = 1000) -> List[int]:
        """Batch insert tracks with progress tracking"""
        result_ids = []
        
        for i in range(0, len(tracks), batch_size):
            batch = tracks[i:i + batch_size]
            batch_ids = []
            
            async with self.get_connection() as conn:
                async with conn.transaction():
                    for track in batch:
                        track_id = await self.insert_track(track)
                        if track_id:
                            batch_ids.append(track_id)
            
            result_ids.extend(batch_ids)
            logger.info(f"✅ Migrated batch {i//batch_size + 1}: {len(batch_ids)}/{len(batch)} tracks")
        
        return result_ids
    
    async def save_analysis_result(self, analysis: Dict[str, Any]) -> Optional[int]:
        """Save analysis result"""
        query = """
            INSERT INTO analysis_results (
                track_id, analyzer_type, sentiment, confidence,
                complexity_score, themes, analysis_data,
                processing_time_ms, model_version
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
        """
        
        try:
            async with self.get_connection() as conn:
                result_id = await conn.fetchval(
                    query,
                    analysis['track_id'],
                    analysis.get('analyzer_type', 'unknown'),
                    analysis.get('sentiment'),
                    analysis.get('confidence'),
                    analysis.get('complexity_score'),
                    json.dumps(analysis.get('themes')) if analysis.get('themes') else None,
                    json.dumps(analysis.get('analysis_data')) if analysis.get('analysis_data') else None,
                    analysis.get('processing_time_ms'),
                    analysis.get('model_version')
                )
                return result_id
        except Exception as e:
            logger.error(f"Error saving analysis result: {e}")
            return None
    
    def get_tracks_for_analysis(self, limit: int = 100, analyzer_type: str = None) -> List[Dict]:
        """Get tracks that need analysis (sync)"""
        query = """
            SELECT t.id, t.title, t.artist, t.lyrics
            FROM tracks t
            LEFT JOIN analysis_results ar ON t.id = ar.track_id 
                AND ar.analyzer_type = %s
            WHERE t.lyrics IS NOT NULL 
                AND ar.id IS NULL
            LIMIT %s
        """
        
        try:
            with self.get_sync_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (analyzer_type or 'default', limit))
                    return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting tracks for analysis: {e}")
            return []
    
    async def check_connection(self) -> bool:
        """Health check"""
        try:
            async with self.get_connection() as conn:
                await conn.fetchval("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False
    
    def get_table_stats(self) -> Dict[str, int]:
        """Get table statistics (sync)"""
        try:
            with self.get_sync_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM tracks")
                    tracks_count = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM analysis_results")
                    analysis_count = cursor.fetchone()[0]
                    
                    return {
                        'tracks': tracks_count,
                        'analysis_results': analysis_count
                    }
        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            return {'tracks': 0, 'analysis_results': 0}
    
    def _parse_date(self, date_str: Any) -> Optional[date]:
        """Parse date string to date object"""
        if not date_str:
            return None
        
        if isinstance(date_str, date):
            return date_str
        
        if isinstance(date_str, str):
            try:
                return datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                try:
                    return datetime.strptime(date_str, '%Y').date()
                except ValueError:
                    return None
        
        return None
    
    async def close(self):
        """Close connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("✅ PostgreSQL connection pool closed")

# Factory function
def create_postgres_manager(config: Optional[DatabaseConfig] = None) -> PostgreSQLManager:
    """Create PostgreSQL manager instance"""
    return PostgreSQLManager(config)