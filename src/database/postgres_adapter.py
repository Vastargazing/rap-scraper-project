"""
🎯 PostgreSQL Database Adapter with ML Support
Concurrent-safe database operations with vector embeddings for ML training

НАЗНАЧЕНИЕ: PostgreSQL + pgvector для хранения эмбеддингов и ML features
РЕЗУЛЬТАТ: Production-ready база для обучения генеративных моделей
"""

import asyncio
import logging
import os
from typing import List, Dict, Optional, Any, AsyncGenerator, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager
import json
from datetime import datetime, date
import hashlib
import numpy as np

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
    """PostgreSQL connection configuration with ML extensions"""
    host: str = "localhost"
    port: int = 5432
    database: str = "rap_lyrics"
    username: str = "rap_user"
    password: str = "securepassword123"
    max_connections: int = 20
    min_connections: int = 5
    enable_pgvector: bool = True  # NEW: Vector support flag
    
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
            min_connections=int(os.getenv('POSTGRES_MIN_CONNECTIONS', '5')),
            enable_pgvector=os.getenv('ENABLE_PGVECTOR', 'true').lower() == 'true'
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
    """PostgreSQL database manager with ML vector support"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        if not POSTGRES_AVAILABLE:
            raise ImportError("PostgreSQL dependencies not installed. Run: pip install psycopg2-binary asyncpg")
        
        self.config = config or DatabaseConfig.from_env()
        self.connection_pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        self._vector_enabled = False
    
    async def initialize(self) -> bool:
        """Initialize connection pool and ML extensions"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.config.async_url,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=30
            )
            self._initialized = True
            logger.info(f"✅ PostgreSQL pool initialized: {self.config.max_connections} connections")
            
            # Setup ML extensions if enabled
            if self.config.enable_pgvector:
                await self.setup_ml_extensions()
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize PostgreSQL pool: {e}")
            return False
    
    async def setup_ml_extensions(self) -> bool:
        """Setup pgvector and ML-related schema"""
        try:
            async with self.get_connection() as conn:
                # Enable pgvector extension
                await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
                logger.info("✅ pgvector extension enabled")
                
                # Add vector columns to tracks table
                await conn.execute('''
                    ALTER TABLE tracks 
                    ADD COLUMN IF NOT EXISTS lyrics_embedding vector(768),
                    ADD COLUMN IF NOT EXISTS flow_embedding vector(384),
                    ADD COLUMN IF NOT EXISTS embedding_model VARCHAR(100),
                    ADD COLUMN IF NOT EXISTS embedding_timestamp TIMESTAMP
                ''')
                
                # Create ML features table for normalized metrics
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS ml_features (
                        id SERIAL PRIMARY KEY,
                        track_id INTEGER REFERENCES tracks(id) UNIQUE,
                        rhyme_density REAL,
                        flow_complexity REAL,
                        emotion_joy REAL,
                        emotion_anger REAL,
                        emotion_fear REAL,
                        emotion_sadness REAL,
                        emotion_surprise REAL,
                        emotion_love REAL,
                        semantic_coherence REAL,
                        vocabulary_richness REAL,
                        metaphor_density REAL,
                        feature_version VARCHAR(20),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create index for vector similarity search
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS lyrics_embedding_idx 
                    ON tracks USING ivfflat (lyrics_embedding vector_cosine_ops)
                    WITH (lists = 100)
                ''')
                
                # Create dataset versions table for ML reproducibility
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS dataset_versions (
                        id SERIAL PRIMARY KEY,
                        version_tag VARCHAR(50) UNIQUE,
                        description TEXT,
                        track_count INTEGER,
                        feature_schema JSONB,
                        dataset_hash VARCHAR(64),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                ''')
                
                self._vector_enabled = True
                logger.info("✅ ML schema setup complete")
                return True
                
        except Exception as e:
            logger.warning(f"⚠️ ML extensions setup failed (non-critical): {e}")
            self._vector_enabled = False
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
    
    # NEW: Vector embedding methods
    async def store_embeddings_batch(self, embeddings_data: List[Tuple[int, List[float], str]]) -> int:
        """
        Store embeddings in batch
        Args:
            embeddings_data: List of (track_id, embedding, model_name) tuples
        Returns:
            Number of successfully stored embeddings
        """
        if not self._vector_enabled:
            logger.warning("Vector support not enabled")
            return 0
        
        stored = 0
        async with self.get_connection() as conn:
            for track_id, embedding, model_name in embeddings_data:
                try:
                    # Convert to PostgreSQL vector format
                    embedding_str = f'[{",".join(map(str, embedding))}]'
                    
                    await conn.execute('''
                        UPDATE tracks 
                        SET lyrics_embedding = $1::vector,
                            embedding_model = $2,
                            embedding_timestamp = CURRENT_TIMESTAMP
                        WHERE id = $3
                    ''', embedding_str, model_name, track_id)
                    stored += 1
                except Exception as e:
                    logger.error(f"Failed to store embedding for track {track_id}: {e}")
        
        logger.info(f"✅ Stored {stored}/{len(embeddings_data)} embeddings")
        return stored
    
    async def find_similar_tracks(self, track_id: int, limit: int = 10) -> List[Dict]:
        """Find similar tracks using vector similarity"""
        if not self._vector_enabled:
            return []
        
        async with self.get_connection() as conn:
            results = await conn.fetch('''
                SELECT 
                    t2.id, t2.title, t2.artist,
                    1 - (t1.lyrics_embedding <=> t2.lyrics_embedding) as similarity
                FROM tracks t1, tracks t2
                WHERE t1.id = $1 
                    AND t2.id != $1
                    AND t2.lyrics_embedding IS NOT NULL
                ORDER BY t1.lyrics_embedding <=> t2.lyrics_embedding
                LIMIT $2
            ''', track_id, limit)
            
            return [dict(r) for r in results]
    
    async def store_ml_features(self, features: Dict[str, Any]) -> bool:
        """Store normalized ML features for a track"""
        query = '''
            INSERT INTO ml_features (
                track_id, rhyme_density, flow_complexity,
                emotion_joy, emotion_anger, emotion_fear,
                emotion_sadness, emotion_surprise, emotion_love,
                semantic_coherence, vocabulary_richness, metaphor_density,
                feature_version
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (track_id) DO UPDATE SET
                rhyme_density = EXCLUDED.rhyme_density,
                flow_complexity = EXCLUDED.flow_complexity,
                emotion_joy = EXCLUDED.emotion_joy,
                emotion_anger = EXCLUDED.emotion_anger,
                emotion_fear = EXCLUDED.emotion_fear,
                emotion_sadness = EXCLUDED.emotion_sadness,
                emotion_surprise = EXCLUDED.emotion_surprise,
                emotion_love = EXCLUDED.emotion_love,
                semantic_coherence = EXCLUDED.semantic_coherence,
                vocabulary_richness = EXCLUDED.vocabulary_richness,
                metaphor_density = EXCLUDED.metaphor_density,
                feature_version = EXCLUDED.feature_version,
                updated_at = CURRENT_TIMESTAMP
        '''
        
        try:
            async with self.get_connection() as conn:
                await conn.execute(
                    query,
                    features['track_id'],
                    features.get('rhyme_density', 0.0),
                    features.get('flow_complexity', 0.0),
                    features.get('emotion_joy', 0.0),
                    features.get('emotion_anger', 0.0),
                    features.get('emotion_fear', 0.0),
                    features.get('emotion_sadness', 0.0),
                    features.get('emotion_surprise', 0.0),
                    features.get('emotion_love', 0.0),
                    features.get('semantic_coherence', 0.0),
                    features.get('vocabulary_richness', 0.0),
                    features.get('metaphor_density', 0.0),
                    features.get('feature_version', 'v1.0')
                )
                return True
        except Exception as e:
            logger.error(f"Failed to store ML features: {e}")
            return False
    
    async def create_dataset_version(self, version_tag: str, description: str = "") -> str:
        """Create a versioned snapshot of the dataset for ML reproducibility"""
        async with self.get_connection() as conn:
            # Get current dataset stats
            track_count = await conn.fetchval("SELECT COUNT(*) FROM tracks WHERE lyrics IS NOT NULL")
            
            # Generate dataset hash for reproducibility
            track_ids = await conn.fetch("SELECT id FROM tracks ORDER BY id")
            ids_str = ','.join(str(r['id']) for r in track_ids)
            dataset_hash = hashlib.sha256(ids_str.encode()).hexdigest()
            
            # Get feature schema
            feature_cols = await conn.fetch('''
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'ml_features'
            ''')
            feature_schema = {r['column_name']: r['data_type'] for r in feature_cols}
            
            # Store version
            await conn.execute('''
                INSERT INTO dataset_versions (
                    version_tag, description, track_count, 
                    feature_schema, dataset_hash
                ) VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (version_tag) DO NOTHING
            ''', version_tag, description, track_count, 
                json.dumps(feature_schema), dataset_hash)
            
            logger.info(f"✅ Created dataset version: {version_tag} (hash: {dataset_hash[:8]}...)")
            return dataset_hash
    
    async def get_unembedded_tracks(self, limit: int = 100) -> List[Dict]:
        """Get tracks without embeddings for processing"""
        async with self.get_connection() as conn:
            results = await conn.fetch('''
                SELECT id, title, artist, lyrics
                FROM tracks
                WHERE lyrics IS NOT NULL 
                    AND lyrics_embedding IS NULL
                LIMIT $1
            ''', limit)
            return [dict(r) for r in results]
    
    async def export_ml_dataset(self, version_tag: str = None) -> Dict[str, Any]:
        """Export dataset in ML-ready format"""
        async with self.get_connection() as conn:
            # Get all tracks with features
            query = '''
                SELECT 
                    t.id, t.title, t.artist, t.lyrics,
                    t.lyrics_embedding, t.flow_embedding,
                    f.rhyme_density, f.flow_complexity,
                    f.emotion_joy, f.emotion_anger, f.emotion_fear,
                    f.emotion_sadness, f.emotion_surprise, f.emotion_love,
                    f.semantic_coherence, f.vocabulary_richness
                FROM tracks t
                LEFT JOIN ml_features f ON t.id = f.track_id
                WHERE t.lyrics IS NOT NULL
            '''
            
            results = await conn.fetch(query)
            
            # Convert to ML-ready format
            dataset = {
                'version': version_tag or datetime.now().strftime('%Y%m%d_%H%M%S'),
                'tracks': [],
                'features': [],
                'embeddings': []
            }
            
            for row in results:
                track_data = {
                    'id': row['id'],
                    'text': row['lyrics'][:1024],  # Truncate for training
                    'artist': row['artist'],
                    'title': row['title']
                }
                dataset['tracks'].append(track_data)
                
                if row['rhyme_density'] is not None:
                    features = [
                        row['rhyme_density'], row['flow_complexity'],
                        row['emotion_joy'], row['emotion_anger'], 
                        row['emotion_fear'], row['emotion_sadness'],
                        row['emotion_surprise'], row['emotion_love'],
                        row['semantic_coherence'], row['vocabulary_richness']
                    ]
                    dataset['features'].append(features)
                
                if row['lyrics_embedding']:
                    dataset['embeddings'].append(row['lyrics_embedding'])
            
            logger.info(f"✅ Exported {len(dataset['tracks'])} tracks for ML training")
            return dataset
    
    # Keep existing methods...
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
    
    async def get_ml_statistics(self) -> Dict[str, Any]:
        """Get ML-related database statistics"""
        stats = {}
        async with self.get_connection() as conn:
            stats['total_tracks'] = await conn.fetchval(
                "SELECT COUNT(*) FROM tracks WHERE lyrics IS NOT NULL"
            )
            stats['tracks_with_embeddings'] = await conn.fetchval(
                "SELECT COUNT(*) FROM tracks WHERE lyrics_embedding IS NOT NULL"
            )
            stats['tracks_with_features'] = await conn.fetchval(
                "SELECT COUNT(*) FROM ml_features"
            )
            stats['dataset_versions'] = await conn.fetchval(
                "SELECT COUNT(*) FROM dataset_versions"
            )
            
            # Get embedding models used
            models = await conn.fetch(
                "SELECT DISTINCT embedding_model, COUNT(*) as count FROM tracks WHERE embedding_model IS NOT NULL GROUP BY embedding_model"
            )
            stats['embedding_models'] = {r['embedding_model']: r['count'] for r in models}
            
        return stats
    
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
    
    # Дополнительные методы для совместимости с emotion_analyzer
    async def get_track_count(self) -> int:
        """Получить общее количество треков"""
        try:
            async with self.connection_pool.acquire() as conn:
                result = await conn.fetchrow("SELECT COUNT(*) as count FROM tracks")
                return result['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting track count: {e}")
            return 0
    
    async def get_table_stats(self) -> dict:
        """Получить статистику таблиц"""
        try:
            stats = {}
            
            async with self.connection_pool.acquire() as conn:
                # Статистика треков
                tracks_result = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_tracks,
                        COUNT(CASE WHEN lyrics IS NOT NULL THEN 1 END) as tracks_with_lyrics,
                        COALESCE(AVG(LENGTH(lyrics)), 0) as avg_lyrics_length
                    FROM tracks
                """)
                if tracks_result:
                    stats['tracks'] = dict(tracks_result)
                
                # Статистика анализов
                analysis_result = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_analyses,
                        COUNT(DISTINCT track_id) as analyzed_tracks,
                        COUNT(DISTINCT analyzer_type) as analysis_types
                    FROM analysis_results
                """)
                if analysis_result:
                    stats['analyses'] = dict(analysis_result)
                
                # Статистика по типам анализа
                types_results = await conn.fetch("""
                    SELECT 
                        analyzer_type as analysis_type,
                        COUNT(*) as count,
                        MAX(created_at) as last_analysis
                    FROM analysis_results
                    GROUP BY analyzer_type
                    ORDER BY count DESC
                """)
                if types_results:
                    stats['analysis_types'] = [dict(row) for row in types_results]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting table stats: {e}")
            return {'error': str(e)}
    
    async def fetch_one(self, query: str, params: tuple = None) -> dict:
        """Выполнить запрос и вернуть одну запись"""
        try:
            async with self.connection_pool.acquire() as conn:
                if params:
                    result = await conn.fetchrow(query, *params)
                else:
                    result = await conn.fetchrow(query)
                return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error in fetch_one: {e}")
            return None
    
    async def fetch_all(self, query: str, params: tuple = None) -> list:
        """Выполнить запрос и вернуть все записи"""
        try:
            async with self.connection_pool.acquire() as conn:
                if params:
                    results = await conn.fetch(query, *params)
                else:
                    results = await conn.fetch(query)
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error in fetch_all: {e}")
            return []
    
    async def close(self):
        """Close connection pool"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("✅ PostgreSQL connection pool closed")

# Factory function
def create_postgres_manager(config: Optional[DatabaseConfig] = None) -> PostgreSQLManager:
    """Create PostgreSQL manager instance"""
    return PostgreSQLManager(config)