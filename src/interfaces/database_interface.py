"""
Database interface and management module.

Provides unified database access layer with support for SQLite operations,
connection pooling, and transaction management.

Унифицированный доступ к базе данных:

DatabaseInterface - абстрактный интерфейс
SQLiteManager - полная реализация для SQLite
Автоматическое создание таблиц и индексов
Thread-safe соединения и транзакции
Удобные методы для работы с песнями и анализом
"""

import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Tuple, Generator
from dataclasses import dataclass
import logging
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_path: str
    timeout: float = 30.0
    check_same_thread: bool = False
    isolation_level: Optional[str] = None
    pool_size: int = 5


class DatabaseInterface(ABC):
    """Abstract interface for database operations"""
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query and return results"""
        pass
    
    @abstractmethod
    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """Execute INSERT/UPDATE/DELETE and return affected rows"""
        pass
    
    @abstractmethod
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Execute query with multiple parameter sets"""
        pass
    
    @abstractmethod
    def begin_transaction(self) -> None:
        """Begin transaction"""
        pass
    
    @abstractmethod
    def commit_transaction(self) -> None:
        """Commit transaction"""
        pass
    
    @abstractmethod
    def rollback_transaction(self) -> None:
        """Rollback transaction"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close database connection"""
        pass


class SQLiteManager(DatabaseInterface):
    """
    SQLite database manager with connection pooling and thread safety.
    
    Provides unified access to SQLite operations with proper error handling
    and connection management.
    """
    
    def __init__(self, config: DatabaseConfig):
        """Initialize database manager with configuration"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._local = threading.local()
        
        # Ensure database file exists and has proper structure
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database file and create tables if needed"""
        db_path = Path(self.config.db_path)
        
        # Create directory if it doesn't exist
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if database exists and has tables
        if not db_path.exists():
            self.logger.info(f"Creating new database: {db_path}")
            self._create_tables()
        else:
            # Verify tables exist
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='songs'
                """)
                if not cursor.fetchone():
                    self.logger.info("Database exists but missing tables, creating...")
                    self._create_tables()
    
    def _create_tables(self) -> None:
        """Create database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Songs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS songs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    artist TEXT NOT NULL,
                    title TEXT NOT NULL,
                    lyrics TEXT,
                    genius_id INTEGER,
                    url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(artist, title)
                )
            """)
            
            # Enhanced songs table for analysis results
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enhanced_songs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_id INTEGER NOT NULL,
                    analyzer_type TEXT NOT NULL,
                    analysis_data TEXT,  -- JSON data
                    confidence REAL,
                    processing_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (song_id) REFERENCES songs (id),
                    UNIQUE(song_id, analyzer_type)
                )
            """)
            
            # Artists table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS artists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    genius_id INTEGER,
                    url TEXT,
                    metadata TEXT,  -- JSON data
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Analysis queue table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    song_id INTEGER NOT NULL,
                    analyzer_type TEXT NOT NULL,
                    priority INTEGER DEFAULT 5,
                    status TEXT DEFAULT 'pending',
                    attempts INTEGER DEFAULT 0,
                    last_attempt TIMESTAMP,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (song_id) REFERENCES songs (id)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_artist ON songs(artist)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_songs_title ON songs(title)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_songs_song_id ON enhanced_songs(song_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_enhanced_songs_analyzer ON enhanced_songs(analyzer_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON analysis_queue(status)")
            
            conn.commit()
            self.logger.info("Database tables created successfully")
    
    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                self.config.db_path,
                timeout=self.config.timeout,
                check_same_thread=self.config.check_same_thread,
                isolation_level=self.config.isolation_level
            )
            
            # Configure connection
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            
        return self._local.connection
    
    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database transactions"""
        conn = self.get_connection()
        try:
            conn.execute("BEGIN")
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Transaction rolled back: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """Execute SELECT query and return results as list of dicts"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Convert rows to dictionaries
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
            
        except sqlite3.Error as e:
            self.logger.error(f"Query execution failed: {e}")
            raise
    
    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """Execute INSERT/UPDATE/DELETE and return affected rows"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            conn.commit()
            return cursor.rowcount
            
        except sqlite3.Error as e:
            self.logger.error(f"Update execution failed: {e}")
            raise
    
    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        """Execute query with multiple parameter sets"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.executemany(query, params_list)
            conn.commit()
            return cursor.rowcount
            
        except sqlite3.Error as e:
            self.logger.error(f"Batch execution failed: {e}")
            raise
    
    def begin_transaction(self) -> None:
        """Begin transaction"""
        conn = self.get_connection()
        conn.execute("BEGIN")
    
    def commit_transaction(self) -> None:
        """Commit transaction"""
        conn = self.get_connection()
        conn.commit()
    
    def rollback_transaction(self) -> None:
        """Rollback transaction"""
        conn = self.get_connection()
        conn.rollback()
    
    def close(self) -> None:
        """Close database connection"""
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
    
    # Convenience methods for common operations
    
    def get_song(self, artist: str, title: str) -> Optional[Dict[str, Any]]:
        """Get song by artist and title"""
        results = self.execute_query(
            "SELECT * FROM songs WHERE artist = ? AND title = ?",
            (artist, title)
        )
        return results[0] if results else None
    
    def get_song_by_id(self, song_id: int) -> Optional[Dict[str, Any]]:
        """Get song by ID"""
        results = self.execute_query(
            "SELECT * FROM songs WHERE id = ?",
            (song_id,)
        )
        return results[0] if results else None
    
    def insert_song(self, artist: str, title: str, lyrics: str, 
                   genius_id: Optional[int] = None, url: Optional[str] = None) -> int:
        """Insert new song and return ID"""
        try:
            cursor = self.get_connection().cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO songs (artist, title, lyrics, genius_id, url)
                VALUES (?, ?, ?, ?, ?)
            """, (artist, title, lyrics, genius_id, url))
            
            if cursor.rowcount == 0:
                # Song already exists, get existing ID
                song = self.get_song(artist, title)
                return song['id'] if song else None
            
            self.get_connection().commit()
            return cursor.lastrowid
            
        except sqlite3.Error as e:
            self.logger.error(f"Failed to insert song: {e}")
            raise
    
    def update_song_lyrics(self, song_id: int, lyrics: str) -> bool:
        """Update lyrics for existing song"""
        affected = self.execute_update(
            "UPDATE songs SET lyrics = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (lyrics, song_id)
        )
        return affected > 0
    
    def get_songs_without_analysis(self, analyzer_type: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get songs that haven't been analyzed by specific analyzer"""
        query = """
            SELECT s.* FROM songs s
            LEFT JOIN enhanced_songs es ON s.id = es.song_id AND es.analyzer_type = ?
            WHERE es.id IS NULL AND s.lyrics IS NOT NULL AND s.lyrics != ''
        """
        
        params = [analyzer_type]
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        return self.execute_query(query, tuple(params))
    
    def save_analysis_result(self, song_id: int, analyzer_type: str, 
                           analysis_data: str, confidence: float, 
                           processing_time: float) -> bool:
        """Save analysis result"""
        try:
            affected = self.execute_update("""
                INSERT OR REPLACE INTO enhanced_songs 
                (song_id, analyzer_type, analysis_data, confidence, processing_time)
                VALUES (?, ?, ?, ?, ?)
            """, (song_id, analyzer_type, analysis_data, confidence, processing_time))
            
            return affected > 0
            
        except sqlite3.Error as e:
            self.logger.error(f"Failed to save analysis result: {e}")
            raise
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        stats = {}
        
        # Total songs
        result = self.execute_query("SELECT COUNT(*) as count FROM songs")
        stats['total_songs'] = result[0]['count']
        
        # Songs with lyrics
        result = self.execute_query(
            "SELECT COUNT(*) as count FROM songs WHERE lyrics IS NOT NULL AND lyrics != ''"
        )
        stats['songs_with_lyrics'] = result[0]['count']
        
        # Analysis coverage by type
        result = self.execute_query("""
            SELECT analyzer_type, COUNT(*) as count 
            FROM enhanced_songs 
            GROUP BY analyzer_type
        """)
        stats['analysis_coverage'] = {row['analyzer_type']: row['count'] for row in result}
        
        return stats


# Factory function for easy database creation
def create_database_manager(db_path: str, **kwargs) -> SQLiteManager:
    """
    Create database manager with default configuration.
    
    Args:
        db_path: Path to SQLite database file
        **kwargs: Additional configuration options
        
    Returns:
        Configured SQLiteManager instance
    """
    config = DatabaseConfig(
        db_path=db_path,
        **kwargs
    )
    
    return SQLiteManager(config)
