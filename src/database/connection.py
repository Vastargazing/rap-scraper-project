"""
🗄️ Database Connection Module with Config Loader Integration
Type-safe PostgreSQL connection management

Features:
- Integration with config_loader for type-safe settings
- Connection pooling with SQLAlchemy
- Session management for FastAPI
- Environment variable support

Author: Vastargazing
Version: 2.0.0
"""

from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from typing import Generator
import logging

from src.config.config_loader import get_config

logger = logging.getLogger(__name__)

# Base for SQLAlchemy models
Base = declarative_base()

# Global variables (initialized on first use)
_engine = None
_SessionLocal = None


def get_engine():
    """
    Get SQLAlchemy engine (singleton pattern)
    
    Returns:
        Engine: Configured SQLAlchemy engine with connection pooling
    """
    global _engine
    
    if _engine is None:
        config = get_config()
        
        logger.info(f"🔧 Creating database engine...")
        logger.info(f"   Host: {config.database.host}")
        logger.info(f"   Database: {config.database.database_name}")
        logger.info(f"   Pool size: {config.database.pool_size}")
        
        _engine = create_engine(
            config.database.connection_string,
            pool_size=config.database.pool_size,
            max_overflow=config.database.max_overflow,
            pool_pre_ping=config.database.pool_pre_ping,
            pool_recycle=config.database.pool_recycle,
            echo=config.database.echo,
            # Connection pool class
            poolclass=pool.QueuePool,
        )
        
        logger.info("✅ Database engine created successfully!")
    
    return _engine


def get_session_local():
    """
    Get SessionLocal class (singleton pattern)
    
    Returns:
        sessionmaker: Configured session factory
    """
    global _SessionLocal
    
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )
        logger.info("✅ Session factory created!")
    
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency for database sessions
    
    Usage:
        from src.database.connection import get_db
        
        @app.get("/songs")
        def get_songs(db: Session = Depends(get_db)):
            return db.query(Song).all()
    
    Yields:
        Session: Database session
    """
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def test_connection() -> bool:
    """
    Test database connection
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            logger.info("✅ Database connection test successful!")
            return True
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False


def get_pool_status() -> dict:
    """
    Get connection pool status
    
    Returns:
        dict: Pool statistics
    """
    engine = get_engine()
    pool = engine.pool
    
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total_connections": pool.size() + pool.overflow()
    }


# Backward compatibility with legacy code
class DatabaseConnection:
    """
    Legacy database connection wrapper
    Provides backward compatibility with old code
    
    Usage (legacy):
        from src.database.connection import DatabaseConnection
        db = DatabaseConnection()
        db.execute("SELECT * FROM songs")
    """
    
    def __init__(self):
        self.engine = get_engine()
        self.SessionLocal = get_session_local()
        self.session = None
    
    def get_session(self) -> Session:
        """Get a new session"""
        if self.session is None:
            self.session = self.SessionLocal()
        return self.session
    
    def close_session(self):
        """Close current session"""
        if self.session:
            self.session.close()
            self.session = None
    
    def execute(self, query: str, params: dict = None):
        """Execute raw SQL query"""
        with self.engine.connect() as conn:
            if params:
                return conn.execute(query, params)
            else:
                return conn.execute(query)
    
    def __enter__(self):
        """Context manager support"""
        self.session = self.get_session()
        return self.session
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.close_session()


if __name__ == "__main__":
    # Test database connection
    print("🧪 Testing database connection...")
    print("=" * 60)
    
    # Load config
    config = get_config()
    print(f"\n📊 Database Configuration:")
    print(f"   Type: {config.database.type}")
    print(f"   Host: {config.database.host}")
    print(f"   Database: {config.database.database_name}")
    print(f"   Pool Size: {config.database.pool_size}")
    print(f"   Max Overflow: {config.database.max_overflow}")
    
    # Test connection
    print(f"\n🔌 Testing connection...")
    if test_connection():
        print("✅ Connection successful!")
        
        # Show pool status
        pool_status = get_pool_status()
        print(f"\n📊 Pool Status:")
        print(f"   Size: {pool_status['size']}")
        print(f"   Checked In: {pool_status['checked_in']}")
        print(f"   Checked Out: {pool_status['checked_out']}")
        print(f"   Overflow: {pool_status['overflow']}")
        print(f"   Total: {pool_status['total_connections']}")
    else:
        print("❌ Connection failed!")
        exit(1)
