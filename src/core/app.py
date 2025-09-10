"""
Application factory and core initialization module.

Provides centralized application setup, dependency injection,
and component initialization.

Фабрика приложения и lifecycle management:

Application - главный класс приложения
Dependency injection для всех компонентов
AppContext - context manager для lifecycle
Централизованная инициализация и shutdown
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Добавляем корневую директорию в sys.path если необходимо
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Core imports
from src.core.config import AppConfig, get_config, load_config
from src.database.postgres_adapter import PostgreSQLManager, DatabaseConfig
from src.interfaces.analyzer_interface import AnalyzerFactory


class ApplicationError(Exception):
    """Base application exception"""
    pass


class ConfigurationError(ApplicationError):
    """Configuration-related errors"""
    pass


class DatabaseError(ApplicationError):
    """Database-related errors"""
    pass


class Application:
    """
    Main application class that orchestrates all components.
    
    Handles initialization, configuration, and provides access
    to core services like database, analyzers, etc.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize application with configuration"""
        self.config = config or get_config()
        self.logger = None
        self.database = None
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all application components"""
        if self._initialized:
            return
        
        try:
            # Setup logging first
            self._setup_logging()
            self.logger.info(f"Initializing {self.config.project_name} v{self.config.version}")
            
            # Initialize database
            self._setup_database()
            
            # Initialize analyzers
            self._setup_analyzers()
            
            # Validate configuration
            self._validate_configuration()
            
            self._initialized = True
            self.logger.info("Application initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize application: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"ERROR: {error_msg}", file=sys.stderr)
            raise ApplicationError(error_msg) from e
    
    def _setup_logging(self) -> None:
        """Setup application logging"""
        log_config = self.config.logging
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_config.level.upper()),
            format=log_config.format,
            force=True,
            encoding=getattr(log_config, 'encoding', 'utf-8')
        )
        
        # Setup file logging if configured
        if log_config.file_path:
            log_path = Path(log_config.file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=log_config.max_file_size,
                backupCount=log_config.backup_count,
                encoding=getattr(log_config, 'encoding', 'utf-8')
            )
            file_handler.setFormatter(logging.Formatter(log_config.format))
            
            root_logger = logging.getLogger()
            root_logger.addHandler(file_handler)
        
        self.logger = logging.getLogger(__name__)
        
        # Log configuration info (без эмодзи для Windows)
        self.logger.info(f"Logging configured: level={log_config.level}")
        if log_config.file_path:
            self.logger.info(f"Log file: {log_config.file_path}")
    
    def _setup_database(self) -> None:
        """Initialize database connection"""
        try:
            # PostgreSQL configuration from environment
            self.database = PostgreSQLManager()
            
            # PostgreSQL будет инициализирован при первом использовании
            # Тестируем, что конфигурация корректная
            try:
                config = self.database.config
                logging.info(f"✅ PostgreSQL configured: {config.host}:{config.port}/{config.database}")
            except Exception as e:
                logging.error(f"PostgreSQL configuration failed: {e}")
                raise
            
        except Exception as e:
            raise DatabaseError(f"Failed to setup database: {e}") from e
    
    def _setup_analyzers(self) -> None:
        """Setup analyzer factory and register available analyzers"""
        try:
            # Анализаторы уже импортированы в начале файла
            available_analyzers = AnalyzerFactory.list_available()
            self.logger.info(f"Analyzers available: {available_analyzers}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup analyzers: {e}")
            # Don't fail initialization for analyzer setup issues
    
    def _validate_configuration(self) -> None:
        """Validate application configuration"""
        errors = []
        
        # Check required directories
        required_dirs = [
            self.config.data_dir,
            self.config.results_dir,
            self.config.logs_dir
        ]
        
        for directory in required_dirs:
            path = Path(directory)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created directory: {directory}")
                except Exception as e:
                    errors.append(f"Cannot create directory {directory}: {e}")
        
        # Check database path
        db_path = Path(self.config.database.path)
        db_dir = db_path.parent
        if not db_dir.exists():
            try:
                db_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create database directory {db_dir}: {e}")
        
        # Check API keys if needed
        if not self.config.api.genius_token:
            self.logger.warning("Genius API token not configured - scraping may be limited")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {err}" for err in errors)
            raise ConfigurationError(error_msg)
    
    def get_database(self) -> PostgreSQLManager:
        """Get database manager instance"""
        if not self._initialized:
            self.initialize()
        
        if not self.database:
            raise DatabaseError("Database not initialized")
        
        return self.database
    
    def get_analyzer(self, name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Get analyzer instance"""
        if not self._initialized:
            self.initialize()
        
        try:
            return AnalyzerFactory.create(name, config)
        except Exception as e:
            raise ApplicationError(f"Failed to create analyzer '{name}': {e}") from e
    
    def list_analyzers(self) -> list:
        """List available analyzers"""
        return AnalyzerFactory.list_available()
    
    def shutdown(self) -> None:
        """Shutdown application and cleanup resources"""
        if self.logger:
            self.logger.info("Shutting down application")
        
        # Close database connections
        if self.database:
            try:
                self.database.close()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error closing database: {e}")
        
        self._initialized = False
        
        if self.logger:
            self.logger.info("Application shutdown complete")


# Global application instance
_app_instance: Optional[Application] = None


def init_analyzers():
    """Инициализируем анализаторы"""
    try:
        # Регистрируем анализаторы напрямую (обход проблем с issubclass)
        from analyzers.algorithmic_analyzer import AlgorithmicAnalyzer
        from analyzers.qwen_analyzer import QwenAnalyzer
        from analyzers.ollama_analyzer import OllamaAnalyzer
        from analyzers.hybrid_analyzer import HybridAnalyzer
        from analyzers.emotion_analyzer import EmotionAnalyzer
        
        AnalyzerFactory._analyzers["algorithmic_basic"] = AlgorithmicAnalyzer
        AnalyzerFactory._analyzers["qwen"] = QwenAnalyzer
        AnalyzerFactory._analyzers["ollama"] = OllamaAnalyzer
        AnalyzerFactory._analyzers["hybrid"] = HybridAnalyzer
        AnalyzerFactory._analyzers["emotion_analyzer"] = EmotionAnalyzer
        
        registered = list(AnalyzerFactory._analyzers.keys())
        logging.info(f"Analyzers registered: {registered}")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize analyzers: {e}")
        return False


def create_app(config: Optional[AppConfig] = None, 
               config_file: Optional[str] = None,
               **config_kwargs) -> Application:
    """
    Create and configure application instance.
    
    Args:
        config: Pre-configured AppConfig instance
        config_file: Path to configuration file
        **config_kwargs: Additional configuration parameters
        
    Returns:
        Configured Application instance
    """
    global _app_instance
    
    if config is None:
        # Load configuration
        config = load_config(config_file=config_file, **config_kwargs)
    
    # Инициализируем анализаторы перед созданием приложения
    init_analyzers()
    
    _app_instance = Application(config)
    return _app_instance


def get_app() -> Application:
    """
    Get global application instance.
    
    Returns:
        Global Application instance
        
    Raises:
        ApplicationError: If no application instance exists
    """
    global _app_instance
    
    if _app_instance is None:
        # Create with default configuration
        _app_instance = create_app()
    
    return _app_instance


def init_app(**kwargs) -> Application:
    """
    Initialize application with custom parameters.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        Initialized Application instance
    """
    app = create_app(**kwargs)
    app.initialize()
    return app


# Context manager for application lifecycle
class AppContext:
    """Context manager for application lifecycle"""
    
    def __init__(self, **kwargs):
        """Initialize context with app configuration"""
        self.kwargs = kwargs
        self.app = None
    
    def __enter__(self) -> Application:
        """Enter context and initialize application"""
        self.app = init_app(**self.kwargs)
        return self.app
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and shutdown application"""
        if self.app:
            self.app.shutdown()


# Convenience function for common usage
def with_app(**config_kwargs):
    """
    Decorator for functions that need application context.
    
    Usage:
        @with_app(config_file="custom_config.json")
        def my_function():
            app = get_app()
            # use app...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with AppContext(**config_kwargs) as app:
                return func(*args, **kwargs)
        return wrapper
    return decorator
