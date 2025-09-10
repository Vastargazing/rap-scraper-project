"""
Configuration management module.

Provides centralized configuration handling with support for:
- Environment variables
- Configuration files (JSON, YAML)
- Default values
- Validation
- Type safety

Централизованное управление конфигурацией:

AppConfig - основная конфигурация приложения
Поддержка JSON/YAML файлов и переменных окружения
Автоматическое создание директорий
Type-safe конфигурация с валидацией
"""

import os
import json
from pathlib import Path

# yaml - опциональная зависимость
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import logging


@dataclass
class DatabaseConfig:
    """Database configuration"""
    path: str = "data/rap_lyrics.db"
    timeout: float = 30.0
    pool_size: int = 5
    check_same_thread: bool = False


@dataclass
class ScrapingConfig:
    """Web scraping configuration"""
    base_delay: float = 1.0
    max_delay: float = 5.0
    max_retries: int = 3
    timeout: int = 30
    headers: Dict[str, str] = field(default_factory=lambda: {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })


@dataclass
class AnalysisConfig:
    """Analysis configuration"""
    batch_size: int = 10
    max_workers: int = 4
    default_confidence_threshold: float = 0.7
    supported_analyzers: List[str] = field(default_factory=lambda: [
        "gemma", "algorithmic", "hybrid"
    ])


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    encoding: str = "utf-8"  # Явная кодировка для Windows


@dataclass
class APIConfig:
    """API configuration for external services"""
    genius_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    request_delay: float = 1.0
    max_requests_per_minute: int = 60


@dataclass
class AppConfig:
    """Main application configuration"""
    # Core settings
    project_name: str = "rap-scraper"
    version: str = "1.0.0"
    environment: str = "development"  # development, staging, production
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    scraping: ScrapingConfig = field(default_factory=ScrapingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Paths
    data_dir: str = "data"
    results_dir: str = "results"
    logs_dir: str = "logs"
    cache_dir: str = "cache"
    
    # Additional settings
    debug: bool = False
    enable_monitoring: bool = True
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Ensure paths are absolute
        base_path = Path.cwd()
        
        self.data_dir = str(base_path / self.data_dir)
        self.results_dir = str(base_path / self.results_dir)
        self.logs_dir = str(base_path / self.logs_dir)
        self.cache_dir = str(base_path / self.cache_dir)
        
        # Update database path to be absolute
        if not Path(self.database.path).is_absolute():
            self.database.path = str(base_path / self.database.path)
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.results_dir, self.logs_dir, self.cache_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self)
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to file"""
        file_path = Path(file_path)
        
        config_dict = self.to_dict()
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            if not HAS_YAML:
                raise ValueError("PyYAML package required for YAML support. Install with: pip install PyYAML")
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")


class ConfigLoader:
    """Configuration loader with multiple sources support"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        try:
            if file_path.suffix.lower() == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_path.suffix.lower() in ['.yml', '.yaml']:
                if not HAS_YAML:
                    raise ValueError("PyYAML package required for YAML support. Install with: pip install PyYAML")
                with open(file_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {file_path}: {e}")
            raise
    
    def load_from_env(self, prefix: str = "RAP_SCRAPER_") -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}
        
        # Database settings
        if os.getenv(f"{prefix}DB_PATH"):
            config["database"] = {"path": os.getenv(f"{prefix}DB_PATH")}
        
        # API keys
        api_config = {}
        if os.getenv(f"{prefix}GENIUS_TOKEN"):
            api_config["genius_token"] = os.getenv(f"{prefix}GENIUS_TOKEN")
        if os.getenv(f"{prefix}OPENAI_API_KEY"):
            api_config["openai_api_key"] = os.getenv(f"{prefix}OPENAI_API_KEY")
        if os.getenv(f"{prefix}HUGGINGFACE_TOKEN"):
            api_config["huggingface_token"] = os.getenv(f"{prefix}HUGGINGFACE_TOKEN")
        
        if api_config:
            config["api"] = api_config
        
        # Environment
        if os.getenv(f"{prefix}ENVIRONMENT"):
            config["environment"] = os.getenv(f"{prefix}ENVIRONMENT")
        
        # Debug mode
        if os.getenv(f"{prefix}DEBUG"):
            config["debug"] = os.getenv(f"{prefix}DEBUG").lower() in ["true", "1", "yes"]
        
        # Logging level
        if os.getenv(f"{prefix}LOG_LEVEL"):
            config["logging"] = {"level": os.getenv(f"{prefix}LOG_LEVEL")}
        
        return config
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries"""
        result = {}
        
        for config in configs:
            self._deep_merge(result, config)
        
        return result
    
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> None:
        """Deep merge configuration dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


class ConfigManager:
    """
    Central configuration manager.
    
    Handles loading, merging, and providing access to application configuration
    from multiple sources with proper precedence.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.loader = ConfigLoader()
        self._config: Optional[AppConfig] = None
    
    def load_config(self, 
                   config_file: Optional[Union[str, Path]] = None,
                   env_prefix: str = "RAP_SCRAPER_",
                   defaults: Optional[Dict[str, Any]] = None) -> AppConfig:
        """
        Load configuration from multiple sources.
        
        Precedence (highest to lowest):
        1. Environment variables
        2. Configuration file
        3. Default values
        
        Args:
            config_file: Path to configuration file (optional)
            env_prefix: Prefix for environment variables
            defaults: Default configuration values
            
        Returns:
            Loaded and merged configuration
        """
        configs = []
        
        # Start with defaults
        if defaults:
            configs.append(defaults)
        
        # Load from file if provided
        if config_file:
            try:
                file_config = self.loader.load_from_file(config_file)
                configs.append(file_config)
                self.logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                self.logger.warning(f"Could not load config file {config_file}: {e}")
        
        # Load from environment (highest priority)
        env_config = self.loader.load_from_env(env_prefix)
        if env_config:
            configs.append(env_config)
            self.logger.info("Loaded configuration from environment variables")
        
        # Merge all configurations
        merged_config = self.loader.merge_configs(*configs)
        
        # Create AppConfig instance
        try:
            # Handle nested configuration by flattening and reconstructing
            self._config = self._dict_to_appconfig(merged_config)
            self.logger.info("Configuration loaded successfully")
            return self._config
            
        except Exception as e:
            self.logger.error(f"Failed to create configuration object: {e}")
            # Fallback to default configuration
            self._config = AppConfig()
            return self._config
    
    def _dict_to_appconfig(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig object"""
        # Create component configs
        database_config = DatabaseConfig()
        if "database" in config_dict:
            db_dict = config_dict["database"]
            for key, value in db_dict.items():
                if hasattr(database_config, key):
                    setattr(database_config, key, value)
        
        scraping_config = ScrapingConfig()
        if "scraping" in config_dict:
            scrape_dict = config_dict["scraping"]
            for key, value in scrape_dict.items():
                if hasattr(scraping_config, key):
                    setattr(scraping_config, key, value)
        
        analysis_config = AnalysisConfig()
        if "analysis" in config_dict:
            analysis_dict = config_dict["analysis"]
            for key, value in analysis_dict.items():
                if hasattr(analysis_config, key):
                    setattr(analysis_config, key, value)
        
        logging_config = LoggingConfig()
        if "logging" in config_dict:
            log_dict = config_dict["logging"]
            for key, value in log_dict.items():
                if hasattr(logging_config, key):
                    setattr(logging_config, key, value)
        
        api_config = APIConfig()
        if "api" in config_dict:
            api_dict = config_dict["api"]
            for key, value in api_dict.items():
                if hasattr(api_config, key):
                    setattr(api_config, key, value)
        
        # Create main config
        app_config = AppConfig(
            database=database_config,
            scraping=scraping_config,
            analysis=analysis_config,
            logging=logging_config,
            api=api_config
        )
        
        # Set top-level attributes
        for key, value in config_dict.items():
            if hasattr(app_config, key) and key not in ["database", "scraping", "analysis", "logging", "api"]:
                setattr(app_config, key, value)
        
        return app_config
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self, **kwargs) -> AppConfig:
        """Reload configuration"""
        self._config = None
        return self.load_config(**kwargs)
    
    def save_current_config(self, file_path: Union[str, Path]) -> None:
        """Save current configuration to file"""
        if self._config is None:
            raise RuntimeError("No configuration loaded")
        
        self._config.save_to_file(file_path)
        self.logger.info(f"Configuration saved to {file_path}")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get global configuration instance"""
    return config_manager.get_config()


def load_config(**kwargs) -> AppConfig:
    """Load configuration with custom parameters"""
    return config_manager.load_config(**kwargs)


def reload_config(**kwargs) -> AppConfig:
    """Reload global configuration"""
    return config_manager.reload_config(**kwargs)
