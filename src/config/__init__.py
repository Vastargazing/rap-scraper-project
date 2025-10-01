"""
Configuration module with type-safe Pydantic models

Usage:
    from src.config import get_config
    
    config = get_config()
    db_string = config.database.connection_string
    api_port = config.api.port
"""

from src.config.config_loader import (
    get_config,
    Config,
    load_config,
    get_environment,
    # Individual config classes for type hints
    ApplicationConfig,
    DatabaseConfig,
    VectorSearchConfig,
    LoggingConfig,
    AnalyzersConfig,
    APIConfig,
    RedisConfig,
    MonitoringConfig,
    CICDConfig
)

__all__ = [
    "get_config",
    "Config",
    "load_config",
    "get_environment",
    "ApplicationConfig",
    "DatabaseConfig",
    "VectorSearchConfig",
    "LoggingConfig",
    "AnalyzersConfig",
    "APIConfig",
    "RedisConfig",
    "MonitoringConfig",
    "CICDConfig"
]