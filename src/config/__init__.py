"""
Configuration module with type-safe Pydantic models

Usage:
    from src.config import get_config

    config = get_config()
    db_string = config.database.connection_string
    api_port = config.api.port
"""

from src.config.config_loader import (
    AnalyzersConfig,
    APIConfig,
    # Individual config classes for type hints
    ApplicationConfig,
    CICDConfig,
    Config,
    DatabaseConfig,
    LoggingConfig,
    MonitoringConfig,
    RedisConfig,
    VectorSearchConfig,
    get_config,
    get_environment,
    load_config,
)

__all__ = [
    "APIConfig",
    "AnalyzersConfig",
    "ApplicationConfig",
    "CICDConfig",
    "Config",
    "DatabaseConfig",
    "LoggingConfig",
    "MonitoringConfig",
    "RedisConfig",
    "VectorSearchConfig",
    "get_config",
    "get_environment",
    "load_config",
]
