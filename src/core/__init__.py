"""
Core module for the rap scraper project.

This module provides the foundational components for the application:
- Configuration management
- Database interfaces
- Application factory
- Analyzer interfaces
"""

from .app import (
    AppContext,
    Application,
    ApplicationError,
    ConfigurationError,
    DatabaseError,
    create_app,
    get_app,
    init_app,
    with_app,
)
from .config import (
    AnalysisConfig,
    APIConfig,
    AppConfig,
    ConfigManager,
    DatabaseConfig,
    LoggingConfig,
    ScrapingConfig,
    get_config,
    load_config,
    reload_config,
)

__all__ = [
    # Configuration
    "AppConfig",
    "DatabaseConfig",
    "ScrapingConfig",
    "AnalysisConfig",
    "LoggingConfig",
    "APIConfig",
    "ConfigManager",
    "get_config",
    "load_config",
    "reload_config",
    # Application
    "Application",
    "ApplicationError",
    "ConfigurationError",
    "DatabaseError",
    "create_app",
    "get_app",
    "init_app",
    "AppContext",
    "with_app",
]
