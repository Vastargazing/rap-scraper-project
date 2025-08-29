"""
Interfaces module for the rap scraper project.

This module defines the core interfaces and abstract base classes
that ensure consistent APIs across different implementations.
"""

from .analyzer_interface import (
    BaseAnalyzer,
    AnalysisResult,
    AnalyzerFactory,
    register_analyzer
)

from .database_interface import (
    DatabaseInterface,
    SQLiteManager,
    DatabaseConfig,
    create_database_manager
)

__all__ = [
    # Analyzer interfaces
    "BaseAnalyzer",
    "AnalysisResult", 
    "AnalyzerFactory",
    "register_analyzer",
    
    # Database interfaces
    "DatabaseInterface",
    "SQLiteManager",
    "DatabaseConfig",
    "create_database_manager"
]
