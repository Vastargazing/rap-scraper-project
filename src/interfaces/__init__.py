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

__all__ = [
    # Analyzer interfaces
    "BaseAnalyzer",
    "AnalysisResult", 
    "AnalyzerFactory",
    "register_analyzer"
]
