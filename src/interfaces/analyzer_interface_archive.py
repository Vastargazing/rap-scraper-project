"""
🧠 Базовые интерфейсы для всех анализаторов проекта Rap Scraper

НАЗНАЧЕНИЕ:
- Определение абстрактных классов и API для всех типов анализаторов (AI, алгоритмические, гибридные)
- Стандартизация формата результатов анализа
- Фабрика и декораторы для регистрации анализаторов

ИСПОЛЬЗОВАНИЕ:
from src.interfaces.analyzer_interface import BaseAnalyzer, AnalysisResult, AnalyzerFactory

ЗАВИСИМОСТИ:
- Python 3.8+
- abc, dataclasses, enum

РЕЗУЛЬТАТ:
- Единый API для всех анализаторов
- Упрощение интеграции и расширения системы

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

# Import будет исправлен после создания всех компонентов
# from src.models.models import EnhancedSongData


class AnalyzerType(Enum):
    """Enum для типов анализаторов"""
    ALGORITHMIC = "algorithmic_basic"  # Используем правильное имя
    GEMMA = "gemma" 
    OLLAMA = "ollama"
    HYBRID = "hybrid"


@dataclass
class AnalysisResult:
    """Standard result format for all analyzers"""
    artist: str
    title: str
    analysis_type: str
    confidence: float
    metadata: Dict[str, Any]
    raw_output: Dict[str, Any]
    processing_time: float
    timestamp: str


class BaseAnalyzer(ABC):
    """
    Abstract base class for all analyzers.
    
    This ensures consistent interface across:
    - AI-based analyzers (Gemma, GPT, etc.)
    - Algorithmic analyzers (feature extraction)
    - Hybrid analyzers (combination approaches)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize analyzer with configuration"""
        self.config = config or {}
        self.name = self.__class__.__name__
        self.available = True  # Default availability
        self.model_name = None  # Default model name
        self.api_url = None  # Default API URL
    
    @abstractmethod
    def analyze_song(self, artist: str, title: str, lyrics: str) -> AnalysisResult:
        """
        Analyze a single song and return structured results.
        
        Args:
            artist: Artist name
            title: Song title  
            lyrics: Song lyrics text
            
        Returns:
            AnalysisResult with standardized output format
        """
        pass
    
    @abstractmethod
    def get_analyzer_info(self) -> Dict[str, Any]:
        """
        Return metadata about this analyzer.
        
        Returns:
            Dict containing analyzer name, version, capabilities, etc.
        """
        pass
    
    @property
    @abstractmethod
    def analyzer_type(self) -> str:
        """
        Return analyzer type classification.
        
        Must be one of: 'ai', 'algorithmic', 'hybrid'
        """
        pass
    
    @property
    @abstractmethod
    def supported_features(self) -> List[str]:
        """Return list of features this analyzer supports"""
        pass
    
    def validate_input(self, artist: str, title: str, lyrics: str) -> bool:
        """
        Validate input parameters.
        
        Can be overridden by specific analyzers for custom validation.
        """
        if not all([artist, title, lyrics]):
            return False
        
        if len(lyrics.strip()) < 10:  # Minimum lyrics length
            return False
            
        return True
    
    def preprocess_lyrics(self, lyrics: str) -> str:
        """
        Basic lyrics preprocessing.
        
        Can be overridden by specific analyzers.
        """
        # Basic cleanup
        lyrics = lyrics.strip()
        
        # Remove excessive whitespace
        import re
        lyrics = re.sub(r'\s+', ' ', lyrics)
        
        return lyrics


class AnalyzerFactory:
    """
    Factory class for creating and managing analyzers.
    
    Provides centralized registry and creation of analyzer instances.
    """
    
    _analyzers: Dict[str, type] = {}
    _instances: Dict[str, BaseAnalyzer] = {}
    
    @classmethod
    def register(cls, name: str, analyzer_class: type) -> None:
        """
        Register an analyzer class.
        
        Args:
            name: Unique name for the analyzer
            analyzer_class: Class that implements BaseAnalyzer
        """
        if not issubclass(analyzer_class, BaseAnalyzer):
            raise ValueError(f"Analyzer class must inherit from BaseAnalyzer")
        
        cls._analyzers[name] = analyzer_class
    
    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None, 
               singleton: bool = True) -> BaseAnalyzer:
        """
        Create analyzer instance.
        
        Args:
            name: Registered analyzer name
            config: Configuration for the analyzer
            singleton: If True, reuse existing instance
            
        Returns:
            Analyzer instance
        """
        if name not in cls._analyzers:
            available = list(cls._analyzers.keys())
            raise ValueError(f"Unknown analyzer: {name}. Available: {available}")
        
        # Return singleton instance if requested and exists
        if singleton and name in cls._instances:
            return cls._instances[name]
        
        # Create new instance
        analyzer_class = cls._analyzers[name]
        instance = analyzer_class(config)
        
        # Store as singleton if requested
        if singleton:
            cls._instances[name] = instance
        
        return instance
    
    @classmethod
    def list_available(cls) -> List[str]:
        """Return list of registered analyzer names"""
        return list(cls._analyzers.keys())
    
    @classmethod
    def get_analyzer_info(cls, name: str) -> Dict[str, Any]:
        """Get information about a registered analyzer"""
        if name not in cls._analyzers:
            raise ValueError(f"Unknown analyzer: {name}")
        
        analyzer_class = cls._analyzers[name]
        
        # Create temporary instance to get info
        temp_instance = analyzer_class()
        return temp_instance.get_analyzer_info()


# Decorator for easy registration
def register_analyzer(name: str):
    """
    Decorator for automatic analyzer registration.
    
    Usage:
        @register_analyzer("my_analyzer")
        class MyAnalyzer(BaseAnalyzer):
            ...
    """
    def decorator(analyzer_class):
        AnalyzerFactory.register(name, analyzer_class)
        return analyzer_class
    
    return decorator
