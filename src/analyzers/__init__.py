"""Analyzers module for the rap lyrics intelligence platform.

This package provides unified access to all text analyzers used for rap
lyrics analysis, sentiment detection, and AI-powered content understanding.

The analyzers module supports multiple analysis paradigms:
- **Algorithmic**: Rule-based statistical analysis
- **AI-powered**: LLM-based analysis (QWEN, Ollama)
- **Emotion**: Specialized emotion detection models
- **Hybrid**: Combined ML and heuristic approaches (coming soon)

Module Attributes:
    QwenAnalyzer: Type-safe QWEN analyzer with config integration (v2.0.0).
        Supports lyrics analysis, sentiment detection, and theme extraction.
    AdvancedAlgorithmicAnalyzer: Rule-based text analysis with statistical
        features including complexity scoring and readability metrics.
    EmotionAnalyzer: Emotion detection using pre-trained transformer models.
        Provides multi-label emotion classification.
    OllamaAnalyzer: Local LLM integration via Ollama for privacy-first
        analysis without external API calls.
    UnifiedQwenMassAnalyzer: Batch processing analyzer for large-scale
        lyrics analysis with QWEN model.

New in v2.0.0:
    - QwenAnalyzer: Config-integrated, type-safe implementation
    - Pydantic-based configuration support
    - Redis caching for performance optimization
    - Comprehensive error handling and logging

Example:
    Basic usage with QWEN analyzer:

    >>> from src.analyzers import QwenAnalyzer
    >>> analyzer = QwenAnalyzer()
    >>> result = analyzer.analyze_lyrics("Yeah, I'm on top of the game...")
    >>> print(result['sentiment'])
    'positive'

    Batch processing:

    >>> from src.analyzers import UnifiedQwenMassAnalyzer
    >>> batch_analyzer = UnifiedQwenMassAnalyzer()
    >>> results = await batch_analyzer.analyze_batch(lyrics_list)

Note:
    All analyzers support the @register_analyzer decorator for automatic
    registration in the analyzer registry. Import any analyzer class to
    register it automatically.

    HybridAnalyzer is temporarily disabled pending refactoring.
"""

from .algorithmic_analyzer import AdvancedAlgorithmicAnalyzer
from .emotion_analyzer import EmotionAnalyzer
from .mass_qwen_analysis import UnifiedQwenMassAnalyzer
from .ollama_analyzer import OllamaAnalyzer
from .qwen_analyzer import QwenAnalyzer  # v2.0.0 with config integration

# from .hybrid_analyzer import HybridAnalyzer  # Temporarily disabled

# When new analyzers are created, they will be automatically registered
# via the @register_analyzer decorator

__all__ = [
    "AdvancedAlgorithmicAnalyzer",
    "EmotionAnalyzer",
    "OllamaAnalyzer",
    "QwenAnalyzer",  # New v2.0.0
    "UnifiedQwenMassAnalyzer",
    # "HybridAnalyzer",  # Temporarily disabled
]
