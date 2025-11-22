"""Analyzers module for rap lyrics analysis.

This module provides various analyzer implementations for processing and analyzing
rap song lyrics, including algorithmic, AI-powered, and hybrid approaches.

Available Analyzers:
    AdvancedAlgorithmicAnalyzer:
        Rule-based analyzer using linguistic features, sentiment analysis,
        and statistical metrics. No external API dependencies required.

        Best for:
            - Fast analysis without API costs
            - Offline processing
            - Explainable, deterministic results
            - Baseline comparisons

    EmotionAnalyzer:
        Specialized emotion and sentiment analysis with fine-grained
        emotional state detection (joy, anger, sadness, fear, etc.).
        Uses linguistic patterns and emotion lexicons.

        Best for:
            - Emotional profiling and mood classification
            - Sentiment intensity measurement
            - Multi-dimensional emotion analysis

    OllamaAnalyzer:
        Local LLM analyzer using Ollama for free, offline AI inference.
        Supports multiple open-source models (llama3, mistral, gemma, etc.).

        Best for:
            - Privacy-conscious analysis (no data leaves your machine)
            - Zero API costs with local inference
            - Experimentation with different LLM models
            - Offline/air-gapped environments

    UnifiedQwenMassAnalyzer:
        Unified QWEN-based analyzer optimized for batch processing multiple tracks.
        Uses Alibaba's QWEN LLM for deep semantic understanding and mass analysis.

        Best for:
            - Batch processing of large track collections
            - Deep semantic and cultural context understanding
            - Production-scale analysis pipelines
            - Chinese and multilingual rap analysis

    QwenAnalyzer (v2.0.0):
        Type-safe QWEN analyzer with modern config integration, enhanced
        error handling, and Pydantic validation. Recommended for new projects.

        Best for:
            - New production deployments
            - Type-safe codebases with mypy/pyright
            - Projects using centralized configuration management
            - Single-track analysis with high reliability

Module Attributes:
    __all__: List of publicly exported analyzer classes available for import.

Typical Usage:
    Fast algorithmic analysis:
        >>> from src.analyzers import AdvancedAlgorithmicAnalyzer
        >>>
        >>> analyzer = AdvancedAlgorithmicAnalyzer()
        >>> result = analyzer.analyze(
        ...     lyrics="Started from the bottom now we're here",
        ...     artist="Drake",
        ...     title="Started From the Bottom"
        ... )
        >>> print(f"Sentiment: {result.sentiment}")
        >>> print(f"Themes: {result.themes}")

    AI-powered analysis with Ollama (local):
        >>> from src.analyzers import OllamaAnalyzer
        >>>
        >>> analyzer = OllamaAnalyzer(model="llama3")
        >>> result = await analyzer.analyze_async(
        ...     lyrics=lyrics,
        ...     artist="Kendrick Lamar",
        ...     title="HUMBLE."
        ... )
        >>> print(f"AI Analysis: {result.ai_commentary}")

    Type-safe config-based analysis:
        >>> from src.analyzers import QwenAnalyzer
        >>> from src.utils.config_manager import ConfigManager
        >>>
        >>> config = ConfigManager()
        >>> analyzer = QwenAnalyzer(config)
        >>> result = await analyzer.analyze_async(
        ...     lyrics=lyrics,
        ...     artist="Eminem",
        ...     title="Lose Yourself"
        ... )
        >>> print(f"Genre: {result.metadata.genre}")

    Batch processing multiple tracks:
        >>> from src.analyzers import UnifiedQwenMassAnalyzer
        >>>
        >>> analyzer = UnifiedQwenMassAnalyzer()
        >>> results = await analyzer.analyze_batch(
        ...     track_ids=[1, 2, 3, 4, 5],
        ...     save_to_db=True
        ... )
        >>> print(f"Processed {len(results)} tracks")

Note:
    - HybridAnalyzer is temporarily disabled pending refactoring
    - All analyzers should follow the BaseAnalyzer interface pattern
    - For new projects, prefer QwenAnalyzer (v2.0.0) for better type safety
    - Async analyzers require await and proper event loop management

Version History:
    2.0.0:
        - Added QwenAnalyzer with config integration and type safety
        - Improved error handling and validation
    1.x:
        - Legacy analyzers (AdvancedAlgorithmic, Ollama, Emotion, etc.)
        - Initial hybrid analyzer (now deprecated)

See Also:
    - src.analyzers.multi_model_analyzer: Multi-provider AI analysis with fallback
    - src.utils.config_manager: Configuration management for analyzers
"""

from .algorithmic_analyzer import AdvancedAlgorithmicAnalyzer

# from .hybrid_analyzer import HybridAnalyzer  # Temporarily disabled
from .emotion_analyzer import EmotionAnalyzer

try:
    from .mass_qwen_analysis import UnifiedQwenMassAnalyzer
except (ImportError, SystemExit):
    # mass_qwen_analysis requires full app context, may fail during tests
    UnifiedQwenMassAnalyzer = None

from .ollama_analyzer import OllamaAnalyzer

# New config-integrated analyzers (v2.0.0)
from .qwen_analyzer import QwenAnalyzer

__all__ = [
    "AdvancedAlgorithmicAnalyzer",
    "EmotionAnalyzer",
    "OllamaAnalyzer",
    "QwenAnalyzer",
    "UnifiedQwenMassAnalyzer",
    # "HybridAnalyzer",  # Temporarily disabled pending refactoring
]
