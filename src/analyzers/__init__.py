"""
Analyzers module for the rap scraper project.

This package provides unified access to all text analyzers used in the platform.

Features:
    - Algorithmic analyzers (rule-based, statistical)
    - AI-powered analyzers (LLM, Qwen, Ollama, Emotion)
    - Hybrid analyzers (combining ML and heuristics)

New in v2.0.0:
    - QwenAnalyzer: Type-safe, config-integrated QWEN analyzer

Usage:
    from src.analyzers import QwenAnalyzer, AdvancedAlgorithmicAnalyzer

All analyzers are auto-registered via @register_analyzer decorator.
"""

from .algorithmic_analyzer import AdvancedAlgorithmicAnalyzer

# from .hybrid_analyzer import HybridAnalyzer  # Временно отключен
from .emotion_analyzer import EmotionAnalyzer
from .mass_qwen_analysis import UnifiedQwenMassAnalyzer
from .ollama_analyzer import OllamaAnalyzer

# New config-integrated analyzers
from .qwen_analyzer import QwenAnalyzer  # v2.0.0 with config integration

# When new analyzers are created, they will be automatically registered
# thanks to the @register_analyzer decorator

__all__ = [
    "AdvancedAlgorithmicAnalyzer",
    "UnifiedQwenMassAnalyzer",
    "OllamaAnalyzer",
    # "HybridAnalyzer",  # Временно отключен
    "EmotionAnalyzer",
    "QwenAnalyzer",  # New v2.0.0
]
