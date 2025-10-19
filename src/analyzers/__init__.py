"""
Analyzers module for the rap scraper project.

Содержит реализации различных анализаторов текста песен:
- Алгоритмические анализаторы
- AI-анализаторы
- Гибридные анализаторы

New config-integrated analyzers (v2.0.0):
- QwenAnalyzer: Type-safe QWEN with config integration
"""

from .algorithmic_analyzer import AdvancedAlgorithmicAnalyzer

# from .hybrid_analyzer import HybridAnalyzer  # Временно отключен
from .emotion_analyzer import EmotionAnalyzer
from .mass_qwen_analysis import UnifiedQwenMassAnalyzer
from .ollama_analyzer import OllamaAnalyzer

# New config-integrated analyzers
from .qwen_analyzer import QwenAnalyzer  # v2.0.0 with config integration

# При создании новых анализаторов они будут автоматически зарегистрированы
# благодаря декоратору @register_analyzer

__all__ = [
    "AdvancedAlgorithmicAnalyzer",
    "UnifiedQwenMassAnalyzer",
    "OllamaAnalyzer",
    # "HybridAnalyzer",  # Временно отключен
    "EmotionAnalyzer",
    "QwenAnalyzer",  # New v2.0.0
]
