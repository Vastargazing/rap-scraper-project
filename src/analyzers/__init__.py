"""
Analyzers module for the rap scraper project.

Содержит реализации различных анализаторов текста песен:
- Алгоритмические анализаторы
- AI-анализаторы 
- Гибридные анализаторы
"""

from .algorithmic_analyzer import AdvancedAlgorithmicAnalyzer
from .qwen_analyzer import QwenAnalyzer
from .ollama_analyzer import OllamaAnalyzer
# from .hybrid_analyzer import HybridAnalyzer  # Временно отключен
from .emotion_analyzer import EmotionAnalyzer

# При создании новых анализаторов они будут автоматически зарегистрированы
# благодаря декоратору @register_analyzer

__all__ = [
    "AdvancedAlgorithmicAnalyzer",
    "QwenAnalyzer", 
    "OllamaAnalyzer",
    # "HybridAnalyzer",  # Временно отключен
    "EmotionAnalyzer"
]
