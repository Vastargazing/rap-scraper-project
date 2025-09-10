"""
Analyzers module for the rap scraper project.

Содержит реализации различных анализаторов текста песен:
- Алгоритмические анализаторы
- AI-анализаторы 
- Гибридные анализаторы
"""

from .algorithmic_analyzer import AlgorithmicAnalyzer
from .qwen_analyzer import QwenAnalyzer
from .ollama_analyzer import OllamaAnalyzer
from .hybrid_analyzer import HybridAnalyzer
from .emotion_analyzer import EmotionAnalyzer

# При создании новых анализаторов они будут автоматически зарегистрированы
# благодаря декоратору @register_analyzer

__all__ = [
    "AlgorithmicAnalyzer",
    "QwenAnalyzer", 
    "OllamaAnalyzer",
    "HybridAnalyzer",
    "EmotionAnalyzer"
]
