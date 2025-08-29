"""
Analyzers module for the rap scraper project.

Содержит реализации различных анализаторов текста песен:
- Алгоритмические анализаторы
- AI-анализаторы 
- Гибридные анализаторы
"""

from .algorithmic_analyzer import AlgorithmicAnalyzer

# При создании новых анализаторов они будут автоматически зарегистрированы
# благодаря декоратору @register_analyzer

__all__ = [
    "AlgorithmicAnalyzer"
]
