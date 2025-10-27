"""
CLI утилиты для rap-scraper-project

Этот модуль содержит утилиты командной строки для:
- Сравнения производительности анализаторов
- Мониторинга системных ресурсов
- Batch processing с checkpoint'ами
"""

from .batch_processor import BatchProcessor
from .performance_monitor import PerformanceMonitor

__all__ = ["BatchProcessor", "PerformanceMonitor"]
