"""
🔧 Общие утилиты проекта

Централизованные утилиты для:
- Логирование
- Валидация данных

Создано для устранения дублирования кода между компонентами.
"""

from .logging_utils import get_logger, setup_logging
from .validation_utils import (
    clean_text,
    validate_artist_name,
    validate_text,
    validate_track_title,
)

__all__ = [
    "clean_text",
    "get_logger",
    "setup_logging",
    "validate_artist_name",
    "validate_text",
    "validate_track_title",
]
