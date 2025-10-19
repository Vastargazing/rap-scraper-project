"""
🔧 Общие утилиты проекта

Централизованные утилиты для:
- Логирование
- Валидация данных
- Общие функции
- Константы

Создано для устранения дублирования кода между компонентами.
"""

from .file_utils import ensure_directory, get_file_size, safe_json_load, safe_json_save
from .logging_utils import get_logger, setup_logging
from .validation_utils import (
    clean_text,
    validate_artist_name,
    validate_text,
    validate_track_title,
)

__all__ = [
    "clean_text",
    "ensure_directory",
    "get_file_size",
    "get_logger",
    "safe_json_load",
    "safe_json_save",
    "setup_logging",
    "validate_artist_name",
    "validate_text",
    "validate_track_title",
]
