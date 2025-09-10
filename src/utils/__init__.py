"""
🔧 Общие утилиты проекта

Централизованные утилиты для:
- Логирование
- Валидация данных
- Общие функции
- Константы

Создано для устранения дублирования кода между компонентами.
"""

from .logging_utils import setup_logging, get_logger
from .validation_utils import validate_text, clean_text, validate_artist_name, validate_track_title
from .file_utils import ensure_directory, safe_json_load, safe_json_save, get_file_size

__all__ = [
    "setup_logging",
    "get_logger", 
    "validate_text",
    "clean_text",
    "validate_artist_name",
    "validate_track_title", 
    "ensure_directory",
    "safe_json_load", 
    "safe_json_save",
    "get_file_size"
]
