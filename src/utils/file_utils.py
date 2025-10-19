"""
📁 Утилиты для работы с файлами

Безопасные операции с файлами и JSON.
"""

import json
import os
from pathlib import Path
from typing import Any


def ensure_directory(path: str) -> Path:
    """
    Создает директорию если она не существует

    Args:
        path: Путь к директории

    Returns:
        Path объект директории
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def safe_json_load(file_path: str, default: Any = None) -> Any:
    """
    Безопасно загружает JSON файл

    Args:
        file_path: Путь к JSON файлу
        default: Значение по умолчанию при ошибке

    Returns:
        Данные из JSON или default
    """
    try:
        if not os.path.exists(file_path):
            return default

        with open(file_path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def safe_json_save(data: Any, file_path: str) -> bool:
    """
    Безопасно сохраняет данные в JSON файл

    Args:
        data: Данные для сохранения
        file_path: Путь к файлу

    Returns:
        True если сохранение успешно
    """
    try:
        # Создаем директорию если нужно
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except (OSError, TypeError):
        return False


def get_file_size(file_path: str) -> int:
    """Возвращает размер файла в байтах"""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return 0
