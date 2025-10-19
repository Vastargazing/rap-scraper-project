"""
✅ Утилиты валидации

Общие функции для валидации и очистки данных.
"""

import re


def validate_text(text: str, min_length: int = 1, max_length: int = 10000) -> bool:
    """
    Валидирует текст на корректность

    Args:
        text: Текст для валидации
        min_length: Минимальная длина
        max_length: Максимальная длина

    Returns:
        True если текст валидный
    """
    if not isinstance(text, str):
        return False

    if not text.strip():
        return False

    if len(text) < min_length or len(text) > max_length:
        return False

    return True


def clean_text(text: str) -> str:
    """
    Очищает текст от лишних символов

    Args:
        text: Исходный текст

    Returns:
        Очищенный текст
    """
    if not isinstance(text, str):
        return ""

    # Убираем лишние пробелы
    text = re.sub(r"\s+", " ", text.strip())

    # Убираем управляющие символы
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)

    return text


def validate_artist_name(name: str) -> bool:
    """Валидирует имя артиста"""
    return validate_text(name, min_length=1, max_length=200)


def validate_track_title(title: str) -> bool:
    """Валидирует название трека"""
    return validate_text(title, min_length=1, max_length=300)
