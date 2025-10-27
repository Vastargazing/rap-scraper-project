"""
✅ Утилиты валидации данных для рэп-проекта

НАЗНАЧЕНИЕ:
- Валидация текстов, имен артистов и названий треков
- Очистка данных от лишних символов и управляющих последовательностей
- Проверка корректности строковых данных перед сохранением в БД
- Унифицированная обработка пользовательского ввода

ОСНОВНЫЕ ФУНКЦИИ:
validate_text()         # Проверка текста на корректность (длина, тип)
clean_text()            # Удаление лишних пробелов и управляющих символов
validate_artist_name()  # Валидация имени артиста (1-200 символов)
validate_track_title()  # Валидация названия трека (1-300 символов)

ИСПОЛЬЗОВАНИЕ:
from src.utils.validation_utils import validate_text, clean_text

# Валидация
if validate_artist_name("Eminem"):
    print("✅ Имя артиста корректно")

# Очистка
clean_lyrics = clean_text("  Multiple    spaces   ")
# → "Multiple spaces"

ЗАВИСИМОСТИ:
- re (стандартная библиотека Python)
- НЕТ внешних зависимостей

СТАТУС:
⚠️ UTILITY MODULE - готов к использованию, но в данный момент НЕ импортируется
в других частях проекта (по состоянию на 2025-10-27)

ПРИМЕНЕНИЕ:
Использовать для валидации данных при:
- Добавлении новых треков через API
- Скрапинге данных с Genius
- Пользовательском вводе в интерактивных скриптах
- Очистке текстов перед AI-анализом

АВТОР: Vastargazing
ДАТА: Октябрь 2025
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
