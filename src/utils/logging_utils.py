"""
📝 Централизованные утилиты логирования для проекта Rap Scraper

НАЗНАЧЕНИЕ:
- Единообразная настройка логирования для всех компонентов проекта
- Автоматическое создание директорий для логов
- Гибкая конфигурация (console, file, level)
- Предотвращение дублирования handlers

ОСНОВНЫЕ ФУНКЦИИ:
setup_logging(name, level, log_file, console)  # Полная настройка логгера
get_logger(name)                               # Быстрая настройка с дефолтами

ИСПОЛЬЗОВАНИЕ:
from src.utils.logging_utils import setup_logging, get_logger

# Вариант 1: Полная настройка
logger = setup_logging(
    name=__name__,
    level=logging.DEBUG,
    log_file="logs/my_module.log",
    console=True
)

# Вариант 2: Быстрая настройка
logger = get_logger(__name__)  # Uses logs/rap_scraper.log

ЗАВИСИМОСТИ:
- logging (стандартная библиотека Python)
- pathlib (стандартная библиотека Python)
- НЕТ внешних зависимостей

СТАТУС:
✅ PRODUCTION READY - централизованная утилита для всего проекта
🎯 ИСПОЛЬЗУЕТСЯ В: app.py, analyzers, scripts, lint.py

АВТОР: Vastargazing
ДАТА: Октябрь 2025
"""

import logging
from pathlib import Path


def setup_logging(
    name: str,
    level: int = logging.INFO,
    log_file: str | None = None,
    console: bool = True,
) -> logging.Logger:
    """
    Настраивает логгер с единообразными параметрами

    Args:
        name: Имя логгера
        level: Уровень логирования
        log_file: Путь к файлу логов (опционально)
        console: Выводить в консоль

    Returns:
        Настроенный логгер
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Очищаем существующие handlers
    logger.handlers.clear()

    # Создаем форматтер
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        # Создаем директорию логов если нужно
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Получает логгер с базовой настройкой"""
    return setup_logging(name, log_file="logs/rap_scraper.log")
