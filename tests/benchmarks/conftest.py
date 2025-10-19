#!/usr/bin/env python3
"""
🧪 Benchmark-специфичная конфигурация для pytest

НАЗНАЧЕНИЕ:
- Настройки только для benchmark тестов
- Фикстуры для анализаторов
- Кастомные метрики и группировка

АВТОР: RapAnalyst AI Assistant
ДАТА: 19.09.2025
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Добавляем src в Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.app import Application

# === BENCHMARK КОНФИГУРАЦИЯ ===


@pytest.fixture(scope="session")
def benchmark_config():
    """Конфигурация бенчмарков для быстрых CI тестов"""
    return {
        "min_rounds": 3,  # Минимум раундов (было 5)
        "max_time": 10.0,  # Максимум времени (было 30)
        "min_time": 0.05,  # Минимум времени (было 0.1)
        "timer": "time.perf_counter",
        "disable_gc": True,
        "warmup": True,
        "warmup_iterations": 2,  # Меньше прогревов (было 3)
    }


# === ФИКСТУРЫ ДЛЯ АНАЛИЗАТОРОВ ===


@pytest.fixture(scope="session")
def rap_app():
    """Фикстура приложения для всей сессии"""
    return Application()


@pytest.fixture(scope="session")
def quick_test_texts():
    """Быстрые тестовые тексты для CI"""
    return [
        "Happy song",
        "Sad lyrics about love",
        "Energetic rap with messages",
    ]


@pytest.fixture(scope="session")
def available_analyzers(rap_app):
    """Доступные анализаторы"""
    available = rap_app.list_analyzers()
    analyzers = {}

    for name in available:
        analyzer = rap_app.get_analyzer(name)
        if analyzer:
            analyzers[name] = analyzer

    return analyzers


# === КАСТОМНЫЕ МЕТРИКИ ===


@pytest.fixture
def enhanced_benchmark(benchmark):
    """Benchmark с дополнительными метриками"""
    import psutil

    def custom_benchmark(func, *args, **kwargs):
        process = psutil.Process()

        # Стартовые метрики
        start_memory = process.memory_info().rss
        start_cpu_time = process.cpu_times()

        # Запуск бенчмарка
        result = benchmark(func, *args, **kwargs)

        # Финальные метрики
        end_memory = process.memory_info().rss
        end_cpu_time = process.cpu_times()

        # Добавляем кастомные метрики
        benchmark.extra_info.update(
            {
                "memory_growth_mb": (end_memory - start_memory) / 1024 / 1024,
                "cpu_time_delta": (end_cpu_time.user + end_cpu_time.system)
                - (start_cpu_time.user + start_cpu_time.system),
            }
        )

        return result

    return custom_benchmark


# === ASYNC ПОДДЕРЖКА ===


@pytest.fixture
def async_benchmark(benchmark):
    """Benchmark для async функций"""

    def run_async_benchmark(async_func, *args, **kwargs):
        def sync_wrapper():
            return asyncio.run(async_func(*args, **kwargs))

        return benchmark(sync_wrapper)

    return run_async_benchmark


# === PYTEST HOOKS ===


def pytest_benchmark_update_machine_info(config, machine_info):
    """Добавляем информацию о машине"""
    import platform

    import psutil

    machine_info.update(
        {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / 1024**3, 1),
            "python_version": platform.python_version(),
            "platform": platform.platform()[:50],  # Обрезаем длинное имя
            "rap_project": "rap-scraper-project",
        }
    )


def pytest_benchmark_group_stats(config, benchmarks, group_by):
    """Группировка статистики по анализаторам"""

    # Если есть информация об анализаторе в extra_info
    analyzer_groups = {}
    for benchmark in benchmarks:
        analyzer = benchmark.extra_info.get("analyzer", "unknown")
        if analyzer not in analyzer_groups:
            analyzer_groups[analyzer] = []
        analyzer_groups[analyzer].append(benchmark)

    return analyzer_groups if analyzer_groups else group_by


# === МАРКЕРЫ ===


def pytest_configure(config):
    """Регистрация маркеров"""
    config.addinivalue_line("markers", "benchmark: mark test as benchmark test")
    config.addinivalue_line("markers", "quick: mark test as quick benchmark for CI")
    config.addinivalue_line("markers", "comparison: mark test as analyzer comparison")
