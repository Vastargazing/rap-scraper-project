"""
🧪 Benchmarks for analyzer performance testing

Эта папка содержит pytest-benchmark тесты для производительности анализаторов.
Используется для автоматизации CI/CD и исторических сравнений.

ИСПОЛЬЗОВАНИЕ:
pytest tests/benchmarks/ --benchmark-only                # Только бенчмарки
pytest tests/benchmarks/ --benchmark-save=baseline       # Сохранить baseline
pytest tests/benchmarks/ --benchmark-compare             # Сравнение

СТРУКТУРА:
- conftest.py                   # Benchmark-специфичная конфигурация
- test_analyzer_benchmarks.py   # Основные тесты производительности
- test_quick_benchmarks.py      # Быстрые тесты для CI/CD
"""
