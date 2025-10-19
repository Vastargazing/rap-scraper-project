#!/usr/bin/env python3
"""
🚀 Быстрые benchmark тесты для CI/CD

НАЗНАЧЕНИЕ:
- Быстрые автоматизированные тесты производительности
- Интеграция с GitHub Actions
- Минимальное дублирование с performance_monitor.py

ИСПОЛЬЗОВАНИЕ:
pytest tests/benchmarks/test_quick_benchmarks.py --benchmark-only
pytest tests/benchmarks/test_quick_benchmarks.py --benchmark-save=baseline
pytest tests/benchmarks/test_quick_benchmarks.py --benchmark-compare

АВТОР: RapAnalyst AI Assistant
ДАТА: 19.09.2025
"""

import sys
from pathlib import Path

import pytest

# Добавляем src в Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from core.app import Application
except ImportError:
    # Fallback если импорт не работает
    Application = None


class TestQuickBenchmarks:
    """Быстрые benchmark тесты для CI/CD"""

    @pytest.mark.quick
    @pytest.mark.benchmark(group="quick")
    def test_algorithmic_analyzer_quick(self, benchmark, rap_app, quick_test_texts):
        """Быстрый тест алгоритмического анализатора"""
        if not rap_app:
            pytest.skip("Application not available")

        analyzer = rap_app.get_analyzer("advanced_algorithmic")
        if not analyzer:
            pytest.skip("advanced_algorithmic analyzer not available")

        text = quick_test_texts[0]  # Самый короткий текст

        def run_analysis():
            if hasattr(analyzer, "analyze_song"):
                return analyzer.analyze_song("Test", "Quick", text)
            # Fallback для других интерфейсов
            return analyzer.analyze(text) if hasattr(analyzer, "analyze") else None

        # Добавляем метаданные
        benchmark.extra_info["analyzer"] = "advanced_algorithmic"
        benchmark.extra_info["text_length"] = len(text)
        benchmark.extra_info["test_type"] = "quick_ci"

        result = benchmark(run_analysis)
        assert result is not None

    @pytest.mark.quick
    @pytest.mark.benchmark(group="quick")
    def test_emotion_analyzer_quick(self, benchmark, rap_app, quick_test_texts):
        """Быстрый тест emotion анализатора"""
        if not rap_app:
            pytest.skip("Application not available")

        analyzer = rap_app.get_analyzer("emotion_analyzer")
        if not analyzer:
            pytest.skip("emotion_analyzer not available")

        text = quick_test_texts[1]

        def run_analysis():
            if hasattr(analyzer, "analyze_song"):
                return analyzer.analyze_song("Test", "Emotion", text)
            return analyzer.analyze(text) if hasattr(analyzer, "analyze") else None

        benchmark.extra_info["analyzer"] = "emotion_analyzer"
        benchmark.extra_info["text_length"] = len(text)
        benchmark.extra_info["test_type"] = "quick_ci"

        result = benchmark(run_analysis)
        assert result is not None

    @pytest.mark.benchmark(group="comparison")
    def test_analyzer_comparison_advanced_algorithmic(
        self, benchmark, rap_app, quick_test_texts
    ):
        """Быстрое тестирование advanced_algorithmic анализатора"""
        if not rap_app:
            pytest.skip("Application not available")

        analyzer = rap_app.get_analyzer("advanced_algorithmic")
        if not analyzer:
            pytest.skip("advanced_algorithmic not available")

        text = quick_test_texts[0]  # Одинаковый текст для сравнения

        def run_analysis():
            if hasattr(analyzer, "analyze_song"):
                return analyzer.analyze_song("Test", "Compare", text)
            return analyzer.analyze(text) if hasattr(analyzer, "analyze") else None

        # Метаданные для группировки
        benchmark.extra_info["analyzer"] = "advanced_algorithmic"
        benchmark.extra_info["text_length"] = len(text)
        benchmark.extra_info["test_type"] = "comparison"

        result = benchmark(run_analysis)
        assert result is not None

    @pytest.mark.benchmark(group="comparison")
    def test_analyzer_comparison_emotion(self, benchmark, rap_app, quick_test_texts):
        """Быстрое тестирование emotion_analyzer анализатора"""
        if not rap_app:
            pytest.skip("Application not available")

        analyzer = rap_app.get_analyzer("emotion_analyzer")
        if not analyzer:
            pytest.skip("emotion_analyzer not available")

        text = quick_test_texts[0]  # Одинаковый текст для сравнения

        def run_analysis():
            if hasattr(analyzer, "analyze_song"):
                return analyzer.analyze_song("Test", "Compare", text)
            return analyzer.analyze(text) if hasattr(analyzer, "analyze") else None

        # Метаданные для группировки
        benchmark.extra_info["analyzer"] = "emotion_analyzer"
        benchmark.extra_info["text_length"] = len(text)
        benchmark.extra_info["test_type"] = "comparison"

        result = benchmark(run_analysis)
        assert result is not None

    @pytest.mark.benchmark(group="stress")
    def test_batch_processing_mini(self, benchmark, rap_app, quick_test_texts):
        """Мини-тест пакетной обработки"""
        if not rap_app:
            pytest.skip("Application not available")

        analyzer = rap_app.get_analyzer("advanced_algorithmic")
        if not analyzer:
            pytest.skip("advanced_algorithmic not available")

        def batch_analysis():
            results = []
            for i, text in enumerate(quick_test_texts):  # Только 3 текста
                if hasattr(analyzer, "analyze_song"):
                    result = analyzer.analyze_song("Test", f"Batch_{i}", text)
                else:
                    result = (
                        analyzer.analyze(text) if hasattr(analyzer, "analyze") else None
                    )
                results.append(result)
            return results

        benchmark.extra_info["analyzer"] = "advanced_algorithmic"
        benchmark.extra_info["batch_size"] = len(quick_test_texts)
        benchmark.extra_info["test_type"] = "mini_batch"

        results = benchmark(batch_analysis)
        assert len(results) == len(quick_test_texts)
        assert all(r is not None for r in results)


class TestMemoryBenchmarks:
    """Тесты потребления памяти"""

    @pytest.mark.benchmark(group="memory")
    def test_memory_efficiency(self, enhanced_benchmark, rap_app, quick_test_texts):
        """Тест эффективности памяти"""
        if not rap_app:
            pytest.skip("Application not available")

        analyzer = rap_app.get_analyzer("advanced_algorithmic")
        if not analyzer:
            pytest.skip("advanced_algorithmic not available")

        def memory_test():
            results = []
            for i in range(5):  # 5 повторов для измерения роста памяти
                text = quick_test_texts[i % len(quick_test_texts)]
                if hasattr(analyzer, "analyze_song"):
                    result = analyzer.analyze_song("Test", f"Memory_{i}", text)
                else:
                    result = (
                        analyzer.analyze(text) if hasattr(analyzer, "analyze") else None
                    )
                results.append(result)
            return results

        results = enhanced_benchmark(memory_test)

        # Проверяем что рост памяти разумный
        memory_growth = enhanced_benchmark.extra_info.get("memory_growth_mb", 0)
        assert memory_growth < 50  # Менее 50MB роста
        assert len(results) == 5


# === ТЕСТ НА ДОСТУПНОСТЬ ===


def test_benchmark_environment():
    """Проверка что benchmark среда готова"""
    try:
        import pytest_benchmark

        assert pytest_benchmark is not None
    except ImportError:
        pytest.skip("pytest-benchmark not installed")

    # Проверяем что src доступен
    src_path = Path(__file__).parent.parent.parent / "src"
    assert src_path.exists(), f"src directory not found: {src_path}"
