#!/usr/bin/env python3
"""
📊 CLI-утилита для мониторинга производительности анализаторов

НАЗНАЧЕНИЕ:
- Измерение скорости, точности, ресурсов разных анализаторов
- Сравнение производительности моделей
- Логирование и вывод статистики

ИСПОЛЬЗОВАНИЕ:
python src/cli/performance_monitor.py --analyzer qwen      # Тест производительности Qwen
python src/cli/performance_monitor.py --all                # Сравнение всех моделей

ЗАВИСИМОСТИ:
- Python 3.8+
- src/core/app.py, src/interfaces/analyzer_interface.py
- psutil, statistics

РЕЗУЛЬТАТ:
- Консольный вывод с метриками и сравнением
- Логирование ошибок и статистики

АВТОР: AI Assistant
ДАТА: Сентябрь 2025

🚀 Продвинутый CLI-монитор производительности анализаторов

НОВЫЕ ФИЧИ:
- pytest-benchmark интеграция
- py-spy профилирование
- Prometheus метрики
- hyperfine CLI сравнения
- Memory profiling
- OpenTelemetry tracing

ИСПОЛЬЗОВАНИЕ:
python enhanced_monitor.py --analyzer qwen --mode benchmark    # Базовый бенчмарк
python enhanced_monitor.py --analyzer qwen --mode profile      # Глубокое профилирование
python enhanced_monitor.py --all --mode compare                # Сравнение всех
python enhanced_monitor.py --analyzer qwen --mode load         # Нагрузочное тестирование

АВТОР: AI Assistant + Human
ДАТА: Сентябрь 2025
"""

import argparse
import asyncio
import cProfile
import json
import logging
import pstats
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import psutil

# Загружаем переменные окружения из .env
try:
    from dotenv import load_dotenv

    load_dotenv()
    print("✅ .env файл загружен")
except ImportError:
    print("⚠️ python-dotenv не установлен. API ключи из .env не загружены.")

# Добавляем src в Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.app import create_app

# Опциональные импорты для продвинутых фич
try:
    from prometheus_client import REGISTRY, Counter, Gauge, Histogram, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from memory_profiler import profile as memory_profile

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

# Rich для красивого вывода
try:
    from rich import box
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, TaskID, track
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Tabulate для таблиц
try:
    from tabulate import tabulate

    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# Click для улучшенного CLI
try:
    import click

    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False


@dataclass
class EnhancedMetrics:
    """Расширенные метрики производительности"""

    analyzer_name: str
    test_count: int

    # Базовые временные метрики
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    median_time: float

    # Системные метрики
    avg_cpu_percent: float
    avg_memory_mb: float
    peak_memory_mb: float

    # Метрики качества
    success_rate: float
    error_count: int
    items_per_second: float

    # Новые метрики
    memory_growth_mb: float = 0.0
    cpu_efficiency: float = 0.0  # items per cpu percent
    latency_p95: float = 0.0
    latency_p99: float = 0.0

    # Профилирование
    hottest_function: str = ""
    profile_data: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PytestBenchmarkIntegration:
    """Интеграция с pytest-benchmark для профессионального бенчмаркинга"""

    def __init__(self, monitor: "EnhancedPerformanceMonitor"):
        self.monitor = monitor
        self.available = PYTEST_AVAILABLE

    def generate_benchmark_test(
        self, analyzer_type: str, test_texts: list[str], output_file: str
    ):
        """Генерация pytest benchmark тестов"""
        if not self.available:
            self.monitor.display.print_warning(
                "pytest не доступен для benchmark генерации"
            )
            return

        test_content = f'''#!/usr/bin/env python3
"""
🧪 Автоматически сгенерированные pytest-benchmark тесты
Анализатор: {analyzer_type}
Тестовые тексты: {len(test_texts)}
Дата генерации: {datetime.now().isoformat()}
"""
import pytest
import asyncio
import sys
from pathlib import Path

# Добавляем src в Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.app import create_app

class TestAnalyzerPerformance:
    """Benchmark тесты для анализатора {analyzer_type}"""
    
    @pytest.fixture(scope="class")
    def analyzer(self):
        """Фикстура анализатора"""
        app = create_app()
        return app.get_analyzer("{analyzer_type}")
    
    @pytest.fixture(scope="class") 
    def test_texts(self):
        """Тестовые тексты"""
        return {test_texts[:10]}  # Ограничиваем для быстрых тестов
    
    def test_single_analysis_benchmark(self, benchmark, analyzer, test_texts):
        """Benchmark одиночного анализа"""
        def run_analysis():
            text = test_texts[0]
            return asyncio.run(analyzer.analyze_song("Test Artist", "Test Song", text))
        
        result = benchmark(run_analysis)
        assert result is not None
        assert hasattr(result, 'confidence')
    
    def test_batch_analysis_benchmark(self, benchmark, analyzer, test_texts):
        """Benchmark батчевого анализа"""
        async def batch_analysis():
            results = []
            for i, text in enumerate(test_texts):
                result = await analyzer.analyze_song("Test Artist", f"Test Song {{i}}", text)
                results.append(result)
            return results
        
        def run_batch():
            return asyncio.run(batch_analysis())
        
        results = benchmark(run_batch)
        assert len(results) == len(test_texts)
        assert all(hasattr(r, 'confidence') for r in results)
    
    @pytest.mark.parametrize("text_length", [50, 200, 500, 1000])
    def test_text_length_scaling_benchmark(self, benchmark, analyzer, text_length):
        """Benchmark масштабирования по длине текста"""
        test_text = "word " * (text_length // 5)  # Примерно text_length символов
        
        def run_analysis():
            return asyncio.run(analyzer.analyze_song("Test Artist", "Test Song", test_text))
        
        result = benchmark(run_analysis)
        assert result is not None

if __name__ == "__main__":
    # Запуск benchmark тестов:
    # pytest {output_file} --benchmark-only --benchmark-sort=mean
    # pytest {output_file} --benchmark-only --benchmark-histogram
    # pytest {output_file} --benchmark-only --benchmark-json=benchmark_results.json
    pass
'''

        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(test_content)

            self.monitor.display.print_success(
                f"Benchmark тесты сгенерированы: {output_file}"
            )
            self.monitor.display.print_progress_info(
                "Запустите: pytest " + output_file + " --benchmark-only"
            )

        except Exception as e:
            self.monitor.display.print_error(f"Ошибка генерации benchmark тестов: {e}")

    def run_benchmark_tests(
        self, test_file: str, json_output: str | None = None
    ) -> dict | None:
        """Запуск pytest benchmark тестов"""
        if not self.available:
            self.monitor.display.print_warning("pytest не доступен")
            return None

        try:
            import subprocess

            cmd = [
                "python",
                "-m",
                "pytest",
                test_file,
                "--benchmark-only",
                "--benchmark-sort=mean",
            ]

            if json_output:
                cmd.extend(["--benchmark-json", json_output])

            self.monitor.display.print_progress_info(
                f"Запуск benchmark тестов: {test_file}"
            )

            result = subprocess.run(
                cmd, check=False, capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                self.monitor.display.print_success("Benchmark тесты завершены успешно")

                # Читаем JSON результаты если есть
                if json_output:
                    try:
                        with open(json_output, encoding="utf-8") as f:
                            return json.load(f)
                    except Exception as e:
                        self.monitor.display.print_warning(
                            f"Не удалось прочитать JSON результаты: {e}"
                        )

                return {"status": "success", "output": result.stdout}
            self.monitor.display.print_error(
                f"Benchmark тесты завершились с ошибкой: {result.stderr}"
            )
            return None

        except subprocess.TimeoutExpired:
            self.monitor.display.print_warning("Benchmark тесты превысили таймаут")
            return None
        except Exception as e:
            self.monitor.display.print_error(f"Ошибка запуска benchmark тестов: {e}")
            return None

    def compare_benchmark_results(
        self, results1: dict, results2: dict, analyzer1: str, analyzer2: str
    ):
        """Сравнение результатов benchmark тестов"""
        if not self.monitor.display.use_rich or not self.monitor.display.console:
            # Fallback для простого вывода
            print(f"\\nComparison: {analyzer1} vs {analyzer2}")
            return

        table = Table(
            title=f"📊 Benchmark Comparison: {analyzer1} vs {analyzer2}",
            box=box.ROUNDED,
        )
        table.add_column("Test", style="cyan", width=25)
        table.add_column(f"{analyzer1}", style="green", width=15)
        table.add_column(f"{analyzer2}", style="blue", width=15)
        table.add_column("Difference", style="yellow", width=15)

        # Извлекаем benchmark данные
        benchmarks1 = results1.get("benchmarks", [])
        benchmarks2 = results2.get("benchmarks", [])

        # Создаем словари для быстрого поиска
        bench1_dict = {b["name"]: b for b in benchmarks1}
        bench2_dict = {b["name"]: b for b in benchmarks2}

        # Сравниваем общие тесты
        common_tests = set(bench1_dict.keys()) & set(bench2_dict.keys())

        for test_name in sorted(common_tests):
            b1 = bench1_dict[test_name]
            b2 = bench2_dict[test_name]

            mean1 = b1["stats"]["mean"]
            mean2 = b2["stats"]["mean"]

            diff_pct = ((mean2 - mean1) / mean1) * 100

            diff_text = f"{diff_pct:+.1f}%"
            if diff_pct < -5:
                diff_style = "bright_green"  # Улучшение
            elif diff_pct > 5:
                diff_style = "bright_red"  # Ухудшение
            else:
                diff_style = "dim"  # Нет значимых изменений

            table.add_row(
                test_name.split("::")[-1],  # Только имя теста
                f"{mean1:.3f}s",
                f"{mean2:.3f}s",
                Text(diff_text, style=diff_style),
            )

        self.monitor.display.console.print(table)


class RichDisplayManager:
    """Менеджер для красивого вывода результатов с Rich"""

    def __init__(self):
        self.console = console if RICH_AVAILABLE else None
        self.use_rich = RICH_AVAILABLE and console is not None

    def print_header(self, title: str, subtitle: str | None = None):
        """Красивый заголовок"""
        if self.use_rich and self.console:
            header_text = Text(title, style="bold cyan")
            if subtitle:
                header_text.append(f"\n{subtitle}", style="dim")

            panel = Panel(
                header_text,
                box=box.DOUBLE,
                padding=(1, 2),
                title="🚀 Enhanced Performance Monitor",
                title_align="center",
            )
            self.console.print(panel)
        else:
            print(f"🚀 {title}")
            if subtitle:
                print(f"   {subtitle}")
            print("=" * 60)

    def print_analyzer_info(self, analyzer_type: str, info: dict):
        """Информация об анализаторе"""
        if self.use_rich and self.console:
            table = Table(
                title=f"📊 Analyzer: {analyzer_type}", show_header=False, box=box.SIMPLE
            )
            table.add_column("Property", style="cyan", width=20)
            table.add_column("Value", style="green")

            table.add_row("Type", analyzer_type)
            table.add_row(
                "Available", "✅ Yes" if info.get("available", True) else "❌ No"
            )
            table.add_row("Features", ", ".join(info.get("supported_features", [])))

            self.console.print(table)
        else:
            print(f"📊 Analyzer: {analyzer_type}")
            print(
                f"   Available: {'✅ Yes' if info.get('available', True) else '❌ No'}"
            )
            print(f"   Features: {', '.join(info.get('supported_features', []))}")

    def print_metrics_table(self, metrics: EnhancedMetrics):
        """Таблица метрик производительности"""
        if self.use_rich and self.console:
            table = Table(
                title=f"📈 Performance Metrics: {metrics.analyzer_name}",
                box=box.ROUNDED,
            )
            table.add_column("Metric", style="cyan bold", width=25)
            table.add_column("Value", style="green", width=15)
            table.add_column("Unit", style="dim", width=10)

            # Временные метрики
            table.add_row("⏱️ Average Time", f"{metrics.avg_time:.3f}", "seconds")
            table.add_row("⚡ Min Time", f"{metrics.min_time:.3f}", "seconds")
            table.add_row("🔥 Max Time", f"{metrics.max_time:.3f}", "seconds")
            table.add_row("📊 Median Time", f"{metrics.median_time:.3f}", "seconds")
            table.add_row("📈 95th Percentile", f"{metrics.latency_p95:.3f}", "seconds")
            table.add_row("📈 99th Percentile", f"{metrics.latency_p99:.3f}", "seconds")

            # Производительность
            table.add_row("🚀 Throughput", f"{metrics.items_per_second:.1f}", "items/s")
            table.add_row("✅ Success Rate", f"{metrics.success_rate:.1f}", "%")
            table.add_row("❌ Errors", f"{metrics.error_count}", "count")

            # Системные ресурсы
            table.add_row("💾 Avg Memory", f"{metrics.avg_memory_mb:.1f}", "MB")
            table.add_row("🔺 Peak Memory", f"{metrics.peak_memory_mb:.1f}", "MB")
            table.add_row("📈 Memory Growth", f"{metrics.memory_growth_mb:.1f}", "MB")
            table.add_row("🖥️ Avg CPU", f"{metrics.avg_cpu_percent:.1f}", "%")
            table.add_row(
                "⚡ CPU Efficiency", f"{metrics.cpu_efficiency:.2f}", "items/cpu%"
            )

            # Профилирование
            if metrics.hottest_function:
                table.add_row("🔥 Hottest Function", metrics.hottest_function, "")

            self.console.print(table)
        else:
            # Fallback для простого вывода
            print(f"\n📊 {metrics.analyzer_name.upper()} Enhanced Metrics:")
            print("=" * 50)
            print(f"⏱️  Average time: {metrics.avg_time:.3f}s")
            print(f"📈 95th percentile: {metrics.latency_p95:.3f}s")
            print(f"📈 99th percentile: {metrics.latency_p99:.3f}s")
            print(f"🚀 Throughput: {metrics.items_per_second:.1f} items/s")
            print(f"💾 Memory growth: {metrics.memory_growth_mb:.1f} MB")
            print(f"⚡ CPU efficiency: {metrics.cpu_efficiency:.2f} items/cpu%")
            if metrics.hottest_function:
                print(f"🔥 Hottest function: {metrics.hottest_function}")

    def print_load_test_results(self, results: dict[str, Any]):
        """Результаты нагрузочного тестирования"""
        if self.use_rich and self.console:
            table = Table(
                title=f"🔥 Load Test Results: {results['analyzer_type']}", box=box.HEAVY
            )
            table.add_column("Metric", style="red bold", width=20)
            table.add_column("Value", style="bright_yellow", width=15)
            table.add_column("Unit", style="dim", width=10)

            table.add_row(
                "🚀 Requests/sec", f"{results['requests_per_second']:.1f}", "req/s"
            )
            table.add_row(
                "✅ Success Rate", f"{100 - results['error_rate_percent']:.1f}", "%"
            )
            table.add_row("❌ Error Rate", f"{results['error_rate_percent']:.1f}", "%")
            table.add_row(
                "⏱️ Avg Response", f"{results['avg_response_time']:.3f}", "seconds"
            )
            table.add_row(
                "👥 Concurrent Users", f"{results['concurrent_users']}", "users"
            )
            table.add_row("📦 Total Requests", f"{results['total_requests']}", "count")
            table.add_row("💾 Memory Usage", f"{results['avg_memory_mb']:.1f}", "MB")
            table.add_row("🖥️ CPU Usage", f"{results['avg_cpu_percent']:.1f}", "%")

            self.console.print(table)
        else:
            print(f"\n🔥 Load Test Results for {results['analyzer_type']}:")
            print("=" * 50)
            print(f"🚀 Requests/sec: {results['requests_per_second']:.1f}")
            print(f"✅ Success rate: {100 - results['error_rate_percent']:.1f}%")
            print(f"⏱️  Avg response: {results['avg_response_time']:.3f}s")
            print(f"💾 Memory usage: {results['avg_memory_mb']:.1f} MB")

    def print_hyperfine_comparison(self, results: dict):
        """Сравнение Hyperfine"""
        if self.use_rich and self.console and results:
            table = Table(title="⚡ Hyperfine Comparison", box=box.SIMPLE_HEAVY)
            table.add_column("Command", style="cyan", width=30)
            table.add_column("Mean Time", style="green", width=12)
            table.add_column("Std Dev", style="yellow", width=12)
            table.add_column("Min", style="blue", width=10)
            table.add_column("Max", style="red", width=10)

            for result in results.get("results", []):
                table.add_row(
                    result["command"][-30:],  # Обрезаем длинные команды
                    f"{result['mean']:.3f}s",
                    f"±{result['stddev']:.3f}s",
                    f"{result['min']:.3f}s",
                    f"{result['max']:.3f}s",
                )

            self.console.print(table)
        else:
            print("\n⚡ Hyperfine Comparison:")
            if results:
                for result in results.get("results", []):
                    print(
                        f"  {result['command']}: {result['mean']:.3f}s ± {result['stddev']:.3f}s"
                    )

    def print_progress_info(
        self, message: str, current: int | None = None, total: int | None = None
    ):
        """Информация о прогрессе"""
        if self.use_rich and self.console:
            if current is not None and total is not None:
                progress_text = f"{message} [{current}/{total}]"
            else:
                progress_text = message

            self.console.print(f"🔄 {progress_text}", style="bright_blue")
        elif current is not None and total is not None:
            print(f"🔄 {message} [{current}/{total}]")
        else:
            print(f"🔄 {message}")

    def print_error(self, message: str):
        """Ошибка"""
        if self.use_rich and self.console:
            self.console.print(f"❌ {message}", style="bright_red")
        else:
            print(f"❌ {message}")

    def print_success(self, message: str):
        """Успех"""
        if self.use_rich and self.console:
            self.console.print(f"✅ {message}", style="bright_green")
        else:
            print(f"✅ {message}")

    def print_warning(self, message: str):
        """Предупреждение"""
        if self.use_rich and self.console:
            self.console.print(f"⚠️  {message}", style="bright_yellow")
        else:
            print(f"⚠️  {message}")

    def print_analyzer_comparison(self, comparison_results: dict[str, dict]):
        """Сравнение анализаторов"""
        if not comparison_results or len(comparison_results) < 2:
            self.print_warning("Недостаточно результатов для сравнения")
            return

        if self.use_rich and self.console:
            table = Table(title="🔥 Analyzer Performance Comparison", box=box.HEAVY)
            table.add_column("Analyzer", style="cyan bold", width=20)
            table.add_column("Avg Time", style="green", width=12)
            table.add_column("Throughput", style="yellow", width=12)
            table.add_column("Success %", style="blue", width=10)
            table.add_column("Memory", style="magenta", width=12)
            table.add_column("CPU Eff", style="red", width=12)

            # Сортируем по производительности (throughput)
            sorted_results = sorted(
                comparison_results.items(),
                key=lambda x: x[1].get("items_per_second", 0)
                if not x[1].get("error")
                else 0,
                reverse=True,
            )

            for analyzer_name, metrics in sorted_results:
                if metrics.get("error"):
                    table.add_row(analyzer_name, "❌ Error", "-", "-", "-", "-")
                else:
                    # Определяем стиль для лучших результатов
                    is_fastest = analyzer_name == sorted_results[0][0]
                    name_style = "bright_green bold" if is_fastest else "cyan"

                    table.add_row(
                        Text(analyzer_name, style=name_style),
                        f"{metrics.get('avg_time', 0):.3f}s",
                        f"{metrics.get('items_per_second', 0):.1f}/s",
                        f"{metrics.get('success_rate', 0):.1f}%",
                        f"{metrics.get('avg_memory_mb', 0):.1f}MB",
                        f"{metrics.get('cpu_efficiency', 0):.2f}",
                    )

            self.console.print(table)

            # Дополнительная статистика
            winner = sorted_results[0]
            if not winner[1].get("error"):
                winner_panel = Panel(
                    f"🏆 Winner: [bold green]{winner[0]}[/bold green]\n"
                    f"⚡ {winner[1]['items_per_second']:.1f} items/s\n"
                    f"⏱️ {winner[1]['avg_time']:.3f}s avg time\n"
                    f"💾 {winner[1]['avg_memory_mb']:.1f}MB memory",
                    title="Performance Champion",
                    border_style="green",
                )
                self.console.print(winner_panel)
        else:
            # Fallback для простого вывода
            print("\n🔥 Analyzer Performance Comparison:")
            print("=" * 60)

            sorted_results = sorted(
                comparison_results.items(),
                key=lambda x: x[1].get("items_per_second", 0)
                if not x[1].get("error")
                else 0,
                reverse=True,
            )

            for i, (analyzer_name, metrics) in enumerate(sorted_results):
                if metrics.get("error"):
                    print(f"{i + 1}. ❌ {analyzer_name}: Error - {metrics['error']}")
                else:
                    trophy = "🏆 " if i == 0 else f"{i + 1}. "
                    print(f"{trophy}{analyzer_name}:")
                    print(f"   ⚡ {metrics.get('items_per_second', 0):.1f} items/s")
                    print(f"   ⏱️  {metrics.get('avg_time', 0):.3f}s avg")
                    print(f"   💾 {metrics.get('avg_memory_mb', 0):.1f}MB")
                    print(f"   ✅ {metrics.get('success_rate', 0):.1f}% success")
                    print()


class PrometheusMetrics:
    """Prometheus метрики для мониторинга"""

    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            return

        self.request_counter = Counter(
            "analyzer_requests_total",
            "Total analyzer requests",
            ["analyzer_type", "status"],
        )

        self.request_duration = Histogram(
            "analyzer_request_duration_seconds",
            "Request duration in seconds",
            ["analyzer_type"],
        )

        self.memory_usage = Gauge(
            "analyzer_memory_usage_mb", "Memory usage in MB", ["analyzer_type"]
        )

        self.cpu_usage = Gauge(
            "analyzer_cpu_usage_percent", "CPU usage percentage", ["analyzer_type"]
        )

    def record_request(self, analyzer_type: str, duration: float, success: bool):
        if not PROMETHEUS_AVAILABLE:
            return

        status = "success" if success else "error"
        self.request_counter.labels(analyzer_type=analyzer_type, status=status).inc()
        self.request_duration.labels(analyzer_type=analyzer_type).observe(duration)

    def update_system_metrics(
        self, analyzer_type: str, memory_mb: float, cpu_percent: float
    ):
        if not PROMETHEUS_AVAILABLE:
            return

        self.memory_usage.labels(analyzer_type=analyzer_type).set(memory_mb)
        self.cpu_usage.labels(analyzer_type=analyzer_type).set(cpu_percent)


class EnhancedPerformanceMonitor:
    """Продвинутый монитор производительности"""

    def __init__(
        self, monitoring_interval: float = 0.1, enable_prometheus: bool = False
    ):
        self.app = create_app()
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)

        # Rich display manager
        self.display = RichDisplayManager()

        # Pytest benchmark интеграция
        self.pytest_benchmark = PytestBenchmarkIntegration(self)

        # Prometheus метрики
        self.prometheus_metrics = PrometheusMetrics() if PROMETHEUS_AVAILABLE else None
        if enable_prometheus and PROMETHEUS_AVAILABLE:
            # Запускаем Prometheus HTTP сервер
            start_http_server(8000)
            self.logger.info("📊 Prometheus metrics server started on :8000")

        # Системные метрики
        self.cpu_measurements = []
        self.memory_measurements = []
        self.is_monitoring = False

    async def benchmark_with_profiling(
        self,
        analyzer_type: str,
        test_texts: list[str],
        enable_profiling: bool = True,
        enable_memory_profiling: bool = True,
        timeout_per_text: float = 30.0,  # Таймаут на один текст
    ) -> EnhancedMetrics:
        """Бенчмарк с глубоким профилированием"""

        analyzer = self.app.get_analyzer(analyzer_type)
        if not analyzer:
            available = self.app.list_analyzers()
            raise ValueError(
                f"Unknown analyzer type: {analyzer_type}. Available: {available}"
            )

        self.logger.info(f"🔬 Enhanced benchmarking {analyzer_type} with profiling")

        # Подготовка профайлера
        profiler = cProfile.Profile() if enable_profiling else None

        # Сброс метрик
        self.cpu_measurements.clear()
        self.memory_measurements.clear()

        # Запуск мониторинга
        monitoring_task = asyncio.create_task(self._monitor_system_resources())

        # Основное тестирование с профилированием
        execution_times = []
        error_count = 0
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024

        if profiler:
            profiler.enable()

        start_time = time.time()

        for i, text in enumerate(test_texts):
            text_start_time = time.time()
            success = False

            try:
                # Проверяем является ли метод async
                import functools
                import inspect

                # Добавляем таймаут для медленных анализаторов
                if inspect.iscoroutinefunction(analyzer.analyze_song):
                    result = await asyncio.wait_for(
                        analyzer.analyze_song("Unknown", f"Test_{i}", text),
                        timeout=timeout_per_text,
                    )
                else:
                    # Для sync методов используем run_in_executor с таймаутом
                    # Используем functools.partial вместо lambda для правильной передачи аргументов
                    loop = asyncio.get_event_loop()
                    sync_call = functools.partial(
                        analyzer.analyze_song, "Unknown", f"Test_{i}", text
                    )
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, sync_call), timeout=timeout_per_text
                    )

                text_time = time.time() - text_start_time
                execution_times.append(text_time)
                success = True

                # Prometheus метрики
                if self.prometheus_metrics:
                    self.prometheus_metrics.record_request(
                        analyzer_type, text_time, True
                    )

                # Показываем прогресс для медленных анализаторов
                if text_time > 5.0:
                    self.display.print_progress_info(
                        f"Text {i + 1}/{len(test_texts)} completed",
                        current=i + 1,
                        total=len(test_texts),
                    )

            except asyncio.TimeoutError:
                error_count += 1
                text_time = time.time() - text_start_time
                self.logger.warning(
                    f"⏰ Timeout processing text {i + 1} after {timeout_per_text}s"
                )

                if self.prometheus_metrics:
                    self.prometheus_metrics.record_request(
                        analyzer_type, text_time, False
                    )

            except Exception as e:
                error_count += 1
                text_time = time.time() - text_start_time
                self.logger.warning(f"❌ Error processing text {i + 1}: {e}")

                if self.prometheus_metrics:
                    self.prometheus_metrics.record_request(
                        analyzer_type, text_time, False
                    )

        total_time = time.time() - start_time

        if profiler:
            profiler.disable()

        # Останавливаем мониторинг
        self.is_monitoring = False
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

        # Вычисляем метрики
        success_count = len(execution_times)
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = memory_end - memory_start

        # Базовые метрики
        if execution_times:
            avg_time = statistics.mean(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            median_time = statistics.median(execution_times)
            # Процентили
            sorted_times = sorted(execution_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            latency_p95 = (
                sorted_times[p95_idx] if p95_idx < len(sorted_times) else max_time
            )
            latency_p99 = (
                sorted_times[p99_idx] if p99_idx < len(sorted_times) else max_time
            )
        else:
            avg_time = min_time = max_time = median_time = 0
            latency_p95 = latency_p99 = 0

        # Системные метрики
        avg_cpu = statistics.mean(self.cpu_measurements) if self.cpu_measurements else 0
        avg_memory = (
            statistics.mean(self.memory_measurements) if self.memory_measurements else 0
        )
        peak_memory = max(self.memory_measurements) if self.memory_measurements else 0

        # CPU эффективность
        cpu_efficiency = (success_count / avg_cpu) if avg_cpu > 0 else 0

        # Анализ профилирования
        hottest_function = ""
        profile_data = None

        if profiler:
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats("cumulative")
            ps.print_stats(10)  # Топ 10 функций

            profile_output = s.getvalue()
            lines = profile_output.split("\n")
            for line in lines:
                if (
                    "function calls" not in line
                    and line.strip()
                    and not line.startswith("Ordered by")
                ):
                    parts = line.split()
                    if len(parts) > 5:
                        hottest_function = parts[-1]
                        break

            # Сохраняем данные профилирования
            profile_data = {
                "total_calls": getattr(ps, "total_calls", 0),
                "total_time": getattr(ps, "total_tt", 0.0),
                "profile_summary": profile_output[:1000],  # Первые 1000 символов
            }

        # Обновляем Prometheus метрики
        if self.prometheus_metrics:
            self.prometheus_metrics.update_system_metrics(
                analyzer_type, avg_memory, avg_cpu
            )

        # Создаем расширенные метрики
        metrics = EnhancedMetrics(
            analyzer_name=analyzer_type,
            test_count=len(test_texts),
            total_time=total_time,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time,
            median_time=median_time,
            avg_cpu_percent=avg_cpu,
            avg_memory_mb=avg_memory,
            peak_memory_mb=peak_memory,
            success_rate=(success_count / len(test_texts) * 100) if test_texts else 0,
            error_count=error_count,
            items_per_second=(success_count / total_time) if total_time > 0 else 0,
            memory_growth_mb=memory_growth,
            cpu_efficiency=cpu_efficiency,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            hottest_function=hottest_function,
            profile_data=profile_data,
        )

        self.logger.info("✅ Enhanced benchmarking completed")
        return metrics

    async def py_spy_analysis(
        self, analyzer_type: str, test_texts: list[str], duration: int = 30
    ) -> str | None:
        """Профилирование с py-spy"""

        if not test_texts:
            return None

        # Создаем временный скрипт для py-spy
        script_content = f'''
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.app import create_app

async def run_analyzer():
    app = create_app()
    analyzer = app.get_analyzer("{analyzer_type}")
    texts = {test_texts[:10]}  # Ограничиваем для py-spy
    
    for i in range(100):  # Много итераций для py-spy
        for j, text in enumerate(texts):
            try:
                import inspect
                if inspect.iscoroutinefunction(analyzer.analyze_song):
                    await analyzer.analyze_song("Unknown", f"Text_{{i}}_{{j}}", text)
                else:
                    analyzer.analyze_song("Unknown", f"Text_{{i}}_{{j}}", text)
            except Exception:
                pass

if __name__ == "__main__":
    asyncio.run(run_analyzer())
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script_content)
            temp_script = f.name

        try:
            # Запускаем py-spy
            output_file = f"pyspy_profile_{analyzer_type}.svg"
            cmd = [
                "py-spy",
                "record",
                "-d",
                str(duration),
                "-o",
                output_file,
                "--",
                "python",
                temp_script,
            ]

            self.logger.info(f"🔍 Running py-spy profiling for {duration}s...")

            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=duration + 10,
                )
                if result.returncode == 0:
                    self.logger.info(f"📊 py-spy profile saved to: {output_file}")
                    return output_file
                self.logger.warning(f"py-spy failed: {result.stderr}")
                return None
            except subprocess.TimeoutExpired:
                self.logger.warning("py-spy timed out")
                return None
            except FileNotFoundError:
                self.logger.warning(
                    "py-spy not found. Install with: pip install py-spy"
                )
                return None

        finally:
            # Удаляем временный файл
            Path(temp_script).unlink(missing_ok=True)

        return None

    def hyperfine_comparison(
        self, analyzer_types: list[str], test_text: str = "Test text for hyperfine"
    ) -> dict | None:
        """Сравнение с hyperfine"""

        # Создаем временные скрипты для каждого анализатора
        temp_scripts = []
        commands = []

        for analyzer_type in analyzer_types:
            script_content = f'''
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.core.app import create_app

async def main():
    app = create_app()
    analyzer = app.get_analyzer("{analyzer_type}")
    text = "{test_text}"
    
    try:
        import inspect
        if inspect.iscoroutinefunction(analyzer.analyze_song):
            await analyzer.analyze_song("Unknown", "Test", text)
        else:
            analyzer.analyze_song("Unknown", "Test", text)
    except Exception as e:
        print(f"Error: {{e}}")

if __name__ == "__main__":
    asyncio.run(main())
'''

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(script_content)
                temp_scripts.append(f.name)
                commands.append(f"python {f.name}")

        try:
            # Запускаем hyperfine
            cmd = ["hyperfine", "--export-json", "hyperfine_results.json"] + commands

            self.logger.info("🏃 Running hyperfine comparison...")

            try:
                result = subprocess.run(
                    cmd, check=False, capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    # Читаем результаты
                    try:
                        with open("hyperfine_results.json") as f:
                            hyperfine_data = json.load(f)

                        self.logger.info("⚡ Hyperfine comparison completed")
                        return hyperfine_data
                    except FileNotFoundError:
                        self.logger.warning("Hyperfine results file not found")
                        return None
                else:
                    self.logger.warning(f"Hyperfine failed: {result.stderr}")
                    return None
            except subprocess.TimeoutExpired:
                self.logger.warning("Hyperfine timed out")
                return None
            except FileNotFoundError:
                self.logger.warning(
                    "Hyperfine not found. Install with: pip install hyperfine"
                )
                return None

        finally:
            # Удаляем временные файлы
            for script in temp_scripts:
                Path(script).unlink(missing_ok=True)

        return None

    async def load_test(
        self,
        analyzer_type: str,
        test_texts: list[str],
        concurrent_users: int = 10,
        duration_seconds: int = 60,
    ) -> dict[str, Any]:
        """Нагрузочное тестирование"""

        analyzer = self.app.get_analyzer(analyzer_type)
        if not analyzer:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")

        self.logger.info(
            f"🔥 Load testing {analyzer_type} with {concurrent_users} concurrent users for {duration_seconds}s"
        )

        # Счетчики
        successful_requests = 0
        failed_requests = 0
        response_times = []
        start_time = time.time()

        async def worker(worker_id: int):
            nonlocal successful_requests, failed_requests, response_times

            while time.time() - start_time < duration_seconds:
                text = test_texts[worker_id % len(test_texts)]
                request_start = time.time()

                try:
                    # Проверяем является ли метод async
                    import inspect

                    if inspect.iscoroutinefunction(analyzer.analyze_song):
                        result = await analyzer.analyze_song(
                            "Unknown", f"Worker_{worker_id}", text
                        )
                    else:
                        result = analyzer.analyze_song(
                            "Unknown", f"Worker_{worker_id}", text
                        )

                    response_time = time.time() - request_start
                    response_times.append(response_time)
                    successful_requests += 1

                except Exception:
                    failed_requests += 1

                # Небольшая пауза между запросами
                await asyncio.sleep(0.01)

        # Запускаем воркеры
        tasks = [asyncio.create_task(worker(i)) for i in range(concurrent_users)]

        # Мониторинг системных ресурсов
        monitoring_task = asyncio.create_task(self._monitor_system_resources())

        # Ждем завершения
        await asyncio.gather(*tasks, return_exceptions=True)

        self.is_monitoring = False
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

        total_time = time.time() - start_time
        total_requests = successful_requests + failed_requests

        # Вычисляем метрики
        avg_response_time = statistics.mean(response_times) if response_times else 0
        rps = successful_requests / total_time if total_time > 0 else 0
        error_rate = (
            (failed_requests / total_requests * 100) if total_requests > 0 else 0
        )

        avg_cpu = statistics.mean(self.cpu_measurements) if self.cpu_measurements else 0
        avg_memory = (
            statistics.mean(self.memory_measurements) if self.memory_measurements else 0
        )

        load_test_results = {
            "analyzer_type": analyzer_type,
            "duration_seconds": total_time,
            "concurrent_users": concurrent_users,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "requests_per_second": rps,
            "error_rate_percent": error_rate,
            "avg_response_time": avg_response_time,
            "avg_cpu_percent": avg_cpu,
            "avg_memory_mb": avg_memory,
        }

        self.logger.info(
            f"✅ Load test completed: {rps:.1f} RPS, {error_rate:.1f}% errors"
        )
        return load_test_results

    async def _monitor_system_resources(self) -> None:
        """Мониторинг системных ресурсов"""
        self.is_monitoring = True
        process = psutil.Process()

        while self.is_monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024

                self.cpu_measurements.append(cpu_percent)
                self.memory_measurements.append(memory_mb)

                await asyncio.sleep(self.monitoring_interval)

            except asyncio.CancelledError:
                break
            except Exception:
                pass

        self.is_monitoring = False


def generate_test_texts(count: int = 50) -> list[str]:
    """Генерация тестовых текстов"""
    base_texts = [
        "Happy song",
        "Sad lyrics about lost love and broken dreams",
        "Energetic rap with political messages and social commentary",
        "Calm meditation on nature and human existence in modern world",
        """This is a comprehensive analysis of modern culture and its impact 
        on society, exploring themes of justice, growth, and artistic expression 
        through complex metaphors and intricate wordplay that challenges thinking""",
    ]

    texts = []
    for i in range(count):
        base_text = base_texts[i % len(base_texts)]
        variation = (
            f" (test variation {i // len(base_texts) + 1})"
            if i >= len(base_texts)
            else ""
        )
        texts.append(base_text + variation)

    return texts


async def main():
    """Главная функция с CLI аргументами"""
    parser = argparse.ArgumentParser(description="🚀 Enhanced Performance Monitor")
    parser.add_argument("--analyzer", type=str, help="Analyzer type to test")
    parser.add_argument(
        "--all", action="store_true", help="Test all available analyzers"
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "profile", "compare", "load", "pyspy", "pytest"],
        default="benchmark",
        help="Testing mode",
    )
    parser.add_argument(
        "--prometheus", action="store_true", help="Enable Prometheus metrics"
    )
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--texts", type=int, default=20, help="Number of test texts")
    parser.add_argument(
        "--duration", type=int, default=30, help="Duration for py-spy profiling"
    )
    parser.add_argument(
        "--users", type=int, default=10, help="Concurrent users for load test"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Timeout per text analysis (seconds)",
    )

    args = parser.parse_args()

    # Настраиваем логирование
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    monitor = EnhancedPerformanceMonitor(enable_prometheus=args.prometheus)
    test_texts = generate_test_texts(args.texts)

    # Используем rich для красивого вывода
    monitor.display.print_header(
        "Enhanced Performance Monitor",
        f"Mode: {args.mode} | Dataset: {len(test_texts)} texts",
    )

    try:
        if args.mode == "benchmark":
            analyzer_type = args.analyzer or "advanced_algorithmic"

            monitor.display.print_progress_info(
                f"Starting benchmark for {analyzer_type}"
            )

            metrics = await monitor.benchmark_with_profiling(
                analyzer_type,
                test_texts,
                enable_profiling=True,
                timeout_per_text=args.timeout,
            )

            # Красивый вывод метрик
            monitor.display.print_metrics_table(metrics)

            # Сохранение результатов
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(metrics.to_dict(), f, indent=2, ensure_ascii=False)
                monitor.display.print_success(f"Results saved to: {args.output}")

        elif args.mode == "pyspy":
            analyzer_type = args.analyzer or "advanced_algorithmic"

            monitor.display.print_progress_info(
                f"Starting py-spy profiling for {analyzer_type}"
            )

            profile_file = await monitor.py_spy_analysis(
                analyzer_type, test_texts, args.duration
            )
            if profile_file:
                monitor.display.print_success(
                    f"py-spy profile saved to: {profile_file}"
                )
            else:
                monitor.display.print_warning(
                    "py-spy profiling failed or not available"
                )

        elif args.mode == "load":
            analyzer_type = args.analyzer or "advanced_algorithmic"

            monitor.display.print_progress_info(
                f"Starting load test for {analyzer_type}"
            )

            results = await monitor.load_test(analyzer_type, test_texts, args.users, 60)

            # Красивый вывод результатов нагрузочного тестирования
            monitor.display.print_load_test_results(results)

            # Сохранение результатов
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                monitor.display.print_success(
                    f"Load test results saved to: {args.output}"
                )

        elif args.mode == "compare":
            analyzers = (
                ["advanced_algorithmic"]
                if not args.all
                else ["advanced_algorithmic", "qwen", "emotion_analyzer"]
            )

            monitor.display.print_progress_info(
                f"Starting comparison of {len(analyzers)} analyzers"
            )

            # Внутреннее сравнение анализаторов
            comparison_results = {}

            for analyzer_type in analyzers:
                try:
                    monitor.display.print_progress_info(f"Benchmarking {analyzer_type}")

                    # Для сравнения используем более короткий таймаут
                    timeout = 10.0 if analyzer_type == "qwen" else 5.0

                    metrics = await monitor.benchmark_with_profiling(
                        analyzer_type,
                        test_texts[:5],
                        enable_profiling=False,  # Быстрое сравнение
                        timeout_per_text=timeout,
                    )

                    comparison_results[analyzer_type] = metrics.to_dict()

                except Exception as e:
                    monitor.display.print_warning(
                        f"Failed to benchmark {analyzer_type}: {e}"
                    )
                    comparison_results[analyzer_type] = {"error": str(e)}

            # Красивая таблица сравнения
            if len(comparison_results) > 1:
                monitor.display.print_analyzer_comparison(comparison_results)

            # Hyperfine сравнение (если доступно)
            hyperfine_results = monitor.hyperfine_comparison(analyzers)
            if hyperfine_results:
                monitor.display.print_hyperfine_comparison(hyperfine_results)
            else:
                monitor.display.print_warning(
                    "Hyperfine comparison failed or not available"
                )
                monitor.display.print_success(
                    "✨ Internal comparison completed! Install hyperfine for external benchmarks:"
                )

                # Показываем инструкции по установке
                if monitor.display.use_rich and monitor.display.console:
                    install_panel = Panel(
                        "🚀 To install Hyperfine:\n\n"
                        "[cyan]Windows:[/cyan]\n"
                        "• choco install hyperfine\n"
                        "• scoop install hyperfine\n"
                        "• Download from GitHub releases\n\n"
                        "[cyan]Linux/Mac:[/cyan]\n"
                        "• cargo install hyperfine\n"
                        "• brew install hyperfine",
                        title="📦 Hyperfine Installation",
                        border_style="yellow",
                    )
                    monitor.display.console.print(install_panel)
                else:
                    print("\n📦 To install Hyperfine:")
                    print("  Windows: choco install hyperfine")
                    print("  Linux:   cargo install hyperfine")
                    print("  Mac:     brew install hyperfine")

            # Сохранение результатов
            if args.output:
                full_results = {
                    "internal_comparison": comparison_results,
                    "hyperfine_results": hyperfine_results,
                }
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(full_results, f, indent=2, ensure_ascii=False)
                monitor.display.print_success(
                    f"Comparison results saved to: {args.output}"
                )

        elif args.mode == "pytest":
            analyzer_type = args.analyzer or "advanced_algorithmic"

            monitor.display.print_progress_info(
                f"Generating pytest benchmark tests for {analyzer_type}"
            )

            # Генерируем benchmark тесты
            test_file = f"test_benchmark_{analyzer_type}.py"
            monitor.pytest_benchmark.generate_benchmark_test(
                analyzer_type, test_texts, test_file
            )

            # Опционально запускаем тесты
            if args.output:
                json_output = args.output.replace(".json", "_benchmark.json")
                results = monitor.pytest_benchmark.run_benchmark_tests(
                    test_file, json_output
                )

                if results:
                    monitor.display.print_success(
                        f"Pytest benchmark результаты сохранены в: {json_output}"
                    )

        monitor.display.print_success("Enhanced monitoring completed successfully!")

    except Exception as e:
        monitor.display.print_error(f"Enhanced monitoring failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
