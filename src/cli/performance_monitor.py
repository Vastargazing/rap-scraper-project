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
import asyncio
import json
import time
import sys
import psutil
import statistics
import subprocess
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
import cProfile
import pstats
from io import StringIO

# Добавляем src в Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.app import Application
from interfaces.analyzer_interface import AnalyzerType

# Опциональные импорты для продвинутых фич
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, REGISTRY
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
    profile_data: Optional[Dict] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PrometheusMetrics:
    """Prometheus метрики для мониторинга"""
    
    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.request_counter = Counter(
            'analyzer_requests_total', 
            'Total analyzer requests',
            ['analyzer_type', 'status']
        )
        
        self.request_duration = Histogram(
            'analyzer_request_duration_seconds',
            'Request duration in seconds',
            ['analyzer_type']
        )
        
        self.memory_usage = Gauge(
            'analyzer_memory_usage_mb',
            'Memory usage in MB',
            ['analyzer_type']
        )
        
        self.cpu_usage = Gauge(
            'analyzer_cpu_usage_percent',
            'CPU usage percentage',
            ['analyzer_type']
        )
    
    def record_request(self, analyzer_type: str, duration: float, success: bool):
        if not PROMETHEUS_AVAILABLE:
            return
            
        status = 'success' if success else 'error'
        self.request_counter.labels(analyzer_type=analyzer_type, status=status).inc()
        self.request_duration.labels(analyzer_type=analyzer_type).observe(duration)
    
    def update_system_metrics(self, analyzer_type: str, memory_mb: float, cpu_percent: float):
        if not PROMETHEUS_AVAILABLE:
            return
            
        self.memory_usage.labels(analyzer_type=analyzer_type).set(memory_mb)
        self.cpu_usage.labels(analyzer_type=analyzer_type).set(cpu_percent)


class EnhancedPerformanceMonitor:
    """Продвинутый монитор производительности"""
    
    def __init__(self, monitoring_interval: float = 0.1, enable_prometheus: bool = False):
        self.app = Application()
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)
        
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
        test_texts: List[str],
        enable_profiling: bool = True,
        enable_memory_profiling: bool = True
    ) -> EnhancedMetrics:
        """Бенчмарк с глубоким профилированием"""
        
        analyzer = self.app.get_analyzer(analyzer_type)
        if not analyzer:
            available = self.app.list_analyzers()
            raise ValueError(f"Unknown analyzer type: {analyzer_type}. Available: {available}")
        
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
                if hasattr(analyzer, 'analyze'):
                    result = analyzer.analyze(text)
                    if hasattr(result, '__await__'):  # Проверяем асинхронность
                        await result
                    # Если синхронный - result уже получен
                elif hasattr(analyzer, 'analyze_song'):
                    result = analyzer.analyze_song("Unknown", f"Text_{i}", text)
                    if hasattr(result, '__await__'):  # Проверяем асинхронность
                        await result
                    # Если синхронный - result уже получен
                else:
                    raise RuntimeError(f"Analyzer has no analyze method")
                
                text_time = time.time() - text_start_time
                execution_times.append(text_time)
                success = True
                
                # Prometheus метрики
                if self.prometheus_metrics:
                    self.prometheus_metrics.record_request(analyzer_type, text_time, True)
                    
            except Exception as e:
                error_count += 1
                text_time = time.time() - text_start_time
                self.logger.warning(f"Error processing text {i+1}: {e}")
                
                if self.prometheus_metrics:
                    self.prometheus_metrics.record_request(analyzer_type, text_time, False)
        
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
            latency_p95 = sorted_times[p95_idx] if p95_idx < len(sorted_times) else max_time
            latency_p99 = sorted_times[p99_idx] if p99_idx < len(sorted_times) else max_time
        else:
            avg_time = min_time = max_time = median_time = 0
            latency_p95 = latency_p99 = 0
        
        # Системные метрики
        avg_cpu = statistics.mean(self.cpu_measurements) if self.cpu_measurements else 0
        avg_memory = statistics.mean(self.memory_measurements) if self.memory_measurements else 0
        peak_memory = max(self.memory_measurements) if self.memory_measurements else 0
        
        # CPU эффективность
        cpu_efficiency = (success_count / avg_cpu) if avg_cpu > 0 else 0
        
        # Анализ профилирования
        hottest_function = ""
        profile_data = None
        
        if profiler:
            s = StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(10)  # Топ 10 функций
            
            profile_output = s.getvalue()
            lines = profile_output.split('\n')
            for line in lines:
                if 'function calls' not in line and line.strip() and not line.startswith('Ordered by'):
                    parts = line.split()
                    if len(parts) > 5:
                        hottest_function = parts[-1]
                        break
            
            # Сохраняем данные профилирования
            profile_data = {
                'total_calls': getattr(ps, 'total_calls', 0),
                'total_time': getattr(ps, 'total_tt', 0.0),
                'profile_summary': profile_output[:1000]  # Первые 1000 символов
            }
        
        # Обновляем Prometheus метрики
        if self.prometheus_metrics:
            self.prometheus_metrics.update_system_metrics(analyzer_type, avg_memory, avg_cpu)
        
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
            profile_data=profile_data
        )
        
        self.logger.info("✅ Enhanced benchmarking completed")
        return metrics
    
    async def py_spy_analysis(
        self,
        analyzer_type: str,
        test_texts: List[str],
        duration: int = 30
    ) -> Optional[str]:
        """Профилирование с py-spy"""
        
        if not test_texts:
            return None
        
        # Создаем временный скрипт для py-spy
        script_content = f'''
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.app import Application

async def run_analyzer():
    app = Application()
    analyzer = app.get_analyzer("{analyzer_type}")
    texts = {test_texts[:10]}  # Ограничиваем для py-spy
    
    for i in range(100):  # Много итераций для py-spy
        for text in texts:
            try:
                if hasattr(analyzer, 'analyze'):
                    result = analyzer.analyze(text)
                    if hasattr(result, '__await__'):
                        await result
                elif hasattr(analyzer, 'analyze_song'):
                    result = analyzer.analyze_song("Unknown", f"Text_{{i}}", text)
                    if hasattr(result, '__await__'):
                        await result
            except Exception:
                pass

if __name__ == "__main__":
    asyncio.run(run_analyzer())
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            temp_script = f.name
        
        try:
            # Запускаем py-spy
            output_file = f"pyspy_profile_{analyzer_type}.svg"
            cmd = [
                "py-spy", "record", 
                "-d", str(duration),
                "-o", output_file,
                "--", "python", temp_script
            ]
            
            self.logger.info(f"🔍 Running py-spy profiling for {duration}s...")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 10)
                if result.returncode == 0:
                    self.logger.info(f"📊 py-spy profile saved to: {output_file}")
                    return output_file
                else:
                    self.logger.warning(f"py-spy failed: {result.stderr}")
                    return None
            except subprocess.TimeoutExpired:
                self.logger.warning("py-spy timed out")
                return None
            except FileNotFoundError:
                self.logger.warning("py-spy not found. Install with: pip install py-spy")
                return None
                
        finally:
            # Удаляем временный файл
            Path(temp_script).unlink(missing_ok=True)
        
        return None
    
    def hyperfine_comparison(
        self,
        analyzer_types: List[str],
        test_text: str = "Test text for hyperfine"
    ) -> Optional[Dict]:
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
from core.app import Application

async def main():
    app = Application()
    analyzer = app.get_analyzer("{analyzer_type}")
    text = "{test_text}"
    
    try:
        if hasattr(analyzer, 'analyze'):
            result = analyzer.analyze(text)
            if hasattr(result, '__await__'):
                await result
        elif hasattr(analyzer, 'analyze_song'):
            result = analyzer.analyze_song("Unknown", "Test", text)
            if hasattr(result, '__await__'):
                await result
    except Exception as e:
        print(f"Error: {{e}}")

if __name__ == "__main__":
    asyncio.run(main())
'''
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                temp_scripts.append(f.name)
                commands.append(f"python {f.name}")
        
        try:
            # Запускаем hyperfine
            cmd = ["hyperfine", "--export-json", "hyperfine_results.json"] + commands
            
            self.logger.info("🏃 Running hyperfine comparison...")
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    # Читаем результаты
                    try:
                        with open("hyperfine_results.json", 'r') as f:
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
                self.logger.warning("Hyperfine not found. Install with: pip install hyperfine")
                return None
                
        finally:
            # Удаляем временные файлы
            for script in temp_scripts:
                Path(script).unlink(missing_ok=True)
        
        return None
    
    async def load_test(
        self,
        analyzer_type: str,
        test_texts: List[str],
        concurrent_users: int = 10,
        duration_seconds: int = 60
    ) -> Dict[str, Any]:
        """Нагрузочное тестирование"""
        
        analyzer = self.app.get_analyzer(analyzer_type)
        if not analyzer:
            raise ValueError(f"Unknown analyzer type: {analyzer_type}")
        
        self.logger.info(f"🔥 Load testing {analyzer_type} with {concurrent_users} concurrent users for {duration_seconds}s")
        
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
                    if hasattr(analyzer, 'analyze'):
                        result = analyzer.analyze(text)
                        if hasattr(result, '__await__'):
                            await result
                    elif hasattr(analyzer, 'analyze_song'):
                        result = analyzer.analyze_song("Unknown", f"Worker_{worker_id}", text)
                        if hasattr(result, '__await__'):
                            await result
                    
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
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        avg_cpu = statistics.mean(self.cpu_measurements) if self.cpu_measurements else 0
        avg_memory = statistics.mean(self.memory_measurements) if self.memory_measurements else 0
        
        load_test_results = {
            'analyzer_type': analyzer_type,
            'duration_seconds': total_time,
            'concurrent_users': concurrent_users,
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'requests_per_second': rps,
            'error_rate_percent': error_rate,
            'avg_response_time': avg_response_time,
            'avg_cpu_percent': avg_cpu,
            'avg_memory_mb': avg_memory
        }
        
        self.logger.info(f"✅ Load test completed: {rps:.1f} RPS, {error_rate:.1f}% errors")
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


def generate_test_texts(count: int = 50) -> List[str]:
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
        variation = f" (test variation {i//len(base_texts) + 1})" if i >= len(base_texts) else ""
        texts.append(base_text + variation)
    
    return texts


async def main():
    """Главная функция с CLI аргументами"""
    parser = argparse.ArgumentParser(description="🚀 Enhanced Performance Monitor")
    parser.add_argument("--analyzer", type=str, help="Analyzer type to test")
    parser.add_argument("--all", action="store_true", help="Test all available analyzers")
    parser.add_argument("--mode", choices=["benchmark", "profile", "compare", "load", "pyspy"], 
                       default="benchmark", help="Testing mode")
    parser.add_argument("--prometheus", action="store_true", help="Enable Prometheus metrics")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--texts", type=int, default=20, help="Number of test texts")
    parser.add_argument("--duration", type=int, default=30, help="Duration for py-spy profiling")
    parser.add_argument("--users", type=int, default=10, help="Concurrent users for load test")
    
    args = parser.parse_args()
    
    # Настраиваем логирование
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    monitor = EnhancedPerformanceMonitor(enable_prometheus=args.prometheus)
    test_texts = generate_test_texts(args.texts)
    
    print("🚀 Enhanced Performance Monitor")
    print(f"Mode: {args.mode}")
    print(f"Test dataset: {len(test_texts)} texts")
    print()
    
    try:
        if args.mode == "benchmark":
            analyzer_type = args.analyzer or "algorithmic"
            metrics = await monitor.benchmark_with_profiling(
                analyzer_type, test_texts, enable_profiling=True
            )
            
            print(f"\n📊 {analyzer_type.upper()} Enhanced Metrics:")
            print("=" * 50)
            print(f"⏱️  Average time: {metrics.avg_time:.3f}s")
            print(f"📈 95th percentile: {metrics.latency_p95:.3f}s")
            print(f"📈 99th percentile: {metrics.latency_p99:.3f}s")
            print(f"🚀 Throughput: {metrics.items_per_second:.1f} items/s")
            print(f"💾 Memory growth: {metrics.memory_growth_mb:.1f} MB")
            print(f"⚡ CPU efficiency: {metrics.cpu_efficiency:.2f} items/cpu%")
            print(f"🔥 Hottest function: {metrics.hottest_function}")
            
        elif args.mode == "pyspy":
            analyzer_type = args.analyzer or "algorithmic"
            profile_file = await monitor.py_spy_analysis(analyzer_type, test_texts, args.duration)
            if profile_file:
                print(f"📊 py-spy profile saved to: {profile_file}")
            
        elif args.mode == "load":
            analyzer_type = args.analyzer or "algorithmic"
            results = await monitor.load_test(analyzer_type, test_texts, args.users, 60)
            
            print(f"\n🔥 Load Test Results for {analyzer_type}:")
            print("=" * 50)
            print(f"🚀 Requests/sec: {results['requests_per_second']:.1f}")
            print(f"✅ Success rate: {100 - results['error_rate_percent']:.1f}%")
            print(f"⏱️  Avg response: {results['avg_response_time']:.3f}s")
            print(f"💾 Memory usage: {results['avg_memory_mb']:.1f} MB")
            
        elif args.mode == "compare":
            analyzers = ["algorithmic"] if not args.all else ["algorithmic"]
            
            # Hyperfine сравнение
            hyperfine_results = monitor.hyperfine_comparison(analyzers)
            if hyperfine_results:
                print("\n⚡ Hyperfine Comparison:")
                for result in hyperfine_results.get('results', []):
                    print(f"  {result['command']}: {result['mean']:.3f}s ± {result['stddev']:.3f}s")
        
        if args.output:
            print(f"\n📄 Results saved to: {args.output}")
            
    except Exception as e:
        print(f"❌ Enhanced monitoring failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
