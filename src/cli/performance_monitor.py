#!/usr/bin/env python3
"""
Утилита для мониторинга производительности анализаторов
Измеряет скорость, точность и ресурсоемкость разных анализаторов
"""

import asyncio
import json
import time
import sys
import psutil
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

# Добавляем src в Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.app import Application
from interfaces.analyzer_interface import AnalyzerType


@dataclass
class PerformanceMetrics:
    """Метрики производительности анализатора"""
    analyzer_name: str
    test_count: int
    
    # Временные метрики
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
    
    # Метрики пропускной способности
    items_per_second: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PerformanceMonitor:
    """Монитор производительности анализаторов"""
    
    def __init__(self, monitoring_interval: float = 0.1):
        self.app = Application()
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(__name__)
        
        # Системные метрики
        self.cpu_measurements = []
        self.memory_measurements = []
        self.is_monitoring = False
    
    async def benchmark_analyzer(
        self,
        analyzer_type: str,
        test_texts: List[str],
        warmup_runs: int = 3
    ) -> PerformanceMetrics:
        """
        Бенчмарк анализатора
        
        Args:
            analyzer_type: Тип анализатора
            test_texts: Тексты для тестирования
            warmup_runs: Количество прогревочных запусков
        """
        
        # Получаем анализатор напрямую по строке
        analyzer = self.app.get_analyzer(analyzer_type)
        if not analyzer:
            available = self.app.list_analyzers()
            raise ValueError(f"Unknown analyzer type: {analyzer_type}. Available: {available}")
        
        self.logger.info(f"🔬 Benchmarking {analyzer_type} analyzer with {len(test_texts)} texts")
        
        # Прогревочные запуски
        if warmup_runs > 0:
            self.logger.info(f"🔥 Warmup runs: {warmup_runs}")
            warmup_text = test_texts[0] if test_texts else "Warmup text"
            for i in range(warmup_runs):
                try:
                    await analyzer.analyze(warmup_text)
                except Exception:
                    pass
        
        # Основное тестирование
        self.logger.info("📊 Starting performance measurement...")
        
        # Сброс метрик
        self.cpu_measurements.clear()
        self.memory_measurements.clear()
        
        # Запуск мониторинга системных ресурсов
        monitoring_task = asyncio.create_task(self._monitor_system_resources())
        
        # Выполняем тесты
        execution_times = []
        error_count = 0
        
        start_time = time.time()
        
        for i, text in enumerate(test_texts):
            self.logger.debug(f"Processing text {i+1}/{len(test_texts)}")
            
            text_start_time = time.time()
            try:
                # Адаптируем к разным API анализаторов
                if hasattr(analyzer, 'analyze'):
                    await analyzer.analyze(text)
                elif hasattr(analyzer, 'analyze_song'):
                    # Старый API - передаем фиктивные данные
                    analyzer.analyze_song("Unknown", f"Text_{i}", text)
                else:
                    raise RuntimeError(f"Analyzer has no analyze method")
                
                text_time = time.time() - text_start_time
                execution_times.append(text_time)
            except Exception as e:
                error_count += 1
                self.logger.warning(f"Error processing text {i+1}: {e}")
        
        total_time = time.time() - start_time
        
        # Останавливаем мониторинг
        self.is_monitoring = False
        await monitoring_task
        
        # Вычисляем метрики
        success_count = len(execution_times)
        
        if execution_times:
            avg_time = statistics.mean(execution_times)
            min_time = min(execution_times)
            max_time = max(execution_times)
            median_time = statistics.median(execution_times)
        else:
            avg_time = min_time = max_time = median_time = 0
        
        # Системные метрики
        avg_cpu = statistics.mean(self.cpu_measurements) if self.cpu_measurements else 0
        avg_memory = statistics.mean(self.memory_measurements) if self.memory_measurements else 0
        peak_memory = max(self.memory_measurements) if self.memory_measurements else 0
        
        # Создаем метрики
        metrics = PerformanceMetrics(
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
            items_per_second=(success_count / total_time) if total_time > 0 else 0
        )
        
        self.logger.info("✅ Benchmarking completed")
        return metrics
    
    async def compare_analyzers(
        self,
        analyzer_types: List[str],
        test_texts: List[str],
        output_file: Optional[str] = None
    ) -> Dict[str, PerformanceMetrics]:
        """Сравнение производительности нескольких анализаторов"""
        
        self.logger.info(f"🏁 Comparing {len(analyzer_types)} analyzers")
        
        results = {}
        
        for analyzer_type in analyzer_types:
            self.logger.info(f"Testing {analyzer_type}...")
            try:
                metrics = await self.benchmark_analyzer(analyzer_type, test_texts)
                results[analyzer_type] = metrics
                
                # Краткий отчет
                self.logger.info(f"  ⏱️  Avg time: {metrics.avg_time:.3f}s")
                self.logger.info(f"  📈 Success rate: {metrics.success_rate:.1f}%")
                self.logger.info(f"  🚀 Throughput: {metrics.items_per_second:.1f} items/s")
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark {analyzer_type}: {e}")
                continue
        
        # Создаем отчет сравнения
        comparison_report = self._create_comparison_report(results)
        
        # Сохраняем результаты
        if output_file:
            report_data = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'test_config': {
                    'analyzers': analyzer_types,
                    'test_count': len(test_texts),
                    'total_chars': sum(len(text) for text in test_texts)
                },
                'individual_results': {k: v.to_dict() for k, v in results.items()},
                'comparison': comparison_report
            }
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📄 Comparison report saved to: {output_file}")
        
        return results
    
    async def _monitor_system_resources(self) -> None:
        """Мониторинг системных ресурсов во время выполнения"""
        self.is_monitoring = True
        process = psutil.Process()
        
        while self.is_monitoring:
            try:
                # CPU (в процентах)
                cpu_percent = process.cpu_percent()
                self.cpu_measurements.append(cpu_percent)
                
                # Память (в MB)
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_measurements.append(memory_mb)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception:
                # Игнорируем ошибки мониторинга
                pass
    
    def _create_comparison_report(
        self,
        results: Dict[str, PerformanceMetrics]
    ) -> Dict[str, Any]:
        """Создание отчета сравнения анализаторов"""
        
        if not results:
            return {}
        
        # Сортируем по производительности
        by_speed = sorted(results.items(), key=lambda x: x[1].avg_time)
        by_throughput = sorted(results.items(), key=lambda x: x[1].items_per_second, reverse=True)
        by_success_rate = sorted(results.items(), key=lambda x: x[1].success_rate, reverse=True)
        by_memory = sorted(results.items(), key=lambda x: x[1].avg_memory_mb)
        
        comparison = {
            'fastest': {
                'analyzer': by_speed[0][0],
                'avg_time': by_speed[0][1].avg_time,
                'improvement_factor': by_speed[-1][1].avg_time / by_speed[0][1].avg_time if by_speed[0][1].avg_time > 0 else 1
            },
            'highest_throughput': {
                'analyzer': by_throughput[0][0],
                'items_per_second': by_throughput[0][1].items_per_second
            },
            'most_reliable': {
                'analyzer': by_success_rate[0][0],
                'success_rate': by_success_rate[0][1].success_rate
            },
            'most_memory_efficient': {
                'analyzer': by_memory[0][0],
                'avg_memory_mb': by_memory[0][1].avg_memory_mb
            },
            'rankings': {
                'by_speed': [(name, metrics.avg_time) for name, metrics in by_speed],
                'by_throughput': [(name, metrics.items_per_second) for name, metrics in by_throughput],
                'by_success_rate': [(name, metrics.success_rate) for name, metrics in by_success_rate],
                'by_memory': [(name, metrics.avg_memory_mb) for name, metrics in by_memory]
            }
        }
        
        return comparison


def generate_test_texts(count: int = 50) -> List[str]:
    """Генерация тестовых текстов разной длины и сложности"""
    
    base_texts = [
        # Короткие тексты
        "Happy song",
        "Sad lyrics",
        "Love ballad",
        "Angry rap",
        "Chill vibes",
        
        # Средние тексты
        "This is a beautiful song about love and happiness in life",
        "Dark and mysterious lyrics expressing deep emotional pain and sorrow",
        "Uplifting and motivational text encouraging people to follow their dreams",
        "Aggressive and intense rap lyrics with strong political messages",
        "Calm and peaceful meditation on nature and human existence",
        
        # Длинные тексты
        """This is a comprehensive analysis of modern hip-hop culture and its impact 
        on society, exploring themes of social justice, personal growth, and artistic expression 
        through complex metaphors and intricate wordplay that challenges conventional thinking 
        about music and its role in contemporary discourse.""",
        
        """A deeply emotional ballad that tells the story of loss, recovery, and finding 
        hope in the darkest moments of life, weaving together personal experiences with 
        universal themes of human resilience and the power of love to overcome adversity 
        and transform our understanding of what it means to be truly alive."""
    ]
    
    # Расширяем до нужного количества
    texts = []
    for i in range(count):
        base_text = base_texts[i % len(base_texts)]
        # Добавляем вариативность
        variation = f" (variation {i//len(base_texts) + 1})" if i >= len(base_texts) else ""
        texts.append(base_text + variation)
    
    return texts


async def main():
    """Демонстрация мониторинга производительности"""
    
    # Настраиваем логирование
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    monitor = PerformanceMonitor()
    
    # Генерируем тестовые данные
    test_texts = generate_test_texts(20)
    
    print("🔬 Performance Monitoring Demo")
    print(f"Test dataset: {len(test_texts)} texts")
    print()
    
    try:
        # Сравниваем доступные анализаторы
        available_analyzers = ["algorithmic"]  # Начнем с одного работающего
        
        results = await monitor.compare_analyzers(
            analyzer_types=available_analyzers,
            test_texts=test_texts,
            output_file="performance_report.json"
        )
        
        # Выводим краткий отчет
        print("\n📊 Performance Summary:")
        print("=" * 50)
        
        for analyzer_name, metrics in results.items():
            print(f"\n{analyzer_name.upper()} Analyzer:")
            print(f"  ⏱️  Average time: {metrics.avg_time:.3f}s")
            print(f"  📏 Time range: {metrics.min_time:.3f}s - {metrics.max_time:.3f}s")
            print(f"  🚀 Throughput: {metrics.items_per_second:.1f} items/s")
            print(f"  💾 Memory usage: {metrics.avg_memory_mb:.1f} MB (peak: {metrics.peak_memory_mb:.1f} MB)")
            print(f"  📈 Success rate: {metrics.success_rate:.1f}%")
        
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
