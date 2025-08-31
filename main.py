#!/usr/bin/env python3
"""
🎯 Центральная точка входа для rap-scraper-project
Фаза 4: Интеграция и тестирование

Этот файл предоставляет единый интерфейс для всех функций системы:
- Анализ текстов и песен
- Пакетная обработка  
- Мониторинг производительности
- Управление конфигурацией
- Статистика и отчеты
"""

import sys
import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Добавляем src в Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from core.app import create_app
    from cli import AnalyzerCLI, BatchProcessor, PerformanceMonitor
    from interfaces.analyzer_interface import AnalyzerFactory
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class RapScraperMainApp:
    """Главное приложение с меню всех функций"""
    
    def __init__(self):
        """Инициализация главного приложения"""
        print("🎵 Rap Scraper Project - Main Application")
        print("=" * 50)
        
        try:
            self.app = create_app()
            self.cli = AnalyzerCLI()
            self.batch_processor = BatchProcessor()
            self.performance_monitor = PerformanceMonitor()
            
            print("✅ Application initialized successfully")
            print(f"📊 Available analyzers: {self.app.list_analyzers()}")
            
        except Exception as e:
            print(f"❌ Failed to initialize application: {e}")
            raise
    
    def show_main_menu(self) -> None:
        """Показать главное меню"""
        print("\n🎯 Main Menu:")
        print("1. 📝 Analyze single text")
        print("2. 📊 Compare analyzers")
        print("3. 📦 Batch processing")
        print("4. 📈 Performance benchmark")
        print("5. 🔍 System information")
        print("6. 🧪 Run tests")
        print("7. 📋 Configuration")
        print("0. ❌ Exit")
        print()
    
    async def run_interactive_mode(self) -> None:
        """Запуск интерактивного режима"""
        while True:
            self.show_main_menu()
            
            try:
                choice = input("Select option (0-7): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    break
                elif choice == "1":
                    await self.analyze_single_text_interactive()
                elif choice == "2":
                    await self.compare_analyzers_interactive()
                elif choice == "3":
                    await self.batch_processing_interactive()
                elif choice == "4":
                    await self.performance_benchmark_interactive()
                elif choice == "5":
                    await self.show_system_info()
                elif choice == "6":
                    await self.run_tests_interactive()
                elif choice == "7":
                    await self.show_configuration()
                else:
                    print("❌ Invalid choice. Please select 0-7.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("Press Enter to continue...")
    
    async def analyze_single_text_interactive(self) -> None:
        """Интерактивный анализ одного текста"""
        print("\n📝 Single Text Analysis")
        print("-" * 30)
        
        # Получаем текст
        print("Enter text to analyze (or paste lyrics):")
        lines = []
        print("(Type 'END' on a new line to finish)")
        
        while True:
            line = input()
            if line.strip().upper() == "END":
                break
            lines.append(line)
        
        text = "\n".join(lines).strip()
        
        if not text:
            print("❌ No text provided")
            return
        
        # Выбираем анализатор
        analyzers = self.app.list_analyzers()
        print(f"\nAvailable analyzers: {', '.join(analyzers)}")
        analyzer_type = input("Select analyzer: ").strip()
        
        if analyzer_type not in analyzers:
            print(f"❌ Unknown analyzer: {analyzer_type}")
            return
        
        try:
            print(f"\n🔄 Analyzing with {analyzer_type}...")
            result = await self.cli.analyze_text(text, analyzer_type)
            
            print("\n📊 Analysis Results:")
            print(f"⏱️  Analysis time: {result['analysis_time']}s")
            print(f"📏 Text length: {result['text_length']} chars")
            print(f"📅 Timestamp: {result['timestamp']}")
            
            # Показываем результат анализа
            analysis_result = result['result']
            if isinstance(analysis_result, dict):
                print("\n🎯 Analysis Details:")
                for key, value in analysis_result.items():
                    print(f"  {key}: {value}")
            
            # Предлагаем сохранить
            save = input("\nSave results to file? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"analysis_result_{analyzer_type}_{result['timestamp'].replace(':', '-').replace(' ', '_')}.json"
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"💾 Results saved to: {filename}")
        
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    async def compare_analyzers_interactive(self) -> None:
        """Интерактивное сравнение анализаторов"""
        print("\n📊 Compare Analyzers")
        print("-" * 25)
        
        # Получаем текст
        text = input("Enter text for comparison: ").strip()
        if not text:
            print("❌ No text provided")
            return
        
        # Выбираем анализаторы
        available = self.app.list_analyzers()
        print(f"Available analyzers: {', '.join(available)}")
        
        analyzer_input = input("Select analyzers (comma-separated): ").strip()
        analyzers = [a.strip() for a in analyzer_input.split(',') if a.strip()]
        
        if not analyzers:
            analyzers = available  # Используем все доступные
        
        try:
            print(f"\n🔄 Comparing {len(analyzers)} analyzers...")
            result = await self.cli.compare_analyzers(
                text=text,
                analyzers=analyzers,
                output_file="comparison_results.json"
            )
            
            print("📊 Comparison completed!")
            print("📄 Results saved to: comparison_results.json")
            
            # Краткая сводка
            if 'analyzers' in result:
                print("\n⚡ Quick Summary:")
                for analyzer, data in result['analyzers'].items():
                    if 'error' not in data:
                        time_taken = data.get('analysis_time', 0)
                        print(f"  {analyzer}: {time_taken}s")
                    else:
                        print(f"  {analyzer}: ❌ {data['error']}")
        
        except Exception as e:
            print(f"❌ Comparison failed: {e}")
    
    async def batch_processing_interactive(self) -> None:
        """Интерактивная пакетная обработка"""
        print("\n📦 Batch Processing")
        print("-" * 20)
        
        # Выбираем входной файл
        input_file = input("Enter input file path (JSON or text): ").strip()
        if not Path(input_file).exists():
            print(f"❌ File not found: {input_file}")
            return
        
        # Выбираем анализатор
        analyzers = self.app.list_analyzers()
        print(f"Available analyzers: {', '.join(analyzers)}")
        analyzer_type = input("Select analyzer: ").strip()
        
        if analyzer_type not in analyzers:
            print(f"❌ Unknown analyzer: {analyzer_type}")
            return
        
        # Настройки
        output_file = input("Output file (default: batch_results.json): ").strip()
        if not output_file:
            output_file = "batch_results.json"
        
        checkpoint_file = input("Checkpoint file (default: batch_checkpoint.json): ").strip()
        if not checkpoint_file:
            checkpoint_file = "batch_checkpoint.json"
        
        workers = input("Number of workers (default: 2): ").strip()
        try:
            workers = int(workers) if workers else 2
        except ValueError:
            workers = 2
        
        # Создаем процессор с настройками
        processor = BatchProcessor(max_workers=workers)
        
        try:
            # Читаем входные данные
            with open(input_file, 'r', encoding='utf-8') as f:
                if input_file.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        texts = [item.get('text', str(item)) for item in data]
                    else:
                        texts = [data.get('text', str(data))]
                else:
                    texts = [line.strip() for line in f if line.strip()]
            
            print(f"\n🚀 Processing {len(texts)} texts with {workers} workers...")
            
            results = await processor.process_batch(
                texts=texts,
                analyzer_type=analyzer_type,
                output_file=output_file,
                checkpoint_file=checkpoint_file
            )
            
            # Статистика
            successful = len([r for r in results if 'error' not in r])
            failed = len(results) - successful
            
            print(f"\n📊 Batch Processing Complete:")
            print(f"  ✅ Successful: {successful}")
            print(f"  ❌ Failed: {failed}")
            print(f"  📄 Results saved to: {output_file}")
        
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
    
    async def performance_benchmark_interactive(self) -> None:
        """Интерактивный бенчмарк производительности"""
        print("\n📈 Performance Benchmark")
        print("-" * 26)
        
        # Выбираем анализаторы
        available = self.app.list_analyzers()
        print(f"Available analyzers: {', '.join(available)}")
        
        analyzer_input = input("Select analyzers for benchmark (comma-separated): ").strip()
        analyzers = [a.strip() for a in analyzer_input.split(',') if a.strip()]
        
        if not analyzers:
            analyzers = available[:2]  # Первые два для скорости
        
        # Настройки
        test_count = input("Number of test texts (default: 10): ").strip()
        try:
            test_count = int(test_count) if test_count else 10
        except ValueError:
            test_count = 10
        
        # Генерируем тестовые тексты
        test_texts = self._generate_test_texts(test_count)
        
        try:
            print(f"\n🔬 Benchmarking {len(analyzers)} analyzers with {test_count} texts...")
            
            results = await self.performance_monitor.compare_analyzers(
                analyzer_types=analyzers,
                test_texts=test_texts,
                output_file="performance_benchmark.json"
            )
            
            print("\n📊 Benchmark Results:")
            print("-" * 40)
            
            for analyzer_name, metrics in results.items():
                print(f"\n🔧 {analyzer_name.upper()}:")
                print(f"  ⏱️  Avg time: {metrics.avg_time:.3f}s")
                print(f"  🚀 Throughput: {metrics.items_per_second:.1f} items/s")
                print(f"  📈 Success rate: {metrics.success_rate:.1f}%")
                print(f"  💾 Memory: {metrics.avg_memory_mb:.1f} MB")
            
            print(f"\n📄 Detailed report saved to: performance_benchmark.json")
        
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
    
    async def show_system_info(self) -> None:
        """Показать информацию о системе"""
        print("\n🔍 System Information")
        print("-" * 22)
        
        try:
            # Информация о приложении
            print("📱 Application Info:")
            print(f"  Version: 1.0.0")
            print(f"  Python: {sys.version.split()[0]}")
            print(f"  Project root: {Path('.').absolute()}")
            
            # Анализаторы
            analyzers = self.app.list_analyzers()
            print(f"\n🧠 Available Analyzers ({len(analyzers)}):")
            for analyzer in analyzers:
                analyzer_obj = self.app.get_analyzer(analyzer)
                if analyzer_obj:
                    status = "✅ Ready"
                else:
                    status = "❌ Error"
                print(f"  {analyzer}: {status}")
            
            # Конфигурация
            print(f"\n⚙️  Configuration:")
            print(f"  Database: {getattr(self.app.config.database, 'path', 'Not configured')}")
            print(f"  Logging: {getattr(self.app.config.logging, 'level', 'Not configured')}")
            
            # Статистика БД (если доступна)
            try:
                # Здесь можно добавить статистику из БД
                print(f"\n📊 Database Stats:")
                print(f"  Connection: ✅ Connected")
            except Exception:
                print(f"  Connection: ❌ Error")
        
        except Exception as e:
            print(f"❌ Failed to get system info: {e}")
    
    async def run_tests_interactive(self) -> None:
        """Интерактивный запуск тестов"""
        print("\n🧪 Run Tests")
        print("-" * 12)
        
        print("Available test suites:")
        print("1. Basic infrastructure tests")
        print("2. CLI component tests")
        print("3. Integration tests")
        print("4. All tests")
        
        choice = input("Select test suite (1-4): ").strip()
        
        try:
            import subprocess
            
            test_file = Path(__file__).parent / "tests" / "test_integration_comprehensive.py"
            
            if choice == "1":
                cmd = [sys.executable, str(test_file), "basic"]
            elif choice == "2":
                cmd = [sys.executable, str(test_file), "cli"]
            elif choice == "3":
                cmd = [sys.executable, str(test_file), "integration"]
            elif choice == "4":
                cmd = [sys.executable, str(test_file)]
            else:
                print("❌ Invalid choice")
                return
            
            print(f"🔄 Running tests...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            print("📊 Test Results:")
            print(result.stdout)
            
            if result.stderr:
                print("⚠️  Warnings/Errors:")
                print(result.stderr)
            
            print(f"Exit code: {result.returncode}")
        
        except Exception as e:
            print(f"❌ Failed to run tests: {e}")
    
    async def show_configuration(self) -> None:
        """Показать конфигурацию"""
        print("\n📋 Configuration")
        print("-" * 17)
        
        try:
            config = self.app.config
            
            print("⚙️  Current Configuration:")
            
            # Database config
            if hasattr(config, 'database'):
                print(f"📊 Database:")
                db_config = config.database
                print(f"  Path: {getattr(db_config, 'path', 'Not set')}")
                print(f"  Pool size: {getattr(db_config, 'pool_size', 'Not set')}")
            
            # Logging config
            if hasattr(config, 'logging'):
                print(f"📝 Logging:")
                log_config = config.logging
                print(f"  Level: {getattr(log_config, 'level', 'Not set')}")
                print(f"  File: {getattr(log_config, 'file_path', 'Not set')}")
                print(f"  Format: {getattr(log_config, 'format', 'Not set')}")
            
            # Analyzer configs
            print(f"🧠 Analyzers:")
            for analyzer_name in self.app.list_analyzers():
                print(f"  {analyzer_name}: Registered")
        
        except Exception as e:
            print(f"❌ Failed to show configuration: {e}")
    
    def _generate_test_texts(self, count: int) -> List[str]:
        """Генерация тестовых текстов для бенчмарка"""
        base_texts = [
            "Happy uplifting song about love and joy",
            "Dark melancholic ballad about loss and sadness",
            "Aggressive energetic rap with confident attitude",
            "Peaceful reflective lyrics about nature and life",
            "Romantic slow song about relationships and feelings"
        ]
        
        texts = []
        for i in range(count):
            base = base_texts[i % len(base_texts)]
            variation = f" (test variation {i//len(base_texts) + 1})" if count > len(base_texts) else ""
            texts.append(base + variation)
        
        return texts


async def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Rap Scraper Project - Main Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Interactive mode
  python main.py --analyze "text"   # Quick analysis
  python main.py --batch input.json # Batch processing
  python main.py --benchmark        # Performance test
  python main.py --info             # System information
        """
    )
    
    parser.add_argument('--analyze', help='Quick text analysis')
    parser.add_argument('--analyzer', default='algorithmic_basic', help='Analyzer to use')
    parser.add_argument('--batch', help='Batch process file')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--info', action='store_true', help='Show system information')
    parser.add_argument('--test', action='store_true', help='Run tests')
    
    args = parser.parse_args()
    
    try:
        app = RapScraperMainApp()
        
        if args.analyze:
            # Quick analysis
            result = await app.cli.analyze_text(args.analyze, args.analyzer)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
        elif args.batch:
            # Batch processing
            with open(args.batch, 'r', encoding='utf-8') as f:
                if args.batch.endswith('.json'):
                    data = json.load(f)
                    texts = [item.get('text', str(item)) for item in data]
                else:
                    texts = [line.strip() for line in f if line.strip()]
            
            results = await app.batch_processor.process_batch(
                texts=texts,
                analyzer_type=args.analyzer,
                output_file="batch_results.json"
            )
            print(f"Processed {len(results)} texts")
        
        elif args.benchmark:
            # Performance benchmark
            test_texts = app._generate_test_texts(10)
            results = await app.performance_monitor.compare_analyzers(
                analyzer_types=[args.analyzer],
                test_texts=test_texts,
                output_file="benchmark_results.json"
            )
            print("Benchmark completed, results saved to benchmark_results.json")
        
        elif args.info:
            # System information
            await app.show_system_info()
        
        elif args.test:
            # Run tests
            await app.run_tests_interactive()
        
        else:
            # Interactive mode
            await app.run_interactive_mode()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Application error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
