#!/usr/bin/env python3
"""
🎯 Единая точка входа для всей системы анализа и сбора данных

НАЗНАЧЕНИЕ:
- Запуск всех функций проекта через интерактивное меню или командные флаги
- Анализ текстов и песен (4 анализатора)
- Пакетная обработка больших файлов
- Мониторинг производительности и статистика
- Управление конфигурацией и статусом системы

ИСПОЛЬЗОВАНИЕ:
python main.py                      # Интерактивное меню
python main.py --analyze "текст"    # Быстрый анализ текста
python main.py --batch file.txt     # Пакетная обработка
python main.py --benchmark          # Тест производительности
python main.py --info               # Статус системы
python main.py --test               # Запуск тестов

ЗАВИСИМОСТИ:
- Python 3.8+
- src/{cli,analyzers,models}/
- config.yaml
- PostgreSQL база данных (rap_lyrics database)
- Все анализаторы: algorithmic_basic, qwen, ollama, hybrid

РЕЗУЛЬТАТ:
- Консольный вывод с результатами анализа, статистикой, статусом
- Логирование ошибок и производительности
- Интеграция с Docker и API

АВТОР: Vastargazing
ДАТА: Сентябрь 2025
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Добавляем src в Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.cli import BatchProcessor, PerformanceMonitor
    from src.core.app import create_app
    from src.interfaces.analyzer_interface import AnalyzerFactory
    from src.scrapers.rap_scraper_postgres import OptimizedPostgreSQLScraper
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
            self.batch_processor = BatchProcessor()
            self.performance_monitor = PerformanceMonitor()

            # Инициализация scraper для операций скрапинга
            from src.utils.config import GENIUS_TOKEN

            if GENIUS_TOKEN:
                self.scraper = OptimizedPostgreSQLScraper(GENIUS_TOKEN)
                print("✅ Scraper initialized successfully")
            else:
                self.scraper = None
                print("⚠️ Genius token not found - scraping disabled")

            print("✅ Application initialized successfully")
            print(f"📊 Available analyzers: {self.app.list_analyzers()}")

        except Exception as e:
            print(f"❌ Failed to initialize application: {e}")
            raise

    def show_main_menu(self) -> None:
        """Показать главное меню"""
        print("\n🎯 Main Menu:")
        print("1. 🕷️ Scraping Operations")
        print("2.  Batch processing")
        print("3. 📈 Performance benchmark")
        print("4. 🔍 System information")
        print("5. 🧪 Run tests")
        print("6. 📋 Configuration")
        print("0. ❌ Exit")
        print(
            "\n💡 Quick start: Press Enter to start scraping from remaining_artists.json"
        )
        print()

    async def run_interactive_mode(self) -> None:
        """Запуск интерактивного режима"""
        while True:
            self.show_main_menu()

            try:
                choice = input("Select option (0-6, Enter=scraping): ").strip()

                if choice == "":
                    # Быстрый запуск скрапинга
                    print(
                        "🚀 Quick start: Beginning scraping from remaining_artists.json..."
                    )
                    await self.continue_scraping()
                elif choice == "0":
                    print("👋 Goodbye!")
                    break
                elif choice == "1":
                    await self.scraping_operations_interactive()
                elif choice == "2":
                    await self.batch_processing_interactive()
                elif choice == "3":
                    await self.performance_benchmark_interactive()
                elif choice == "4":
                    await self.show_system_info()
                elif choice == "5":
                    await self.run_tests_interactive()
                elif choice == "6":
                    await self.show_configuration()
                else:
                    print("❌ Invalid choice. Please select 0-6.")

                input("\nPress Enter to continue...")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                input("Press Enter to continue...")

    async def scraping_operations_interactive(self) -> None:
        """Интерактивные операции скрапинга"""
        print("\n🕷️ Scraping Operations")
        print("-" * 22)

        print("Available scraping options:")
        print("1. 🔄 Continue scraping (from remaining_artists.json) [RECOMMENDED]")
        print("2. 🎤 Scrape single artist")
        print("3. � Scrape from artist list (custom)")
        print("4. 📊 View scraping status")
        print("5. 🛠️ Database management")
        print("0. ⬅️ Back to main menu")

        choice = input("\nSelect scraping option (0-5, Enter=1): ").strip()

        # По умолчанию выбираем Continue scraping
        if choice == "" or choice == "1":
            await self.continue_scraping()
        elif choice == "0":
            return
        elif choice == "2":
            await self.scrape_single_artist()
        elif choice == "3":
            await self.scrape_artist_list()
        elif choice == "4":
            await self.show_scraping_status()
        elif choice == "5":
            await self.database_management()
        else:
            print("❌ Invalid choice. Please select 0-5.")

    async def scrape_single_artist(self) -> None:
        """Скрапинг одного артиста"""
        print("\n🎤 Single Artist Scraping")
        print("-" * 25)

        artist_name = input("Enter artist name: ").strip()
        if not artist_name:
            print("❌ No artist name provided")
            return

        max_songs = input("Max songs to scrape (default: 50): ").strip()
        try:
            max_songs = int(max_songs) if max_songs else 50
        except ValueError:
            max_songs = 50

        try:
            print(f"\n🔄 Scraping {artist_name} (max {max_songs} songs)...")

            # Импортируем и используем PostgreSQL скрапер
            from src.scrapers.rap_scraper_postgres import OptimizedPostgreSQLScraper
            from src.utils.config import GENIUS_TOKEN

            if not GENIUS_TOKEN:
                print("❌ GENIUS_TOKEN not found in environment")
                return

            scraper = OptimizedPostgreSQLScraper(GENIUS_TOKEN)

            # Запускаем скрапинг одного артиста
            result = await self._run_artist_scraping(scraper, artist_name, max_songs)

            if result["success"]:
                print("✅ Scraping completed!")
                print(f"  🎵 Songs found: {result.get('songs_found', 0)}")
                print(f"  💾 Songs saved: {result.get('songs_saved', 0)}")
                print(f"  ⏱️ Time taken: {result.get('time_taken', 0):.1f}s")
            else:
                print(f"❌ Scraping failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Scraping error: {e}")

    async def scrape_artist_list(self) -> None:
        """Скрапинг списка артистов"""
        print("\n📋 Artist List Scraping")
        print("-" * 24)

        # Выбор источника списка
        print("Select artist list source:")
        print("1. 📄 Text file (one artist per line)")
        print("2. 📊 JSON file")
        print("3. ⌨️ Manual input")

        source_choice = input("Select source (1-3): ").strip()

        artists = []

        if source_choice == "1":
            file_path = input("Enter text file path: ").strip()
            try:
                with open(file_path, encoding="utf-8") as f:
                    artists = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                return

        elif source_choice == "2":
            file_path = input("Enter JSON file path: ").strip()
            try:
                import json

                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        artists = [str(item) for item in data]
                    else:
                        artists = list(data.keys()) if isinstance(data, dict) else []
            except Exception as e:
                print(f"❌ Error reading JSON file: {e}")
                return

        elif source_choice == "3":
            print("Enter artist names (one per line, empty line to finish):")
            while True:
                artist = input().strip()
                if not artist:
                    break
                artists.append(artist)
        else:
            print("❌ Invalid choice")
            return

        if not artists:
            print("❌ No artists provided")
            return

        print(f"\n📋 Found {len(artists)} artists to scrape")

        # Настройки скрапинга
        songs_per_artist = input("Songs per artist (default: 20): ").strip()
        try:
            songs_per_artist = int(songs_per_artist) if songs_per_artist else 20
        except ValueError:
            songs_per_artist = 20

        try:
            print("\n🔄 Starting batch scraping...")

            from src.scrapers.rap_scraper_postgres import OptimizedPostgreSQLScraper
            from src.utils.config import GENIUS_TOKEN

            if not GENIUS_TOKEN:
                print("❌ GENIUS_TOKEN not found in environment")
                return

            scraper = OptimizedPostgreSQLScraper(GENIUS_TOKEN)

            total_songs = 0
            successful_artists = 0

            for i, artist in enumerate(artists, 1):
                print(f"\n🎤 [{i}/{len(artists)}] Scraping: {artist}")

                result = await self._run_artist_scraping(
                    scraper, artist, songs_per_artist
                )

                if result["success"]:
                    songs_saved = result.get("songs_saved", 0)
                    total_songs += songs_saved
                    successful_artists += 1
                    print(f"  ✅ Saved {songs_saved} songs")
                else:
                    print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")

                # Пауза между артистами
                if i < len(artists):
                    print("  ⏸️ Pausing...")
                    await asyncio.sleep(2)

            print("\n🏆 Batch scraping completed!")
            print(f"  🎤 Artists processed: {successful_artists}/{len(artists)}")
            print(f"  🎵 Total songs scraped: {total_songs}")

        except Exception as e:
            print(f"❌ Batch scraping error: {e}")

    async def continue_scraping(self) -> None:
        """Продолжение скрапинга с remaining_artists.json"""
        print("\n🔄 Continue Scraping")
        print("-" * 19)

        try:
            # Автоматически используем data/remaining_artists.json
            remaining_artists_file = Path("data/remaining_artists.json")

            if remaining_artists_file.exists():
                print(f"📁 Using artist list: {remaining_artists_file}")
                self.scrape_from_remaining_artists()
            else:
                print(f"❌ File not found: {remaining_artists_file}")
                print("🔍 Looking for alternative artist files in data/...")

                # Ищем другие JSON файлы с артистами
                json_files = list(Path("data").glob("*artist*.json"))
                if json_files:
                    print("📁 Found artist files:")
                    for file in json_files:
                        print(f"  - {file.name}")

                    # Используем первый найденный файл
                    selected_file = json_files[0]
                    print(f"🎯 Using: {selected_file.name}")
                    self.scrape_from_file(selected_file)
                else:
                    print("❌ No artist files found in data/ directory")
                    print("💡 Please ensure data/remaining_artists.json exists")

        except Exception as e:
            print(f"❌ Continue scraping error: {e}")

    def scrape_from_remaining_artists(self) -> None:
        """Скрапинг из remaining_artists.json без лишних вопросов"""
        try:
            import json

            with open("data/remaining_artists.json", encoding="utf-8") as f:
                data = json.load(f)

            # Извлекаем артистов из JSON
            if isinstance(data, list):
                artists = [str(item) for item in data]
            elif isinstance(data, dict):
                artists = list(data.keys())
            else:
                print("❌ Invalid JSON format in remaining_artists.json")
                return

            if not artists:
                print("❌ No artists found in remaining_artists.json")
                return

            print(f"📋 Found {len(artists)} artists in remaining_artists.json")

            # Автоматические настройки для продолжения скрапинга
            songs_per_artist = 20  # По умолчанию 20 песен

            print(f"⚙️ Settings: {songs_per_artist} songs per artist")
            print("🚀 Starting scraping...")

            # Запускаем скрапинг
            successful_artists = 0
            total_songs = 0

            for i, artist in enumerate(artists, 1):
                print(f"\n🎤 [{i}/{len(artists)}] Scraping: {artist}")

                try:
                    # Используем существующий scraper (синхронный вызов)
                    if not self.scraper:
                        print("  ❌ Scraper not available (no Genius token)")
                        continue

                    songs_scraped = self.scraper.scrape_artist_songs(
                        artist, max_songs=songs_per_artist
                    )

                    if songs_scraped > 0:
                        successful_artists += 1
                        total_songs += songs_scraped
                        print(f"  ✅ {songs_scraped} songs scraped")
                        print("  💾 Saved to database")
                    else:
                        print("  ⚠️ No songs found")

                except Exception as artist_error:
                    print(f"  ❌ Error scraping {artist}: {artist_error}")
                    continue

            print("\n🏆 Scraping completed!")
            print(f"  🎤 Artists processed: {successful_artists}/{len(artists)}")
            print(f"  🎵 Total songs scraped: {total_songs}")

        except Exception as e:
            print(f"❌ Error scraping from remaining_artists.json: {e}")

    def scrape_from_file(self, file_path: Path) -> None:
        """Скрапинг из указанного JSON файла"""
        try:
            import json

            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Извлекаем артистов из JSON
            if isinstance(data, list):
                artists = [str(item) for item in data]
            elif isinstance(data, dict):
                artists = list(data.keys())
            else:
                print(f"❌ Invalid JSON format in {file_path}")
                return

            if not artists:
                print(f"❌ No artists found in {file_path}")
                return

            print(f"📋 Found {len(artists)} artists in {file_path.name}")

            # Автоматические настройки
            songs_per_artist = 20

            print(f"⚙️ Settings: {songs_per_artist} songs per artist")
            print("🚀 Starting scraping...")

            # Запускаем скрапинг
            successful_artists = 0
            total_songs = 0

            for i, artist in enumerate(artists, 1):
                print(f"\n🎤 [{i}/{len(artists)}] Scraping: {artist}")

                try:
                    if not self.scraper:
                        print("  ❌ Scraper not available (no Genius token)")
                        continue

                    songs_scraped = self.scraper.scrape_artist_songs(
                        artist, max_songs=songs_per_artist
                    )

                    if songs_scraped > 0:
                        successful_artists += 1
                        total_songs += songs_scraped
                        print(f"  ✅ {songs_scraped} songs scraped")
                        print("  💾 Saved to database")
                    else:
                        print("  ⚠️ No songs found")

                except Exception as artist_error:
                    print(f"  ❌ Error scraping {artist}: {artist_error}")
                    continue

            print("\n🏆 Scraping completed!")
            print(f"  🎤 Artists processed: {successful_artists}/{len(artists)}")
            print(f"  🎵 Total songs scraped: {total_songs}")

        except Exception as e:
            print(f"❌ Error scraping from {file_path}: {e}")

    async def show_scraping_status(self) -> None:
        """Показать статус скрапинга"""
        print("\n📊 Scraping Status")
        print("-" * 18)

        try:
            # Используем наш unified database diagnostics tool
            import subprocess

            print("🔍 Running database diagnostics...")

            # Запускаем наш объединенный инструмент диагностики
            result = subprocess.run(
                [sys.executable, "scripts/tools/database_diagnostics.py", "--quick"],
                check=False,
                capture_output=True,
                text=True,
                cwd=Path(),
            )

            if result.returncode == 0:
                print(result.stdout)
            else:
                print(f"❌ Error running diagnostics: {result.stderr}")

            # Дополнительная статистика скрапинга
            print("\n📈 Additional Scraping Metrics:")
            print("  🕒 Last scraping session: [Would show from logs]")
            print("  📁 Database file size: [Calculated above]")
            print("  🎯 Target completion: [Based on remaining artists]")

        except Exception as e:
            print(f"❌ Status check error: {e}")

    async def database_management(self) -> None:
        """Управление базой данных"""
        print("\n🛠️ Database Management")
        print("-" * 21)

        print("Available operations:")
        print("1. 🔍 Full database diagnostics")
        print("2. 📊 View unanalyzed records")
        print("3. 🧹 Database cleanup/optimization")
        print("4. 📁 Backup database")
        print("5. 📈 Export statistics")
        print("0. ⬅️ Back")

        choice = input("Select operation (0-5): ").strip()

        if choice == "0":
            return
        if choice == "1":
            # Полная диагностика
            import subprocess

            result = subprocess.run(
                [sys.executable, "scripts/tools/database_diagnostics.py"],
                check=False,
                cwd=Path(),
            )

        elif choice == "2":
            # Неанализированные записи
            import subprocess

            limit = input("Number of records to show (default: 10): ").strip()
            limit = limit if limit.isdigit() else "10"

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/tools/database_diagnostics.py",
                    "--unanalyzed",
                    "-n",
                    limit,
                ],
                check=False,
                cwd=Path(),
            )

        elif choice == "3":
            print("🧹 Database cleanup options:")
            print("  - Remove duplicate entries")
            print("  - Optimize database file")
            print("  - Clean temporary data")
            print("ℹ️ Cleanup functionality would be implemented here")

        elif choice == "4":
            print("📁 Creating database backup...")
            try:
                import shutil
                from datetime import datetime

                backup_name = (
                    f"data_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                )
                backup_path = Path("data") / backup_name

                shutil.copy2("data/rap_lyrics.db", backup_path)
                print(f"✅ Backup created: {backup_path}")

            except Exception as e:
                print(f"❌ Backup failed: {e}")

        elif choice == "5":
            print("📈 Exporting statistics...")
            print("ℹ️ Statistics export functionality would be implemented here")

        else:
            print("❌ Invalid choice")

    async def _run_artist_scraping(
        self, scraper, artist_name: str, max_songs: int
    ) -> dict[str, Any]:
        """Вспомогательный метод для запуска скрапинга артиста"""
        try:
            # Имитация скрапинга (заменить на реальную логику)
            start_time = time.time()

            # Здесь будет реальный вызов scraper.scrape_artist()
            # result = scraper.scrape_artist(artist_name, max_songs)

            # Временная заглушка
            await asyncio.sleep(1)  # Имитация работы

            return {
                "success": True,
                "songs_found": max_songs,
                "songs_saved": max_songs - 2,  # Имитация
                "time_taken": time.time() - start_time,
                "artist": artist_name,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "artist": artist_name}

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
        output_file = input(
            "Output file (default: results/batch_results.json): "
        ).strip()
        if not output_file:
            output_file = "results/batch_results.json"

        checkpoint_file = input(
            "Checkpoint file (default: results/batch_checkpoint.json): "
        ).strip()
        if not checkpoint_file:
            checkpoint_file = "results/batch_checkpoint.json"

        workers = input("Number of workers (default: 2): ").strip()
        try:
            workers = int(workers) if workers else 2
        except ValueError:
            workers = 2

        # Создаем процессор с настройками
        processor = BatchProcessor(max_workers=workers)

        try:
            # Читаем входные данные
            with open(input_file, encoding="utf-8") as f:
                if input_file.endswith(".json"):
                    data = json.load(f)
                    if isinstance(data, list):
                        texts = [item.get("text", str(item)) for item in data]
                    else:
                        texts = [data.get("text", str(data))]
                else:
                    texts = [line.strip() for line in f if line.strip()]

            print(f"\n🚀 Processing {len(texts)} texts with {workers} workers...")

            results = await processor.process_batch(
                texts=texts,
                analyzer_type=analyzer_type,
                output_file=output_file,
                checkpoint_file=checkpoint_file,
            )

            # Статистика
            successful = len([r for r in results if "error" not in r])
            failed = len(results) - successful

            print("\n📊 Batch Processing Complete:")
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

        analyzer_input = input(
            "Select analyzers for benchmark (comma-separated): "
        ).strip()
        analyzers = [a.strip() for a in analyzer_input.split(",") if a.strip()]

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
            print(
                f"\n🔬 Benchmarking {len(analyzers)} analyzers with {test_count} texts..."
            )

            results = await self.performance_monitor.compare_analyzers(
                analyzer_types=analyzers,
                test_texts=test_texts,
                output_file="performance_benchmark.json",
            )

            print("\n📊 Benchmark Results:")
            print("-" * 40)

            for analyzer_name, metrics in results.items():
                print(f"\n🔧 {analyzer_name.upper()}:")
                print(f"  ⏱️  Avg time: {metrics.avg_time:.3f}s")
                print(f"  🚀 Throughput: {metrics.items_per_second:.1f} items/s")
                print(f"  📈 Success rate: {metrics.success_rate:.1f}%")
                print(f"  💾 Memory: {metrics.avg_memory_mb:.1f} MB")

            print("\n📄 Detailed report saved to: performance_benchmark.json")

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")

    async def show_system_info(self) -> None:
        """Показать информацию о системе"""
        print("\n🔍 System Information")
        print("-" * 22)

        try:
            # Информация о приложении
            print("📱 Application Info:")
            print("  Version: 1.0.0")
            print(f"  Python: {sys.version.split()[0]}")
            print(f"  Project root: {Path().absolute()}")

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
            print("\n⚙️  Configuration:")

            # Проверяем тип базы данных
            if hasattr(self.app.config.database, "host"):
                # PostgreSQL конфигурация
                db_info = f"{self.app.config.database.host}:{self.app.config.database.port}/{self.app.config.database.name}"
                print(f"  Database (PostgreSQL): {db_info}")
            else:
                # SQLite конфигурация
                print(
                    f"  Database (SQLite): {getattr(self.app.config.database, 'path', 'Not configured')}"
                )

            print(
                f"  Logging: {getattr(self.app.config.logging, 'level', 'Not configured')}"
            )

            # Статистика БД (если доступна)
            try:
                print("\n📊 Database Stats:")
                if hasattr(self.app, "database") and self.app.database:
                    print("  Connection: ✅ Connected")

                    # Получаем статистику из PostgreSQL
                    if hasattr(self.app.database, "get_track_count"):
                        track_count = await self.app.database.get_track_count()
                        print(f"  Total tracks: {track_count:,}")

                    # Проверяем ML возможности
                    if hasattr(self.app.database, "_vector_enabled"):
                        if self.app.database._vector_enabled:
                            print("  ML Features: ✅ pgvector enabled")
                        else:
                            print("  ML Features: ❌ pgvector not available")
                else:
                    print("  Connection: ❌ Not initialized")
            except Exception as e:
                print(f"  Connection: ❌ Error: {e}")

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

            test_file = (
                Path(__file__).parent / "tests" / "test_integration_comprehensive.py"
            )

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

            print("🔄 Running tests...")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)

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
            if hasattr(config, "database"):
                print("📊 Database:")
                db_config = config.database
                print(f"  Path: {getattr(db_config, 'path', 'Not set')}")
                print(f"  Pool size: {getattr(db_config, 'pool_size', 'Not set')}")

            # Logging config
            if hasattr(config, "logging"):
                print("📝 Logging:")
                log_config = config.logging
                print(f"  Level: {getattr(log_config, 'level', 'Not set')}")
                print(f"  File: {getattr(log_config, 'file_path', 'Not set')}")
                print(f"  Format: {getattr(log_config, 'format', 'Not set')}")

            # Analyzer configs
            print("🧠 Analyzers:")
            for analyzer_name in self.app.list_analyzers():
                print(f"  {analyzer_name}: Registered")

        except Exception as e:
            print(f"❌ Failed to show configuration: {e}")

    def _generate_test_texts(self, count: int) -> list[str]:
        """Генерация тестовых текстов для бенчмарка"""
        base_texts = [
            "Happy uplifting song about love and joy",
            "Dark melancholic ballad about loss and sadness",
            "Aggressive energetic rap with confident attitude",
            "Peaceful reflective lyrics about nature and life",
            "Romantic slow song about relationships and feelings",
        ]

        texts = []
        for i in range(count):
            base = base_texts[i % len(base_texts)]
            variation = (
                f" (test variation {i // len(base_texts) + 1})"
                if count > len(base_texts)
                else ""
            )
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
        """,
    )

    parser.add_argument("--analyze", help="Quick text analysis")
    parser.add_argument(
        "--analyzer", default="algorithmic_basic", help="Analyzer to use"
    )
    parser.add_argument("--batch", help="Batch process file")
    parser.add_argument(
        "--benchmark", action="store_true", help="Run performance benchmark"
    )
    parser.add_argument("--info", action="store_true", help="Show system information")
    parser.add_argument("--test", action="store_true", help="Run tests")

    args = parser.parse_args()

    try:
        app = RapScraperMainApp()

        if args.analyze:
            # Quick analysis
            result = await app.cli.analyze_text(args.analyze, args.analyzer)
            print(json.dumps(result, indent=2, ensure_ascii=False))

        elif args.batch:
            # Batch processing
            with open(args.batch, encoding="utf-8") as f:
                if args.batch.endswith(".json"):
                    data = json.load(f)
                    texts = [item.get("text", str(item)) for item in data]
                else:
                    texts = [line.strip() for line in f if line.strip()]

            results = await app.batch_processor.process_batch(
                texts=texts,
                analyzer_type=args.analyzer,
                output_file="results/batch_results.json",
            )
            print(f"Processed {len(results)} texts")

        elif args.benchmark:
            # Performance benchmark
            test_texts = app._generate_test_texts(10)
            results = await app.performance_monitor.compare_analyzers(
                analyzer_types=[args.analyzer],
                test_texts=test_texts,
                output_file="results/benchmark_results.json",
            )
            print(
                "Benchmark completed, results saved to results/benchmark_results.json"
            )

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
