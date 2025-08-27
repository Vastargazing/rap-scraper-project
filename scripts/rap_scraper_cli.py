#!/usr/bin/env python3
"""
🎯 Rap Scraper Project - Главный CLI интерфейс
Единая точка входа для всех операций проекта
"""

import sys
import os
import argparse
from pathlib import Path

# Добавляем корневую папку в path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def print_banner():
    """Красивый баннер проекта"""
    print("""
🎤 ═══════════════════════════════════════════════════════════════
   RAP SCRAPER PROJECT - ML Pipeline for Hip-Hop Lyrics Analysis
   ═══════════════════════════════════════════════════════════════
   📊 48K+ tracks | 263 artists | Spotify enriched | AI analyzed
   🏗️ Production-ready ML pipeline with structured metadata
   ═══════════════════════════════════════════════════════════════
""")

def show_status():
    """Показать текущий статус проекта"""
    try:
        from src.utils.check_db import check_database
        print("📊 ТЕКУЩИЙ СТАТУС ПРОЕКТА:")
        print("-" * 50)
        check_database()
    except Exception as e:
        print(f"❌ Ошибка получения статуса: {e}")

def run_scraping(args):
    """Запуск скрапинга с различными режимами"""
    print("🕷️ Запуск системы скрапинга...")
    
    try:
        if args.artist:
            # Режим одного артиста
            print(f"🎤 Скрапинг одного артиста: {args.artist}")
            import subprocess
            script_path = Path(__file__).parent / "development" / "scrape_artist_one.py"
            result = subprocess.run([sys.executable, str(script_path), args.artist], 
                                  capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(f"❌ Ошибки: {result.stderr}")
                
        elif args.test:
            # Тестовый режим
            print("🧪 Тестовый режим скрапинга")
            import subprocess
            script_path = Path(__file__).parent / "development" / "test_fixed_scraper.py"
            subprocess.run([sys.executable, str(script_path)])
            
        elif args.debug:
            # Отладочный режим
            print("🔍 Отладочный режим скрапинга")
            import subprocess
            script_path = Path(__file__).parent / "development" / "run_scraping_debug.py"
            subprocess.run([sys.executable, str(script_path)])
            
        elif args.continue_mode:
            # Режим продолжения
            print("🔄 Продолжение скрапинга оставшихся артистов")
            import subprocess
            script_path = Path(__file__).parent / "development" / "run_remaining_artists.py"
            subprocess.run([sys.executable, str(script_path)])
            
        else:
            # Обычный режим
            print("🚀 Полный скрапинг")
            from src.scrapers.rap_scraper_optimized import main
            main()
            
    except KeyboardInterrupt:
        print("\n⏹️ Скрапинг остановлен пользователем")
    except Exception as e:
        print(f"❌ Ошибка скрапинга: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()

def run_spotify_enhancement(args):
    """Запуск Spotify обогащения"""
    print("🎵 Запуск Spotify enhancement...")
    
    if args.continue_mode:
        print("🔄 Режим продолжения - обработка оставшихся данных")
        try:
            import subprocess
            subprocess.run([sys.executable, "scripts/continue_spotify_enhancement.py"])
        except Exception as e:
            print(f"❌ Ошибка Spotify enhancement: {e}")
    else:
        print("🆕 Новый запуск Spotify enhancement")
        try:
            import subprocess
            subprocess.run([sys.executable, "scripts/run_spotify_enhancement.py"])
        except Exception as e:
            print(f"❌ Ошибка Spotify enhancement: {e}")

def run_analysis(args):
    """Запуск ML анализа"""
    print("🤖 Запуск ML анализа...")
    
    if args.analyzer == "gemma":
        print("🔥 Используем Gemma 27B для анализа")
        try:
            from src.analyzers.gemma_27b_fixed import main
            main()
        except Exception as e:
            print(f"❌ Ошибка Gemma анализа: {e}")
    
    elif args.analyzer == "multi":
        print("🔄 Сравнительный анализ нескольких моделей")
        try:
            from src.analyzers.multi_model_analyzer import main
            main()
        except Exception as e:
            print(f"❌ Ошибка multi-model анализа: {e}")
    
    elif args.analyzer == "langchain":
        print("⛓️ LangChain анализ с OpenAI")
        try:
            # Загружаем архивированный скрипт
            import subprocess
            subprocess.run([sys.executable, "scripts/archive/test_langchain.py"])
        except Exception as e:
            print(f"❌ Ошибка LangChain анализа: {e}")
    
    else:
        print("📋 Доступные анализаторы:")
        print("   🔥 gemma    - Gemma 27B (локальный)")
        print("   🔄 multi    - Сравнение моделей")
        print("   ⛓️ langchain - LangChain + OpenAI")
        print("\nПример: python scripts/rap_scraper_cli.py analysis --analyzer gemma")

def run_monitoring(args):
    """Мониторинг и статистика"""
    print("📊 Система мониторинга...")
    
    if args.component == "database":
        print("🗄️ Проверка состояния базы данных")
        show_status()
    
    elif args.component == "analysis":
        print("🤖 Статус AI анализа")
        try:
            import subprocess
            subprocess.run([sys.executable, "monitoring/check_analysis_status.py"])
        except Exception as e:
            print(f"❌ Ошибка мониторинга анализа: {e}")
    
    elif args.component == "gemma":
        print("🔥 Мониторинг прогресса Gemma")
        try:
            import subprocess
            subprocess.run([sys.executable, "monitoring/monitor_gemma_progress.py"])
        except Exception as e:
            print(f"❌ Ошибка мониторинга Gemma: {e}")
    
    else:
        print("📋 Доступные компоненты мониторинга:")
        print("   🗄️ database - Состояние БД")
        print("   🤖 analysis - AI анализ")
        print("   🔥 gemma    - Gemma прогресс")

def run_utilities(args):
    """Утилиты и сервисные функции"""
    print("🛠️ Утилиты проекта...")
    
    if args.utility == "cleanup":
        print("🗑️ Очистка проекта")
        try:
            import subprocess
            # cleanup_project.py was moved to scripts/utils/
            # Build a robust path relative to this CLI file and fall back to repo scripts/ if missing
            cleanup_path = Path(__file__).parent / "utils" / "cleanup_project.py"
            if not cleanup_path.exists():
                cleanup_path = Path.cwd() / "scripts" / "cleanup_project.py"
            cmd = [sys.executable, str(cleanup_path)]
            if args.execute:
                cmd.append("--execute")
            subprocess.run(cmd)
        except Exception as e:
            print(f"❌ Ошибка очистки: {e}")
    
    elif args.utility == "migrate":
        print("🔄 Миграция базы данных")
        try:
            from src.utils.migrate_database import main
            main()
        except Exception as e:
            print(f"❌ Ошибка миграции: {e}")
    
    elif args.utility == "spotify-setup":
        print("🎵 Настройка Spotify API")
        try:
            from src.utils.setup_spotify import main
            main()
        except Exception as e:
            print(f"❌ Ошибка настройки Spotify: {e}")
    
    else:
        print("📋 Доступные утилиты:")
        print("   🗑️ cleanup       - Очистка проекта")
        print("   🔄 migrate       - Миграция БД")
        print("   🎵 spotify-setup - Настройка Spotify")

def create_parser():
    """Создание парсера аргументов"""
    parser = argparse.ArgumentParser(
        description="🎤 Rap Scraper Project CLI - ML Pipeline for Hip-Hop Analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  
  📊 Проверка статуса:
    python scripts/rap_scraper_cli.py status
  
  🕷️ Скрапинг данных:
    python scripts/rap_scraper_cli.py scraping                    # Полный скрапинг
    python scripts/rap_scraper_cli.py scraping --continue         # Продолжить оставшихся
    python scripts/rap_scraper_cli.py scraping --artist "Drake"   # Один артист
    python scripts/rap_scraper_cli.py scraping --test             # Тестовый режим
    python scripts/rap_scraper_cli.py scraping --debug           # Отладочный режим
  
  🎵 Обогащение Spotify метаданными:
    python scripts/rap_scraper_cli.py spotify
    python scripts/rap_scraper_cli.py spotify --continue
  
  🤖 ML анализ:
    python scripts/rap_scraper_cli.py analysis --analyzer gemma
    python scripts/rap_scraper_cli.py analysis --analyzer multi
  
  📊 Мониторинг:
    python scripts/rap_scraper_cli.py monitoring --component database
    python scripts/rap_scraper_cli.py monitoring --component analysis
  
  🛠️ Утилиты:
    python scripts/rap_scraper_cli.py utils --utility cleanup
    python scripts/rap_scraper_cli.py utils --utility cleanup --execute
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='📊 Статус проекта')
    
    # Scraping command
    scraping_parser = subparsers.add_parser('scraping', help='🕷️ Скрапинг данных')
    scraping_parser.add_argument('--limit', type=int, help='Лимит треков для скрапинга')
    scraping_parser.add_argument('--artist', type=str, help='Скрапинг одного артиста')
    scraping_parser.add_argument('--test', action='store_true', help='Тестовый режим (малый набор данных)')
    scraping_parser.add_argument('--debug', action='store_true', help='Отладочный режим с детальными логами')
    scraping_parser.add_argument('--continue', dest='continue_mode', action='store_true', 
                               help='Продолжить скрапинг оставшихся артистов')
    
    # Spotify command
    spotify_parser = subparsers.add_parser('spotify', help='🎵 Spotify enhancement')
    spotify_parser.add_argument('--continue', dest='continue_mode', action='store_true', 
                               help='Продолжить обогащение оставшихся данных')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analysis', help='🤖 ML анализ')
    analysis_parser.add_argument('--analyzer', choices=['gemma', 'multi', 'langchain'],
                                help='Выбор анализатора')
    
    # Monitoring command
    monitoring_parser = subparsers.add_parser('monitoring', help='📊 Мониторинг')
    monitoring_parser.add_argument('--component', choices=['database', 'analysis', 'gemma'],
                                  help='Компонент для мониторинга')
    
    # Utils command
    utils_parser = subparsers.add_parser('utils', help='🛠️ Утилиты')
    utils_parser.add_argument('--utility', choices=['cleanup', 'migrate', 'spotify-setup'],
                             help='Выбор утилиты')
    utils_parser.add_argument('--execute', action='store_true',
                             help='Выполнить действие (для cleanup)')
    
    return parser

def main():
    """Главная функция CLI"""
    parser = create_parser()
    args = parser.parse_args()
    
    print_banner()
    
    if not args.command:
        print("💡 Используйте --help для просмотра доступных команд")
        print("🚀 Быстрый старт: python scripts/rap_scraper_cli.py status")
        return
    
    if args.command == 'status':
        show_status()
    elif args.command == 'scraping':
        run_scraping(args)
    elif args.command == 'spotify':
        run_spotify_enhancement(args)
    elif args.command == 'analysis':
        run_analysis(args)
    elif args.command == 'monitoring':
        run_monitoring(args)
    elif args.command == 'utils':
        run_utilities(args)
    else:
        print(f"❌ Неизвестная команда: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()
