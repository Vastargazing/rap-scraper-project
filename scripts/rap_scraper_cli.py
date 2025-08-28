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
   📊 54.5K+ tracks | 345 artists | Spotify enriched | AI analyzed
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

def run_mlfeatures(args):
    """Извлечение расширенных ML-фичей"""
    print("🎯 Система извлечения ML-фичей...")
    
    if args.demo:
        print("📊 Запуск полной демонстрации возможностей")
        try:
            import subprocess
            script_path = Path(__file__).parent / "development" / "demo_simplified_ml_features.py"
            subprocess.run([sys.executable, str(script_path)])
        except Exception as e:
            print(f"❌ Ошибка демонстрации: {e}")
    
    elif args.text:
        print(f"📝 Анализ текста: '{args.text[:50]}...'")
        try:
            from src.analyzers.simplified_feature_analyzer import extract_simplified_features
            import json
            
            features = extract_simplified_features(args.text)
            
            print("\n🔍 ИЗВЛЕЧЕННЫЕ ФИЧИ:")
            print("-" * 40)
            for key, value in features.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
            
            if args.export and args.output:
                if args.export == "json":
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump({'text': args.text, 'features': features}, f, ensure_ascii=False, indent=2)
                    print(f"\n💾 Результаты сохранены в: {args.output}")
                    
        except Exception as e:
            print(f"❌ Ошибка анализа текста: {e}")
    
    elif args.file:
        print(f"📄 Анализ файла: {args.file}")
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            from src.analyzers.simplified_feature_analyzer import extract_simplified_features
            import json
            
            features = extract_simplified_features(text)
            
            print("\n🔍 ИЗВЛЕЧЕННЫЕ ФИЧИ:")
            print("-" * 40)
            for key, value in features.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
            
            if args.export and args.output:
                if args.export == "json":
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump({'file': args.file, 'text': text, 'features': features}, f, ensure_ascii=False, indent=2)
                    print(f"\n💾 Результаты сохранены в: {args.output}")
                        
        except Exception as e:
            print(f"❌ Ошибка анализа файла: {e}")
    
    elif args.batch:
        print(f"📦 Пакетная обработка {args.batch} записей из БД")
        try:
            import sqlite3
            from src.analyzers.simplified_feature_analyzer import extract_simplified_features
            import json
            import time
            
            # Подключаемся к БД
            db_path = "data/rap_lyrics.db"
            conn = sqlite3.connect(db_path)
            
            # Получаем записи
            query = "SELECT artist, title, lyrics FROM songs WHERE lyrics IS NOT NULL LIMIT ?"
            cursor = conn.execute(query, (args.batch,))
            songs = cursor.fetchall()
            conn.close()
            
            print(f"📊 Загружено {len(songs)} песен из БД")
            
            # Обрабатываем
            results = []
            start_time = time.time()
            
            for i, (artist, title, lyrics) in enumerate(songs):
                try:
                    features = extract_simplified_features(lyrics)
                    results.append({
                        'artist': artist,
                        'title': title,
                        'features': features
                    })
                    
                    if (i + 1) % 10 == 0:
                        print(f"   Обработано: {i + 1}/{len(songs)}")
                        
                except Exception as e:
                    print(f"   ⚠️ Ошибка обработки '{artist} - {title}': {e}")
            
            processing_time = time.time() - start_time
            
            print(f"\n✅ Пакетная обработка завершена:")
            print(f"   Время: {processing_time:.2f}с")
            print(f"   Успешно: {len(results)}/{len(songs)}")
            print(f"   Скорость: {len(results)/processing_time:.1f} треков/сек")
            
            # Сохраняем результаты
            if args.export and args.output:
                if args.export == "json":
                    with open(args.output, 'w', encoding='utf-8') as f:
                        json.dump({
                            'processing_info': {
                                'total_processed': len(results),
                                'processing_time': processing_time,
                                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                            },
                            'results': results
                        }, f, ensure_ascii=False, indent=2)
                    print(f"💾 Результаты сохранены в: {args.output}")
                elif args.export == "csv":
                    try:
                        import pandas as pd
                        
                        # Создаем плоскую структуру для CSV
                        flat_data = []
                        for result in results:
                            row = {
                                'artist': result['artist'],
                                'title': result['title']
                            }
                            row.update(result['features'])
                            flat_data.append(row)
                        
                        df = pd.DataFrame(flat_data)
                        df.to_csv(args.output, index=False, encoding='utf-8')
                        print(f"💾 CSV сохранен в: {args.output}")
                    except ImportError:
                        print("❌ Pandas не установлен. Используйте JSON экспорт.")
                        
        except Exception as e:
            print(f"❌ Ошибка пакетной обработки: {e}")
    
    else:
        print("📋 Доступные опции ML-фичей:")
        print("   📊 --demo              - Полная демонстрация")
        print("   📝 --text 'текст'      - Анализ текста")
        print("   📄 --file путь         - Анализ файла")
        print("   📦 --batch N           - Пакетная обработка N записей")
        print("   💾 --export json/csv   - Экспорт результатов")
        print("   📁 --output путь       - Файл для сохранения")
        print("\nПримеры:")
        print("   python scripts/rap_scraper_cli.py mlfeatures --demo")
        print("   python scripts/rap_scraper_cli.py mlfeatures --text 'мой рэп текст'")
        print("   python scripts/rap_scraper_cli.py mlfeatures --batch 100 --export json --output features.json")

def run_monitoring(args):
    """Мониторинг и статистика"""
    print("📊 Система мониторинга...")
    
    if args.component == "database" or args.component == "all":
        print("🗄️ Проверка состояния базы данных")
        show_status()
        if args.component == "all":
            print("\n" + "="*50 + "\n")
    
    if args.component == "analysis" or args.component == "all":
        print("🤖 Статус AI анализа")
        try:
            import subprocess
            subprocess.run([sys.executable, "monitoring/check_analysis_status.py"])
        except Exception as e:
            print(f"❌ Ошибка мониторинга анализа: {e}")
        if args.component == "all":
            print("\n" + "="*50 + "\n")
    
    if args.component == "gemma" or args.component == "all":
        print("🔥 Мониторинг прогресса Gemma")
        try:
            import subprocess
            subprocess.run([sys.executable, "monitoring/monitor_gemma_progress.py"])
        except Exception as e:
            print(f"❌ Ошибка мониторинга Gemma: {e}")
        if args.component == "all":
            print("\n" + "="*50 + "\n")
    
    if args.component not in ["database", "analysis", "gemma", "all"]:
        print("📋 Доступные компоненты мониторинга:")
        print("   🗄️ database - Состояние БД")
        print("   🤖 analysis - AI анализ")
        print("   🔥 gemma    - Gemma прогресс")
        print("   📊 all      - Все компоненты")

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
  
  🎯 Извлечение ML-фичей (НОВОЕ!):
    python scripts/rap_scraper_cli.py mlfeatures --demo
    python scripts/rap_scraper_cli.py mlfeatures --text "мой рэп текст"
    python scripts/rap_scraper_cli.py mlfeatures --batch 100 --export json --output features.json
  
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
    
    # ML Features command (NEW!)
    mlfeatures_parser = subparsers.add_parser('mlfeatures', help='🎯 Извлечение ML-фичей')
    mlfeatures_parser.add_argument('--demo', action='store_true', 
                                  help='Демонстрация всех возможностей')
    mlfeatures_parser.add_argument('--text', type=str, 
                                  help='Анализ конкретного текста')
    mlfeatures_parser.add_argument('--file', type=str, 
                                  help='Анализ текста из файла')
    mlfeatures_parser.add_argument('--batch', type=int, 
                                  help='Пакетная обработка N записей из БД')
    mlfeatures_parser.add_argument('--export', choices=['json', 'csv'], 
                                  help='Экспорт результатов')
    mlfeatures_parser.add_argument('--output', type=str, 
                                  help='Путь для сохранения результатов')
    
    # Monitoring command
    monitoring_parser = subparsers.add_parser('monitoring', help='📊 Мониторинг')
    monitoring_parser.add_argument('--component', choices=['database', 'analysis', 'gemma', 'all'],
                                  help='Компонент для мониторинга (all = все компоненты)')
    
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
    elif args.command == 'mlfeatures':
        run_mlfeatures(args)
    elif args.command == 'monitoring':
        run_monitoring(args)
    elif args.command == 'utils':
        run_utilities(args)
    else:
        print(f"❌ Неизвестная команда: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main()
