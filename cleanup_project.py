#!/usr/bin/env python3
"""
Автоматическая очистка проекта от ненужных файлов
"""

import os
import shutil
from pathlib import Path

def cleanup_project(dry_run=True):
    """Удаляет ненужные файлы из проекта
    
    Args:
        dry_run: Если True, только показывает что будет удалено
    """
    
    project_dir = Path(".")
    
    # Файлы для удаления (ОСТОРОЖНО - проверяем перед удалением!)
    files_to_remove = [
        # Deprecated тестовые файлы (НЕ ТРОГАЕМ АКТУАЛЬНЫЕ!)
        "test_gemma.py",
        # "test_langchain.py",  # ВАЖНО! Активный файл для Case 9
        "test_ollama.py",
        "test_ollama_connection.py",
        # "test_optimized_scraper.py",  # ВАЖНО! Активный тестер
        "test_analysis.py",
        # "test_artists.json",  # ВАЖНО! Тестовые данные
        
        # Устаревшие анализаторы (можно удалять)
        "check_gemini.py",
        # "langchain_analyzer.py",  # Оставляем - может быть полезен
        # "ollama_analyzer.py",     # Оставляем - может быть полезен
        # "optimized_analyzer.py",  # Оставляем - может быть полезен
        "gemini_simple_analyzer.py",
        "gemma_analyzer.py",
        
        # Вспомогательные/экспериментальные
        "analyze_results.py",
        "clean_lyrics.py",
        "enhance_existing_songs.py", 
        "hybrid_strategy.py",
        "ml_architecture_explanation.py",
        
        # Старые скраперы
        "rap_scraper.py",
        
        # Логи (осторожно - могут содержать важную информацию)
        "ai_analysis.log",
        "gemini_simple_analysis.log", 
        "gemma_analysis.log",
        
        # Временные файлы БД (НЕ ОСНОВНУЮ БД!)
        "rap_lyrics.db-shm",
        "rap_lyrics.db-wal",
        
        # План очистки (после выполнения)
        "cleanup_plan.md",
        "calculate_analysis_time.py"
    ]
    
    # Папки для удаления
    dirs_to_remove = [
        "__pycache__"
    ]
    
    # Бэкапы БД (по маске)
    backup_pattern = "rap_lyrics_backup_*.db"
    
    # КРИТИЧЕСКИ ВАЖНЫЕ ФАЙЛЫ - НЕ УДАЛЯТЬ!
    protected_files = [
        # Основная инфраструктура
        "rap_lyrics.db",                    # Основная база данных
        "models.py",                        # Pydantic модели
        "requirements.txt",                 # Зависимости
        "claude.md",                        # Контекст проекта
        "AI_ONBOARDING_CHECKLIST.md",       # Onboarding guide
        "Makefile",                         # TDD workflow
        
        # Активные скраперы и enhancers
        "rap_scraper_optimized.py",         # Основной скрапер
        "spotify_enhancer.py",              # Spotify API
        "bulk_spotify_enhancement.py",      # Массовая обработка
        "setup_spotify.py",                 # Spotify setup
        
        # Активные тесты
        "test_langchain.py",                # LangChain анализ (Case 9)
        "test_optimized_scraper.py",        # Scraper testing  
        "test_artists.json",                # Тестовые данные
        
        # Тестовая инфраструктура
        "tests/conftest.py",                # Pytest fixtures
        "tests/test_models.py",             # Model tests
        "tests/test_spotify_enhancer.py",   # API tests
        
        # Анализаторы (могут быть полезны)
        "langchain_analyzer.py",            # LangChain integration
        "ollama_analyzer.py",               # Local AI
        "optimized_analyzer.py",            # Optimized analysis
        "multi_model_analyzer.py",          # Model comparison
        
        # Утилиты
        "check_db.py",                      # Database status
        "check_analysis_status.py",         # Analysis monitoring
        "monitor_gemma_progress.py",        # Progress tracking
        "gemma_27b_fixed.py",               # Gemma integration
        
        # Документация
        "AI_Engineer_Journal/"              # Вся документация
    ]
    
    print("🗑️  ОЧИСТКА ПРОЕКТА")
    print("=" * 40)
    
    if dry_run:
        print("🧪 DRY RUN MODE - файлы НЕ будут удалены")
        print("   Для реального удаления: cleanup_project(dry_run=False)")
        print("-" * 40)
    
    # Проверяем защищенные файлы
    print("🛡️  Проверка защищенных файлов...")
    protected_found = []
    for protected_file in protected_files:
        if (project_dir / protected_file).exists():
            protected_found.append(protected_file)
    
    print(f"✅ Найдено {len(protected_found)}/{len(protected_files)} защищенных файлов")
    
    # Проверяем, не пытаемся ли мы удалить защищенные файлы
    conflicts = []
    for file_to_remove in files_to_remove:
        if file_to_remove in protected_files:
            conflicts.append(file_to_remove)
    
    if conflicts:
        print("⚠️  ВНИМАНИЕ! Конфликт с защищенными файлами:")
        for conflict in conflicts:
            print(f"   ❌ {conflict} - в списке удаления, но защищен!")
        print("   Удаление отменено для безопасности.")
        return False
    
    removed_files = []
    removed_dirs = []
    
    # Удаляем файлы
    for file_name in files_to_remove:
        file_path = project_dir / file_name
        if file_path.exists():
            if dry_run:
                print(f"🔍 Будет удален: {file_name}")
                removed_files.append(file_name)
            else:
                try:
                    file_path.unlink()
                    removed_files.append(file_name)
                    print(f"✅ Удален файл: {file_name}")
                except Exception as e:
                    print(f"❌ Ошибка удаления {file_name}: {e}")
    
    # Удаляем папки
    for dir_name in dirs_to_remove:
        dir_path = project_dir / dir_name
        if dir_path.exists():
            if dry_run:
                print(f"🔍 Будет удалена папка: {dir_name}")
                removed_dirs.append(dir_name)
            else:
                try:
                    shutil.rmtree(dir_path)
                    removed_dirs.append(dir_name)
                    print(f"✅ Удалена папка: {dir_name}")
                except Exception as e:
                    print(f"❌ Ошибка удаления папки {dir_name}: {e}")
    
    # Удаляем бэкапы БД
    backup_files = list(project_dir.glob(backup_pattern))
    for backup_file in backup_files:
        try:
            backup_file.unlink()
            removed_files.append(backup_file.name)
            print(f"✅ Удален бэкап: {backup_file.name}")
        except Exception as e:
            print(f"❌ Ошибка удаления бэкапа {backup_file.name}: {e}")
    
    print(f"\n📊 СТАТИСТИКА ОЧИСТКИ:")
    print(f"   🗃️  Удалено файлов: {len(removed_files)}")
    print(f"   📁 Удалено папок: {len(removed_dirs)}")
    
    if removed_files:
        print(f"\n🗑️  Удаленные файлы:")
        for file_name in sorted(removed_files):
            print(f"   - {file_name}")
    
    if removed_dirs:
        print(f"\n📁 Удаленные папки:")
        for dir_name in removed_dirs:
            print(f"   - {dir_name}")
    
    # Показываем оставшиеся основные файлы
    important_files = [
        # Core infrastructure
        "rap_lyrics.db",
        "models.py", 
        "claude.md",
        "Makefile",
        
        # Active scrapers
        "rap_scraper_optimized.py",
        "spotify_enhancer.py",
        "bulk_spotify_enhancement.py",
        
        # Active tests
        "test_langchain.py",
        "test_optimized_scraper.py", 
        "test_artists.json",
        
        # Test infrastructure
        "tests/test_models.py",
        "tests/test_spotify_enhancer.py",
        
        # Analysis tools
        "multi_model_analyzer.py",
        "langchain_analyzer.py",
        "check_db.py",
        
        # Config
        "requirements.txt",
        ".env.example"
    ]
    
    print(f"\n✅ ОСНОВНЫЕ ФАЙЛЫ (оставлены):")
    for file_name in important_files:
        if (project_dir / file_name).exists():
            print(f"   ✓ {file_name}")
        else:
            print(f"   ❌ {file_name} (отсутствует)")
    
    print(f"\n🎉 Очистка завершена! Проект готов к работе.")

if __name__ == "__main__":
    print("🛡️  БЕЗОПАСНАЯ ОЧИСТКА ПРОЕКТА")
    print("=" * 50)
    
    # Сначала dry run
    print("1️⃣ ПРОВЕРКА (Dry Run):")
    cleanup_project(dry_run=True)
    
    print("\n" + "=" * 50)
    response = input("\n⚠️  Выполнить реальное удаление? (y/N): ").strip().lower()
    
    if response == 'y':
        print("\n2️⃣ РЕАЛЬНОЕ УДАЛЕНИЕ:")
        cleanup_project(dry_run=False)
    else:
        print("✅ Удаление отменено. Файлы сохранены.")
