#!/usr/bin/env python3
"""
Автоматическая очистка проекта от ненужных файлов
Версия: 2.0 с категоризацией и smart detection
"""

import os
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import json
from datetime import datetime

class ProjectCleaner:
    """Smart project cleanup с категоризацией файлов"""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.cleanup_stats = {
            "files_removed": 0,
            "dirs_removed": 0, 
            "size_saved": 0,
            "categories": {}
        }
    
    def get_file_categories(self) -> Dict[str, List[str]]:
        """Категории файлов для удаления"""
        return {
            # Deprecated тестовые файлы
            "deprecated_tests": [
                "test_gemma.py",
                "test_ollama.py", 
                "test_ollama_connection.py",
                "test_analysis.py"
            ],
            
            # Устаревшие анализаторы
            "deprecated_analyzers": [
                "check_gemini.py",
                "gemini_simple_analyzer.py", 
                "gemma_analyzer.py"
            ],
            
            # Экспериментальные/временные скрипты  
            "experimental": [
                "analyze_results.py",
                "clean_lyrics.py", 
                "enhance_existing_songs.py",
                "hybrid_strategy.py",
                "ml_architecture_explanation.py",
                "calculate_analysis_time.py"
            ],
            
            # Устаревшие основные файлы
            "deprecated_core": [
                "rap_scraper.py"  # Заменен на rap_scraper_optimized.py
            ],
            
            # Логи и временные файлы
            "temp_files": [
                "ai_analysis.log",
                "gemini_simple_analysis.log",
                "gemma_analysis.log",
                "cleanup_plan.md"
            ],
            
            # SQLite временные файлы  
            "sqlite_temp": [
                "rap_lyrics.db-shm",
                "rap_lyrics.db-wal"
            ],
            
            # Системные папки
            "system_dirs": [
                "__pycache__",
                ".pytest_cache", 
                ".mypy_cache"
            ]
        }
    
    def get_protected_files(self) -> List[str]:
        """Критически важные файлы - НЕ ТРОГАТЬ!"""
        return [
            # Core infrastructure  
            "rap_lyrics.db",                    # Основная база данных
            "models.py",                        # Pydantic модели
            "requirements.txt",                 # Зависимости
            "claude.md",                        # AI контекст
            "AI_ONBOARDING_CHECKLIST.md",       # AI onboarding
            "Makefile",                         # TDD workflow
            ".env",                             # API credentials
            ".env.example",                     # Template
            ".gitignore",                       # Git config
            
            # Active scrapers & enhancers
            "rap_scraper_optimized.py",         # Основной скрапер
            "spotify_enhancer.py",              # Spotify API
            "bulk_spotify_enhancement.py",      # Массовая обработка
            "setup_spotify.py",                 # Spotify setup
            "migrate_database.py",              # DB migrations
            
            # Active tests (Case 9+)
            "test_langchain.py",                # LangChain анализ
            "test_optimized_scraper.py",        # Scraper testing
            "test_artists.json",                # Тестовые данные
            
            # Test infrastructure
            "tests/conftest.py",                # Pytest fixtures
            "tests/test_models.py",             # Model tests  
            "tests/test_spotify_enhancer.py",   # API tests
            "tests/test_database.py",           # DB tests
            
            # Analyzers (полезны для ML)
            "langchain_analyzer.py",            # LangChain integration
            "ollama_analyzer.py",               # Local AI
            "optimized_analyzer.py",            # Optimized analysis
            "multi_model_analyzer.py",          # Model comparison
            
            # Utilities
            "check_db.py",                      # Database status
            "check_analysis_status.py",         # Analysis monitoring
            "monitor_gemma_progress.py",        # Progress tracking
            "gemma_27b_fixed.py",               # Gemma integration
            
            # Documentation (вся папка)
            "AI_Engineer_Journal/",             # Полная документация
            
            # Data directories
            "analysis_results/",                # ML outputs
            "enhanced_data/",                   # Enriched datasets
            "tests/"                            # Test suite
        ]
    
    def find_backup_files(self) -> List[Path]:
        """Найти файлы бэкапов по pattern"""
        patterns = [
            "rap_lyrics_backup_*.db",
            "*_backup_*.json",
            "*.bak",
            "*.old"
        ]
        
        backup_files = []
        for pattern in patterns:
            backup_files.extend(self.project_dir.glob(pattern))
        return backup_files
    
    def get_file_size(self, file_path: Path) -> int:
        """Получить размер файла в bytes"""
        try:
            return file_path.stat().st_size
        except OSError:
            return 0
    
    def format_size(self, size_bytes: int) -> str:
        """Форматировать размер для человека"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def preview_cleanup(self) -> Dict:
        """Preview что будет удалено"""
        categories = self.get_file_categories()
        protected = set(self.get_protected_files())
        
        preview = {
            "files": {},
            "dirs": {},
            "backups": [],
            "total_size": 0,
            "conflicts": []
        }
        
        # Анализируем файлы по категориям
        for category, files in categories.items():
            preview["files"][category] = []
            
            for file_name in files:
                file_path = self.project_dir / file_name
                
                # Проверяем конфликты с protected files
                if file_name in protected:
                    preview["conflicts"].append(file_name)
                    continue
                
                if file_path.exists():
                    size = self.get_file_size(file_path)
                    preview["files"][category].append({
                        "name": file_name,
                        "size": size,
                        "size_human": self.format_size(size)
                    })
                    preview["total_size"] += size
        
        # Анализируем папки
        for dir_name in categories["system_dirs"]:
            dir_path = self.project_dir / dir_name
            if dir_path.exists():
                dir_size = sum(
                    self.get_file_size(f) 
                    for f in dir_path.rglob("*") 
                    if f.is_file()
                )
                preview["dirs"][dir_name] = {
                    "size": dir_size,
                    "size_human": self.format_size(dir_size)
                }
                preview["total_size"] += dir_size
        
        # Анализируем бэкапы
        backup_files = self.find_backup_files()
        for backup in backup_files:
            size = self.get_file_size(backup)
            preview["backups"].append({
                "name": backup.name,
                "size": size, 
                "size_human": self.format_size(size)
            })
            preview["total_size"] += size
        
        return preview
    
    def execute_cleanup(self, dry_run: bool = True) -> bool:
        """Выполнить очистку"""
        preview = self.preview_cleanup()
        
        print("🗑️  PROJECT CLEANUP")
        print("=" * 50)
        
        if dry_run:
            print("🧪 DRY RUN MODE - файлы НЕ будут удалены")
            print("   Для реального удаления: --execute")
            print("-" * 50)
        
        # Проверяем конфликты
        if preview["conflicts"]:
            print("⚠️  ВНИМАНИЕ! Конфликты с protected files:")
            for conflict in preview["conflicts"]:
                print(f"   ❌ {conflict} - в списке удаления, но защищен!")
            print("   Cleanup отменен для безопасности.")
            return False
        
        # Показываем план по категориям
        total_files = 0
        for category, files in preview["files"].items():
            if files:
                print(f"\n📁 {category.upper()}:")
                for file_info in files:
                    status = "🔍" if dry_run else "✅"
                    print(f"   {status} {file_info['name']} ({file_info['size_human']})")
                    total_files += 1
        
        # Показываем папки
        if preview["dirs"]:
            print("\n📂 SYSTEM DIRECTORIES:")
            for dir_name, dir_info in preview["dirs"].items():
                status = "🔍" if dry_run else "✅"
                print(f"   {status} {dir_name}/ ({dir_info['size_human']})")
        
        # Показываем бэкапы
        if preview["backups"]:
            print("\n💾 BACKUP FILES:")
            for backup in preview["backups"]:
                status = "🔍" if dry_run else "✅"
                print(f"   {status} {backup['name']} ({backup['size_human']})")
        
        print(f"\n📊 SUMMARY:")
        print(f"   🗃️  Files to clean: {total_files}")
        print(f"   📁 Dirs to clean: {len(preview['dirs'])}")
        print(f"   💾 Backups found: {len(preview['backups'])}")
        print(f"   💾 Total size: {self.format_size(preview['total_size'])}")
        
        if dry_run:
            return True
        
        # Реальное удаление
        success = self._execute_removal(preview)
        
        # Сохраняем статистику
        self._save_cleanup_log(preview)
        
        return success
    
    def _execute_removal(self, preview: Dict) -> bool:
        """Выполнить реальное удаление"""
        try:
            # Удаляем файлы по категориям
            for category, files in preview["files"].items():
                for file_info in files:
                    file_path = self.project_dir / file_info["name"]
                    if file_path.exists():
                        file_path.unlink()
                        self.cleanup_stats["files_removed"] += 1
                        self.cleanup_stats["size_saved"] += file_info["size"]
            
            # Удаляем папки
            for dir_name in preview["dirs"]:
                dir_path = self.project_dir / dir_name
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    self.cleanup_stats["dirs_removed"] += 1
            
            # Удаляем бэкапы
            for backup in preview["backups"]:
                backup_path = self.project_dir / backup["name"]
                if backup_path.exists():
                    backup_path.unlink()
                    self.cleanup_stats["files_removed"] += 1
                    self.cleanup_stats["size_saved"] += backup["size"]
            
            print(f"\n🎉 Cleanup успешно завершен!")
            print(f"   📁 Удалено файлов: {self.cleanup_stats['files_removed']}")
            print(f"   🗂️  Удалено папок: {self.cleanup_stats['dirs_removed']}")
            print(f"   💾 Освобождено места: {self.format_size(self.cleanup_stats['size_saved'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка при очистке: {e}")
            return False
    
    def _save_cleanup_log(self, preview: Dict):
        """Сохранить лог очистки для истории"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "files_removed": self.cleanup_stats["files_removed"],
            "dirs_removed": self.cleanup_stats["dirs_removed"],
            "size_saved": self.cleanup_stats["size_saved"],
            "size_saved_human": self.format_size(self.cleanup_stats["size_saved"]),
            "categories_cleaned": list(preview["files"].keys())
        }
        
        log_file = self.project_dir / "cleanup_history.json"
        
        # Читаем существующую историю
        history = []
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                history = []
        
        # Добавляем новую запись
        history.append(log_entry)
        
        # Сохраняем (последние 10 записей)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(history[-10:], f, indent=2, ensure_ascii=False)
    
    def show_protected_files(self):
        """Показать список защищенных файлов"""
        protected = self.get_protected_files()
        
        print("🛡️  PROTECTED FILES (НЕ будут удалены):")
        print("-" * 50)
        
        categories = {
            "Core": ["rap_lyrics.db", "models.py", "claude.md", "requirements.txt"],
            "Scrapers": ["rap_scraper_optimized.py", "spotify_enhancer.py", "bulk_spotify_enhancement.py"],
            "Analyzers": ["langchain_analyzer.py", "ollama_analyzer.py", "multi_model_analyzer.py"],
            "Tests": ["test_langchain.py", "test_optimized_scraper.py", "tests/"],
            "Utils": ["check_db.py", "setup_spotify.py", "migrate_database.py"]
        }
        
        for category, examples in categories.items():
            print(f"\n📁 {category}:")
            for file_name in examples:
                if file_name in protected:
                    exists = "✅" if (self.project_dir / file_name).exists() else "❌"
                    print(f"   {exists} {file_name}")
    
    def smart_scan(self) -> Dict:
        """Умное сканирование для обнаружения кандидатов на удаление"""
        candidates = {
            "large_logs": [],
            "old_backups": [], 
            "unused_scripts": [],
            "temp_outputs": []
        }
        
        # Ищем большие лог файлы (>10MB)
        for log_file in self.project_dir.rglob("*.log"):
            if self.get_file_size(log_file) > 10 * 1024 * 1024:  # 10MB
                candidates["large_logs"].append({
                    "name": log_file.name,
                    "size": self.format_size(self.get_file_size(log_file))
                })
        
        # Ищем старые бэкапы (>30 дней)
        import time
        current_time = time.time()
        for backup in self.find_backup_files():
            file_age = current_time - backup.stat().st_mtime
            if file_age > 30 * 24 * 3600:  # 30 дней
                candidates["old_backups"].append({
                    "name": backup.name,
                    "age_days": int(file_age / (24 * 3600)),
                    "size": self.format_size(self.get_file_size(backup))
                })
        
        # Ищем потенциально неиспользуемые скрипты
        for py_file in self.project_dir.glob("*.py"):
            if py_file.name.startswith("temp_") or py_file.name.startswith("test_"):
                # Проверяем, есть ли references в активных файлах
                if not self._has_references(py_file.name):
                    candidates["unused_scripts"].append({
                        "name": py_file.name,
                        "size": self.format_size(self.get_file_size(py_file))
                    })
        
        # Ищем временные выходные файлы
        for pattern in ["*.tmp", "*.temp", "*_output.json", "*_results.csv"]:
            for temp_file in self.project_dir.glob(pattern):
                candidates["temp_outputs"].append({
                    "name": temp_file.name,
                    "size": self.format_size(self.get_file_size(temp_file))
                })
        
        return candidates
    
    def _has_references(self, filename: str) -> bool:
        """Проверить, есть ли ссылки на файл в активном коде"""
        filename_base = filename.replace('.py', '')
        
        # Проверяем imports и mentions в основных файлах
        main_files = [
            "rap_scraper_optimized.py",
            "spotify_enhancer.py", 
            "bulk_spotify_enhancement.py",
            "multi_model_analyzer.py"
        ]
        
        for main_file in main_files:
            file_path = self.project_dir / main_file
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    if filename_base in content:
                        return True
                except UnicodeDecodeError:
                    continue
        
        return False

def main():
    """Main cleanup interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart project cleanup tool")
    parser.add_argument("--execute", action="store_true", help="Выполнить реальное удаление")
    parser.add_argument("--show-protected", action="store_true", help="Показать защищенные файлы")
    parser.add_argument("--smart-scan", action="store_true", help="Умное сканирование кандидатов")
    parser.add_argument("--category", type=str, help="Удалить только определенную категорию")
    
    args = parser.parse_args()
    
    cleaner = ProjectCleaner()
    
    if args.show_protected:
        cleaner.show_protected_files()
        return
    
    if args.smart_scan:
        print("🔍 SMART SCAN - поиск кандидатов на удаление")
        print("=" * 50)
        
        candidates = cleaner.smart_scan()
        
        for category, items in candidates.items():
            if items:
                print(f"\n📁 {category.upper()}:")
                for item in items:
                    print(f"   🔍 {item['name']} ({item.get('size', 'N/A')})")
        
        print("\nИспользуй --execute для удаления стандартных категорий")
        return
    
    # Основная очистка
    if args.category:
        print(f"🎯 Очистка только категории: {args.category}")
        # TODO: Implement category-specific cleanup
        return
    
    # Стандартная очистка
    success = cleaner.execute_cleanup(dry_run=not args.execute)
    
    if not args.execute and success:
        print("\n" + "=" * 50)
        print("💡 Для выполнения реального удаления:")
        print("   python cleanup_project.py --execute")
        print("\n💡 Другие опции:")
        print("   python cleanup_project.py --show-protected")
        print("   python cleanup_project.py --smart-scan")

if __name__ == "__main__":
    main()