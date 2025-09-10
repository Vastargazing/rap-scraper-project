#!/usr/bin/env python3
"""
🤖 AI Context Manager — интеллектуальный менеджер контекста для AI ассистентов в VS Code

НАЗНАЧЕНИЕ:
- Динамическое управление контекстом для разных типов задач
- Приоритезация файлов по важности (1-5)
- Генерация VS Code workspace с оптимальными настройками
- Предупреждения для AI о legacy коде и архитектурных ограничениях

ИСПОЛЬЗОВАНИЕ:
python scripts/tools/ai_context_manager.py --task debug

ЗАВИСИМОСТИ:
- Python 3.8+
- dataclasses, pathlib

РЕЗУЛЬТАТ:
- Автоматизированное управление контекстом для AI
- Предупреждения о legacy и архитектурных рисках

АВТОР: Vastargazing
ДАТА: Сентябрь 2025
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import re

@dataclass
class FileContext:
    path: str
    priority: int  # 1-5, где 5 = критично важный
    category: str  # database, analyzer, cli, config, test, legacy
    description: str
    last_modified: str
    size_lines: int
    dependencies: List[str]
    
@dataclass
class AIPromptContext:
    task_type: str  # debug, develop, analyze, refactor
    relevant_files: List[str]
    context_summary: str
    suggested_commands: List[str]
    warnings: List[str]

class AIContextManager:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        # Сохраняем контекст в results/ согласно архитектуре проекта
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(exist_ok=True)
        self.context_file = self.results_dir / ".ai_context.json"
        self.file_contexts = {}
        self.load_context()
        
    def load_context(self):
        """Загружает сохраненный контекст или создает новый"""
        if self.context_file.exists():
            with open(self.context_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.file_contexts = {
                    path: FileContext(**ctx) for path, ctx in data.items()
                }
        else:
            self._build_initial_context()
    
    def save_context(self):
        """Сохраняет контекст в файл"""
        data = {
            path: asdict(ctx) for path, ctx in self.file_contexts.items()
        }
        with open(self.context_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _build_initial_context(self):
        """Строит первоначальный контекст проекта"""
        
        # Критически важные файлы
        critical_files = {
            "main.py": FileContext(
                path="main.py",
                priority=5,
                category="cli",
                description="Единая точка входа, центральная оркестрация",
                last_modified="",
                size_lines=0,
                dependencies=["src.analyzers", "src.cli", "src.models"]
            ),
            "src/database/postgres_adapter.py": FileContext(
                path="src/database/postgres_adapter.py",
                priority=5,
                category="database",
                description="PostgreSQL адаптер, connection pooling, async операции",
                last_modified="",
                size_lines=0,
                dependencies=["asyncpg", "psycopg2"]
            ),
            "config.yaml": FileContext(
                path="config.yaml",
                priority=4,
                category="config",
                description="Централизованная конфигурация системы",
                last_modified="",
                size_lines=0,
                dependencies=[]
            ),
            "docs/claude.md": FileContext(
                path="docs/claude.md",
                priority=5,
                category="docs",
                description="Основной контекст для AI ассистентов",
                last_modified="",
                size_lines=0,
                dependencies=[]
            )
        }
        
        # Добавляем все анализаторы
        analyzer_dir = self.project_root / "src" / "analyzers"
        if analyzer_dir.exists():
            for analyzer_file in analyzer_dir.glob("*.py"):
                if analyzer_file.name != "__init__.py":
                    self.file_contexts[str(analyzer_file)] = FileContext(
                        path=str(analyzer_file),
                        priority=3,
                        category="analyzer",
                        description=f"AI анализатор: {analyzer_file.stem}",
                        last_modified="",
                        size_lines=0,
                        dependencies=["src.models"]
                    )
        
        self.file_contexts.update(critical_files)
        self._update_file_stats()
        self.save_context()
    
    def _update_file_stats(self):
        """Обновляет статистику файлов"""
        for path, context in self.file_contexts.items():
            file_path = Path(path)
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        context.size_lines = len(lines)
                    
                    context.last_modified = datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).strftime("%Y-%m-%d %H:%M")
                except:
                    pass
    
    def generate_ai_context(self, task_type: str, query: str = "") -> AIPromptContext:
        """Генерирует оптимальный контекст для AI ассистента"""
        
        relevant_files = self._select_relevant_files(task_type, query)
        context_summary = self._generate_context_summary(relevant_files, task_type)
        commands = self._suggest_commands(task_type, relevant_files)
        warnings = self._generate_warnings(task_type, relevant_files)
        
        return AIPromptContext(
            task_type=task_type,
            relevant_files=relevant_files,
            context_summary=context_summary,
            suggested_commands=commands,
            warnings=warnings
        )
    
    def _select_relevant_files(self, task_type: str, query: str) -> List[str]:
        """Выбирает релевантные файлы для задачи"""
        
        # Всегда включаем критичные файлы
        always_include = [
            path for path, ctx in self.file_contexts.items() 
            if ctx.priority >= 4
        ]
        
        task_specific = []
        
        if task_type == "debug":
            # Для отладки важны логи, тесты, главные компоненты
            task_specific = [
                path for path, ctx in self.file_contexts.items()
                if ctx.category in ["database", "analyzer", "cli"] or "test" in path.lower()
            ]
        
        elif task_type == "develop":
            # Для разработки важна архитектура
            task_specific = [
                path for path, ctx in self.file_contexts.items()
                if ctx.category in ["analyzer", "cli", "models"]
            ]
        
        elif task_type == "analyze":
            # Для анализа данных важны анализаторы и БД
            task_specific = [
                path for path, ctx in self.file_contexts.items()
                if ctx.category in ["analyzer", "database"]
            ]
        
        elif task_type == "refactor":
            # Для рефакторинга важно видеть всю архитектуру
            task_specific = list(self.file_contexts.keys())
        
        # Поиск по ключевым словам в запросе
        if query:
            query_lower = query.lower()
            query_specific = [
                path for path, ctx in self.file_contexts.items()
                if any(word in path.lower() or word in ctx.description.lower() 
                       for word in query_lower.split())
            ]
            task_specific.extend(query_specific)
        
        # Удаляем дубликаты и сортируем по приоритету
        all_files = list(set(always_include + task_specific))
        all_files.sort(key=lambda x: self.file_contexts[x].priority, reverse=True)
        
        return all_files[:15]  # Ограничиваем до 15 файлов
    
    def _generate_context_summary(self, files: List[str], task_type: str) -> str:
        """Генерирует краткое описание контекста"""
        
        categories = {}
        for file_path in files:
            ctx = self.file_contexts[file_path]
            categories.setdefault(ctx.category, []).append(ctx)
        
        summary_parts = [
            f"Контекст для задачи: {task_type.upper()}",
            f"Файлов в контексте: {len(files)}"
        ]
        
        for category, contexts in categories.items():
            files_in_category = len(contexts)
            high_priority = sum(1 for ctx in contexts if ctx.priority >= 4)
            summary_parts.append(
                f"- {category}: {files_in_category} файлов ({high_priority} высокого приоритета)"
            )
        
        # Добавляем специфичную информацию для задачи
        if task_type == "debug":
            summary_parts.append("Фокус: диагностика проблем, анализ логики, проверка интеграции")
        elif task_type == "develop":
            summary_parts.append("Фокус: разработка новых функций, архитектурные решения")
        elif task_type == "analyze":
            summary_parts.append("Фокус: анализ данных, производительность AI моделей")
        elif task_type == "refactor":
            summary_parts.append("Фокус: улучшение кода, устранение дубликатов, архитектура")
        
        return "\n".join(summary_parts)
    
    def _suggest_commands(self, task_type: str, files: List[str]) -> List[str]:
        """Предлагает полезные команды для задачи"""
        
        base_commands = [
            "python main.py --info  # Статус системы",
            "python main.py --test  # Валидация компонентов"
        ]
        
        if task_type == "debug":
            return base_commands + [
                "python check_stats.py  # PostgreSQL статистика", 
                "python check_overlap.py  # Проверка консистентности данных",
                "grep -r 'ERROR\\|Exception' src/ --include='*.py'  # Поиск ошибок",
                "docker-compose logs rap-analyzer-api  # Логи контейнера (если Docker)"
            ]
        
        elif task_type == "develop":
            return base_commands + [
                "python main.py --analyze 'test text' --analyzer hybrid  # Тест анализатора",
                "python main.py --benchmark  # Тест производительности",
                "find src/ -name '*.py' | head -10  # Структура компонентов",
                "pytest tests/ -v  # Запуск тестов"
            ]
        
        elif task_type == "analyze":
            return base_commands + [
                "python scripts/mass_qwen_analysis.py --test  # Тест массового анализа",
                "python scripts/db_browser.py  # Интерактивный браузер БД",
                "python main.py --benchmark  # Сравнение анализаторов"
            ]
        
        elif task_type == "refactor":
            return base_commands + [
                "python scripts/ai_code_audit.py  # Аудит кода (если создан)",
                "grep -r 'TODO\\|FIXME' src/ --include='*.py'  # Найти места для улучшения",
                "find src/ -name '*.py' -exec wc -l {} + | sort -nr  # Найти большие файлы"
            ]
        
        return base_commands
    
    def _generate_warnings(self, task_type: str, files: List[str]) -> List[str]:
        """Генерирует предупреждения для AI ассистента"""
        
        warnings = []
        
        # Проверяем наличие legacy файлов в контексте
        legacy_files = [
            f for f in files 
            if self.file_contexts[f].category == "legacy"
        ]
        if legacy_files:
            warnings.append(
                f"⚠️ В контексте {len(legacy_files)} legacy файлов. "
                "Используй их только для справки, не копируй устаревшие паттерны."
            )
        
        # Проверяем архитектурную целостность
        if task_type == "develop":
            has_database = any(
                self.file_contexts[f].category == "database" for f in files
            )
            has_models = any(
                self.file_contexts[f].category == "models" for f in files
            )
            if not has_database or not has_models:
                warnings.append(
                    "⚠️ Неполный архитектурный контекст. "
                    "Убедись, что используешь PostgreSQL (не SQLite) и актуальные модели."
                )
        
        # Специфичные предупреждения для PostgreSQL проекта
        warnings.extend([
            "🔄 Проект мигрирован на PostgreSQL. Избегай sqlite3 импортов.",
            "🏗️ Используй main.py как единую точку входа, не запускай компоненты напрямую.",
            "🧪 Всегда тестируй изменения через python main.py --test."
        ])
        
        return warnings
    
    def create_ai_workspace_file(self) -> str:
        """Создает файл рабочего пространства для AI ассистента"""
        
        workspace_data = {
            "name": "Rap Scraper AI Workspace",
            "description": "PostgreSQL-powered rap lyrics analysis with microservices architecture",
            "version": "2.0.0-postgresql",
            "architecture": {
                "type": "microservices",
                "database": "PostgreSQL 15",
                "entry_point": "main.py",
                "key_components": [
                    "src/database/postgres_adapter.py",
                    "src/analyzers/",
                    "src/cli/",
                    "src/models/"
                ]
            },
            "ai_context": {
                "priority_files": [
                    {
                        "path": path,
                        "priority": ctx.priority,
                        "category": ctx.category,
                        "description": ctx.description
                    }
                    for path, ctx in self.file_contexts.items()
                    if ctx.priority >= 4
                ],
                "task_workflows": {
                    "debug": "Focus on database connectivity, error handling, integration issues",
                    "develop": "Use microservices patterns, PostgreSQL backend, test-driven development", 
                    "analyze": "Leverage 5 AI analyzers, batch processing, performance optimization",
                    "refactor": "Maintain architectural boundaries, eliminate SQLite legacy, DRY principles"
                }
            },
            "warnings": [
                "Project migrated from SQLite to PostgreSQL - avoid legacy patterns",
                "Use main.py unified interface instead of direct component calls",
                "Test all changes with comprehensive test suite",
                "Maintain microservices architectural boundaries"
            ]
        }
        
        # Сохраняем workspace в results/ согласно архитектуре
        workspace_file = self.results_dir / "ai_workspace.json"
        
        with open(workspace_file, 'w', encoding='utf-8') as f:
            json.dump(workspace_data, f, indent=2, ensure_ascii=False)
        
        return str(workspace_file)

def create_ai_friendly_readme() -> str:
    """Создает AI-friendly README с быстрым контекстом"""
    
    ai_readme_content = """# 🤖 AI Assistant Quick Context

> **CRITICAL:** This project uses PostgreSQL, not SQLite. All legacy SQLite code is archived.

## ⚡ 30-Second Context
- **Architecture:** Microservices with main.py orchestration
- **Database:** PostgreSQL 15 with connection pooling  
- **AI Models:** 5 analyzers (algorithmic_basic, qwen, ollama, emotion, hybrid)
- **Entry Point:** `python main.py` (unified interface)
- **Data:** 57,718 tracks, 54,170+ analyzed, concurrent processing capable

## 🎯 AI Assistant Commands

### Quick Status
```bash
python main.py --info          # Complete system status
python main.py --test          # Validate all components
python check_stats.py          # PostgreSQL database health
```

### Analysis Tasks  
```bash
python main.py --analyze "text" --analyzer qwen     # Single analysis
python main.py --benchmark                          # Performance comparison
python scripts/mass_qwen_analysis.py --test         # Batch analysis test
```

### Development
```bash
python main.py --help          # All available options
pytest tests/ -v               # Run test suite
docker-compose up -d            # Deploy full stack
```

## 🚨 Critical Reminders for AI

1. **PostgreSQL ONLY** - No sqlite3 imports, use PostgreSQLManager
2. **Unified Interface** - Use main.py, not direct component calls
3. **Microservices** - Respect src/{analyzers,cli,models}/ boundaries
4. **Testing** - Always validate with python main.py --test
5. **Concurrent Processing** - Multiple scripts can run simultaneously

## 📁 Priority Files for AI Analysis

| File | Priority | Purpose |
|------|----------|---------|
| `docs/claude.md` | 🔥🔥🔥🔥🔥 | Complete AI context |
| `main.py` | 🔥🔥🔥🔥🔥 | Central orchestration |  
| `src/database/postgres_adapter.py` | 🔥🔥🔥🔥🔥 | Database layer |
| `config.yaml` | 🔥🔥🔥🔥 | System configuration |
| `scripts/mass_qwen_analysis.py` | 🔥🔥🔥🔥 | Main analysis script |

## ⚠️ Deprecated/Legacy (Reference Only)
- `scripts/archive/` - SQLite legacy code
- `data/data_backup_*.db` - SQLite backups  
- Any file with `_sqlite.py` suffix

---
*Auto-generated for AI assistants. Human-readable docs in README.md*
"""
    
    ai_readme_path = Path("docs") / "AI_README.md"
    ai_readme_path.parent.mkdir(exist_ok=True)
    with open(ai_readme_path, 'w', encoding='utf-8') as f:
        f.write(ai_readme_content)
    
    return str(ai_readme_path)

def main():
    """Главная функция - создание AI-friendly окружения"""
    print("🤖 Создание AI-friendly рабочего пространства...")
    
    # Инициализируем менеджер контекста
    context_manager = AIContextManager()
    
    # Создаем различные контексты
    debug_context = context_manager.generate_ai_context("debug")
    develop_context = context_manager.generate_ai_context("develop") 
    analyze_context = context_manager.generate_ai_context("analyze")
    refactor_context = context_manager.generate_ai_context("refactor")
    
    print("📊 Созданы контексты для задач:")
    print(f"  - Debug: {len(debug_context.relevant_files)} файлов")
    print(f"  - Develop: {len(develop_context.relevant_files)} файлов")
    print(f"  - Analyze: {len(analyze_context.relevant_files)} файлов") 
    print(f"  - Refactor: {len(refactor_context.relevant_files)} файлов")
    
    # Создаем AI workspace
    workspace_file = context_manager.create_ai_workspace_file()
    print(f"📁 AI workspace создан: results\\ai_workspace.json")
    
    # Создаем AI-friendly README
    ai_readme = create_ai_friendly_readme()
    print(f"📄 AI README создан: {ai_readme}")
    
    # Сохраняем примеры контекстов в results/
    examples_dir = context_manager.results_dir / "ai_examples"
    examples_dir.mkdir(exist_ok=True)
    
    contexts = {
        "debug": debug_context,
        "develop": develop_context,
        "analyze": analyze_context,
        "refactor": refactor_context
    }
    
    for task_type, context in contexts.items():
        example_file = examples_dir / f"{task_type}_context.json"
        with open(example_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(context), f, indent=2, ensure_ascii=False)
    
    print(f"💡 Примеры контекстов сохранены в results\\ai_examples\\")
    
    # Создаем VS Code task для быстрого переключения контекстов
    vscode_tasks = {
        "version": "2.0.0",
        "tasks": [
            {
                "label": "AI: Debug Context",
                "type": "shell",
                "command": "python",
                "args": ["scripts/tools/ai_context_manager.py", "debug"],
                "group": "build",
                "problemMatcher": []
            },
            {
                "label": "AI: Analysis Context", 
                "type": "shell",
                "command": "python",
                "args": ["scripts/tools/ai_context_manager.py", "analyze"],
                "group": "build",
                "problemMatcher": []
            },
            {
                "label": "AI: System Status",
                "type": "shell", 
                "command": "python",
                "args": ["main.py", "--info"],
                "group": "build",
                "problemMatcher": []
            }
        ]
    }
    
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    with open(vscode_dir / "tasks.json", 'w', encoding='utf-8') as f:
        json.dump(vscode_tasks, f, indent=2)
    
    print("⚙️ VS Code tasks настроены для быстрого переключения контекстов")
    
    print("\n✅ AI-friendly окружение готово!")
    print("\nТеперь AI ассистент может:")
    print("  1. Быстро понять контекст через AI_README.md")
    print("  2. Использовать подходящие файлы для каждой задачи")
    print("  3. Получать релевантные предупреждения и команды")
    print("  4. Переключаться между контекстами через VS Code tasks")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Режим генерации контекста для конкретной задачи
        task_type = sys.argv[1]
        context_manager = AIContextManager()
        context = context_manager.generate_ai_context(task_type)
        
        print(f"\n🎯 AI КОНТЕКСТ ДЛЯ ЗАДАЧИ: {task_type.upper()}")
        print("=" * 50)
        print(context.context_summary)
        print(f"\nРелевантные файлы ({len(context.relevant_files)}):")
        for file_path in context.relevant_files[:10]:  # Показываем топ-10
            ctx = context_manager.file_contexts[file_path]
            print(f"  🔥{'🔥' * ctx.priority} {Path(file_path).name} - {ctx.description}")
        
        print(f"\nРекомендуемые команды:")
        for cmd in context.suggested_commands:
            print(f"  $ {cmd}")
        
        print(f"\nПредупреждения:")
        for warning in context.warnings:
            print(f"  {warning}")
    else:
        # Полная инициализация
        main()
