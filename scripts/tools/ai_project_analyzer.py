#!/usr/bin/env python3
"""
🧠 AI Project Analyzer — интеллектуальная система анализа проекта для AI ассистентов

НАЗНАЧЕНИЕ:
- Семантический анализ дубликатов через AST-парсинг вместо примитивного grep
- Анализ архитектурных нарушений с учетом PostgreSQL миграции
- Контекстно-зависимый поиск неиспользуемых файлов
- Специализированные метрики для микросервисной архитектуры

ИСПОЛЬЗОВАНИЕ:
python scripts/tools/ai_project_analyzer.py --analyze

ЗАВИСИМОСТИ:
- Python 3.8+
- ast, dataclasses, pathlib

РЕЗУЛЬТАТ:
- Семантический анализ архитектуры и дубликатов
- Метрики для микросервисной архитектуры

АВТОР: Vastargazing
ДАТА: Сентябрь 2025
"""

import os
import ast
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import hashlib

@dataclass
class CodeMetrics:
    file_path: str
    lines_of_code: int
    functions: List[str]
    classes: List[str]
    imports: List[str]
    complexity_score: float
    last_modified: float
    is_test: bool
    is_legacy: bool

@dataclass
class DuplicationResult:
    similarity_score: float
    file1: str
    file2: str
    duplicate_lines: List[str]
    suggestion: str

class ProjectIntelligence:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        # Сохраняем результаты в results/ согласно архитектуре проекта
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.metrics = {}
        self.duplicates = []
        self.unused_files = set()
        self.architecture_violations = []
        
        # Определяем архитектурные слои проекта
        self.architecture_layers = {
            "database": ["src/database/", "postgres_adapter.py"],
            "analyzers": ["src/analyzers/"],
            "cli": ["src/cli/", "main.py"],
            "models": ["src/models/"],
            "scripts": ["scripts/"],
            "legacy": ["scripts/archive/", "_sqlite.py", "data_backup_"],
            "tests": ["tests/", "test_"]
        }
    
    def analyze_project(self) -> Dict:
        """Главная функция анализа проекта"""
        print("🔍 Запуск интеллектуального анализа проекта...")
        
        # 1. Сбор метрик по файлам
        self._collect_file_metrics()
        
        # 2. Поиск семантических дубликатов
        self._find_semantic_duplicates()
        
        # 3. Поиск неиспользуемых файлов
        self._find_unused_files()
        
        # 4. Проверка архитектурных нарушений
        self._check_architecture_violations()
        
        # 5. Анализ PostgreSQL migration status
        self._analyze_postgresql_migration()
        
        return self._generate_report()
    
    def _collect_file_metrics(self):
        """Собирает метрики по всем Python файлам"""
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                metrics = CodeMetrics(
                    file_path=str(py_file),
                    lines_of_code=len(content.splitlines()),
                    functions=self._extract_functions(tree),
                    classes=self._extract_classes(tree),
                    imports=self._extract_imports(tree),
                    complexity_score=self._calculate_complexity(tree),
                    last_modified=py_file.stat().st_mtime,
                    is_test="test" in str(py_file).lower(),
                    is_legacy=self._is_legacy_file(py_file)
                )
                
                self.metrics[str(py_file)] = metrics
                
            except (SyntaxError, UnicodeDecodeError) as e:
                print(f"⚠️ Не удалось проанализировать {py_file}: {e}")
    
    def _find_semantic_duplicates(self):
        """Находит семантически похожие блоки кода"""
        function_signatures = defaultdict(list)
        
        # Группируем функции по сигнатурам
        for file_path, metrics in self.metrics.items():
            for func in metrics.functions:
                signature = self._normalize_function_signature(func)
                function_signatures[signature].append((file_path, func))
        
        # Ищем дубликаты
        for signature, occurrences in function_signatures.items():
            if len(occurrences) > 1:
                # Проверяем, действительно ли это дубликаты
                for i, (file1, func1) in enumerate(occurrences):
                    for file2, func2 in occurrences[i+1:]:
                        similarity = self._calculate_similarity(file1, file2, func1, func2)
                        if similarity > 0.7:  # 70% похожести
                            self.duplicates.append(DuplicationResult(
                                similarity_score=similarity,
                                file1=file1,
                                file2=file2,
                                duplicate_lines=[func1, func2],
                                suggestion=self._suggest_refactoring(file1, file2)
                            ))
    
    def _find_unused_files(self):
        """Находит потенциально неиспользуемые файлы"""
        all_imports = set()
        
        # Собираем все импорты
        for metrics in self.metrics.values():
            all_imports.update(metrics.imports)
        
        # Проверяем каждый файл
        for file_path in self.metrics.keys():
            module_name = self._file_to_module_name(file_path)
            
            # Если модуль нигде не импортируется и не является entry point
            if not self._is_imported(module_name, all_imports) and not self._is_entry_point(file_path):
                self.unused_files.add(file_path)
    
    def _check_architecture_violations(self):
        """Проверяет нарушения архитектуры"""
        for file_path, metrics in self.metrics.items():
            # Проверка 1: Legacy код не должен импортироваться в новом коде
            if not metrics.is_legacy:
                for imp in metrics.imports:
                    if "sqlite" in imp.lower() or "archive" in imp:
                        self.architecture_violations.append(
                            f"❌ {file_path}: Импорт legacy кода '{imp}'"
                        )
            
            # Проверка 2: Слои не должны нарушаться
            layer = self._determine_layer(file_path)
            for imp in metrics.imports:
                imp_layer = self._determine_layer_by_import(imp)
                if not self._is_allowed_dependency(layer, imp_layer):
                    self.architecture_violations.append(
                        f"❌ {file_path}: Нарушение архитектуры - {layer} -> {imp_layer}"
                    )
    
    def _analyze_postgresql_migration(self):
        """Анализирует статус миграции на PostgreSQL"""
        sqlite_usage = []
        postgresql_usage = []
        
        for file_path, metrics in self.metrics.items():
            if metrics.is_legacy:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                
                if "sqlite3" in content or "sqlite" in content:
                    sqlite_usage.append(file_path)
                
                if "postgresql" in content or "asyncpg" in content or "psycopg2" in content:
                    postgresql_usage.append(file_path)
                    
            except Exception:
                continue
        
        self.migration_status = {
            "sqlite_files": sqlite_usage,
            "postgresql_files": postgresql_usage,
            "migration_complete": len(sqlite_usage) == 0
        }
    
    def _generate_report(self) -> Dict:
        """Генерирует итоговый отчет для AI"""
        total_files = len(self.metrics)
        legacy_files = sum(1 for m in self.metrics.values() if m.is_legacy)
        
        report = {
            "summary": {
                "total_files": total_files,
                "active_files": total_files - legacy_files,
                "legacy_files": legacy_files,
                "average_complexity": sum(m.complexity_score for m in self.metrics.values()) / total_files,
                "migration_status": self.migration_status.get("migration_complete", False)
            },
            "duplicates": [{
                "similarity": d.similarity_score,
                "files": [d.file1, d.file2],
                "suggestion": d.suggestion
            } for d in self.duplicates],
            "unused_files": list(self.unused_files),
            "architecture_violations": self.architecture_violations,
            "ai_recommendations": self._generate_ai_recommendations()
        }
        
        return report
    
    def _generate_ai_recommendations(self) -> List[str]:
        """Генерирует рекомендации для AI ассистента"""
        recommendations = []
        
        # Рекомендации по дубликатам
        if len(self.duplicates) > 5:
            recommendations.append(
                f"🔄 Найдено {len(self.duplicates)} дубликатов кода. "
                "Приоритет: создать общие утилиты в src/utils/"
            )
        
        # Рекомендации по неиспользуемым файлам
        if len(self.unused_files) > 3:
            recommendations.append(
                f"🗑️ Найдено {len(self.unused_files)} потенциально неиспользуемых файлов. "
                "Рекомендую переместить в scripts/archive/"
            )
        
        # Рекомендации по архитектуре
        if self.architecture_violations:
            recommendations.append(
                "⚠️ Обнаружены нарушения архитектуры. "
                "Приоритет: рефакторинг импортов и зависимостей"
            )
        
        # Рекомендации по миграции
        if not self.migration_status.get("migration_complete", True):
            recommendations.append(
                "🔄 Миграция на PostgreSQL не завершена. "
                "Найдены остатки SQLite кода в активных файлах"
            )
        
        return recommendations
    
    # Вспомогательные методы
    def _should_skip_file(self, file_path: Path) -> bool:
        skip_patterns = ['.venv', '__pycache__', '.git', 'node_modules']
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _extract_functions(self, tree: ast.AST) -> List[str]:
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> List[str]:
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.append(module)
        return imports
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Упрощенный расчет цикломатической сложности"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
        return complexity / 10  # Нормализуем
    
    def _is_legacy_file(self, file_path: Path) -> bool:
        path_str = str(file_path)
        return any(pattern in path_str for pattern in ["archive", "sqlite", "backup"])
    
    def _normalize_function_signature(self, func_name: str) -> str:
        """Нормализует сигнатуру функции для сравнения"""
        return func_name.lower().replace("_", "")
    
    def _calculate_similarity(self, file1: str, file2: str, func1: str, func2: str) -> float:
        """Рассчитывает семантическую похожесть функций"""
        # Упрощенная логика - в реальности можно использовать AST-анализ
        if func1 == func2:
            return 0.9
        return 0.5 if self._normalize_function_signature(func1) == self._normalize_function_signature(func2) else 0.0
    
    def _suggest_refactoring(self, file1: str, file2: str) -> str:
        """Предлагает способ рефакторинга"""
        if "analyzer" in file1 and "analyzer" in file2:
            return "Вынести общую логику в BaseAnalyzer"
        elif "script" in file1 and "script" in file2:
            return "Создать общую утилиту в src/utils/"
        return "Рассмотреть создание общего модуля"
    
    def _file_to_module_name(self, file_path: str) -> str:
        return file_path.replace("/", ".").replace("\\", ".").replace(".py", "")
    
    def _is_imported(self, module_name: str, all_imports: Set[str]) -> bool:
        return any(module_name in imp or imp in module_name for imp in all_imports)
    
    def _is_entry_point(self, file_path: str) -> bool:
        entry_points = ["main.py", "__init__.py", "api.py", "app.py"]
        return any(ep in file_path for ep in entry_points)
    
    def _determine_layer(self, file_path: str) -> str:
        for layer, patterns in self.architecture_layers.items():
            if any(pattern in file_path for pattern in patterns):
                return layer
        return "unknown"
    
    def _determine_layer_by_import(self, import_name: str) -> str:
        for layer, patterns in self.architecture_layers.items():
            if any(pattern.replace("/", ".") in import_name for pattern in patterns):
                return layer
        return "external"
    
    def _is_allowed_dependency(self, from_layer: str, to_layer: str) -> bool:
        """Определяет допустимые зависимости между слоями"""
        allowed_deps = {
            "cli": ["models", "analyzers", "database", "external"],
            "analyzers": ["models", "database", "external"],
            "database": ["models", "external"],
            "models": ["external"],
            "scripts": ["cli", "analyzers", "database", "models", "external"],
            "tests": ["cli", "analyzers", "database", "models", "external"]
        }
        return to_layer in allowed_deps.get(from_layer, [])

def generate_vscode_config(project_analysis: Dict) -> Dict:
    """Генерирует оптимальную конфигурацию VS Code на основе анализа"""
    
    # Определяем файлы для исключения
    exclude_patterns = {
        "**/__pycache__": True,
        "**/.pytest_cache": True,
        "**/venv": True,
        "**/.venv": True,
    }
    
    # Добавляем legacy файлы в исключения
    if project_analysis["summary"]["legacy_files"] > 0:
        exclude_patterns.update({
            "scripts/archive/**": True,
            "data/data_backup_*.db": True,
            "**/*sqlite*.py": True
        })
    
    config = {
        "python.defaultInterpreterPath": "./venv/bin/python",
        "python.linting.enabled": True,
        "python.formatting.provider": "black",
        "python.analysis.autoImportCompletions": True,
        "python.analysis.typeCheckingMode": "basic",
        
        # Настройки файлов
        "files.exclude": exclude_patterns,
        "search.exclude": exclude_patterns,
        
        # Настройки для AI
        "github.copilot.enable": {
            "*": True,
            "yaml": True,
            "plaintext": False,
            "markdown": True
        },
        
        # Настройки поиска для больших проектов
        "search.maxResults": 2000,
        "search.smartCase": True,
        
        # Подсветка важных файлов
        "workbench.colorCustomizations": {
            "tab.activeBorder": "#ff6b6b",
            "tab.unfocusedActiveBorder": "#ff6b6b50"
        },
        
        # Настройки для анализа
        "todo-tree.tree.showScanModeButton": False,
        "todo-tree.highlights.defaultHighlight": {
            "icon": "alert",
            "type": "tag",
            "foreground": "red",
            "background": "white",
            "opacity": 50,
            "iconColour": "blue"
        }
    }
    
    return config

def main():
    """Главная функция - запуск анализа и генерация конфигурации"""
    print("🚀 Запуск интеллектуального анализа проекта...")
    
    analyzer = ProjectIntelligence()
    analysis_result = analyzer.analyze_project()
    
    # Генерируем отчет
    print("\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
    print(f"Всего файлов: {analysis_result['summary']['total_files']}")
    print(f"Активных файлов: {analysis_result['summary']['active_files']}")
    print(f"Legacy файлов: {analysis_result['summary']['legacy_files']}")
    print(f"Средняя сложность: {analysis_result['summary']['average_complexity']:.2f}")
    print(f"PostgreSQL миграция: {'✅ Завершена' if analysis_result['summary']['migration_status'] else '❌ В процессе'}")
    
    if analysis_result['duplicates']:
        print(f"\n🔄 Найдено дубликатов: {len(analysis_result['duplicates'])}")
        for dup in analysis_result['duplicates'][:3]:  # Показываем топ-3
            print(f"  - {dup['similarity']:.1%} схожесть: {Path(dup['files'][0]).name} <-> {Path(dup['files'][1]).name}")
    
    if analysis_result['unused_files']:
        print(f"\n🗑️ Потенциально неиспользуемые файлы: {len(analysis_result['unused_files'])}")
        for file in list(analysis_result['unused_files'])[:3]:  # Показываем топ-3
            print(f"  - {Path(file).name}")
    
    if analysis_result['architecture_violations']:
        print(f"\n⚠️ Нарушения архитектуры: {len(analysis_result['architecture_violations'])}")
        for violation in analysis_result['architecture_violations'][:3]:  # Показываем топ-3
            print(f"  - {violation}")
    
    print("\n💡 РЕКОМЕНДАЦИИ ДЛЯ AI:")
    for rec in analysis_result['ai_recommendations']:
        print(f"  {rec}")
    
    # Генерируем VS Code конфигурацию
    vscode_config = generate_vscode_config(analysis_result)
    
    # Сохраняем результаты согласно архитектуре проекта
    os.makedirs('.vscode', exist_ok=True)
    
    with open('.vscode/settings.json', 'w', encoding='utf-8') as f:
        json.dump(vscode_config, f, indent=2, ensure_ascii=False)
    
    # Основные результаты в results/
    with open(analyzer.results_dir / 'project_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)
    
    print("\n✅ Анализ завершен!")
    print(f"📁 Результаты сохранены в {analyzer.results_dir / 'project_analysis.json'}")
    print("⚙️ VS Code конфигурация обновлена в .vscode/settings.json")

if __name__ == "__main__":
    main()
