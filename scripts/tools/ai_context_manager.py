#!/usr/bin/env python3
"""
🤖 AI Context Manager PRO — интеллектуальный менеджер контекста с ML и интеграциями

НАЗНАЧЕНИЕ:
🎯 Динамическое управление контекстом для разных типов задач с AI-powered возможностями
📊 Git-based приоритизация файлов на основе истории коммитов и активности разработчиков
🧠 ML семантический поиск через TF-IDF векторизацию для поиска релевантных файлов
🔄 Умное кеширование с автоматической инвалидацией при изменении файлов
🔗 Интеграция с ai_project_analyzer для получения метрик качества кода
📈 Динамические метрики сложности и связанности модулей

НОВЫЕ ВОЗМОЖНОСТИ PRO:
✨ Git-based приоритизация - файлы с частыми изменениями получают выше приоритет
🧠 Семантический поиск - находит релевантные файлы по смыслу, а не только по ключевым словам
🤖 Автоопределение типа задачи - анализирует запрос и выбирает debug/develop/analyze/refactor
💾 Умное кеширование - MD5 хеширование + инвалидация при изменениях
📊 Enterprise метрики - сложность кода, связанность модулей, git статистика
🔗 Project Analyzer интеграция - дубликаты кода, архитектурные нарушения
🎨 Интерактивный CLI - удобный интерфейс вместо только командной строки

ИСПОЛЬЗОВАНИЕ:
# Интерактивный режим (рекомендуется)
python scripts/tools/ai_context_manager.py --interactive

# Автоопределение типа задачи
python scripts/tools/ai_context_manager.py --query "fix database connection error"

# Семантический поиск
python scripts/tools/ai_context_manager.py --semantic-search "analyzer performance"

# Статистика проекта с ML метриками
python scripts/tools/ai_context_manager.py --stats

# Интеграция с Project Analyzer
python scripts/tools/ai_context_manager.py --integrate

ЗАВИСИМОСТИ:
- Python 3.8+
- dataclasses, pathlib, ast, subprocess (базовые)
- scikit-learn, numpy (для ML семантического поиска)
- pyperclip (опционально, для копирования в буфер)

РЕЗУЛЬТАТ:
🎯 Автоматизированное управление контекстом с ML инсайтами
⚠️ Предупреждения о legacy коде и архитектурных рисках
📊 Динамические приоритеты на основе git активности
🧠 Семантический поиск по кодовой базе
🔗 Интеграция с анализом качества кода

ЭВОЛЮЦИЯ ОТ БАЗОВОЙ ВЕРСИИ:
🔄 Было: Статические приоритеты (int 1-5) → Стало: Динамические (float 0-5.0)
🔄 Было: Простой grep поиск → Стало: ML семантический поиск с TF-IDF
🔄 Было: Ручная категоризация → Стало: Git-based автоматическая приоритизация
🔄 Было: Базовый CLI → Стало: Интерактивный режим + автоопределение задач
🔄 Было: Изолированный инструмент → Стало: Интеграция с Project Analyzer

АВТОР: Vastargazing
ДАТА: Сентябрь 2025 (PRO upgrade)
ВЕРСИЯ: 2.0 PRO
"""

import ast
import hashlib
import json
import pickle
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# Опциональные зависимости для ML features
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ sklearn не установлен, семантический поиск будет упрощенным")

# Опциональные зависимости для advanced features
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False


@dataclass
class EnhancedFileContext:
    """Расширенный контекст файла с ML метриками"""

    path: str
    priority: float  # Теперь float для динамической корректировки
    category: str
    description: str
    last_modified: str
    size_lines: int
    dependencies: list[str]

    # Новые поля для ML
    git_commits_count: int = 0
    git_last_commit: str = ""
    git_authors: list[str] = field(default_factory=list)
    complexity_score: float = 0.0
    coupling_score: float = 0.0  # Связанность с другими модулями
    usage_frequency: int = 0  # Как часто файл используется в проекте
    semantic_embedding: list[float] | None = None  # Для семантического поиска


@dataclass
class ContextCache:
    """Кеш для ускорения работы"""

    file_hashes: dict[str, str] = field(default_factory=dict)
    embeddings: dict[str, list[float]] = field(default_factory=dict)
    git_data: dict[str, dict] = field(default_factory=dict)
    last_update: float = 0


class GitAnalyzer:
    """Анализатор git истории для приоритезации"""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def get_file_stats(self, file_path: str) -> dict:
        """Получает git статистику для файла"""
        try:
            # Количество коммитов
            commits = (
                subprocess.run(
                    ["git", "log", "--oneline", file_path],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path,
                )
                .stdout.strip()
                .split("\n")
            )
            commit_count = len([c for c in commits if c])

            # Последний коммит
            last_commit = subprocess.run(
                ["git", "log", "-1", "--format=%ar", file_path],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            ).stdout.strip()

            # Авторы
            authors = (
                subprocess.run(
                    ["git", "log", "--format=%an", file_path],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=self.repo_path,
                )
                .stdout.strip()
                .split("\n")
            )
            unique_authors = list(set([a for a in authors if a]))

            return {
                "commits": commit_count,
                "last_commit": last_commit,
                "authors": unique_authors,
            }
        except Exception:
            return {"commits": 0, "last_commit": "unknown", "authors": []}

    def get_recent_changes(self, days: int = 7) -> list[str]:
        """Получает список недавно измененных файлов"""
        try:
            since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            result = subprocess.run(
                ["git", "log", f"--since={since}", "--name-only", "--format="],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
            )
            files = [
                f for f in result.stdout.strip().split("\n") if f and f.endswith(".py")
            ]
            return list(set(files))
        except:
            return []


class SemanticSearchEngine:
    """Семантический поиск по коду"""

    def __init__(self):
        self.vectorizer = None
        self.file_contents = {}
        self.embeddings = None

    def build_index(self, files: dict[str, EnhancedFileContext]) -> None:
        """Строит индекс для семантического поиска"""
        if not ML_AVAILABLE:
            return

        # Собираем контент файлов
        contents = []
        file_paths = []

        for path, context in files.items():
            if Path(path).exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        content = f.read()
                        # Извлекаем docstrings и комментарии для лучшего понимания
                        content = self._extract_semantic_content(content)
                        contents.append(content)
                        file_paths.append(path)
                        self.file_contents[path] = content
                except:
                    pass

        if contents:
            # Создаем TF-IDF векторы
            self.vectorizer = TfidfVectorizer(
                max_features=500, stop_words="english", ngram_range=(1, 2)
            )
            self.embeddings = self.vectorizer.fit_transform(contents)
            self.file_paths = file_paths

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Семантический поиск по запросу"""
        if not ML_AVAILABLE or self.vectorizer is None:
            return []

        # Векторизуем запрос
        query_vec = self.vectorizer.transform([query])

        # Вычисляем схожесть
        similarities = cosine_similarity(query_vec, self.embeddings).flatten()

        # Сортируем и возвращаем топ результаты
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Порог релевантности
                results.append((self.file_paths[idx], float(similarities[idx])))

        return results

    def _extract_semantic_content(self, code: str) -> str:
        """Извлекает семантически важный контент из кода"""
        try:
            tree = ast.parse(code)
            semantic_parts = []

            for node in ast.walk(tree):
                # Извлекаем docstrings
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    docstring = ast.get_docstring(node)
                    if docstring:
                        semantic_parts.append(docstring)
                    semantic_parts.append(node.name)

                # Извлекаем комментарии
                for line in code.split("\n"):
                    if "#" in line:
                        comment = line.split("#")[1].strip()
                        if len(comment) > 10:  # Игнорируем короткие комментарии
                            semantic_parts.append(comment)

            return " ".join(semantic_parts)
        except:
            return code[:1000]  # Fallback к первым 1000 символам


class DynamicPrioritizer:
    """Динамическая приоритезация файлов"""

    def __init__(self, git_analyzer: GitAnalyzer):
        self.git_analyzer = git_analyzer
        self.usage_patterns = defaultdict(int)

    def calculate_priority(self, context: EnhancedFileContext) -> float:
        """Вычисляет динамический приоритет файла"""

        base_priority = self._get_base_priority(context.category)

        # Факторы, влияющие на приоритет
        factors = []

        # 1. Частота изменений (git commits)
        if context.git_commits_count > 0:
            commit_factor = min(context.git_commits_count / 50, 1.0)  # Нормализуем до 1
            factors.append(commit_factor * 0.3)

        # 2. Недавность изменений
        if "hour" in context.git_last_commit or "minute" in context.git_last_commit:
            factors.append(0.5)  # Очень недавние изменения
        elif "day" in context.git_last_commit:
            factors.append(0.3)
        elif "week" in context.git_last_commit:
            factors.append(0.1)

        # 3. Количество авторов (популярность файла)
        if len(context.git_authors) > 1:
            author_factor = min(len(context.git_authors) / 5, 1.0)
            factors.append(author_factor * 0.2)

        # 4. Сложность кода
        if context.complexity_score > 0:
            complexity_factor = min(context.complexity_score / 100, 1.0)
            factors.append(complexity_factor * 0.2)

        # 5. Связанность с другими модулями
        if context.coupling_score > 0:
            coupling_factor = min(context.coupling_score / 10, 1.0)
            factors.append(coupling_factor * 0.3)

        # 6. Размер файла (большие файлы часто важнее)
        if context.size_lines > 200:
            size_factor = min(context.size_lines / 1000, 1.0)
            factors.append(size_factor * 0.1)

        # Вычисляем финальный приоритет
        dynamic_boost = sum(factors)
        final_priority = base_priority + dynamic_boost

        # Ограничиваем максимальное значение
        return min(final_priority, 5.0)

    def _get_base_priority(self, category: str) -> float:
        """Базовые приоритеты по категориям"""
        priorities = {
            "database": 4.0,
            "cli": 4.0,
            "analyzer": 3.5,
            "config": 3.5,
            "models": 3.0,
            "docs": 3.0,
            "tests": 2.5,
            "scripts": 2.0,
            "legacy": 1.0,
        }
        return priorities.get(category, 2.5)


class LLMDescriptionGenerator:
    """Генератор умных описаний файлов через LLM"""

    def __init__(self, provider="ollama", model="codellama"):
        self.provider = provider
        self.model = model
        self.cache_dir = Path("results/.llm_cache")
        self.cache_dir.mkdir(exist_ok=True)

    async def generate_file_description(self, file_path: Path) -> str:
        """Генерирует умное описание файла через LLM"""

        # Проверяем кеш
        cache_key = hashlib.md5(str(file_path).encode()).hexdigest()
        cache_file = self.cache_dir / f"{cache_key}.txt"

        if cache_file.exists():
            return cache_file.read_text()

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()[:2000]  # Первые 2000 символов

            description = await self._generate_description(file_path.name, content)

            # Кешируем результат
            cache_file.write_text(description)
            return description

        except Exception:
            return self._fallback_description(file_path, "")

    async def _generate_description(self, filename: str, content: str) -> str:
        """Генерирует описание через доступный LLM"""

        prompt = f"""Analyze this Python file and provide a concise description (max 100 chars):
Filename: {filename}
Content preview:
{content[:500]}

Describe the main purpose and key functionality:"""

        if self.provider == "ollama" and HTTPX_AVAILABLE:
            return await self._call_ollama(prompt)
        return self._fallback_description(Path(filename), content)

    async def _call_ollama(self, prompt: str) -> str:
        """Вызов Ollama API"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1},
                    },
                    timeout=30,
                )
                if response.status_code == 200:
                    result = response.json()["response"]
                    return result.strip()[:100]
        except Exception as e:
            print(f"⚠️ Ollama error: {e}")

        return "AI-generated description unavailable"

    def _fallback_description(self, file_path: Path, content: str) -> str:
        """Fallback описание без LLM"""
        if "class" in content and "def" in content:
            return f"Python module with classes and functions: {file_path.name}"
        if "class" in content:
            return f"Class definitions in {file_path.name}"
        if "def" in content:
            return f"Function definitions in {file_path.name}"
        if "import" in content:
            return f"Python module with imports: {file_path.name}"
        return f"Python file: {file_path.name}"

    def generate_descriptions(self, file_contexts: dict) -> dict[str, str]:
        """Генерирует описания для всех файлов в контексте"""
        descriptions = {}
        print(f"🤖 Генерируем AI описания для {len(file_contexts)} файлов...")

        for file_path, context in file_contexts.items():
            try:
                # Асинхронный вызов нужно обернуть
                import asyncio

                description = asyncio.run(
                    self.generate_file_description(Path(file_path))
                )
                descriptions[file_path] = description
                print(f"✅ {file_path}: {description[:50]}...")
            except Exception as e:
                print(f"❌ Ошибка для {file_path}: {e}")
                descriptions[file_path] = self._fallback_description(
                    Path(file_path), context.content[:500]
                )

        return descriptions


class DependencyVisualizer:
    """Простая визуализация зависимостей"""

    def __init__(self, file_contexts: dict):
        self.file_contexts = file_contexts
        self.output_dir = Path("results/visualizations")
        self.output_dir.mkdir(exist_ok=True)

    def generate_dependency_graph(self, focus_files: list[str] | None = None) -> str:
        """Генерирует граф зависимостей в DOT формате"""

        dot_lines = [
            "digraph Dependencies {",
            "  rankdir=LR;",
            "  node [shape=box, style=rounded, fontname=Arial];",
            "  edge [color=gray50];",
            "",
        ]

        # Цвета для категорий
        category_colors = {
            "database": "#3498db",
            "analyzer": "#2ecc71",
            "cli": "#e74c3c",
            "models": "#f39c12",
            "config": "#9b59b6",
            "tests": "#95a5a6",
            "legacy": "#7f8c8d",
        }

        # Фильтруем файлы
        files_to_show = (
            focus_files if focus_files else list(self.file_contexts.keys())[:20]
        )

        # Добавляем узлы
        for file_path in files_to_show:
            if file_path not in self.file_contexts:
                continue

            ctx = self.file_contexts[file_path]
            node_name = Path(file_path).name.replace(".", "_").replace("-", "_")
            color = category_colors.get(ctx.category, "#34495e")

            # Размер зависит от приоритета
            width = 0.5 + (ctx.priority * 0.2)

            dot_lines.append(
                f'  "{node_name}" [fillcolor="{color}", style=filled, '
                f'width={width:.1f}, tooltip="{ctx.description[:50]}"];'
            )

        # Добавляем связи
        for file_path in files_to_show:
            if file_path not in self.file_contexts:
                continue

            ctx = self.file_contexts[file_path]
            node_name = Path(file_path).name.replace(".", "_").replace("-", "_")

            for dep in ctx.dependencies[:3]:  # Ограничиваем связи
                dep_files = [
                    f
                    for f in files_to_show
                    if dep.lower() in Path(f).name.lower() and f != file_path
                ]

                for dep_file in dep_files[:1]:  # Максимум 1 связь на зависимость
                    dep_name = Path(dep_file).name.replace(".", "_").replace("-", "_")
                    dot_lines.append(f'  "{node_name}" -> "{dep_name}";')

        dot_lines.append("}")
        return "\n".join(dot_lines)

    def save_graph(self, focus_category: str | None = None) -> Path:
        """Сохраняет граф в файл"""

        focus_files = None
        if focus_category:
            focus_files = [
                path
                for path, ctx in self.file_contexts.items()
                if ctx.category == focus_category
            ]

        dot_content = self.generate_dependency_graph(focus_files)

        # Сохраняем DOT файл
        suffix = f"_{focus_category}" if focus_category else ""
        dot_file = self.output_dir / f"dependencies{suffix}.dot"
        dot_file.write_text(dot_content, encoding="utf-8")

        print(f"✅ Граф зависимостей сохранен: {dot_file}")
        print(
            f"💡 Для рендеринга: dot -Tsvg {dot_file} -o {dot_file.with_suffix('.svg')}"
        )

        return dot_file


class SimpleAPI:
    """Простой REST API для интеграции с IDE"""

    def __init__(self, context_manager):
        if not API_AVAILABLE:
            print("⚠️ FastAPI не установлен - API недоступен")
            return

        self.context_manager = context_manager
        self.app = FastAPI(title="AI Context Manager API", version="2.0")

        # CORS для работы с IDE
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()

    def _setup_routes(self):
        """Настройка API endpoints"""

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "files_indexed": len(self.context_manager.file_contexts),
                "ml_available": ML_AVAILABLE,
                "version": "2.0",
            }

        @self.app.post("/context")
        async def generate_context(
            task_type: str = "develop", query: str = "", max_files: int = 15
        ):
            """Генерирует контекст для задачи"""
            context = self.context_manager.generate_ai_context(task_type, query)

            return {
                "relevant_files": context["relevant_files"][:max_files],
                "context_summary": context["context_summary"],
                "suggested_commands": context["suggested_commands"],
                "warnings": context["warnings"],
                "ml_insights": context.get("ml_insights", []),
            }

        @self.app.get("/files")
        async def list_files(category: str | None = None, min_priority: float = 0.0):
            """Список файлов с фильтрацией"""
            files = []
            for path, ctx in self.context_manager.file_contexts.items():
                if category and ctx.category != category:
                    continue
                if ctx.priority < min_priority:
                    continue

                files.append(
                    {
                        "path": path,
                        "name": Path(path).name,
                        "category": ctx.category,
                        "priority": round(ctx.priority, 2),
                        "description": ctx.description,
                        "git_commits": ctx.git_commits_count,
                    }
                )

            return {"files": files, "total": len(files)}

        @self.app.get("/stats")
        async def get_stats():
            """Статистика проекта"""
            categories = defaultdict(int)
            total_complexity = 0
            high_priority = 0

            for ctx in self.context_manager.file_contexts.values():
                categories[ctx.category] += 1
                total_complexity += ctx.complexity_score
                if ctx.priority >= 4.0:
                    high_priority += 1

            return {
                "total_files": len(self.context_manager.file_contexts),
                "high_priority_files": high_priority,
                "avg_complexity": round(
                    total_complexity / len(self.context_manager.file_contexts), 1
                ),
                "categories": dict(categories),
            }

        @self.app.get("/visualize/{category}")
        async def visualize_category(category: str):
            """Генерирует граф для категории"""
            visualizer = DependencyVisualizer(self.context_manager.file_contexts)
            dot_content = visualizer.generate_dependency_graph(
                focus_files=[
                    path
                    for path, ctx in self.context_manager.file_contexts.items()
                    if ctx.category == category
                ]
            )
            return {"dot": dot_content, "category": category}

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Запускает API сервер"""
        if not API_AVAILABLE:
            print("❌ FastAPI не установлен. Установите: pip install fastapi uvicorn")
            return

        print(f"🚀 API сервер запущен на http://{host}:{port}")
        print(f"📚 Документация: http://{host}:{port}/docs")
        print(f"💚 Health check: http://{host}:{port}/health")
        uvicorn.run(self.app, host=host, port=port)


class AIContextManagerPro:
    """Улучшенный менеджер контекста с ML и интеграциями"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(exist_ok=True)

        # Файлы для хранения данных
        self.context_file = self.results_dir / ".ai_context_pro.json"
        self.cache_file = self.results_dir / ".ai_context_cache.pkl"

        # Компоненты
        self.git_analyzer = GitAnalyzer(self.project_root)
        self.semantic_engine = SemanticSearchEngine()
        self.prioritizer = DynamicPrioritizer(self.git_analyzer)

        # Данные
        self.file_contexts = {}
        self.cache = self._load_cache()

        # Интеграция с project analyzer
        self.project_analyzer = None
        self.analyzer_metrics = {}

        # Advanced features
        self.llm_generator = None
        self.visualizer = None
        self.api_server = None

        # Загружаем контекст
        self.load_context()

    def setup_advanced_features(
        self, enable_llm: bool = False, enable_api: bool = False
    ):
        """Настройка продвинутых возможностей"""

        if enable_llm:
            self.llm_generator = LLMDescriptionGenerator()
            print("✅ LLM генератор описаний настроен")

        if enable_api:
            self.api_server = SimpleAPI(self)
            print("✅ REST API настроен")

        # Визуализатор всегда доступен
        self.visualizer = DependencyVisualizer(self.file_contexts)
        print("✅ Визуализатор зависимостей настроен")

    def integrate_with_project_analyzer(self) -> bool:
        """Интеграция с ai_project_analyzer для получения метрик"""
        try:
            # Импортируем project analyzer
            import sys

            sys.path.append(str(self.project_root / "scripts" / "tools"))

            try:
                from ai_project_analyzer import ProjectIntelligence

                self.project_analyzer = ProjectIntelligence(str(self.project_root))

                # Загружаем существующие результаты анализа если есть
                analyzer_results_file = self.results_dir / "project_analysis.json"
                if analyzer_results_file.exists():
                    with open(analyzer_results_file, encoding="utf-8") as f:
                        self.analyzer_metrics = json.load(f)
                else:
                    # Запускаем анализ
                    print("🔄 Запуск интеграции с Project Analyzer...")
                    self.analyzer_metrics = self.project_analyzer.analyze_project()

                    # Сохраняем результаты
                    with open(analyzer_results_file, "w", encoding="utf-8") as f:
                        json.dump(
                            self.analyzer_metrics, f, indent=2, ensure_ascii=False
                        )

                return True

            except ImportError as e:
                print(f"⚠️ Не удалось импортировать ProjectIntelligence: {e}")
                return False

        except Exception as e:
            print(f"⚠️ Ошибка интеграции с project analyzer: {e}")
            return False

    def _enhance_context_with_analyzer_data(
        self, file_path: str, context: EnhancedFileContext
    ):
        """Обогащает контекст данными из project analyzer"""
        if not self.analyzer_metrics:
            return

        # Получаем метрики файла из project analyzer
        file_metrics = self.analyzer_metrics.get("file_metrics", {})
        if file_path in file_metrics:
            metrics = file_metrics[file_path]

            # Обновляем сложность более точными данными
            if "complexity_score" in metrics:
                context.complexity_score = max(
                    context.complexity_score, metrics["complexity_score"]
                )

            # Добавляем информацию о дубликатах
            duplicates = self.analyzer_metrics.get("duplicates", [])
            for dup in duplicates:
                if file_path in [dup.get("file1"), dup.get("file2")]:
                    context.coupling_score += 1  # Файл участвует в дубликации

        # Проверяем архитектурные нарушения
        violations = self.analyzer_metrics.get("architecture_violations", [])
        for violation in violations:
            if file_path in violation.get("description", ""):
                context.priority += 0.5  # Повышаем приоритет файлов с нарушениями

    def _generate_enhanced_insights(self, relevant_files: list[str]) -> list[str]:
        """Генерирует улучшенные инсайты с данными project analyzer"""
        insights = []

        if not self.analyzer_metrics:
            return self._generate_ml_insights(relevant_files, "")

        # Анализ дубликатов в релевантных файлах
        duplicates = self.analyzer_metrics.get("duplicates", [])
        relevant_duplicates = [
            dup
            for dup in duplicates
            if any(rf in [dup.get("file1"), dup.get("file2")] for rf in relevant_files)
        ]

        if relevant_duplicates:
            insights.append(
                f"🔍 Найдено {len(relevant_duplicates)} дубликатов в релевантных файлах - "
                f"возможна рефакторизация"
            )

        # Анализ архитектурных нарушений
        violations = self.analyzer_metrics.get("architecture_violations", [])
        relevant_violations = [
            v
            for v in violations
            if any(rf in v.get("description", "") for rf in relevant_files)
        ]

        if relevant_violations:
            insights.append(
                f"⚠️ {len(relevant_violations)} архитектурных нарушений - "
                f"проверь соответствие PostgreSQL архитектуре"
            )

        # Анализ legacy файлов
        unused_files = self.analyzer_metrics.get("unused_files", [])
        relevant_unused = [f for f in unused_files if f in relevant_files]

        if relevant_unused:
            insights.append(
                f"🗑️ {len(relevant_unused)} неиспользуемых файлов в контексте - "
                f"возможно устаревший код"
            )

        # Анализ качества кода
        summary = self.analyzer_metrics.get("summary", {})
        if summary:
            avg_complexity = summary.get("average_complexity", 0)
            if avg_complexity > 10:
                insights.append(
                    f"🔥 Высокая средняя сложность ({avg_complexity:.1f}) - "
                    f"рассмотри упрощение"
                )

        # Добавляем базовые ML инсайты
        base_insights = self._generate_ml_insights(relevant_files, "")
        insights.extend(base_insights)

        return insights

    def _load_cache(self) -> ContextCache:
        """Загружает кеш из файла"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except:
                pass
        return ContextCache()

    def _save_cache(self):
        """Сохраняет кеш в файл"""
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except:
            pass

    def load_context(self):
        """Загружает контекст с учетом кеша"""
        if self.context_file.exists():
            with open(self.context_file, encoding="utf-8") as f:
                data = json.load(f)
                self.file_contexts = {
                    path: EnhancedFileContext(**ctx) for path, ctx in data.items()
                }
        else:
            self._build_initial_context()

        # Обновляем динамические данные
        self._update_dynamic_data()

        # Строим семантический индекс
        self.semantic_engine.build_index(self.file_contexts)

    def _update_dynamic_data(self):
        """Обновляет динамические данные (git, метрики и т.д.)"""

        # Получаем недавно измененные файлы для приоритезации
        recent_files = set(self.git_analyzer.get_recent_changes(days=30))

        for path, context in self.file_contexts.items():
            # Проверяем, нужно ли обновить данные
            file_path = Path(path)
            if not file_path.exists():
                continue

            # Проверяем хеш файла
            current_hash = self._get_file_hash(file_path)
            cached_hash = self.cache.file_hashes.get(path)

            if current_hash != cached_hash or path in recent_files:
                # Файл изменился, обновляем все метрики

                # Git статистика
                git_stats = self.git_analyzer.get_file_stats(path)
                context.git_commits_count = git_stats["commits"]
                context.git_last_commit = git_stats["last_commit"]
                context.git_authors = git_stats["authors"]

                # Анализ сложности
                context.complexity_score = self._calculate_complexity(file_path)

                # Анализ связанности
                context.coupling_score = self._calculate_coupling(file_path)

                # Обновляем кеш
                self.cache.file_hashes[path] = current_hash

            # Обогащаем данными из project analyzer
            self._enhance_context_with_analyzer_data(path, context)

            # Пересчитываем приоритет
            context.priority = self.prioritizer.calculate_priority(context)

        # Сохраняем кеш
        self._save_cache()

    def _get_file_hash(self, file_path: Path) -> str:
        """Вычисляет хеш файла для проверки изменений"""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""

    def _calculate_complexity(self, file_path: Path) -> float:
        """Вычисляет сложность файла"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            complexity = 0

            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
                    complexity += 1
                elif isinstance(node, ast.FunctionDef):
                    complexity += 0.5
                elif isinstance(node, ast.ClassDef):
                    complexity += 2

            return complexity
        except:
            return 0.0

    def _calculate_coupling(self, file_path: Path) -> float:
        """Вычисляет связанность модуля с другими"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module)

            # Считаем внутренние импорты
            internal_imports = sum(
                1 for imp in imports if not imp.startswith(("python", "sys", "os"))
            )

            return float(internal_imports)
        except:
            return 0.0

    def generate_ai_context(self, task_type: str, query: str = "") -> dict:
        """Генерирует оптимальный контекст с ML улучшениями"""

        # Базовая селекция файлов
        relevant_files = self._select_relevant_files_smart(task_type, query)

        # Семантический поиск, если есть запрос
        if query and ML_AVAILABLE:
            semantic_results = self.semantic_engine.search(query, top_k=5)
            for file_path, score in semantic_results:
                if file_path not in relevant_files and score > 0.3:
                    relevant_files.append(file_path)

        # Генерируем контекст
        context_summary = self._generate_smart_summary(relevant_files, task_type, query)
        commands = self._suggest_commands_smart(task_type, relevant_files, query)
        warnings = self._generate_warnings_smart(task_type, relevant_files)

        # Добавляем ML insights с интеграцией project analyzer
        if self.analyzer_metrics:
            insights = self._generate_enhanced_insights(relevant_files)
        else:
            insights = self._generate_ml_insights(relevant_files, query)

        return {
            "task_type": task_type,
            "relevant_files": relevant_files[:20],  # Ограничиваем до 20
            "context_summary": context_summary,
            "suggested_commands": commands,
            "warnings": warnings,
            "ml_insights": insights,
            "semantic_matches": semantic_results[:3] if query and ML_AVAILABLE else [],
        }

    def _select_relevant_files_smart(self, task_type: str, query: str) -> list[str]:
        """Умная селекция файлов с учетом динамических приоритетов"""

        # Сортируем файлы по приоритету
        sorted_files = sorted(
            self.file_contexts.items(), key=lambda x: x[1].priority, reverse=True
        )

        relevant = []

        # Добавляем критичные файлы (приоритет >= 4)
        for path, ctx in sorted_files:
            if ctx.priority >= 4.0:
                relevant.append(path)

        # Добавляем файлы по типу задачи
        task_categories = {
            "debug": ["database", "analyzer", "cli", "tests"],
            "develop": ["analyzer", "cli", "models", "database"],
            "analyze": ["analyzer", "database", "models"],
            "refactor": ["analyzer", "cli", "models", "database", "scripts"],
        }

        for path, ctx in sorted_files:
            if ctx.category in task_categories.get(task_type, []):
                if path not in relevant:
                    relevant.append(path)

        # Добавляем недавно измененные файлы
        recent_files = self.git_analyzer.get_recent_changes(days=7)
        for file in recent_files[:5]:  # Топ 5 недавних
            full_path = str(self.project_root / file)
            if full_path in self.file_contexts and full_path not in relevant:
                relevant.append(full_path)

        return relevant

    def _generate_smart_summary(
        self, files: list[str], task_type: str, query: str
    ) -> str:
        """Генерирует умное описание контекста"""

        summary_parts = [
            f"🎯 Контекст для: {task_type.upper()}",
            f"📁 Файлов: {len(files)}",
        ]

        # Анализируем распределение по категориям
        categories = defaultdict(list)
        total_complexity = 0
        recent_count = 0

        for file_path in files:
            ctx = self.file_contexts.get(file_path)
            if ctx:
                categories[ctx.category].append(ctx)
                total_complexity += ctx.complexity_score
                if "hour" in ctx.git_last_commit or "day" in ctx.git_last_commit:
                    recent_count += 1

        # Добавляем статистику
        if recent_count > 0:
            summary_parts.append(f"🔥 Недавно изменено: {recent_count} файлов")

        avg_complexity = total_complexity / len(files) if files else 0
        if avg_complexity > 50:
            summary_parts.append(
                f"⚠️ Высокая сложность кода (avg: {avg_complexity:.1f})"
            )

        # Категории
        for category, contexts in categories.items():
            high_priority = sum(1 for ctx in contexts if ctx.priority >= 4.0)
            summary_parts.append(
                f"• {category}: {len(contexts)} файлов ({high_priority} критичных)"
            )

        # Добавляем контекст запроса
        if query:
            summary_parts.append(f"\n🔍 Поиск: '{query[:50]}...'")

        return "\n".join(summary_parts)

    def _suggest_commands_smart(
        self, task_type: str, files: list[str], query: str
    ) -> list[str]:
        """Предлагает команды с учетом контекста"""

        commands = []

        # Базовые команды всегда полезны
        commands.extend(
            [
                "python main.py --info  # Статус системы",
                "python main.py --test  # Валидация",
            ]
        )

        # Анализируем, какие компоненты затронуты
        has_db = any("database" in f for f in files)
        has_analyzer = any("analyzer" in f for f in files)
        has_tests = any("test" in f for f in files)

        if task_type == "debug":
            if has_db:
                commands.append("python check_stats.py  # Проверка БД")
            if has_analyzer:
                commands.append("python main.py --benchmark  # Тест анализаторов")
            if not has_tests:
                commands.append("pytest tests/ -v  # Запустить тесты")

        elif task_type == "develop":
            if has_analyzer:
                commands.append(
                    f"python main.py --analyze '{query or 'test'}' --analyzer hybrid"
                )
            commands.append(
                "python scripts/tools/ai_project_analyzer.py  # Аудит архитектуры"
            )

        elif task_type == "analyze":
            commands.append("python scripts/mass_qwen_analysis.py --test")
            if has_db:
                commands.append("python scripts/db_browser.py  # Браузер БД")

        elif task_type == "refactor":
            commands.append("grep -r 'TODO\\|FIXME' src/  # Найти TODO")
            commands.append("python scripts/tools/ai_project_analyzer.py --duplicates")

        return commands

    def _generate_warnings_smart(self, task_type: str, files: list[str]) -> list[str]:
        """Генерирует умные предупреждения"""

        warnings = []

        # Проверяем наличие legacy кода
        legacy_count = sum(
            1
            for f in files
            if "legacy"
            in self.file_contexts.get(
                f, EnhancedFileContext("", 0, "", "", "", 0, [])
            ).category
        )
        if legacy_count > 0:
            warnings.append(
                f"⚠️ {legacy_count} legacy файлов в контексте - используй только для справки"
            )

        # Проверяем сложность
        high_complexity_files = [
            f
            for f in files
            if self.file_contexts.get(
                f, EnhancedFileContext("", 0, "", "", "", 0, [])
            ).complexity_score
            > 100
        ]
        if high_complexity_files:
            warnings.append(
                f"🔥 {len(high_complexity_files)} файлов с высокой сложностью - будь внимателен"
            )

        # Проверяем недавние изменения
        recent_critical = [
            f
            for f in files
            if self.file_contexts.get(f)
            and self.file_contexts[f].priority >= 4.0
            and "hour" in self.file_contexts[f].git_last_commit
        ]
        if recent_critical:
            warnings.append(
                f"🚨 {len(recent_critical)} критичных файлов изменены недавно - проверь совместимость"
            )

        # Стандартные предупреждения
        warnings.extend(
            [
                "🔄 Используй PostgreSQL, не SQLite",
                "🎯 main.py - единая точка входа",
                "🧪 Тестируй через python main.py --test",
            ]
        )

        return warnings

    def _generate_ml_insights(self, files: list[str], query: str) -> list[str]:
        """Генерирует ML-основанные инсайты"""

        insights = []

        # Анализируем паттерны
        if len(files) > 10:
            # Находим самые связанные модули
            most_coupled = sorted(
                [
                    (f, self.file_contexts[f].coupling_score)
                    for f in files
                    if f in self.file_contexts
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:3]

            if most_coupled and most_coupled[0][1] > 10:
                insights.append(
                    f"🔗 Высокая связанность: {Path(most_coupled[0][0]).name} "
                    f"импортирует {int(most_coupled[0][1])} модулей"
                )

        # Анализируем git активность
        active_authors = set()
        for f in files:
            if f in self.file_contexts:
                active_authors.update(self.file_contexts[f].git_authors)

        if len(active_authors) > 3:
            insights.append(
                f"👥 Над этими файлами работали {len(active_authors)} разработчиков"
            )

        # Семантические инсайты
        if query and ML_AVAILABLE:
            insights.append(f"🧠 Семантический поиск активен для '{query[:30]}...'")

        return insights

    def save_context(self):
        """Сохраняет контекст с новыми данными"""
        data = {path: asdict(ctx) for path, ctx in self.file_contexts.items()}
        with open(self.context_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _build_initial_context(self):
        """Строит начальный контекст с улучшенной логикой"""

        # Сканируем проект более интеллектуально
        python_files = list(self.project_root.rglob("*.py"))
        config_files = list(self.project_root.glob("*.yaml")) + list(
            self.project_root.glob("*.yml")
        )
        doc_files = list(self.project_root.glob("docs/*.md"))

        for file_path in python_files + config_files + doc_files:
            if self._should_skip(file_path):
                continue

            category = self._determine_category(file_path)

            context = EnhancedFileContext(
                path=str(file_path),
                priority=2.5,  # Начальный приоритет
                category=category,
                description=self._generate_description(file_path),
                last_modified="",
                size_lines=self._count_lines(file_path),
                dependencies=self._extract_dependencies(file_path),
            )

            self.file_contexts[str(file_path)] = context

        # Обновляем все метрики
        self._update_dynamic_data()
        self.save_context()

    def _should_skip(self, file_path: Path) -> bool:
        """Проверяет, нужно ли пропустить файл"""
        skip_patterns = [
            "__pycache__",
            ".git",
            "venv",
            ".venv",
            "node_modules",
            ".pytest_cache",
            "*.pyc",
        ]
        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _determine_category(self, file_path: Path) -> str:
        """Определяет категорию файла"""
        path_str = str(file_path).lower()

        if "database" in path_str or "postgres" in path_str:
            return "database"
        if "analyzer" in path_str:
            return "analyzer"
        if "cli" in path_str or file_path.name == "main.py":
            return "cli"
        if "model" in path_str:
            return "models"
        if "test" in path_str:
            return "tests"
        if "archive" in path_str or "sqlite" in path_str:
            return "legacy"
        if "script" in path_str:
            return "scripts"
        if file_path.suffix in [".yaml", ".yml", ".json", ".env"]:
            return "config"
        if file_path.suffix == ".md":
            return "docs"
        return "other"

    def _generate_description(self, file_path: Path) -> str:
        """Генерирует описание файла"""
        try:
            with open(file_path, encoding="utf-8") as f:
                first_lines = f.read(500)

            # Ищем docstring или комментарии
            if '"""' in first_lines:
                docstring = (
                    first_lines.split('"""')[1]
                    if len(first_lines.split('"""')) > 1
                    else ""
                )
                return docstring.strip()[:100]
            if "#" in first_lines:
                comment = first_lines.split("\n")[0].replace("#", "").strip()
                return comment[:100]
        except:
            pass

        return f"Файл {file_path.name}"

    def _count_lines(self, file_path: Path) -> int:
        """Подсчитывает количество строк в файле"""
        try:
            with open(file_path, encoding="utf-8") as f:
                return len(f.readlines())
        except:
            return 0

    def _extract_dependencies(self, file_path: Path) -> list[str]:
        """Извлекает зависимости из файла"""
        if file_path.suffix != ".py":
            return []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            imports = set()

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])

            return list(imports)
        except:
            return []

    def generate_llm_descriptions(self):
        """Генерация LLM описаний для файлов"""

        if not self.llm_generator:
            print("❌ LLM генератор не настроен. Используйте --llm-descriptions")
            return

        print("🤖 Генерируем AI описания для файлов...")
        descriptions = self.llm_generator.generate_descriptions(self.file_contexts)
        print(f"✅ Сгенерировано описаний: {len(descriptions)}")

    def create_dependency_graph(self, output_path: str = "dependency_graph.dot"):
        """Создание графа зависимостей"""

        if not self.visualizer:
            self.visualizer = DependencyVisualizer(self.file_contexts)

        graph_content = self.visualizer.generate_dependency_graph()
        output_file = self.visualizer.save_graph()
        print(f"📊 Граф зависимостей создан: {output_file}")
        return output_file

    def start_api_server(self, host: str = "127.0.0.1", port: int = 8000):
        """Запуск REST API сервера"""

        if not self.api_server:
            print("❌ API сервер не настроен. Используйте --api")
            return

        print(f"🚀 Запускаем API сервер на http://{host}:{port}")
        self.api_server.run(host, port)


def auto_detect_task_type(query: str) -> str:
    """Автоматически определяет тип задачи по запросу"""
    query_lower = query.lower()

    debug_keywords = [
        "error",
        "bug",
        "fix",
        "broken",
        "crash",
        "timeout",
        "exception",
        "fail",
    ]
    develop_keywords = ["add", "create", "implement", "feature", "new", "build"]
    analyze_keywords = ["analyze", "stats", "metrics", "report", "check", "examine"]
    refactor_keywords = ["refactor", "optimize", "improve", "clean", "restructure"]

    if any(kw in query_lower for kw in debug_keywords):
        return "debug"
    if any(kw in query_lower for kw in develop_keywords):
        return "develop"
    if any(kw in query_lower for kw in analyze_keywords):
        return "analyze"
    if any(kw in query_lower for kw in refactor_keywords):
        return "refactor"
    return "debug"  # Default


def interactive_mode():
    """Интерактивный режим работы с контекстом"""
    print("🤖 AI Context Manager PRO - Интерактивный режим")
    print("=" * 50)

    manager = AIContextManagerPro()

    while True:
        print("\n🎯 Выберите действие:")
        print("1. Генерация контекста по запросу")
        print("2. Семантический поиск файлов")
        print("3. Анализ git активности")
        print("4. Обновить кеш")
        print("5. Статистика проекта")
        print("6. Интеграция с Project Analyzer")
        print("0. Выход")

        choice = input("\nВыбор: ").strip()

        if choice == "0":
            break
        if choice == "1":
            query = input("🔍 Опишите задачу: ").strip()
            if query:
                task_type = auto_detect_task_type(query)
                print(f"📝 Определен тип задачи: {task_type.upper()}")

                context = manager.generate_ai_context(task_type, query)
                print_context_pretty(context)

                export = input("\n💾 Экспорт в файл? (y/N): ").strip().lower()
                if export == "y":
                    export_context(
                        context,
                        f"context_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    )

        elif choice == "2":
            if not ML_AVAILABLE:
                print("❌ Семантический поиск недоступен - установите scikit-learn")
                continue

            query = input("🧠 Запрос для семантического поиска: ").strip()
            if query:
                results = manager.semantic_engine.search(query, top_k=10)
                print(f"\n🔍 Найдено {len(results)} релевантных файлов:")
                for i, (path, score) in enumerate(results, 1):
                    filename = Path(path).name
                    print(f"{i}. {filename} (релевантность: {score:.3f})")

        elif choice == "3":
            recent_files = manager.git_analyzer.get_recent_changes(days=7)
            print(f"\n📅 Недавно измененные файлы (7 дней): {len(recent_files)}")
            for i, file in enumerate(recent_files[:10], 1):
                print(f"{i}. {file}")

        elif choice == "4":
            print("🔄 Обновление кеша...")
            manager._update_dynamic_data()
            manager.save_context()
            print("✅ Кеш обновлен!")

        elif choice == "5":
            print_project_stats(manager)

        elif choice == "6":
            print("🔗 Интеграция с Project Analyzer...")
            success = manager.integrate_with_project_analyzer()
            if success:
                print(
                    "✅ Интеграция успешна! Данные Project Analyzer добавлены в контекст"
                )
                manager._update_dynamic_data()
                manager.save_context()
            else:
                print(
                    "❌ Интеграция не удалась - проверьте наличие ai_project_analyzer.py"
                )


def print_context_pretty(context: dict):
    """Красивый вывод контекста"""
    print("\n" + "=" * 60)
    print(f"🎯 {context['task_type'].upper()} КОНТЕКСТ")
    print("=" * 60)

    print(f"\n📋 {context['context_summary']}")

    print(f"\n📁 Релевантные файлы ({len(context['relevant_files'])}):")
    for i, file_path in enumerate(context["relevant_files"][:15], 1):
        filename = Path(file_path).name
        print(f"  {i:2d}. {filename}")

    if context.get("semantic_matches"):
        print("\n🧠 Семантические совпадения:")
        for path, score in context["semantic_matches"]:
            filename = Path(path).name
            print(f"  • {filename} (relevance: {score:.3f})")

    print("\n💡 Рекомендуемые команды:")
    for i, cmd in enumerate(context["suggested_commands"][:5], 1):
        print(f"  {i}. {cmd}")

    print("\n⚠️ Предупреждения:")
    for warning in context["warnings"][:3]:
        print(f"  • {warning}")

    if context.get("ml_insights"):
        print("\n🚀 ML Инсайты:")
        for insight in context["ml_insights"]:
            print(f"  • {insight}")


def print_project_stats(manager: AIContextManagerPro):
    """Выводит статистику проекта"""
    print("\n📊 СТАТИСТИКА ПРОЕКТА")
    print("=" * 40)

    # Статистика по категориям
    categories = defaultdict(list)
    total_complexity = 0
    high_priority_count = 0
    recent_count = 0

    for path, ctx in manager.file_contexts.items():
        categories[ctx.category].append(ctx)
        total_complexity += ctx.complexity_score
        if ctx.priority >= 4.0:
            high_priority_count += 1
        if "hour" in ctx.git_last_commit or "day" in ctx.git_last_commit:
            recent_count += 1

    print(f"📁 Всего файлов: {len(manager.file_contexts)}")
    print(f"🔥 Критичных (priority >= 4): {high_priority_count}")
    print(f"⏰ Недавно изменены: {recent_count}")
    print(f"🧮 Средняя сложность: {total_complexity / len(manager.file_contexts):.1f}")

    print("\n📂 Распределение по категориям:")
    for category, contexts in categories.items():
        avg_priority = sum(ctx.priority for ctx in contexts) / len(contexts)
        print(
            f"  • {category}: {len(contexts)} файлов (avg priority: {avg_priority:.1f})"
        )


def export_context(context: dict, filename: str):
    """Экспорт контекста в файл"""
    try:
        output_path = Path("results") / filename
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2, ensure_ascii=False)

        print(f"✅ Контекст экспортирован: {output_path}")

        # Опциональное копирование в буфер обмена
        try:
            import pyperclip

            context_text = json.dumps(context, indent=2, ensure_ascii=False)
            pyperclip.copy(context_text)
            print("📋 Скопировано в буфер обмена")
        except ImportError:
            pass

    except Exception as e:
        print(f"❌ Ошибка экспорта: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="🤖 AI Context Manager PRO - продвинутый менеджер контекста с ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🎯 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:

# Интерактивный режим (рекомендуется)
python scripts/tools/ai_context_manager_pro.py --interactive

# Автоматическое определение задачи
python scripts/tools/ai_context_manager_pro.py --query "fix database connection error"

# Конкретная задача
python scripts/tools/ai_context_manager_pro.py --task debug --query "postgres timeout"

# Семантический поиск
python scripts/tools/ai_context_manager_pro.py --semantic-search "analyzer performance"

# Обновить кеш
python scripts/tools/ai_context_manager_pro.py --update-cache

# Экспорт контекста
python scripts/tools/ai_context_manager_pro.py --task develop --export context.json

🚀 НОВЫЕ ВОЗМОЖНОСТИ PRO:
• Git-based приоритизация файлов
• ML семантический поиск с TF-IDF
• Умное кеширование с инвалидацией
• Автоопределение типа задачи
• Динамические метрики сложности
        """,
    )

    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Интерактивный режим с CLI меню",
    )
    parser.add_argument(
        "--task",
        "-t",
        choices=["debug", "develop", "analyze", "refactor"],
        help="Тип задачи",
    )
    parser.add_argument(
        "--query", "-q", type=str, help="Описание задачи для автоопределения типа"
    )
    parser.add_argument(
        "--semantic-search", "-s", type=str, help="Семантический поиск по файлам"
    )
    parser.add_argument(
        "--update-cache",
        "-u",
        action="store_true",
        help="Обновить кеш без генерации контекста",
    )
    parser.add_argument("--export", "-e", type=str, help="Экспорт результата в файл")
    parser.add_argument(
        "--integrate", action="store_true", help="Интеграция с ai_project_analyzer"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Показать статистику проекта"
    )

    # Advanced features
    parser.add_argument(
        "--llm-descriptions",
        action="store_true",
        help="Генерация AI описаний через LLM (Ollama)",
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true", help="Создать граф зависимостей"
    )
    parser.add_argument("--api", action="store_true", help="Запустить REST API сервер")
    parser.add_argument(
        "--api-host",
        default="127.0.0.1",
        help="Хост для API сервера (по умолчанию: 127.0.0.1)",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Порт для API сервера (по умолчанию: 8000)",
    )

    args = parser.parse_args()

    # Интерактивный режим
    if args.interactive:
        interactive_mode()
        exit()

    # Создаем менеджер
    try:
        manager = AIContextManagerPro()
        print("🤖 AI Context Manager PRO инициализирован")

        if ML_AVAILABLE:
            print("✅ ML возможности доступны (scikit-learn)")
        else:
            print("⚠️ ML возможности ограничены - установите: pip install scikit-learn")

    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        exit(1)

    # Обновление кеша
    if args.update_cache:
        print("🔄 Обновление кеша...")
        manager._update_dynamic_data()
        manager.save_context()
        print("✅ Кеш обновлен!")
        exit()

    # Статистика проекта
    if args.stats:
        print_project_stats(manager)
        exit()

    # Семантический поиск
    if args.semantic_search:
        if not ML_AVAILABLE:
            print("❌ Семантический поиск недоступен - установите scikit-learn")
            exit(1)

        print(f"🧠 Семантический поиск: '{args.semantic_search}'")
        results = manager.semantic_engine.search(args.semantic_search, top_k=10)

        if results:
            print(f"\n🔍 Найдено {len(results)} релевантных файлов:")
            for i, (path, score) in enumerate(results, 1):
                filename = Path(path).name
                print(f"  {i:2d}. {filename} (relevance: {score:.3f})")
        else:
            print("🤷 Релевантные файлы не найдены")
        exit()

    # Интеграция с project analyzer
    if args.integrate:
        print("🔗 Интеграция с Project Analyzer...")
        success = manager.integrate_with_project_analyzer()
        if success:
            print("✅ Интеграция успешна!")
            manager._update_dynamic_data()
            manager.save_context()
        else:
            print("❌ Интеграция не удалась")
        exit()

    # Настройка продвинутых возможностей
    if args.llm_descriptions or args.api:
        enable_llm = args.llm_descriptions
        enable_api = args.api
        manager.setup_advanced_features(enable_llm=enable_llm, enable_api=enable_api)

    # LLM описания
    if args.llm_descriptions:
        print("🤖 Генерация AI описаний...")
        manager.generate_llm_descriptions()
        exit()

    # Визуализация зависимостей
    if args.visualize:
        print("📊 Создание графа зависимостей...")
        output_file = manager.create_dependency_graph()
        print(f"📁 Граф сохранен: {output_file}")
        exit()

    # Запуск API сервера
    if args.api:
        print("🚀 Запуск API сервера...")
        manager.start_api_server(args.api_host, args.api_port)
        exit()

    # Генерация контекста
    if args.query:
        task_type = args.task or auto_detect_task_type(args.query)
        print(f"🎯 Генерация контекста для: {task_type.upper()}")
        print(f"🔍 Запрос: '{args.query}'")

        context = manager.generate_ai_context(task_type, args.query)
        print_context_pretty(context)

        # Экспорт
        if args.export:
            export_context(context, args.export)

    elif args.task:
        print(f"🎯 Генерация контекста для: {args.task.upper()}")
        context = manager.generate_ai_context(args.task, "")
        print_context_pretty(context)

        # Экспорт
        if args.export:
            export_context(context, args.export)

    else:
        print("🤖 AI Context Manager PRO")
        print(
            "Используйте --help для списка команд или --interactive для интерактивного режима"
        )
        print("\n🚀 Быстрый старт:")
        print("  python scripts/tools/ai_context_manager_pro.py --interactive")
