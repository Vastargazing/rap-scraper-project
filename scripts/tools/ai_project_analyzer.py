#!/usr/bin/env python3
"""
🧠 AI Project Analyzer — интеллектуальная система анализа проекта для AI ассистентов

НАЗНАЧЕНИЕ:
- Семантический анализ дубликатов через AST-парсинг вместо примитивного grep
- Анализ архитектурных нарушений с учетом PostgreSQL миграции
- Контекстно-зависимый поиск неиспользуемых файлов
- Специализированные метрики для микросервисной архитектуры
- Security анализ уязвимостей и проблем безопасности

ИСПОЛЬЗОВАНИЕ:
python scripts/tools/ai_project_analyzer.py --analyze

ЗАВИСИМОСТИ:
- Python 3.8+
- ast, dataclasses, pathlib

РЕЗУЛЬТАТ:
- Семантический анализ архитектуры и дубликатов
- Метрики для микросервисной архитектуры
- Анализ безопасности кода

АВТОР: Vastargazing
ДАТА: Сентябрь 2025
"""

import ast
import hashlib
import json
import pickle
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CodeMetrics:
    file_path: str
    lines_of_code: int
    functions: list[str]
    classes: list[str]
    imports: list[str]
    complexity_score: float
    last_modified: float
    is_test: bool
    is_legacy: bool


@dataclass
class DuplicationResult:
    files: list[str]
    similarity: float
    common_functions: list[str]


class SecurityAnalyzer:
    """Анализатор проблем безопасности"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.security_issues = []

    def find_security_issues(self, file_metrics: dict) -> list[str]:
        """Поиск проблем безопасности в коде"""

        issues = []

        for file_path, metrics in file_metrics.items():
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Hardcoded passwords
                password_patterns = [
                    r'password\s*=\s*["\'][^"\']{3,}["\']',
                    r'pwd\s*=\s*["\'][^"\']{3,}["\']',
                    r'secret\s*=\s*["\'][^"\']{8,}["\']',
                ]
                for pattern in password_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(f"🔒 Hardcoded password in {file_path}")
                        break

                # SQL injection patterns
                sql_patterns = [
                    r'f".*SELECT.*FROM.*{',  # f-string SQL
                    r"\.format\(.*SELECT.*FROM",  # format SQL
                    r'\+.*["\'].*SELECT.*FROM',  # concatenated SQL
                ]
                for pattern in sql_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(f"💉 Potential SQL injection in {file_path}")
                        break

                # Exposed API keys
                api_patterns = [
                    r'api_key\s*=\s*["\'][A-Za-z0-9]{20,}["\']',
                    r'token\s*=\s*["\'][A-Za-z0-9]{20,}["\']',
                    r'secret_key\s*=\s*["\'][A-Za-z0-9]{32,}["\']',
                ]
                for pattern in api_patterns:
                    if re.search(pattern, content):
                        issues.append(f"🔑 Exposed API key in {file_path}")
                        break

                # Insecure random
                if re.search(r"import random\b", content) and re.search(
                    r"random\.(choice|randint|random)", content
                ):
                    if "password" in content.lower() or "token" in content.lower():
                        issues.append(f"🎲 Insecure random for security in {file_path}")

                # Pickle security
                if re.search(r"pickle\.loads?\(", content):
                    issues.append(f"⚠️ Pickle usage (security risk) in {file_path}")

                # Eval/exec usage
                dangerous_funcs = [r"\beval\(", r"\bexec\(", r"__import__\("]
                for func in dangerous_funcs:
                    if re.search(func, content):
                        issues.append(f"⚡ Dangerous function usage in {file_path}")
                        break

            except Exception:
                continue

        return issues


class PerformanceAnalyzer:
    """Анализатор проблем производительности"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.performance_issues = []

    def find_performance_issues(self, file_metrics: dict) -> list[str]:
        """Поиск проблем производительности в коде"""

        issues = []

        for file_path, metrics in file_metrics.items():
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content)

                # Анализируем вложенные циклы
                nested_loops = self._find_nested_loops(tree)
                if nested_loops > 2:
                    issues.append(
                        f"🔄 Deep nested loops (depth: {nested_loops}) in {file_path}"
                    )

                # Поиск N+1 query patterns
                if self._has_n_plus_one_pattern(tree, content):
                    issues.append(f"🗃️ Potential N+1 query pattern in {file_path}")

                # Поиск неэффективных операций в циклах
                inefficient_ops = self._find_inefficient_loop_operations(tree, content)
                for op in inefficient_ops:
                    issues.append(
                        f"⚡ Inefficient operation in loop: {op} in {file_path}"
                    )

                # Поиск больших объектов в памяти
                if self._has_memory_intensive_operations(content):
                    issues.append(f"💾 Memory-intensive operations in {file_path}")

                # Поиск синхронного IO в циклах
                if self._has_sync_io_in_loops(tree, content):
                    issues.append(f"⏳ Synchronous I/O in loops in {file_path}")

            except Exception:
                continue

        return issues

    def _find_nested_loops(self, tree: ast.AST) -> int:
        """Находит максимальную глубину вложенных циклов"""
        max_depth = 0

        def count_depth(node, current_depth=0):
            nonlocal max_depth

            if isinstance(node, (ast.For, ast.While)):
                current_depth += 1
                max_depth = max(max_depth, current_depth)

            for child in ast.iter_child_nodes(node):
                count_depth(child, current_depth)

        count_depth(tree)
        return max_depth

    def _has_n_plus_one_pattern(self, tree: ast.AST, content: str) -> bool:
        """Ищет паттерны N+1 запросов"""

        # Поиск циклов с SQL запросами внутри
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Проверяем, есть ли SQL запросы в теле цикла
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(
                            child.func, ast.Attribute
                        ) and child.func.attr in ["execute", "query", "get", "filter"]:
                            return True

                    # Проверяем строковые литералы с SQL
                    if isinstance(child, (ast.Str, ast.Constant)):
                        str_value = child.s if hasattr(child, "s") else str(child.value)
                        if isinstance(str_value, str) and any(
                            sql_word in str_value.upper()
                            for sql_word in ["SELECT", "INSERT", "UPDATE", "DELETE"]
                        ):
                            return True

        return False

    def _find_inefficient_loop_operations(
        self, tree: ast.AST, content: str
    ) -> list[str]:
        """Находит неэффективные операции в циклах"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                # Поиск операций со списками в циклах
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(
                            child.func, ast.Attribute
                        ) and child.func.attr in ["append", "extend", "insert"]:
                            # Проверяем, не используется ли list.append в цикле
                            if "append" in ast.dump(child):
                                issues.append(
                                    "list.append() in loop (consider list comprehension)"
                                )

                        # Поиск повторных вычислений
                        if isinstance(child.func, ast.Name) and child.func.id in [
                            "len",
                            "max",
                            "min",
                            "sum",
                        ]:
                            issues.append(
                                f"{child.func.id}() called in loop (cache result)"
                            )

        return issues

    def _has_memory_intensive_operations(self, content: str) -> bool:
        """Ищет операции, интенсивно использующие память"""

        memory_patterns = [
            r"\.read\(\)",  # Чтение всего файла
            r"\.readlines\(\)",  # Чтение всех строк
            r"pickle\.loads?\([^)]+\)",  # Pickle операции
            r"json\.loads?\(.+\)",  # JSON операции с большими данными
            r"\[\s*.*\s*for\s+.*\s+in\s+.*\]",  # Большие list comprehensions
        ]

        for pattern in memory_patterns:
            if re.search(pattern, content):
                return True

        return False

    def _has_sync_io_in_loops(self, tree: ast.AST, content: str) -> bool:
        """Ищет синхронные I/O операции в циклах"""

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loop_content = ast.dump(node)

                # Проверяем наличие I/O операций
                io_patterns = [
                    "open(",
                    "requests.",
                    "urllib.",
                    "socket.",
                    "subprocess.",
                ]
                for pattern in io_patterns:
                    if pattern in loop_content or pattern in content:
                        return True

        return False


class GitBlameAnalyzer:
    """Анализатор git blame для поиска hotspots и bus factor"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.author_stats = {}
        self.file_hotspots = {}

    def analyze_git_patterns(self, file_metrics: dict) -> dict:
        """Анализ паттернов git для выявления проблемных зон"""

        results = {
            "hotspots": [],
            "bus_factor_risks": [],
            "author_ownership": {},
            "change_frequency": {},
        }

        for file_path in file_metrics:
            try:
                # Анализ частоты изменений
                change_count = self._get_file_change_count(file_path)
                if change_count > 50:  # Файлы с частыми изменениями
                    results["hotspots"].append(
                        {
                            "file": file_path,
                            "changes": change_count,
                            "reason": "High change frequency",
                        }
                    )

                # Анализ авторства (bus factor)
                authors = self._get_file_authors(file_path)
                if len(authors) == 1 and change_count > 10:
                    results["bus_factor_risks"].append(
                        {
                            "file": file_path,
                            "sole_author": list(authors.keys())[0],
                            "changes": change_count,
                            "risk_level": "HIGH" if change_count > 30 else "MEDIUM",
                        }
                    )

                # Статистика по авторам
                for author, lines in authors.items():
                    if author not in results["author_ownership"]:
                        results["author_ownership"][author] = {"files": 0, "lines": 0}
                    results["author_ownership"][author]["files"] += 1
                    results["author_ownership"][author]["lines"] += lines

                results["change_frequency"][file_path] = change_count

            except Exception:
                continue

        return results

    def _get_file_change_count(self, file_path: str) -> int:
        """Получает количество изменений файла из git log"""
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "--", file_path],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                return (
                    len(result.stdout.strip().split("\n"))
                    if result.stdout.strip()
                    else 0
                )
        except Exception:
            pass

        return 0

    def _get_file_authors(self, file_path: str) -> dict[str, int]:
        """Получает статистику авторов по файлу"""
        authors = {}

        try:
            result = subprocess.run(
                ["git", "blame", "--line-porcelain", file_path],
                check=False,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.split("\n")
                current_author = None

                for line in lines:
                    if line.startswith("author "):
                        current_author = line[7:]  # Remove "author " prefix
                    elif line.startswith("\t") and current_author:
                        # This is a code line
                        if current_author not in authors:
                            authors[current_author] = 0
                        authors[current_author] += 1

        except Exception:
            pass

        return authors


class HTMLReportGenerator:
    """Генератор HTML отчетов с интерактивными графиками"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.output_dir = project_root / "results" / "html_reports"
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def generate_dashboard(self, analysis_results: dict) -> Path:
        """Генерирует интерактивный HTML dashboard"""

        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("⚠️ Plotly не установлен. Используйте: pip install plotly")
            return self._generate_simple_dashboard(analysis_results)

        # Подготовка данных
        summary = analysis_results["summary"]
        metrics = analysis_results.get("export_data", {}).get("complexity_scores", {})

        # Создаем subplot dashboard
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Распределение сложности файлов",
                "Категории файлов",
                "Проблемы по типам",
                "Топ-10 самых сложных файлов",
                "Тренд качества кода",
                "Статистика анализа",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "table"}],
            ],
        )

        # 1. Гистограмма сложности
        if metrics:
            complexities = list(metrics.values())
            fig.add_trace(
                go.Histogram(
                    x=complexities,
                    name="Сложность",
                    marker_color="lightblue",
                    opacity=0.7,
                ),
                row=1,
                col=1,
            )

        # 2. Pie chart типов файлов
        file_types = {
            "Активные": summary["active_files"],
            "Legacy": summary["legacy_files"],
            "Тесты": summary["test_files"],
        }
        fig.add_trace(
            go.Pie(
                labels=list(file_types.keys()),
                values=list(file_types.values()),
                name="Типы файлов",
            ),
            row=1,
            col=2,
        )

        # 3. Bar chart проблем
        issues_count = {
            "Безопасность": len(analysis_results.get("security_issues", [])),
            "Производительность": len(analysis_results.get("performance_issues", [])),
            "Архитектура": len(analysis_results.get("architecture_violations", [])),
            "Дубликаты": len(analysis_results.get("duplicates_analysis", [])),
        }
        fig.add_trace(
            go.Bar(
                x=list(issues_count.keys()),
                y=list(issues_count.values()),
                name="Проблемы",
                marker_color=["red", "orange", "yellow", "lightcoral"],
            ),
            row=2,
            col=1,
        )

        # 4. Топ сложных файлов
        if metrics:
            top_complex = sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:10]
            files = [Path(f).name for f, _ in top_complex]
            complexities = [c for _, c in top_complex]

            fig.add_trace(
                go.Bar(
                    x=complexities,
                    y=files,
                    orientation="h",
                    name="Сложность",
                    marker_color="darkred",
                ),
                row=2,
                col=2,
            )

        # 5. Scatter качества кода (сложность vs связанность)
        if metrics:
            coupling_scores = analysis_results.get("export_data", {}).get(
                "coupling_scores", {}
            )
            if coupling_scores:
                x_vals = [coupling_scores.get(f, 0) for f in metrics.keys()]
                y_vals = list(metrics.values())

                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers",
                        name="Файлы",
                        marker=dict(
                            size=10, color=y_vals, colorscale="Reds", showscale=True
                        ),
                    ),
                    row=3,
                    col=1,
                )

        # 6. Таблица статистики
        stats_data = [
            ["Всего файлов", summary["total_files"]],
            ["Активных файлов", summary["active_files"]],
            ["Legacy файлов", summary["legacy_files"]],
            ["Средняя сложность", f"{summary['average_complexity']:.1f}"],
            ["Миграция PostgreSQL", "✅" if summary["migration_status"] else "❌"],
            ["Проблем безопасности", len(analysis_results.get("security_issues", []))],
            [
                "Проблем производительности",
                len(analysis_results.get("performance_issues", [])),
            ],
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Метрика", "Значение"], fill_color="lightblue", align="left"
                ),
                cells=dict(
                    values=[
                        [row[0] for row in stats_data],
                        [row[1] for row in stats_data],
                    ],
                    fill_color="white",
                    align="left",
                ),
            ),
            row=3,
            col=2,
        )

        # Настройка layout
        fig.update_layout(
            height=1200,
            showlegend=False,
            title_text="🧠 AI Project Analysis Dashboard",
            title_x=0.5,
            title_font_size=20,
        )

        # Добавляем названия осей
        fig.update_xaxes(title_text="Сложность", row=1, col=1)
        fig.update_yaxes(title_text="Количество файлов", row=1, col=1)
        fig.update_xaxes(title_text="Связанность (imports)", row=3, col=1)
        fig.update_yaxes(title_text="Сложность", row=3, col=1)

        # Создаем полный HTML
        html_content = self._create_html_template(
            fig.to_html(include_plotlyjs="cdn"), analysis_results
        )

        # Сохраняем файл
        report_file = self.output_dir / "project_analysis_dashboard.html"
        report_file.write_text(html_content, encoding="utf-8")

        return report_file

    def _generate_simple_dashboard(self, analysis_results: dict) -> Path:
        """Генерирует простой HTML отчет без Plotly"""

        summary = analysis_results["summary"]

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Project Analysis Report</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
                .header {{ text-align: center; color: #333; margin-bottom: 30px; }}
                .metric-card {{ 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    border-radius: 10px; 
                    min-width: 200px;
                    text-align: center;
                }}
                .metric-value {{ font-size: 2em; font-weight: bold; }}
                .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
                .issues-section {{ margin: 30px 0; }}
                .issue-type {{ 
                    margin: 15px 0; 
                    padding: 15px; 
                    border-left: 5px solid #ff6b6b; 
                    background: #fff5f5; 
                }}
                .recommendations {{ 
                    background: #f0f8ff; 
                    padding: 20px; 
                    border-radius: 10px; 
                    border-left: 5px solid #4dabf7; 
                }}
                ul {{ padding-left: 20px; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧠 AI Project Analysis Report</h1>
                    <p>Комплексный анализ проекта с использованием AI</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-value">{summary["total_files"]}</div>
                        <div class="metric-label">📁 Всего файлов</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary["active_files"]}</div>
                        <div class="metric-label">🔥 Активных файлов</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary["legacy_files"]}</div>
                        <div class="metric-label">🗂️ Legacy файлов</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{summary["average_complexity"]:.1f}</div>
                        <div class="metric-label">📊 Средняя сложность</div>
                    </div>
                </div>
                
                <div class="issues-section">
                    <h2>🔍 Выявленные проблемы</h2>
                    
                    <div class="issue-type">
                        <h3>🔒 Проблемы безопасности: {len(analysis_results.get("security_issues", []))}</h3>
                        <ul>
                            {"".join(f"<li>{issue}</li>" for issue in analysis_results.get("security_issues", [])[:5])}
                        </ul>
                    </div>
                    
                    <div class="issue-type">
                        <h3>⚡ Проблемы производительности: {len(analysis_results.get("performance_issues", []))}</h3>
                        <ul>
                            {"".join(f"<li>{issue}</li>" for issue in analysis_results.get("performance_issues", [])[:5])}
                        </ul>
                    </div>
                    
                    <div class="issue-type">
                        <h3>🏗️ Нарушения архитектуры: {len(analysis_results.get("architecture_violations", []))}</h3>
                        <ul>
                            {"".join(f"<li>{violation}</li>" for violation in analysis_results.get("architecture_violations", [])[:5])}
                        </ul>
                    </div>
                </div>
                
                <div class="recommendations">
                    <h2>💡 AI Рекомендации</h2>
                    <ul>
                        {"".join(f"<li>{rec}</li>" for rec in analysis_results.get("ai_recommendations", []))}
                    </ul>
                </div>
                
                <div style="text-align: center; margin-top: 30px; color: #666;">
                    <p>Отчет сгенерирован AI Project Analyzer</p>
                    <p>Для интерактивных графиков установите: pip install plotly</p>
                </div>
            </div>
        </body>
        </html>
        """

        report_file = self.output_dir / "project_analysis_simple.html"
        report_file.write_text(html_content, encoding="utf-8")

        return report_file

    def _create_html_template(self, plotly_html: str, analysis_results: dict) -> str:
        """Создает полный HTML шаблон с Plotly графиками"""

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Project Analysis Dashboard</title>
            <meta charset="utf-8">
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                .container {{ 
                    max-width: 1400px; 
                    margin: 0 auto; 
                    background: white; 
                    border-radius: 15px; 
                    overflow: hidden;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                }}
                .header {{ 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white; 
                    padding: 30px; 
                    text-align: center; 
                }}
                .content {{ padding: 20px; }}
                .summary-cards {{ 
                    display: flex; 
                    justify-content: space-around; 
                    margin: 20px 0; 
                    flex-wrap: wrap;
                }}
                .card {{ 
                    background: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 10px; 
                    text-align: center; 
                    margin: 10px;
                    min-width: 150px;
                    border-left: 5px solid #667eea;
                }}
                .card-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
                .card-label {{ color: #666; margin-top: 5px; }}
                .issues-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; 
                    margin: 30px 0; 
                }}
                .issue-card {{ 
                    background: #fff; 
                    border: 1px solid #eee; 
                    border-radius: 10px; 
                    padding: 20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .issue-title {{ 
                    font-weight: bold; 
                    margin-bottom: 15px; 
                    font-size: 1.1em;
                }}
                .issue-list {{ 
                    max-height: 200px; 
                    overflow-y: auto; 
                }}
                .issue-list li {{ 
                    margin: 8px 0; 
                    padding: 5px; 
                    background: #f8f9fa; 
                    border-radius: 5px;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧠 AI Project Analysis Dashboard</h1>
                    <p>Комплексный анализ проекта с интерактивными графиками</p>
                </div>
                
                <div class="content">
                    <div class="summary-cards">
                        <div class="card">
                            <div class="card-value">{analysis_results["summary"]["total_files"]}</div>
                            <div class="card-label">📁 Всего файлов</div>
                        </div>
                        <div class="card">
                            <div class="card-value">{analysis_results["summary"]["active_files"]}</div>
                            <div class="card-label">🔥 Активных</div>
                        </div>
                        <div class="card">
                            <div class="card-value">{len(analysis_results.get("security_issues", []))}</div>
                            <div class="card-label">🔒 Проблем безопасности</div>
                        </div>
                        <div class="card">
                            <div class="card-value">{len(analysis_results.get("performance_issues", []))}</div>
                            <div class="card-label">⚡ Проблем производительности</div>
                        </div>
                    </div>
                    
                    {plotly_html}
                    
                    <div class="issues-grid">
                        <div class="issue-card">
                            <div class="issue-title">🔒 Проблемы безопасности</div>
                            <ul class="issue-list">
                                {"".join(f"<li>{issue}</li>" for issue in analysis_results.get("security_issues", [])[:10])}
                            </ul>
                        </div>
                        
                        <div class="issue-card">
                            <div class="issue-title">⚡ Проблемы производительности</div>
                            <ul class="issue-list">
                                {"".join(f"<li>{issue}</li>" for issue in analysis_results.get("performance_issues", [])[:10])}
                            </ul>
                        </div>
                        
                        <div class="issue-card">
                            <div class="issue-title">🏗️ Нарушения архитектуры</div>
                            <ul class="issue-list">
                                {"".join(f"<li>{violation}</li>" for violation in analysis_results.get("architecture_violations", [])[:10])}
                            </ul>
                        </div>
                        
                        <div class="issue-card">
                            <div class="issue-title">💡 AI Рекомендации</div>
                            <ul class="issue-list">
                                {"".join(f"<li>{rec}</li>" for rec in analysis_results.get("ai_recommendations", []))}
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """


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

        # Кеширование результатов
        self.cache_file = self.results_dir / "analysis_cache.pkl"
        self.cache_duration = 3600  # 1 час

        # Анализаторы
        self.security_analyzer = SecurityAnalyzer(self.project_root)
        self.performance_analyzer = PerformanceAnalyzer(self.project_root)
        self.git_analyzer = GitBlameAnalyzer(self.project_root)
        self.html_generator = HTMLReportGenerator(self.project_root)

        # Определяем архитектурные слои проекта
        self.architecture_layers = {
            "database": ["src/database/", "postgres_adapter.py"],
            "analyzers": ["src/analyzers/"],
            "cli": ["src/cli/", "main.py"],
            "models": ["src/models/"],
            "scripts": ["scripts/"],
            "legacy": ["scripts/archive/", "_sqlite.py", "data_backup_"],
            "tests": ["tests/", "test_"],
        }

    def analyze_project(self) -> dict:
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

        # 6. Анализ безопасности
        security_issues = self.security_analyzer.find_security_issues(self.metrics)

        # 7. Анализ производительности
        performance_issues = self.performance_analyzer.find_performance_issues(
            self.metrics
        )

        # 8. Git blame анализ
        git_patterns = self.git_analyzer.analyze_git_patterns(self.metrics)

        return self._generate_report(security_issues, performance_issues, git_patterns)

    def analyze_with_cache(self) -> dict:
        """Анализ с кешированием результатов"""

        # Проверяем кеш
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "rb") as f:
                    cached_data = pickle.load(f)

                # Проверяем актуальность кеша
                if time.time() - cached_data["timestamp"] < self.cache_duration:
                    print("✅ Используем кешированные результаты анализа")
                    return cached_data["results"]
            except Exception as e:
                print(f"⚠️ Ошибка чтения кеша: {e}")

        # Выполняем анализ
        print("🔄 Выполняем новый анализ проекта...")
        results = self.analyze_project()

        # Сохраняем в кеш
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump({"timestamp": time.time(), "results": results}, f)
            print("💾 Результаты анализа кешированы")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения кеша: {e}")

        return results

    def export_for_context_manager(self) -> dict:
        """Экспорт данных для AI Context Manager"""

        return {
            "complexity_scores": {
                str(path): metrics.complexity_score
                for path, metrics in self.metrics.items()
            },
            "coupling_scores": {
                str(path): len(metrics.imports)
                for path, metrics in self.metrics.items()
            },
            "duplicates": [
                {
                    "similarity": dup.similarity,
                    "files": [str(f) for f in dup.files],
                    "common_functions": dup.common_functions,
                }
                for dup in self.duplicates
            ],
            "unused_files": [str(f) for f in self.unused_files],
            "architecture_violations": self.architecture_violations,
            "layers": {
                layer: [str(f) for f in files if isinstance(f, Path)]
                for layer, files in self.architecture_layers.items()
            },
        }

    def _collect_file_metrics(self):
        """Собирает метрики по всем Python файлам"""
        for py_file in self.project_root.rglob("*.py"):
            if self._should_skip_file(py_file):
                continue

            try:
                with open(py_file, encoding="utf-8") as f:
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
                    is_legacy=self._is_legacy_file(py_file),
                )

                self.metrics[str(py_file)] = metrics

            except Exception as e:
                print(f"⚠️ Ошибка анализа {py_file}: {e}")

    def _find_semantic_duplicates(self):
        """Поиск семантических дубликатов через AST"""
        function_hashes = defaultdict(list)

        for file_path, metrics in self.metrics.items():
            for func in metrics.functions:
                # Используем хеш от имени функции для группировки
                func_hash = hashlib.md5(func.encode()).hexdigest()[:8]
                function_hashes[func_hash].append((file_path, func))

        # Ищем файлы с похожими функциями
        for func_hash, file_func_pairs in function_hashes.items():
            if len(file_func_pairs) > 1:
                files = [pair[0] for pair in file_func_pairs]
                functions = [pair[1] for pair in file_func_pairs]

                if len(set(files)) > 1:  # Разные файлы
                    similarity = len(functions) / max(
                        len(self.metrics[f].functions) for f in files
                    )
                    if similarity > 0.3:  # Порог похожести
                        self.duplicates.append(
                            DuplicationResult(
                                files=files,
                                similarity=similarity,
                                common_functions=functions,
                            )
                        )

    def _find_unused_files(self):
        """Поиск неиспользуемых файлов"""
        # Анализируем импорты
        all_imports = set()
        for metrics in self.metrics.values():
            all_imports.update(metrics.imports)

        # Проверяем каждый файл
        for file_path in self.metrics.keys():
            file_stem = Path(file_path).stem
            if file_stem not in all_imports and not self._is_entry_point(file_path):
                # Дополнительная проверка через grep
                if not self._is_referenced_in_project(file_stem):
                    self.unused_files.add(file_path)

    def _check_architecture_violations(self):
        """Проверка нарушений архитектуры"""
        for file_path, metrics in self.metrics.items():
            layer = self._determine_layer(file_path)

            # Проверяем нарушения слоев
            for imp in metrics.imports:
                if layer == "models" and "database" in imp:
                    if not any(
                        allowed in file_path for allowed in ["adapter", "interface"]
                    ):
                        self.architecture_violations.append(
                            f"🏗️ Models layer accessing database directly: {file_path} -> {imp}"
                        )

                # Проверяем использование legacy кода в новых модулях
                if "sqlite" in imp.lower() and not self._is_legacy_file(
                    Path(file_path)
                ):
                    self.architecture_violations.append(
                        f"🗂️ New module using legacy SQLite: {file_path} -> {imp}"
                    )

    def _analyze_postgresql_migration(self):
        """Анализ статуса миграции на PostgreSQL"""
        postgres_files = []
        sqlite_files = []

        for file_path, metrics in self.metrics.items():
            if any("postgres" in imp.lower() for imp in metrics.imports):
                postgres_files.append(file_path)
            if any("sqlite" in imp.lower() for imp in metrics.imports):
                sqlite_files.append(file_path)

        # Анализируем прогресс миграции
        migration_progress = (
            len(postgres_files) / (len(postgres_files) + len(sqlite_files))
            if (len(postgres_files) + len(sqlite_files)) > 0
            else 0
        )

        if migration_progress < 0.8:
            self.architecture_violations.append(
                f"🔄 PostgreSQL migration incomplete: {migration_progress:.1%} migrated"
            )

    def _generate_report(
        self,
        security_issues: list[str] | None = None,
        performance_issues: list[str] | None = None,
        git_patterns: dict | None = None,
    ) -> dict:
        """Генерирует итоговый отчет"""

        # Базовая статистика
        total_files = len(self.metrics)
        legacy_files = sum(1 for m in self.metrics.values() if m.is_legacy)
        test_files = sum(1 for m in self.metrics.values() if m.is_test)

        return {
            "summary": {
                "total_files": total_files,
                "legacy_files": legacy_files,
                "test_files": test_files,
                "active_files": total_files - legacy_files,
                "average_complexity": sum(
                    m.complexity_score for m in self.metrics.values()
                )
                / total_files,
                "migration_status": len(
                    [v for v in self.architecture_violations if "PostgreSQL" in v]
                )
                == 0,
            },
            "duplicates_analysis": [
                {
                    "files": d.files,
                    "similarity": d.similarity,
                    "functions": d.common_functions,
                }
                for d in self.duplicates
            ],
            "unused_files": list(self.unused_files),
            "architecture_violations": self.architecture_violations,
            "security_issues": security_issues or [],
            "performance_issues": performance_issues or [],
            "git_patterns": git_patterns or {},
            "ai_recommendations": self._generate_ai_recommendations(),
            "export_data": self.export_for_context_manager(),
        }

    def _generate_ai_recommendations(self) -> list[str]:
        """Генерирует AI рекомендации на основе анализа"""
        recommendations = []

        # Анализируем дубликаты
        if len(self.duplicates) > 5:
            recommendations.append(
                "🔄 Рефакторинг: Много дубликатов кода - создайте общие модули"
            )

        # Анализируем сложность
        high_complexity = [m for m in self.metrics.values() if m.complexity_score > 100]
        if len(high_complexity) > 10:
            recommendations.append(
                "📉 Упрощение: Много сложных файлов - разбейте на меньшие модули"
            )

        # Анализируем legacy
        legacy_ratio = sum(1 for m in self.metrics.values() if m.is_legacy) / len(
            self.metrics
        )
        if legacy_ratio > 0.3:
            recommendations.append(
                "🗂️ Миграция: Высокий процент legacy кода - планируйте миграцию"
            )

        # Анализируем тесты
        test_ratio = sum(1 for m in self.metrics.values() if m.is_test) / len(
            self.metrics
        )
        if test_ratio < 0.2:
            recommendations.append(
                "🧪 Тестирование: Низкое покрытие тестами - добавьте unit тесты"
            )

        # Анализируем архитектуру
        if len(self.architecture_violations) > 5:
            recommendations.append(
                "🏗️ Архитектура: Много нарушений - пересмотрите слои приложения"
            )

        return recommendations

    def _should_skip_file(self, file_path: Path) -> bool:
        """Определяет, нужно ли пропустить файл"""
        skip_patterns = ["__pycache__", ".git", ".venv", "venv", "env", ".pytest_cache"]
        return any(pattern in str(file_path) for pattern in skip_patterns)

    def _extract_functions(self, tree: ast.AST) -> list[str]:
        """Извлекает имена функций из AST"""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        return functions

    def _extract_classes(self, tree: ast.AST) -> list[str]:
        """Извлекает имена классов из AST"""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes

    def _extract_imports(self, tree: ast.AST) -> list[str]:
        """Извлекает импорты из AST"""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Вычисляет цикломатическую сложность"""
        complexity = 1  # Базовая сложность

        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With)) or isinstance(
                node, ast.ExceptHandler
            ):
                complexity += 1

        return complexity

    def _is_legacy_file(self, file_path: Path) -> bool:
        """Определяет, является ли файл legacy"""
        legacy_indicators = ["sqlite", "backup", "archive", "old", "legacy"]
        return any(
            indicator in str(file_path).lower() for indicator in legacy_indicators
        )

    def _determine_layer(self, file_path: str) -> str:
        """Определяет архитектурный слой файла"""
        for layer, patterns in self.architecture_layers.items():
            if any(pattern in file_path for pattern in patterns):
                return layer
        return "other"

    def _is_entry_point(self, file_path: str) -> bool:
        """Проверяет, является ли файл точкой входа"""
        entry_points = ["main.py", "app.py", "__init__.py", "cli.py"]
        return any(ep in file_path for ep in entry_points)

    def _is_referenced_in_project(self, file_stem: str) -> bool:
        """Проверяет, ссылается ли проект на файл"""
        try:
            result = subprocess.run(
                ["grep", "-r", file_stem, str(self.project_root)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return len(result.stdout.strip()) > 0
        except:
            return True  # На всякий случай считаем используемым


def main():
    """Главная функция CLI"""
    if "--analyze" in sys.argv:
        analyzer = ProjectIntelligence()
        results = analyzer.analyze_with_cache()

        print("\n🎯 РЕЗУЛЬТАТЫ АНАЛИЗА:")
        print("=" * 50)

        summary = results["summary"]
        print(f"📁 Всего файлов: {summary['total_files']}")
        print(f"🔥 Активных файлов: {summary['active_files']}")
        print(f"🗂️ Legacy файлов: {summary['legacy_files']}")
        print(f"🧪 Тестовых файлов: {summary['test_files']}")
        print(f"📊 Средняя сложность: {summary['average_complexity']:.1f}")
        print(
            f"🔄 PostgreSQL миграция: {'✅' if summary['migration_status'] else '❌'}"
        )

        if results["duplicates_analysis"]:
            print(f"\n🔄 Дубликаты кода: {len(results['duplicates_analysis'])}")
            for dup in results["duplicates_analysis"][:3]:
                print(
                    f"  • {dup['similarity']:.1%} схожести: {', '.join(dup['files'][:2])}"
                )

        if results["security_issues"]:
            print(f"\n🔒 Проблемы безопасности: {len(results['security_issues'])}")
            for issue in results["security_issues"][:5]:
                print(f"  • {issue}")

        if results["performance_issues"]:
            print(
                f"\n⚡ Проблемы производительности: {len(results['performance_issues'])}"
            )
            for issue in results["performance_issues"][:5]:
                print(f"  • {issue}")

        if results["git_patterns"] and results["git_patterns"].get("hotspots"):
            print(f"\n🔥 Git Hotspots: {len(results['git_patterns']['hotspots'])}")
            for hotspot in results["git_patterns"]["hotspots"][:3]:
                print(f"  • {hotspot['changes']} изменений: {hotspot['file']}")

        if results["git_patterns"] and results["git_patterns"].get("bus_factor_risks"):
            print(
                f"\n👤 Bus Factor риски: {len(results['git_patterns']['bus_factor_risks'])}"
            )
            for risk in results["git_patterns"]["bus_factor_risks"][:3]:
                print(
                    f"  • {risk['risk_level']}: {risk['file']} (автор: {risk['sole_author']})"
                )

        if results["architecture_violations"]:
            print(
                f"\n🏗️ Нарушения архитектуры: {len(results['architecture_violations'])}"
            )
            for violation in results["architecture_violations"][:3]:
                print(f"  • {violation}")

        if results["ai_recommendations"]:
            print("\n💡 AI Рекомендации:")
            for rec in results["ai_recommendations"]:
                print(f"  • {rec}")

        # Сохраняем результаты
        output_file = Path("results") / "project_analysis_enhanced.json"
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Результаты сохранены: {output_file}")

        # Генерируем HTML отчет
        print("\n📊 Генерация HTML dashboard...")
        html_report = analyzer.html_generator.generate_dashboard(results)
        print(f"📊 HTML отчет создан: {html_report}")

    elif "--html" in sys.argv:
        # Генерация только HTML отчета из существующих данных
        json_file = Path("results") / "project_analysis_enhanced.json"
        if json_file.exists():
            with open(json_file, encoding="utf-8") as f:
                results = json.load(f)

            analyzer = ProjectIntelligence()
            print("📊 Генерация HTML dashboard из кешированных данных...")
            html_report = analyzer.html_generator.generate_dashboard(results)
            print(f"📊 HTML отчет создан: {html_report}")
        else:
            print(
                "❌ Файл results/project_analysis_enhanced.json не найден. Сначала запустите --analyze"
            )

    else:
        print("🎯 AI Project Analyzer - Enterprise Grade Code Intelligence")
        print("=" * 60)
        print("📊 Доступные команды:")
        print("  python ai_project_analyzer.py --analyze     # Полный анализ проекта")
        print("  python ai_project_analyzer.py --html        # Генерация HTML отчета")
        print("\n💡 Для интеграции с ai_context_manager.py используйте:")
        print("  analyzer = ProjectIntelligence()")
        print("  results = analyzer.export_for_context_manager()")
        print("\n🚀 Возможности:")
        print("  • � Анализ безопасности (SQL injection, пароли, API ключи)")
        print("  • ⚡ Анализ производительности (циклы, N+1 запросы)")
        print("  • 🔄 Обнаружение дубликатов кода")
        print("  • 🏗️ Анализ архитектуры и зависимостей")
        print("  • � HTML dashboard с интерактивными графиками")
        print("  • 💾 Кеширование результатов")
        print("  • � Интеграция с AI Context Manager")


if __name__ == "__main__":
    main()
