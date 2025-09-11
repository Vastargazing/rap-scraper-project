#!/usr/bin/env python3
"""
🔐 Dependency Security Manager — автоматическое управление зависимостями и безопасностью

НАЗНАЧЕНИЕ:
- Автоматическая проверка уязвимостей в dependencies
- Мониторинг устаревших пакетов с предложениями обновлений
- Анализ конфликтов версий и совместимости
- Генерация отчетов безопасности для production deployments

ИСПОЛЬЗОВАНИЕ:
python scripts/tools/dependency_manager.py --audit
python scripts/tools/dependency_manager.py --update-safe
python scripts/tools/dependency_manager.py --security-report

РЕЗУЛЬТАТ:
- Автоматическое обнаружение уязвимостей
- Безопасные обновления с rollback capability
- Отчеты для compliance и security audits
"""

import subprocess
import shutil
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
from datetime import datetime, timedelta

@dataclass
class Vulnerability:
    package: str
    installed_version: str
    affected_versions: str
    vulnerability_id: str
    severity: str
    description: str
    fixed_version: Optional[str]

@dataclass
class PackageStatus:
    name: str
    installed_version: str
    latest_version: str
    is_outdated: bool
    days_behind: int
    breaking_changes: bool
    update_priority: str  # "critical", "high", "medium", "low"

class DependencySecurityManager:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.requirements_file = self.project_root / "requirements.txt"
        self.security_report = self.results_dir / "security_audit.json"
        self.update_log = self.results_dir / "dependency_updates.json"
        
        # Критически важные пакеты для проекта
        self.critical_packages = {
            "asyncpg", "psycopg2", "fastapi", "uvicorn", 
            "aiohttp", "requests", "pydantic", "sqlalchemy"
        }
        
        # Пакеты, требующие осторожного обновления
        self.sensitive_packages = {
            "tensorflow", "torch", "transformers", "numpy", "pandas"
        }
    
    def get_installed_packages(self) -> Dict[str, str]:
        """Получает список установленных пакетов"""
        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"], 
                capture_output=True, text=True, check=True
            )
            packages = json.loads(result.stdout)
            return {pkg["name"].lower(): pkg["version"] for pkg in packages}
        except Exception as e:
            print(f"Ошибка получения списка пакетов: {e}")
            return {}
    
    def check_vulnerabilities(self) -> List[Vulnerability]:
        """Проверяет уязвимости в установленных пакетах"""
        vulnerabilities = []
        
        try:
            # Проверяем наличие утилиты `safety`
            if shutil.which("safety") is None:
                print("safety не найден в PATH – пропуск проверки уязвимостей. Установите его через `pip install safety` для полного аудита.")
                return []

            # Используем safety для проверки уязвимостей
            result = subprocess.run(
                ["safety", "check", "--json"], 
                capture_output=True, text=True
            )

            if result.returncode == 0:
                return []  # Уязвимости не найдены

            # Парсим JSON вывод safety
            safety_data = json.loads(result.stdout)
            
            for vuln_data in safety_data:
                vulnerability = Vulnerability(
                    package=vuln_data.get("package", "unknown"),
                    installed_version=vuln_data.get("installed_version", ""),
                    affected_versions=vuln_data.get("affected_versions", ""),
                    vulnerability_id=vuln_data.get("vulnerability_id", ""),
                    severity=self._determine_severity(vuln_data),
                    description=vuln_data.get("advisory", ""),
                    fixed_version=self._extract_fixed_version(vuln_data)
                )
                vulnerabilities.append(vulnerability)
                
        except subprocess.CalledProcessError:
            print("Safety вернул ошибку при выполнении")
        except FileNotFoundError:
            # На Windows это был исходный WinError 2 — более дружелюбное сообщение
            print("safety не найден в системе (FileNotFoundError). Установите его: pip install safety")
        except json.JSONDecodeError:
            print("Не удалось парсить вывод safety")
        except Exception as e:
            print(f"Ошибка проверки уязвимостей: {e}")
        
        return vulnerabilities
    
    def check_outdated_packages(self) -> List[PackageStatus]:
        """Проверяет устаревшие пакеты"""
        outdated_packages = []
        
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True, check=True
            )
            
            outdated_data = json.loads(result.stdout)
            
            for pkg_data in outdated_data:
                package_status = PackageStatus(
                    name=pkg_data["name"],
                    installed_version=pkg_data["version"],
                    latest_version=pkg_data["latest_version"],
                    is_outdated=True,
                    days_behind=self._calculate_days_behind(pkg_data),
                    breaking_changes=self._check_breaking_changes(pkg_data),
                    update_priority=self._determine_update_priority(pkg_data)
                )
                outdated_packages.append(package_status)
                
        except Exception as e:
            print(f"Ошибка проверки устаревших пакетов: {e}")
        
        return outdated_packages
    
    def generate_security_report(self) -> Dict:
        """Генерирует полный отчет безопасности"""
        print("Генерация отчета безопасности...")
        
        # Проверяем уязвимости
        vulnerabilities = self.check_vulnerabilities()
        
        # Проверяем устаревшие пакеты
        outdated = self.check_outdated_packages()
        
        # Анализируем конфликты
        conflicts = self._check_dependency_conflicts()
        
        # Проверяем requirements.txt
        requirements_issues = self._audit_requirements_file()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_vulnerabilities": len(vulnerabilities),
                "critical_vulnerabilities": len([v for v in vulnerabilities if v.severity == "critical"]),
                "outdated_packages": len(outdated),
                "critical_updates": len([p for p in outdated if p.update_priority == "critical"]),
                "dependency_conflicts": len(conflicts),
                "security_score": self._calculate_security_score(vulnerabilities, outdated)
            },
            "vulnerabilities": [v.__dict__ for v in vulnerabilities],
            "outdated_packages": [p.__dict__ for p in outdated],
            "conflicts": conflicts,
            "requirements_issues": requirements_issues,
            "recommendations": self._generate_security_recommendations(vulnerabilities, outdated)
        }
        
        # Сохраняем отчет (обязательно в UTF-8, чтобы не падать на эмодзи)
        with open(self.security_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def perform_safe_updates(self, dry_run: bool = True) -> Dict:
        """Выполняет безопасные обновления пакетов"""
        print(f"Запуск {'симуляции' if dry_run else 'реальных'} обновлений...")
        
        outdated = self.check_outdated_packages()
        safe_updates = []
        risky_updates = []
        
        for package in outdated:
            if self._is_safe_to_update(package):
                safe_updates.append(package)
            else:
                risky_updates.append(package)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "safe_updates": [],
            "risky_updates": [p.__dict__ for p in risky_updates],
            "errors": []
        }
        
        if not dry_run:
            # Создаем backup requirements.txt
            self._backup_requirements()
            
            # Выполняем безопасные обновления
            for package in safe_updates:
                try:
                    result = subprocess.run([
                        "pip", "install", "--upgrade", package.name
                    ], capture_output=True, text=True, check=True)
                    
                    results["safe_updates"].append({
                        "package": package.name,
                        "old_version": package.installed_version,
                        "new_version": package.latest_version,
                        "status": "success"
                    })
                    
                except subprocess.CalledProcessError as e:
                    results["errors"].append({
                        "package": package.name,
                        "error": str(e)
                    })
        else:
            # В dry_run режиме просто показываем что будет обновлено
            results["safe_updates"] = [p.__dict__ for p in safe_updates]
        
        # Логируем результаты
        self._log_update_results(results)
        
        return results
    
    def _determine_severity(self, vuln_data: Dict) -> str:
        """Определяет серьезность уязвимости"""
        advisory = vuln_data.get("advisory", "").lower()
        
        if any(word in advisory for word in ["critical", "remote code execution", "rce"]):
            return "critical"
        elif any(word in advisory for word in ["high", "sql injection", "xss"]):
            return "high"
        elif any(word in advisory for word in ["medium", "denial of service"]):
            return "medium"
        else:
            return "low"
    
    def _extract_fixed_version(self, vuln_data: Dict) -> Optional[str]:
        """Извлекает версию с исправлением"""
        advisory = vuln_data.get("advisory", "")
        
        # Ищем паттерны типа "fixed in 1.2.3" или "upgrade to 1.2.3"
        patterns = [
            r"fixed in (\d+\.\d+\.\d+)",
            r"upgrade to (\d+\.\d+\.\d+)",
            r"version (\d+\.\d+\.\d+) and above"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, advisory, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _calculate_days_behind(self, pkg_data: Dict) -> int:
        """Вычисляет количество дней отставания от последней версии"""
        # Упрощенная логика - в реальности нужен API PyPI
        return 30  # заглушка
    
    def _check_breaking_changes(self, pkg_data: Dict) -> bool:
        """Проверяет наличие breaking changes"""
        installed = pkg_data["version"]
        latest = pkg_data["latest_version"]
        
        # Проверяем major version changes
        installed_major = int(installed.split('.')[0]) if '.' in installed else 0
        latest_major = int(latest.split('.')[0]) if '.' in latest else 0
        
        return latest_major > installed_major
    
    def _determine_update_priority(self, pkg_data: Dict) -> str:
        """Определяет приоритет обновления"""
        pkg_name = pkg_data["name"].lower()
        
        if pkg_name in self.critical_packages:
            return "critical" if self._check_breaking_changes(pkg_data) else "high"
        elif pkg_name in self.sensitive_packages:
            return "medium"
        else:
            return "low"
    
    def _check_dependency_conflicts(self) -> List[Dict]:
        """Проверяет конфликты зависимостей"""
        conflicts = []
        
        try:
            result = subprocess.run(
                ["pip", "check"], capture_output=True, text=True
            )
            
            if result.returncode != 0:
                # Парсим вывод pip check
                for line in result.stdout.split('\n'):
                    if "has requirement" in line:
                        conflicts.append({
                            "description": line.strip(),
                            "severity": "medium"
                        })
                        
        except Exception as e:
            print(f"Ошибка проверки конфликтов: {e}")
        
        return conflicts
    
    def _audit_requirements_file(self) -> List[Dict]:
        """Аудит файла requirements.txt"""
        issues = []
        
        if not self.requirements_file.exists():
            issues.append({
                "type": "missing_file",
                "message": "Отсутствует файл requirements.txt"
            })
            return issues
        
        with open(self.requirements_file, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Проверяем закрепленные версии
            if '==' not in line and '~=' not in line and '>=' not in line:
                issues.append({
                    "type": "unpinned_version",
                    "line": i,
                    "package": line,
                    "message": f"Пакет {line} не имеет зафиксированной версии"
                })
        
        return issues
    
    def _calculate_security_score(self, vulnerabilities: List, outdated: List) -> float:
        """Вычисляет общий балл безопасности (0-100)"""
        score = 100
        
        # Снижаем балл за уязвимости
        for vuln in vulnerabilities:
            if vuln.severity == "critical":
                score -= 25
            elif vuln.severity == "high":
                score -= 15
            elif vuln.severity == "medium":
                score -= 5
            else:
                score -= 2
        
        # Снижаем балл за устаревшие критические пакеты
        critical_outdated = [p for p in outdated if p.update_priority == "critical"]
        score -= len(critical_outdated) * 5
        
        return max(0, score)
    
    def _generate_security_recommendations(self, vulnerabilities: List, outdated: List) -> List[str]:
        """Генерирует рекомендации по безопасности"""
        recommendations = []
        
        if vulnerabilities:
            critical_vulns = [v for v in vulnerabilities if v.severity == "critical"]
            if critical_vulns:
                recommendations.append(
                    f"🚨 НЕМЕДЛЕННО обновите {len(critical_vulns)} пакетов с критическими уязвимостями"
                )
        
        critical_updates = [p for p in outdated if p.update_priority == "critical"]
        if critical_updates:
            recommendations.append(
                f"⚠️ Обновите {len(critical_updates)} критически важных пакетов"
            )
        
        if len(outdated) > 10:
            recommendations.append(
                "📅 Запланируйте регулярные обновления зависимостей (еженедельно)"
            )
        
        return recommendations
    
    def _is_safe_to_update(self, package: PackageStatus) -> bool:
        """Определяет безопасно ли обновлять пакет"""
        # Не обновляем если есть breaking changes в критических пакетах
        if package.breaking_changes and package.name.lower() in self.critical_packages:
            return False
        
        # Не обновляем чувствительные пакеты автоматически
        if package.name.lower() in self.sensitive_packages:
            return False
        
        return True
    
    def _backup_requirements(self):
        """Создает backup файла requirements.txt"""
        if self.requirements_file.exists():
            backup_file = self.results_dir / f"requirements_backup_{int(datetime.now().timestamp())}.txt"
            # Читать/писать в UTF-8
            backup_file.write_text(self.requirements_file.read_text(encoding='utf-8'), encoding='utf-8')
    
    def _log_update_results(self, results: Dict):
        """Логирует результаты обновлений"""
        if self.update_log.exists():
            with open(self.update_log, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = []
        
        data.append(results)
        
        # Оставляем последние 50 записей
        if len(data) > 50:
            data = data[-50:]
        
        with open(self.update_log, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    import sys
    
    manager = DependencySecurityManager()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--audit":
        # Полный аудит безопасности
        report = manager.generate_security_report()
        
        print("\n🔐 ОТЧЕТ БЕЗОПАСНОСТИ:")
        print(f"Безопасность: {report['summary']['security_score']}/100")
        print(f"Уязвимости: {report['summary']['total_vulnerabilities']} (критических: {report['summary']['critical_vulnerabilities']})")
        print(f"Устаревших пакетов: {report['summary']['outdated_packages']}")
        
        if report['recommendations']:
            print("\n💡 РЕКОМЕНДАЦИИ:")
            for rec in report['recommendations']:
                print(f"  {rec}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--update-safe":
        # Безопасные обновления
        results = manager.perform_safe_updates(dry_run=False)
        
        print(f"\n✅ Обновлено пакетов: {len(results['safe_updates'])}")
        print(f"⚠️ Пропущено рискованных: {len(results['risky_updates'])}")
        print(f"❌ Ошибок: {len(results['errors'])}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "--security-report":
        # Краткий отчет безопасности
        vulnerabilities = manager.check_vulnerabilities()
        outdated = manager.check_outdated_packages()
        
        if vulnerabilities:
            print(f"\n🚨 Найдено {len(vulnerabilities)} уязвимостей:")
            for vuln in vulnerabilities[:5]:
                print(f"  {vuln.package} {vuln.installed_version}: {vuln.severity}")
        
        if outdated:
            critical = [p for p in outdated if p.update_priority == "critical"]
            if critical:
                print(f"\n⚠️ Критические обновления ({len(critical)}):")
                for pkg in critical[:5]:
                    print(f"  {pkg.name}: {pkg.installed_version} → {pkg.latest_version}")
    
    else:
        print("🔐 Dependency Security Manager")
        print("Использование:")
        print("  --audit          Полный аудит безопасности")
        print("  --update-safe    Безопасные обновления")
        print("  --security-report Краткий отчет")

if __name__ == "__main__":
    main()
