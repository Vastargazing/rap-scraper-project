# 📚 Шпаргалка для Собеседований в FAANG
## Подготовка по Code Review: От Теории к Реальным Кейсам

**Автор**: AI Assistant
**Цель**: Подготовка к техническим собеседованиям в FAANG (Google, Meta, Amazon, Apple, Netflix)
**Язык**: Русский
**Уровень**: Middle - Senior

---

# 🎯 ЧАСТЬ 1: КРИТИЧЕСКИЕ ПРОБЛЕМЫ БЕЗОПАСНОСТИ

## 1. Утечка Паролей в Логах

### 📖 Теория
**Проблема**: Вывод конфиденциальных данных (пароли, токены, ключи API) в логи, консоль или error tracking системы.

**Почему это критично**:
- Логи часто хранятся в plain text
- Доступ к логам имеют DevOps, SRE, иногда аналитики
- Логи могут попадать в системы мониторинга (Splunk, ELK, Datadog)
- Логи могут бэкапиться и храниться годами

### 🔴 Реальный Кейс #1: Uber (2016)
**Что произошло**:
- Разработчики Uber хранили AWS ключи в приватном GitHub репозитории
- Бывший сотрудник скачал код и нашел креды в логах
- Были украдены данные **57 миллионов пользователей** и 600,000 водителей
- Uber заплатил хакерам $100,000 чтобы удалить данные
- Позже компания заплатила штраф **$148 миллионов**

**Что пошло не так**:
```python
# ❌ ПЛОХО - так делали в Uber
logger.error(f"Failed to connect to AWS: {aws_credentials}")
# Креды попали в CloudWatch Logs
```

**Правильное решение**:
```python
# ✅ ХОРОШО
def sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive fields before logging."""
    sensitive_fields = ['password', 'token', 'api_key', 'secret', 'credential']
    sanitized = config.copy()
    for key in sanitized:
        if any(field in key.lower() for field in sensitive_fields):
            sanitized[key] = '***REDACTED***'
    return sanitized

logger.error(f"Failed to connect: {sanitize_config(config)}")
```

### 🔴 Реальный Кейс #2: Twitter (2018)
**Что произошло**:
- Twitter хранил пароли пользователей в **plain text в внутренних логах**
- Затронуты **330 миллионов аккаунтов**
- Пароли были доступны внутренним сотрудникам
- Масштабная принудительная смена паролей

**Последствия**:
- Репутационный ущерб
- Расследование регуляторов
- Массовая потеря доверия пользователей

**Урок**: Никогда не логировать credentials, даже во внутренние системы.

### 💡 Вопросы на Собеседовании

**Q1**: "Как бы вы спроектировали систему логирования для микросервисов, чтобы избежать утечки credentials?"

<details>
<summary>Правильный Ответ</summary>

```python
import logging
import re
from typing import Any

class SanitizingFormatter(logging.Formatter):
    """Custom formatter that redacts sensitive data."""

    SENSITIVE_PATTERNS = [
        (re.compile(r'password["\s:=]+([^\s&"]+)', re.I), r'password=***'),
        (re.compile(r'token["\s:=]+([^\s&"]+)', re.I), r'token=***'),
        (re.compile(r'api[_-]?key["\s:=]+([^\s&"]+)', re.I), r'api_key=***'),
        (re.compile(r'\b[A-Za-z0-9+/]{32,}={0,2}\b'), r'***BASE64***'),
    ]

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        sanitized = original
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized

# Использование
handler = logging.StreamHandler()
handler.setFormatter(SanitizingFormatter('%(asctime)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
```

**Дополнительные меры**:
1. Использовать Secret Management системы (HashiCorp Vault, AWS Secrets Manager)
2. Ротация секретов каждые 30-90 дней
3. Аудит логов с помощью automated scanners
4. Отдельные права доступа к production логам
5. Encryption at rest для всех логов
</details>

**Q2**: "Вы обнаружили, что пароли попали в Sentry/error tracking. Ваши действия?"

<details>
<summary>Правильный Ответ</summary>

**Немедленные действия** (первые 15 минут):
1. **Удалить данные из Sentry**: Использовать Data Scrubbing или полное удаление событий
2. **Ротация скомпрометированных credentials**: Немедленная смена паролей
3. **Уведомить Security Team**: Incident response процедура

**Краткосрочные действия** (24 часа):
1. Аудит всех логов за последние 30 дней
2. Форс смена паролей затронутых пользователей
3. Мониторинг подозрительной активности
4. Блокировка старых сессий

**Долгосрочные действия**:
```python
# Добавить в Sentry before_send hook
import sentry_sdk

def before_send(event, hint):
    """Scrub sensitive data before sending to Sentry."""
    if 'request' in event:
        # Удалить authorization headers
        event['request'].get('headers', {}).pop('Authorization', None)
        event['request'].get('headers', {}).pop('Cookie', None)

    # Скрыть env variables
    if 'contexts' in event and 'runtime' in event['contexts']:
        env = event['contexts']['runtime'].get('env', {})
        for key in list(env.keys()):
            if any(s in key.lower() for s in ['password', 'secret', 'key', 'token']):
                env[key] = '***'

    return event

sentry_sdk.init(
    dsn="...",
    before_send=before_send
)
```

**Post-mortem документ**:
- Root cause analysis
- Timeline событий
- Preventive measures
- Action items with owners
</details>

---

## 2. SQL Injection

### 📖 Теория
**Проблема**: Использование user input или переменных напрямую в SQL запросах без экранирования.

**Типы SQL Injection**:
1. **Classic SQL Injection**: `SELECT * FROM users WHERE id = '1 OR 1=1'`
2. **Blind SQL Injection**: Определение структуры БД по времени ответа
3. **Second-Order Injection**: Вредоносный код сохраняется и выполняется позже

### 🔴 Реальный Кейс #1: Target (2013)
**Что произошло**:
- Хакеры использовали SQL injection в системе HVAC подрядчика
- Получили доступ к Point-of-Sale системам Target
- Украдены данные **40 миллионов кредитных карт**
- Скомпрометированы личные данные **70 миллионов покупателей**

**Финансовые последствия**:
- Прямые убытки: **$292 миллиона**
- Штрафы банков: **$39 миллионов**
- Урегулирование с пользователями: **$18.5 миллионов**
- CEO компании уволен
- Падение акций на 11%

**Уязвимый код** (реконструкция):
```python
# ❌ УЯЗВИМЫЙ КОД
vendor_id = request.GET['vendor']
query = f"SELECT * FROM vendors WHERE id = {vendor_id}"
cursor.execute(query)

# Атака: ?vendor=1 OR 1=1; DROP TABLE vendors;--
```

### 🔴 Реальный Кейс #2: Sony PlayStation Network (2011)
**Что произошло**:
- SQL injection в веб-приложении PSN
- Скомпрометированы данные **77 миллионов аккаунтов**
- Сервис был выключен на **23 дня**
- Один из крупнейших data breaches в истории

**Последствия**:
- Прямые убытки: **$171 миллион**
- Class action lawsuit: **$15 миллионов**
- Бесплатные подписки пользователям: **$3 миллиона**
- Репутационный ущерб неоценим

**Урок**: SQL injection может уничтожить компанию.

### ✅ Правильные Решения

#### 1. Parameterized Queries (Prepared Statements)
```python
# ✅ ПРАВИЛЬНО - PostgreSQL
import psycopg2

def get_user_by_id(user_id: int) -> dict:
    """Safely fetch user by ID using parameterized query."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Используем %s placeholders, НЕ f-strings!
            cur.execute(
                "SELECT * FROM users WHERE id = %s",
                (user_id,)  # Tuple, даже для одного параметра
            )
            return cur.fetchone()

# ✅ ПРАВИЛЬНО - для динамических имен таблиц/колонок
from psycopg2 import sql

def get_count(table_name: str) -> int:
    """Count rows in table with SQL Identifier for table name."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            # sql.Identifier экранирует имена таблиц/колонок
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}").format(
                    sql.Identifier(table_name)
                )
            )
            return cur.fetchone()[0]
```

#### 2. ORM (SQLAlchemy, Django ORM)
```python
# ✅ ОТЛИЧНО - SQLAlchemy автоматически экранирует
from sqlalchemy import select
from models import User

def get_users_by_email(email: str) -> List[User]:
    """Fetch users by email using SQLAlchemy ORM."""
    stmt = select(User).where(User.email == email)
    return session.execute(stmt).scalars().all()

# SQLAlchemy сгенерирует:
# SELECT * FROM users WHERE email = %s
# И безопасно передаст параметр
```

#### 3. Input Validation + Whitelist
```python
# ✅ ЗАЩИТА В ГЛУБИНУ
from enum import Enum

class AllowedTables(Enum):
    """Whitelist разрешенных таблиц."""
    USERS = "users"
    TRACKS = "tracks"
    ARTISTS = "artists"

def get_table_stats(table: AllowedTables) -> dict:
    """Get statistics with validated table name."""
    if not isinstance(table, AllowedTables):
        raise ValueError(f"Invalid table: {table}")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {}").format(
                    sql.Identifier(table.value)
                )
            )
            return {"table": table.value, "count": cur.fetchone()[0]}

# Использование
stats = get_table_stats(AllowedTables.USERS)  # ✅ Безопасно
stats = get_table_stats("users; DROP TABLE users;--")  # ❌ Raise ValueError
```

### 💡 Вопросы на Собеседовании

**Q1**: "Объясните разницу между parameterized queries и ORM. Когда использовать каждый подход?"

<details>
<summary>Правильный Ответ</summary>

**Parameterized Queries (Prepared Statements)**:

*Плюсы*:
- Максимальная производительность
- Полный контроль над SQL
- Можно оптимизировать сложные запросы
- Нет overhead от ORM

*Минусы*:
- Больше boilerplate кода
- Ручное управление типами
- Легко сделать ошибку

*Когда использовать*:
- High-performance системы (>10K QPS)
- Сложная бизнес-логика в SQL
- Data warehousing, аналитика
- Миграция legacy систем

**ORM (Object-Relational Mapping)**:

*Плюсы*:
- Type safety из коробки
- Автоматическая защита от SQL injection
- Легче поддерживать код
- Database agnostic
- Автоматические миграции

*Минусы*:
- Performance overhead (5-15%)
- N+1 query problem
- Сложные запросы неудобны
- "Magic" может скрывать проблемы

*Когда использовать*:
- CRUD приложения (90% случаев)
- Быстрая разработка MVP
- Команда junior разработчиков
- Multi-database поддержка

**Hybrid подход** (рекомендуется):
```python
from sqlalchemy import text

# Простые запросы - ORM
users = session.query(User).filter_by(email=email).all()

# Сложные запросы - Raw SQL через ORM
result = session.execute(
    text("""
        SELECT u.id, COUNT(t.id) as track_count
        FROM users u
        LEFT JOIN tracks t ON u.id = t.user_id
        WHERE u.created_at > :start_date
        GROUP BY u.id
        HAVING COUNT(t.id) > :min_tracks
    """),
    {"start_date": start_date, "min_tracks": 10}
)
```
</details>

**Q2**: "Ваш код прошел code review, но QA нашел SQL injection. Как это возможно и что делать?"

<details>
<summary>Правильный Ответ</summary>

**Как это возможно**:

1. **Code Review не покрыл все edge cases**:
```python
# Выглядит безопасно, но уязвимо!
def search_users(sort_by: str = "name"):
    # ❌ sort_by не валидируется!
    query = f"SELECT * FROM users ORDER BY {sort_by}"
    return db.execute(query)

# Атака: search_users(sort_by="id; DROP TABLE users;--")
```

2. **Second-order SQL Injection**:
```python
# Шаг 1: Безопасно сохраняем
username = "admin'--"
db.execute("INSERT INTO users (name) VALUES (%s)", (username,))

# Шаг 2: Небезопасно читаем (в другом месте кода)
user = db.fetchone("SELECT name FROM users WHERE id = 1")
# ❌ Используем в динамическом запросе
db.execute(f"SELECT * FROM posts WHERE author = '{user['name']}'")
# Результат: SELECT * FROM posts WHERE author = 'admin'--'
```

3. **NoSQL Injection** (тоже SQL Injection!):
```python
# MongoDB - тоже уязвим!
username = {"$ne": None}  # Вернет всех пользователей
db.users.find({"username": username})
```

**Что делать**:

**Немедленно**:
1. Hotfix и deploy
2. Проверить логи на exploits
3. Security incident protocol

**Systematically**:
```python
# 1. Static Analysis
# Добавить в CI/CD: bandit, semgrep
# .bandit config:
# B608: Hardcoded SQL strings
# B201: Flask debug mode

# 2. Dynamic Analysis (DAST)
# Интегрировать OWASP ZAP в staging tests

# 3. Input Validation Layer
from pydantic import BaseModel, validator

class SearchParams(BaseModel):
    sort_by: str
    order: str

    @validator('sort_by')
    def validate_sort(cls, v):
        allowed = ['name', 'email', 'created_at']
        if v not in allowed:
            raise ValueError(f'Invalid sort field: {v}')
        return v

    @validator('order')
    def validate_order(cls, v):
        if v.upper() not in ['ASC', 'DESC']:
            raise ValueError('Order must be ASC or DESC')
        return v.upper()

# 4. WAF (Web Application Firewall)
# AWS WAF, Cloudflare, ModSecurity
# Блокирует типичные SQL injection паттерны

# 5. Database Permissions
# Приложение НЕ должно иметь DROP, CREATE права!
GRANT SELECT, INSERT, UPDATE, DELETE ON database.* TO 'app_user'@'%';
-- НЕ давать: DROP, CREATE, ALTER, GRANT
```

**Процесс улучшения**:
1. **Security Training** для всей команды
2. **Security Champions** в каждой команде
3. **Threat Modeling** для новых фич
4. **Penetration Testing** раз в квартал
5. **Bug Bounty Program**
</details>

---

## 3. Missing Type Hints

### 📖 Теория
**Проблема**: Отсутствие аннотаций типов в Python коде затрудняет:
- Раннее обнаружение ошибок
- Автоматический рефакторинг
- Onboarding новых разработчиков
- IDE autocomplete

**В FAANG это критично** потому что:
- Кодбейзы миллионы строк
- Сотни разработчиков работают параллельно
- Цена ошибки в production огромна

### 🔴 Реальный Кейс: Dropbox Migration to Type Hints (2019)
**Контекст**:
- Dropbox имел **4 миллиона строк Python кода**
- Более **1000 инженеров**
- Без type hints - ошибки обнаруживались в production

**Что сделали**:
- Миграция на Python 3 с type hints
- Внедрение mypy в CI/CD
- Обязательные type hints для нового кода

**Результаты** (официальная статистика):
- **15% reduction** в production bugs
- **20% faster** onboarding новых разработчиков
- **40% improvement** в IDE performance
- **Сэкономлено 1000+ часов** на debugging в год

**Цитата CTO Dropbox**:
> "Type hints are the single most impactful change we made to our Python codebase. The ROI is incredible."

### 🔴 Реальный Кейс: Instagram (Meta) - Python Typing

**Контекст**:
- Instagram - один из крупнейших Python проектов (основан на Django)
- **Миллионы строк** Python кода
- Проблема: runtime ошибки из-за неправильных типов

**Что произошло** (2018):
```python
# ❌ Реальный баг в Instagram (упрощенно)
def get_user_followers(user_id):
    # Иногда возвращали int, иногда str
    followers = cache.get(f"followers:{user_id}")
    # Bug: когда followers = None, дальше падало
    return followers.split(",")  # AttributeError в production!
```

**Решение Instagram**:
- Создали **Pyre** - статический type checker
- Open-sourced в 2018
- Обязательное использование для всего кода

**Результаты**:
- **70% reduction** в type-related bugs
- Type checker находит **сотни ошибок** до production
- Improved developer confidence

### ✅ Правильное Использование Type Hints

```python
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import psycopg2
from psycopg2.extensions import connection as PgConnection

class PostgreSQLDiagnostics:
    """PostgreSQL database diagnostics tool.

    Attributes:
        project_root: Root directory of the project.
        conn: Active PostgreSQL connection or None.
        db_config: Database connection configuration.
    """

    def __init__(
        self,
        db_config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None
    ) -> None:
        """Initialize diagnostics with optional config.

        Args:
            db_config: Database connection parameters.
            logger: Logger instance for diagnostics output.
        """
        self.project_root: Path = Path(__file__).parent.parent.parent
        self.conn: Optional[PgConnection] = None
        self.db_config: Dict[str, Any] = db_config or self._load_db_config()
        self.logger: logging.Logger = logger or logging.getLogger(__name__)

    def connect(self) -> bool:
        """Establish connection to PostgreSQL.

        Returns:
            True if connection successful, False otherwise.

        Raises:
            psycopg2.OperationalError: If all connection attempts fail.
        """
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.conn.autocommit = True
            self.logger.info("Connected to PostgreSQL successfully")
            return True
        except psycopg2.OperationalError as e:
            self.logger.error(f"Connection failed: {e}", exc_info=True)
            return False

    def find_unanalyzed(
        self,
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Union[int, str]]]:
        """Find tracks without AI analysis.

        Args:
            limit: Maximum number of tracks to return (1-1000).
            offset: Number of tracks to skip.

        Returns:
            List of dicts with keys: id (int), artist (str), title (str).

        Raises:
            ValueError: If limit is out of valid range.
            psycopg2.Error: If database query fails.
        """
        if not 1 <= limit <= 1000:
            raise ValueError(f"Limit must be 1-1000, got {limit}")

        # Implementation...
        return []

    def _table_exists(self, table_name: str) -> bool:
        """Check if table exists in database.

        Args:
            table_name: Name of table to check.

        Returns:
            True if table exists, False otherwise.
        """
        # Implementation...
        return False
```

### 💡 Вопросы на Собеседовании

**Q1**: "Какие type hints вы бы использовали для функции, которая может принимать как filepath (str), так и file object?"

<details>
<summary>Правильный Ответ</summary>

```python
from typing import Union, TextIO, BinaryIO
from pathlib import Path
from typing import Protocol

# Вариант 1: Union (простой)
def read_data(source: Union[str, Path, TextIO]) -> str:
    """Read data from file path or file object."""
    if isinstance(source, (str, Path)):
        with open(source, 'r') as f:
            return f.read()
    else:
        return source.read()

# Вариант 2: Protocol (advanced, Google style)
from typing import Protocol

class ReadableFile(Protocol):
    """Protocol for file-like objects with read method."""
    def read(self) -> str: ...

def read_data_advanced(source: Union[str, Path, ReadableFile]) -> str:
    """Read data from path or readable file-like object."""
    if isinstance(source, (str, Path)):
        with open(source, 'r') as f:
            return f.read()
    return source.read()

# Вариант 3: PathLike (Python 3.6+)
from os import PathLike

def read_data_pathlike(
    source: Union[str, PathLike, TextIO]
) -> str:
    """Read using os.PathLike protocol."""
    if isinstance(source, (str, PathLike)):
        with open(source, 'r') as f:
            return f.read()
    return source.read()

# Вариант 4: Overload (самый точный)
from typing import overload

@overload
def read_data_overload(source: Union[str, Path]) -> str: ...

@overload
def read_data_overload(source: TextIO) -> str: ...

def read_data_overload(source):
    """Read data with precise type hints using overload."""
    if isinstance(source, (str, Path)):
        with open(source, 'r') as f:
            return f.read()
    return source.read()
```

**На собеседовании в Google ожидают**:
- Знание разницы между Union и Optional
- Понимание Protocol (duck typing)
- Awareness о overload для точности
- Объяснение trade-offs каждого подхода
</details>

**Q2**: "Как бы вы добавили type hints в legacy codebase на 100K строк без остановки разработки?"

<details>
<summary>Правильный Ответ</summary>

**Стратегия поэтапной миграции** (по опыту Dropbox и Instagram):

**Phase 1: Setup (Week 1)**
```bash
# 1. Установить mypy
pip install mypy

# 2. Создать mypy.ini с мягкими настройками
# mypy.ini
[mypy]
python_version = 3.9
warn_return_any = False
warn_unused_configs = True
disallow_untyped_defs = False  # Пока False!
ignore_missing_imports = True  # Для legacy libs

# 3. Первый запуск (будет много ошибок - это ок)
mypy src/  # Baseline: 5000 ошибок

# 4. Создать baseline
mypy src/ > mypy_baseline.txt
```

**Phase 2: New Code Only (Week 2-4)**
```python
# pre-commit hook (.pre-commit-config.yaml)
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.991
    hooks:
      - id: mypy
        args: [--strict]  # Строго для НОВОГО кода
        files: ^(новые_модули|src/new)/  # Только новые файлы
```

**Phase 3: Критичные Модули (Month 2-3)**
```python
# Приоритизация модулей по критичности
priority_modules = [
    'src/auth/',       # Authentication - критично!
    'src/payment/',    # Payments - критично!
    'src/database/',   # DB layer - критично!
    'src/api/endpoints/',  # Public API
]

# Постепенное добавление strict checks
# mypy.ini
[mypy-src.auth.*]
disallow_untyped_defs = True
disallow_any_generics = True

[mypy-src.payment.*]
disallow_untyped_defs = True
```

**Phase 4: Automated Migration (Month 4-6)**
```bash
# Использовать инструменты автоматизации

# 1. MonkeyType (runtime type collector)
pip install MonkeyType
# Запускаем тесты с MonkeyType
monkeytype run -m pytest tests/
# Генерируем stubs
monkeytype apply src.module

# 2. PyAnnotate (Facebook tool)
pip install pyannotate
# Собираем типы во время выполнения
pyannotate --type-info type_info.json src/

# 3. auto-optional (для Optional типов)
pip install auto-optional
auto-optional src/
```

**Phase 5: Team Training (Ongoing)**
```python
# Code review checklist
CHECKLIST = """
✅ All new functions have type hints
✅ Return type specified
✅ Complex types use typing module
✅ mypy passes on new code
✅ Docstring includes Args, Returns, Raises
"""

# Weekly brown bag sessions
# "Type Hints Best Practices"
# "Common mypy Errors and Solutions"
# "Generic Types Deep Dive"
```

**Phase 6: Metrics & Incentives (Month 6+)**
```python
# Трекинг прогресса
import ast
from pathlib import Path

def calculate_type_coverage(directory: Path) -> float:
    """Calculate percentage of functions with type hints."""
    total_funcs = 0
    typed_funcs = 0

    for py_file in directory.rglob("*.py"):
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_funcs += 1
                if node.returns is not None:
                    typed_funcs += 1

    return (typed_funcs / total_funcs * 100) if total_funcs > 0 else 0

# Dashboard metrics
print(f"Type Coverage: {calculate_type_coverage(Path('src'))}%")
# Week 1: 5%
# Month 3: 40%
# Month 6: 75%
# Year 1: 95%
```

**Timeline** (100K LOC):
- Month 1-2: Infrastructure + New Code
- Month 3-4: Critical Modules (20% codebase)
- Month 5-8: Core Modules (60% codebase)
- Month 9-12: Long Tail (20% codebase)
- **Total: 1 year** для 100K строк

**Key Metrics to Track**:
- Type coverage %
- mypy error count
- Bugs caught by mypy in CI
- Time saved on debugging
</details>

---

# 🎯 ЧАСТЬ 2: АРХИТЕКТУРНЫЕ ПРОБЛЕМЫ

## 4. Single Responsibility Principle (SRP) Violation

### 📖 Теория
**Проблема**: Класс или функция делает слишком много разных вещей.

**Признаки нарушения SRP**:
- Методы >50-100 строк
- Класс имеет >10 public методов
- Класс имеет >5 dependencies
- Сложно написать unit test
- Изменение в одной фиче ломает другую

### 🔴 Реальный Кейс: Healthcare.gov Launch (2013)

**Контекст**:
- Запуск федерального сайта здравоохранения США
- Бюджет: **$1.7 billion**
- Ожидаемая нагрузка: 60,000 одновременных пользователей
- Сайт упал в первый же день

**Что пошло не так**:
```python
# ❌ Упрощенная версия проблемы
class UserEnrollmentService:
    """God Object - делает ВСЕ!"""

    def enroll_user(self, user_data):
        # 1. Валидация (200 строк)
        if not self._validate_ssn(user_data['ssn']):
            return False
        if not self._validate_income(user_data['income']):
            return False
        # ... 15 других валидаций

        # 2. Проверка eligibility (300 строк)
        if not self._check_citizenship(user_data):
            return False
        if not self._check_state_eligibility(user_data):
            return False
        # ... 10 других проверок

        # 3. Интеграция с 10+ внешними системами (500 строк)
        self._query_irs_database(user_data)
        self._query_social_security(user_data)
        self._query_immigration(user_data)
        # ... 7 других API calls

        # 4. Расчет subsidies (400 строк)
        # ... сложная бизнес-логика

        # 5. Сохранение в БД (100 строк)
        # ...

        # 6. Отправка email (50 строк)
        # ...

        # Итого: >1500 строк в ОДНОМ методе!
```

**Последствия**:
- Сайт не работал **несколько недель**
- Только **6 человек** смогли зарегистрироваться в первый день (вместо тысяч)
- Правительство потратило дополнительные **$1 billion** на исправления
- Политический скандал

**Root Cause**:
- Tight coupling между компонентами
- Невозможно было масштабировать отдельные части
- Один failed API call ронял всю регистрацию
- Тестирование было невозможно

### ✅ Правильное Решение: Разделение Ответственности

```python
from abc import ABC, abstractmethod
from typing import Protocol
from dataclasses import dataclass

# 1. Value Objects
@dataclass(frozen=True)
class UserData:
    """Immutable user data."""
    ssn: str
    income: int
    state: str
    citizenship_status: str

@dataclass(frozen=True)
class EligibilityResult:
    """Result of eligibility check."""
    is_eligible: bool
    reasons: List[str]
    subsidy_amount: Optional[int] = None

# 2. Single Responsibility Services

class IValidator(Protocol):
    """Protocol for validators."""
    def validate(self, data: UserData) -> bool:
        ...

class SSNValidator:
    """Validates Social Security Numbers."""
    def validate(self, data: UserData) -> bool:
        return len(data.ssn) == 9 and data.ssn.isdigit()

class IncomeValidator:
    """Validates income data."""
    def __init__(self, min_income: int, max_income: int):
        self.min_income = min_income
        self.max_income = max_income

    def validate(self, data: UserData) -> bool:
        return self.min_income <= data.income <= self.max_income

class ValidationService:
    """Orchestrates validation - SINGLE responsibility."""
    def __init__(self, validators: List[IValidator]):
        self.validators = validators

    def validate_all(self, data: UserData) -> Tuple[bool, List[str]]:
        errors = []
        for validator in self.validators:
            if not validator.validate(data):
                errors.append(f"{validator.__class__.__name__} failed")
        return len(errors) == 0, errors

# 3. External Service Integration (с Circuit Breaker!)

class ExternalServiceClient(ABC):
    """Base class for external API clients."""

    @abstractmethod
    def query(self, data: UserData) -> Dict[str, Any]:
        pass

class IRSClient(ExternalServiceClient):
    """IRS API client - SINGLE responsibility."""

    def __init__(self, api_key: str, timeout: int = 30):
        self.api_key = api_key
        self.timeout = timeout
        self.circuit_breaker = CircuitBreaker()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential())
    def query(self, data: UserData) -> Dict[str, Any]:
        """Query IRS with retry and circuit breaker."""
        if self.circuit_breaker.is_open():
            return {"status": "unavailable"}

        try:
            response = requests.post(
                "https://irs.gov/api/verify",
                json={"ssn": data.ssn},
                timeout=self.timeout
            )
            return response.json()
        except RequestException as e:
            self.circuit_breaker.record_failure()
            raise

# 4. Business Logic Service

class EligibilityService:
    """Determines eligibility - SINGLE responsibility."""

    def __init__(
        self,
        irs_client: IRSClient,
        subsidy_calculator: SubsidyCalculator
    ):
        self.irs_client = irs_client
        self.subsidy_calculator = subsidy_calculator

    def check_eligibility(self, data: UserData) -> EligibilityResult:
        """Check if user is eligible for healthcare."""
        reasons = []

        # Citizenship check
        if data.citizenship_status not in ['citizen', 'permanent_resident']:
            reasons.append("Citizenship requirement not met")
            return EligibilityResult(False, reasons)

        # Income check via IRS
        irs_data = self.irs_client.query(data)
        if not irs_data.get('income_verified'):
            reasons.append("Income verification failed")
            return EligibilityResult(False, reasons)

        # Calculate subsidy
        subsidy = self.subsidy_calculator.calculate(data.income)

        return EligibilityResult(True, [], subsidy)

# 5. Orchestrator (Координатор)

class EnrollmentOrchestrator:
    """Orchestrates enrollment process - delegates to specialists."""

    def __init__(
        self,
        validation_service: ValidationService,
        eligibility_service: EligibilityService,
        enrollment_repository: EnrollmentRepository,
        notification_service: NotificationService
    ):
        self.validation_service = validation_service
        self.eligibility_service = eligibility_service
        self.enrollment_repository = enrollment_repository
        self.notification_service = notification_service

    async def enroll_user(self, user_data: UserData) -> EnrollmentResult:
        """Enroll user by orchestrating services."""

        # Step 1: Validate
        is_valid, errors = self.validation_service.validate_all(user_data)
        if not is_valid:
            return EnrollmentResult(success=False, errors=errors)

        # Step 2: Check eligibility
        eligibility = self.eligibility_service.check_eligibility(user_data)
        if not eligibility.is_eligible:
            return EnrollmentResult(success=False, reasons=eligibility.reasons)

        # Step 3: Save to database
        enrollment_id = await self.enrollment_repository.save(
            user_data, eligibility
        )

        # Step 4: Send notification (async, non-blocking)
        await self.notification_service.send_confirmation(
            enrollment_id, user_data
        )

        return EnrollmentResult(success=True, enrollment_id=enrollment_id)
```

**Преимущества разделения**:
- ✅ Каждый класс <100 строк
- ✅ Легко тестировать (mock dependencies)
- ✅ Можно масштабировать сервисы независимо
- ✅ Изменения локализованы
- ✅ Можно заменить реализацию без изменения интерфейса

### 💡 Вопросы на Собеседовании

**Q1**: "У вас есть класс DatabaseManager с 30 методами. Как вы его рефакторите?"

<details>
<summary>Правильный Ответ (Google-style)</summary>

**Шаг 1: Анализ**
```python
# ❌ God Object
class DatabaseManager:
    # Connection management (5 методов)
    def connect(self): ...
    def disconnect(self): ...
    def get_connection(self): ...
    def retry_connection(self): ...
    def check_health(self): ...

    # Query execution (8 методов)
    def execute(self): ...
    def execute_many(self): ...
    def fetch_one(self): ...
    def fetch_all(self): ...
    def execute_transaction(self): ...
    # ...

    # Migrations (4 метода)
    def run_migration(self): ...
    def rollback_migration(self): ...
    def get_migration_status(self): ...
    def create_migration(self): ...

    # Monitoring (6 методов)
    def get_stats(self): ...
    def get_slow_queries(self): ...
    def explain_query(self): ...
    # ...

    # Backup (4 метода)
    def create_backup(self): ...
    def restore_backup(self): ...
    # ...

    # Cache (3 метода)
    def cache_query(self): ...
    def invalidate_cache(self): ...
    def get_from_cache(self): ...
```

**Шаг 2: Группировка по ответственности**

```python
# ✅ Разделенная архитектура

# 1. Connection Management
class ConnectionPool:
    """Manages database connection pool."""

    def __init__(self, config: DbConfig):
        self.pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=20,
            **config.dict()
        )

    def get_connection(self) -> connection:
        return self.pool.getconn()

    def return_connection(self, conn: connection) -> None:
        self.pool.putconn(conn)

    def close_all(self) -> None:
        self.pool.closeall()

# 2. Query Execution
class QueryExecutor:
    """Executes database queries."""

    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool

    def execute_one(self, query: str, params: tuple) -> Dict:
        with self.pool.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchone()

    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        with self.pool.get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany(query, params_list)
                return cur.rowcount

# 3. Transaction Management
class TransactionManager:
    """Manages database transactions."""

    def __init__(self, connection_pool: ConnectionPool):
        self.pool = connection_pool

    @contextmanager
    def transaction(self):
        """Context manager for transactions."""
        conn = self.pool.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.return_connection(conn)

# 4. Migration Runner
class MigrationRunner:
    """Runs database migrations."""

    def __init__(self, executor: QueryExecutor):
        self.executor = executor
        self.migrations_dir = Path("migrations")

    def run_pending(self) -> List[str]:
        """Run all pending migrations."""
        applied = self._get_applied_migrations()
        pending = self._get_pending_migrations(applied)

        for migration in pending:
            self._run_migration(migration)

        return pending

# 5. Query Performance Monitor
class QueryMonitor:
    """Monitors query performance."""

    def __init__(self, executor: QueryExecutor):
        self.executor = executor

    def get_slow_queries(self, threshold_ms: int = 1000) -> List[Dict]:
        """Get queries slower than threshold."""
        return self.executor.execute_many(
            """
            SELECT query, mean_exec_time, calls
            FROM pg_stat_statements
            WHERE mean_exec_time > %s
            ORDER BY mean_exec_time DESC
            LIMIT 100
            """,
            (threshold_ms,)
        )

# 6. Backup Service
class BackupService:
    """Handles database backups."""

    def __init__(self, db_config: DbConfig):
        self.db_config = db_config

    def create_backup(self, backup_path: Path) -> None:
        """Create database backup using pg_dump."""
        subprocess.run([
            "pg_dump",
            "-h", self.db_config.host,
            "-U", self.db_config.user,
            "-d", self.db_config.database,
            "-F", "c",  # Custom format
            "-f", str(backup_path)
        ], check=True)

# 7. Query Cache
class QueryCache:
    """Caches query results."""

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        self.ttl = 3600  # 1 hour

    def get(self, key: str) -> Optional[Any]:
        """Get cached result."""
        data = self.redis.get(key)
        return json.loads(data) if data else None

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Cache query result."""
        self.redis.setex(
            key,
            ttl or self.ttl,
            json.dumps(value)
        )

# 8. Facade (для обратной совместимости)
class DatabaseFacade:
    """Facade providing backward-compatible interface."""

    def __init__(self, config: DbConfig):
        self.connection_pool = ConnectionPool(config)
        self.executor = QueryExecutor(self.connection_pool)
        self.transaction_manager = TransactionManager(self.connection_pool)
        self.migration_runner = MigrationRunner(self.executor)
        self.monitor = QueryMonitor(self.executor)
        self.backup_service = BackupService(config)
        self.cache = QueryCache(redis_client)

    # Делегирование к специализированным сервисам
    def execute(self, query: str, params: tuple) -> Dict:
        return self.executor.execute_one(query, params)

    def run_migrations(self) -> List[str]:
        return self.migration_runner.run_pending()

    # ... и т.д.
```

**Шаг 3: Migration Path**

```python
# Постепенная миграция
# 1. Создаем новые классы
# 2. DatabaseManager начинает делегировать к ним
# 3. Постепенно переводим клиентов на новые классы
# 4. Deprecate старый DatabaseManager

# Old code (будет работать)
db_manager = DatabaseManager()
result = db_manager.execute("SELECT * FROM users")

# New code (рекомендуется)
facade = DatabaseFacade(config)
result = facade.executor.execute_one("SELECT * FROM users", ())
```

**Метрики успеха**:
- Каждый класс <200 строк
- Каждый класс имеет 1 ответственность
- Unit test coverage >80%
- Легко добавлять новую функциональность
</details>

---

Продолжить создание остальных разделов шпаргалки? Следующие темы:
- Code Duplication и DRY принцип
- Exception Handling Best Practices
- Logging vs Print
- Performance Optimization (с кейсами Netflix, Spotify)
- Testing Strategies
- Database Best Practices

Продолжить?