# 📚 Тетрадь для подготовки к собеседованиям FAANG

## Автор: Vastargazing
## На основе: qwen_analyzer.py code review

---

# 🎯 Содержание

1. [Уровни инженеров в FAANG](#уровни-инженеров)
2. [Критические ошибки и их последствия](#критические-ошибки)
3. [Реальные кейсы из индустрии](#реальные-кейсы)
4. [Вопросы на собеседованиях](#вопросы-на-собесах)
5. [Code Review: что проверяют](#что-проверяют-на-code-review)
6. [Как отвечать на технические вопросы](#как-отвечать)
7. [Примеры до/после рефакторинга](#примеры-кода)

---

# 🏆 Уровни инженеров в FAANG

## L3 - Junior Engineer (0-2 года)

**Что умеет:**
- Пишет рабочий код под присмотром
- Исправляет баги по инструкции
- Делает простые фичи (CRUD)

**Типичные проблемы в коде:**
```python
# ❌ Плохо
def process_data(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result

# Проблемы:
# - Нет типов
# - Нет docstring
# - Нет обработки ошибок
# - Нет валидации
```

**Зарплата:** $120-180k (США)

---

## L4 - Mid-Level Engineer (2-5 лет)

**Что умеет:**
- Работает самостоятельно
- Проектирует фичи среднего размера
- Пишет тесты
- Участвует в code review

**Хороший код:**
```python
# ✅ Хорошо
def process_data(data: list[int]) -> list[int]:
    """Multiply each element by 2.

    Args:
        data: List of integers to process.

    Returns:
        List with doubled values.

    Raises:
        ValueError: If data is empty.
    """
    if not data:
        raise ValueError("Data cannot be empty")

    return [item * 2 for item in data]
```

**Зарплата:** $180-250k (США)

---

## L5 - Senior Engineer (5-8 лет) 👈 **ТЫ ЗДЕСЬ**

**Что умеет:**
- Проектирует системы целиком
- Думает о безопасности и производительности
- Менторит джунов
- Принимает архитектурные решения

**Production-ready код:**
```python
# ✅ Отлично - Senior уровень
import hashlib
import logging
from typing import TypedDict

logger = logging.getLogger(__name__)

class ProcessResult(TypedDict):
    """Result of data processing."""
    data: list[int]
    processed_count: int
    errors: list[str]

def process_data_safely(
    data: list[int],
    multiplier: int = 2
) -> ProcessResult:
    """Process data with error handling and logging.

    Args:
        data: List of integers to process.
        multiplier: Value to multiply by. Defaults to 2.

    Returns:
        ProcessResult with processed data and metadata.

    Example:
        result = process_data_safely([1, 2, 3])
        print(result['data'])  # [2, 4, 6]
    """
    if not data:
        logger.warning("Empty data provided")
        return {"data": [], "processed_count": 0, "errors": ["Empty input"]}

    if multiplier <= 0:
        raise ValueError(f"Multiplier must be positive, got {multiplier}")

    result_data = [item * multiplier for item in data]

    logger.info(f"Processed {len(data)} items successfully")

    return {
        "data": result_data,
        "processed_count": len(result_data),
        "errors": []
    }
```

**Зарплата:** $250-400k (США)

---

## L6 - Staff Engineer (8-12 лет)

**Что умеет:**
- Влияет на направление всей команды/продукта
- Пишет Design Docs
- Думает о масштабировании
- Code review для всей команды

**Enterprise-grade код:**
```python
# ✅ Staff уровень - думает о production
from typing import Protocol, TypedDict
import structlog

logger = structlog.get_logger()

class DataProcessor(Protocol):
    """Interface for data processors."""
    def process(self, data: list[int]) -> list[int]: ...

class ProcessorMetrics(TypedDict):
    """Metrics for monitoring."""
    duration_ms: float
    items_processed: int
    errors_count: int

class ProductionDataProcessor:
    """Enterprise data processor with observability."""

    def __init__(self, multiplier: int = 2, max_items: int = 10000):
        """Initialize processor with limits.

        Args:
            multiplier: Multiplication factor.
            max_items: Max items to prevent memory issues.
        """
        self.multiplier = multiplier
        self.max_items = max_items
        self._metrics: ProcessorMetrics = {
            "duration_ms": 0.0,
            "items_processed": 0,
            "errors_count": 0
        }

    def process(self, data: list[int]) -> ProcessResult:
        """Process with monitoring and circuit breaker."""
        import time
        start = time.time()

        try:
            # Validate
            if len(data) > self.max_items:
                raise ValueError(
                    f"Data size {len(data)} exceeds limit {self.max_items}"
                )

            # Process
            result = [item * self.multiplier for item in data]

            # Metrics
            duration = (time.time() - start) * 1000
            self._metrics["duration_ms"] = duration
            self._metrics["items_processed"] += len(data)

            logger.info(
                "processing_complete",
                items=len(data),
                duration_ms=duration
            )

            return {
                "data": result,
                "processed_count": len(result),
                "errors": []
            }

        except Exception as e:
            self._metrics["errors_count"] += 1
            logger.error("processing_failed", error=str(e))
            raise

    def get_metrics(self) -> ProcessorMetrics:
        """Get current metrics for monitoring."""
        return self._metrics.copy()
```

**Зарплата:** $400-600k (США)

---

# 🔴 Критические ошибки и их последствия

## 1. ❌ Использование hash() вместо hashlib

### Твоя ошибка в коде:
```python
# ❌ ОПАСНО
cached = redis_cache.get_analysis(f"qwen:{hash(lyrics)}")
```

### Почему это плохо:

**Проблема 1: Недетерминированность**
```python
# Python запущен 1 раз
hash("test")  # -1234567890

# Python запущен 2 раз
hash("test")  # 9876543210  ← ДРУГОЕ ЗНАЧЕНИЕ!
```

**Что происходит:**
- Пользователь анализирует текст → сохраняется в кэш
- Сервер перезагружается
- Пользователь запрашивает тот же текст → **кэш НЕ найден!**
- Потеря денег на API запросы

**Проблема 2: Коллизии**
```python
hash("abc")  # 12345
hash("xyz")  # 12345  ← КОЛЛИЗИЯ!
```

**Что происходит:**
- Пользователь A анализирует "rap lyrics about love"
- Пользователь B анализирует "completely different text"
- hash() даёт одинаковые значения
- Пользователь B получает **ЧУЖОЙ результат!**

### ✅ Правильное решение:
```python
import hashlib

def get_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# Всегда одинаковый результат
get_cache_key("test")  # "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
```

**Вероятность коллизии SHA256:** ~0% (практически невозможно)

---

## 2. ❌ Bare Exception Catching

### Твоя ошибка:
```python
# ❌ ОПАСНО
try:
    response = api.call()
except Exception as e:
    retry()
```

### Почему это плохо:

**Ловит ВСЁ, даже системные ошибки:**
```python
try:
    response = api.call()
except Exception:
    retry()  # ← Будет ретраить даже Ctrl+C!

# Что ловится:
# - KeyboardInterrupt (Ctrl+C)
# - SystemExit (выход из программы)
# - MemoryError (нет памяти)
# - KeyError (ошибка в коде)
```

**Что происходит в production:**
- API вернул 401 Unauthorized (неправильный ключ)
- Код пытается **ретраить 10 раз**
- Блокировка IP за слишком много запросов
- **Downtime** на 30 минут

### ✅ Правильное решение:
```python
from openai import (
    APIConnectionError,    # Сеть упала
    APITimeoutError,       # Таймаут
    RateLimitError,        # Лимит запросов
    AuthenticationError    # Неправильный ключ
)

try:
    response = api.call()

# Ретраим только временные ошибки
except (APIConnectionError, APITimeoutError, RateLimitError):
    retry()

# Не ретраим - сразу fail
except AuthenticationError as e:
    logger.error(f"Invalid API key: {e}")
    alert_team()
    raise
```

---

## 3. ❌ Отсутствие валидации

### Твоя ошибка:
```python
# ❌ Нет проверок
def analyze_lyrics(lyrics: str, temperature: float):
    return api.call(lyrics, temperature)
```

### Что может пойти не так:

**Кейс 1: Пустые данные**
```python
analyze_lyrics("", 0.5)
# → Тратим деньги на API запрос пустой строки
# → Возвращаем бесполезный результат
```

**Кейс 2: Неправильные параметры**
```python
analyze_lyrics("text", temperature=999.0)
# → API вернёт ошибку 400
# → Пользователь видит краш
```

**Кейс 3: SQL Injection (если бы использовали БД)**
```python
lyrics = "'; DROP TABLE users; --"
# Без валидации → потеря данных
```

### ✅ Правильное решение:
```python
def analyze_lyrics(
    lyrics: str,
    temperature: float | None = None
) -> AnalysisResult:
    """Analyze lyrics with validation."""

    # Проверка на пустоту
    if not lyrics or not lyrics.strip():
        raise ValueError("Lyrics cannot be empty")

    # Проверка длины (защита от abuse)
    if len(lyrics) > 100_000:
        raise ValueError("Lyrics too long (max 100k chars)")

    # Проверка диапазона
    if temperature is not None:
        if not (0.0 <= temperature <= 2.0):
            raise ValueError(
                f"Temperature must be 0.0-2.0, got {temperature}"
            )

    # Теперь безопасно использовать
    return api.call(lyrics, temperature)
```

---

# 💥 Реальные кейсы из индустрии

## Кейс 1: Knight Capital - $440 миллионов за 45 минут

**Дата:** 1 августа 2012
**Компания:** Knight Capital Group (США)
**Потери:** $440,000,000 💸

### Что произошло:

**Код:**
```python
# Старый код (должен был быть удалён)
if flag == True:  # ❌ Забыли удалить
    # Тестовый код для отладки
    for i in range(1000):
        buy_stock("AAPL", amount=1000000)

# Новый код
if flag == False:
    normal_trading()
```

**Последовательность событий:**
1. Разработчики обновили софт на 7 из 8 серверов
2. На одном сервере остался **старый код**
3. При запуске флаг установился в `True`
4. Старый тестовый код начал покупать акции
5. За **45 минут** куплено акций на $7 миллиардов
6. Компания обанкротилась через 2 дня

### Что нужно было сделать:

```python
# ✅ Правильно
class TradingMode(Enum):
    PRODUCTION = "production"
    TEST = "test"

class TradingSystem:
    def __init__(self, mode: TradingMode):
        if mode == TradingMode.TEST:
            raise RuntimeError(
                "Test mode not allowed in production!"
            )
        self.mode = mode

    def trade(self):
        # Тестовый код ФИЗИЧЕСКИ УДАЛЁН
        # Невозможно запустить по ошибке
        self.execute_production_trade()
```

### Уроки:
- ✅ Удаляй тестовый код из production
- ✅ Используй feature flags правильно
- ✅ Автоматизируй деплой (не руками на 8 серверов)
- ✅ Добавь circuit breakers (автостоп при аномалиях)

**Вопрос на собесе:** "Расскажите про Knight Capital. Как бы вы предотвратили это?"

---

## Кейс 2: AWS S3 Outage - упал весь интернет

**Дата:** 28 февраля 2017
**Компания:** Amazon Web Services
**Downtime:** 4 часа
**Убытки:** ~$150 миллионов для клиентов

### Что произошло:

**Команда инженера:**
```bash
# Хотел отключить несколько серверов для отладки
$ aws s3 remove-servers --num 5

# Опечатка - удалил ВСЕ серверы
$ aws s3 remove-servers --num 5000  # ❌
```

**Последствия:**
- Упали сайты: Netflix, Airbnb, Spotify, Slack
- Не работали умные дверные замки (люди не могли войти домой!)
- Остановились production линии заводов
- Убытки: $150M+ за 4 часа

### Что нужно было сделать:

```python
# ✅ Правильно
def remove_servers(num: int, confirm: bool = False) -> None:
    """Remove servers with safety checks.

    Args:
        num: Number of servers to remove.
        confirm: Must be True for production.

    Raises:
        ValueError: If num exceeds safety threshold.
    """
    MAX_SAFE_REMOVAL = 100

    # Проверка лимита
    if num > MAX_SAFE_REMOVAL:
        raise ValueError(
            f"Cannot remove {num} servers. "
            f"Max allowed: {MAX_SAFE_REMOVAL}. "
            f"Use --force-dangerous flag and get approval."
        )

    # Требуем подтверждение
    if not confirm:
        print(f"Will remove {num} servers. Add --confirm to proceed.")
        return

    # Dry-run сначала
    affected = get_affected_services(num)
    print(f"This will affect: {affected}")

    # Двойное подтверждение
    response = input("Type 'DELETE' to confirm: ")
    if response != "DELETE":
        print("Cancelled")
        return

    # Только теперь удаляем
    logger.info(f"Removing {num} servers", extra={"operator": get_user()})
    actually_remove(num)
```

### Уроки:
- ✅ Валидируй опасные операции
- ✅ Добавляй dry-run режим
- ✅ Требуй явное подтверждение
- ✅ Логируй кто что делает

**Вопрос на собесе:** "Как бы вы защитили критичную операцию от случайного запуска?"

---

## Кейс 3: Cloudflare - 27 минут downtime из-за регулярки

**Дата:** 2 июля 2019
**Компания:** Cloudflare
**Затронуто:** 50% интернет-трафика

### Что произошло:

**Код:**
```python
# ❌ Регулярное выражение с катастрофической сложностью
pattern = r"(.+)*"  # Backtracking hell!

def check_pattern(text: str) -> bool:
    import re
    return bool(re.match(pattern, text))

# Обычный текст - OK (0.001s)
check_pattern("hello")

# Длинный текст без совпадения - КАТАСТРОФА (>60s)
check_pattern("a" * 100 + "!")  # CPU 100% на минуту!
```

**Что произошло:**
- Задеплоили новый WAF (Web Application Firewall)
- В регулярке была катастрофическая сложность O(2^n)
- При определённых запросах CPU уходил в 100%
- Все серверы зависли
- Downtime: 27 минут

### Правильное решение:

```python
# ✅ Правильно - с таймаутом
import re
import signal
from contextlib import contextmanager

class RegexTimeout(Exception):
    pass

@contextmanager
def timeout(seconds: int):
    """Timeout context manager."""
    def handler(signum, frame):
        raise RegexTimeout("Regex took too long")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def check_pattern_safe(text: str, pattern: str) -> bool:
    """Check pattern with timeout protection.

    Args:
        text: Text to check.
        pattern: Regex pattern.

    Returns:
        True if matches, False otherwise.

    Raises:
        RegexTimeout: If regex takes >1 second.
    """
    try:
        with timeout(1):
            return bool(re.match(pattern, text))
    except RegexTimeout:
        logger.error(
            "Regex timeout",
            pattern=pattern,
            text_len=len(text)
        )
        return False
```

### Уроки:
- ✅ Тестируй на edge cases (длинные строки)
- ✅ Добавляй таймауты на опасные операции
- ✅ Используй regex validators перед деплоем
- ✅ Канареечный деплой (не всё сразу)

**Вопрос на собесе:** "Что такое ReDoS атака? Как защититься?"

---

## Кейс 4: Твой проект - как плохой код влияет

### Сценарий 1: hash() коллизия

**Что может случиться:**
```
День 1:
- Пользователь анализирует песню Eminem
- Результат кэшируется с hash("eminem lyrics...")

День 2:
- Другой пользователь анализирует песню 2Pac
- hash("2pac lyrics...") = ТАКОЕ ЖЕ ЗНАЧЕНИЕ (коллизия)
- Получает анализ Eminem!

Последствия:
- Пользователь жалуется: "Ваш AI тупой!"
- Репутация падает
- Отток пользователей
```

**С SHA256 - невозможно!**

---

### Сценарий 2: Retry без разбора ошибок

**Что было:**
```python
# ❌ Плохо
try:
    analyze_lyrics()
except Exception:
    for i in range(10):
        retry()  # Ретраим ВСЁ
```

**Что происходит:**
```
Пользователь вводит неправильный API ключ
→ AuthenticationError
→ Код ретраит 10 раз
→ 10 запросов с неправильным ключом
→ API блокирует ваш IP на 1 час
→ ВСЕ пользователи не могут работать!

Убытки:
- 1000 пользователей × 1 час = 1000 часов простоя
- Потеря выручки
- Негативные отзывы
```

**С правильной обработкой:**
```python
# ✅ Хорошо
except AuthenticationError:
    # Не ретраим - сразу алерт
    alert_admin("Invalid API key!")
    return {"error": "Service temporarily unavailable"}
```

---

# 📝 Вопросы на собеседованиях

## Вопросы по безопасности

### Q1: "Почему нельзя использовать hash() для кэш-ключей?"

**Плохой ответ:**
> "Потому что это небезопасно"

**Хороший ответ (Senior уровень):**
> "hash() имеет две критические проблемы:
>
> 1. **Недетерминированность**: Python рандомизирует hash seed при старте (с Python 3.3) для защиты от DOS атак. Поэтому `hash("test")` даёт разные значения при каждом запуске процесса. Это ломает кэширование между перезапусками.
>
> 2. **Коллизии**: hash() использует простую хеш-функцию с малым пространством значений. Вероятность коллизий высокая, особенно на больших данных.
>
> Вместо этого я использую SHA256: детерминированный, криптографически стойкий, практически нулевая вероятность коллизий (2^256 возможных значений)."

---

### Q2: "Как бы вы защитили API ключ в логах?"

**Плохой ответ:**
> "Не логировать его"

**Хороший ответ:**
> "Я использую несколько уровней защиты:
>
> 1. **Маскировка**: Показываю только первые 8 и последние 4 символа:
>    ```python
>    masked = f"{key[:8]}...{key[-4:]}"  # "sk-proj12...xyz9"
>    ```
>
> 2. **Структурированное логирование**: Использую structlog с фильтрами для автоматической маскировки полей `api_key`, `password`, `token`.
>
> 3. **Переменные окружения**: Ключи только в ENV, никогда в коде:
>    ```python
>    api_key = os.getenv("QWEN_API_KEY")
>    if not api_key:
>        raise ValueError("QWEN_API_KEY not set")
>    ```
>
> 4. **Secrets management**: В продакшене использую AWS Secrets Manager или HashiCorp Vault с автоматической ротацией ключей."

---

## Вопросы по обработке ошибок

### Q3: "В чём разница между retryable и non-retryable ошибками?"

**Хороший ответ:**

**Retryable (временные):**
- `APIConnectionError` - сеть упала, можно повторить
- `APITimeoutError` - таймаут, можно попробовать ещё раз
- `RateLimitError` - превышен лимит, подождём и повторим
- `500 Internal Server Error` - проблема на сервере, может пройти

**Non-retryable (постоянные):**
- `AuthenticationError` - неправильный ключ, повторение бесполезно
- `400 Bad Request` - неправильные данные, не исправится само
- `404 Not Found` - ресурс не существует
- `ValueError` - ошибка в коде, нужен фикс

**Пример:**
```python
try:
    api.call()
except (APIConnectionError, RateLimitError) as e:
    # Temporary - retry with backoff
    time.sleep(2 ** attempt)
    retry()
except (AuthenticationError, ValueError) as e:
    # Permanent - alert and fail fast
    logger.error(f"Permanent failure: {e}")
    alert_team()
    raise
```

---

### Q4: "Что такое exponential backoff и зачем он нужен?"

**Хороший ответ:**

"Exponential backoff - это стратегия повторных попыток с экспоненциально растущими задержками.

**Проблема без backoff:**
```python
# ❌ Плохо - все ретраят одновременно
for i in range(10):
    try:
        api.call()
    except:
        time.sleep(1)  # Все ждут 1 секунду
        retry()

# Что происходит:
# 100 клиентов получили ошибку
# Все ждут 1 секунду
# Все одновременно ретраят через 1 сек
# → Thundering herd problem
# → Сервер падает снова
```

**С exponential backoff:**
```python
# ✅ Хорошо
for attempt in range(1, MAX_RETRIES + 1):
    try:
        return api.call()
    except RetryableError:
        wait = min(2 ** attempt, 32)  # 2s, 4s, 8s, 16s, 32s max
        time.sleep(wait + random.uniform(0, 1))  # Jitter

# Преимущества:
# - Даём серверу время восстановиться
# - Рандомизация (jitter) распределяет нагрузку
# - Ограничение максимальной задержки (32s cap)
```

**Реальный пример:**
AWS SDK использует exponential backoff. Когда DynamoDB перегружен, тысячи клиентов не DDOS'ят его повторными запросами."

---

## Вопросы по тестированию

### Q5: "Какие тесты вы бы написали для analyze_lyrics()?"

**Хороший ответ (приоритизация):**

**Priority 1 (Must have):**
```python
def test_analyze_lyrics_success():
    """Happy path - всё работает."""

def test_empty_lyrics_raises_error():
    """Валидация - пустые данные."""

def test_invalid_temperature_raises():
    """Валидация - неправильный диапазон."""

def test_api_error_retry():
    """Retry logic - повторяет при сбое."""

def test_auth_error_no_retry():
    """Fail fast - не ретраит auth ошибки."""
```

**Priority 2 (Should have):**
```python
def test_cache_hit():
    """Кэширование - возвращает из кэша."""

def test_cache_miss():
    """Кэширование - делает запрос если нет в кэше."""

def test_exponential_backoff():
    """Retry - проверяем 2^n задержки."""
```

**Priority 3 (Nice to have):**
```python
def test_concurrent_requests():
    """Thread safety."""

def test_long_lyrics():
    """Edge case - очень длинный текст."""
```

**Обоснование:**
"Фокусируюсь на критичных путях (80/20 правило). Priority 1 покрывает безопасность и корректность. Priority 2 - производительность. Priority 3 - edge cases."

---

## Вопросы по архитектуре

### Q6: "Зачем нужен Context Manager для QwenAnalyzer?"

**Хороший ответ:**

"Context manager решает проблему управления ресурсами:

**Проблема без него:**
```python
# ❌ Потенциальная утечка ресурсов
analyzer = QwenAnalyzer()
result = analyzer.analyze_lyrics("text")  # Что если тут exception?
# client никогда не закрылся!
```

**С context manager:**
```python
# ✅ Гарантированная очистка
with QwenAnalyzer() as analyzer:
    result = analyzer.analyze_lyrics("text")
# client автоматически закрылся, даже при exception

# Реализация:
def __enter__(self):
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    if hasattr(self.client, 'close'):
        self.client.close()
    return False  # Не подавляем исключения
```

**Преимущества:**
1. Детерминированное освобождение ресурсов
2. Защита от утечек при исключениях
3. Чистый, читаемый код
4. Стандартная практика в Python (как file.open())

**Реальный кейс:**
В продакшене с 1000 RPS без context manager накапливались незакрытые соединения. Через час приложение падало с 'Too many open files'. Context manager решил проблему."

---

# 💡 Как отвечать на собесах

## Структура хорошего ответа (метод STAR)

**S**ituation - ситуация
**T**ask - задача
**A**ction - действие
**R**esult - результат

### Пример:

**Вопрос:** "Расскажите про сложный баг, который вы исправили"

**Плохой ответ:**
> "Был баг, я его исправил"

**Хороший ответ (STAR):**

**S (Situation):**
> "В нашем проекте rap-scraper пользователи жаловались, что анализ одних и тех же текстов даёт разные результаты после перезапуска сервера."

**T (Task):**
> "Нужно было найти причину недетерминированного поведения и исправить, не ломая существующие фичи."

**A (Action):**
> "Я провёл код ревью и обнаружил, что мы используем hash() для генерации кэш-ключей. Проблема в том, что Python рандомизирует hash seed для защиты от DOS атак, поэтому hash() даёт разные значения при каждом запуске.
>
> Я:
> 1. Заменил hash() на hashlib.sha256() - детерминированный
> 2. Написал тесты для проверки стабильности ключей
> 3. Добавил миграцию для очистки старого кэша
> 4. Задокументировал почему hash() опасен"

**R (Result):**
> "После деплоя hit rate кэша вырос с 40% до 85%, жалобы прекратились, сэкономили ~$500/месяц на API запросах."

---

## Распространённые ловушки

### Ловушка 1: "Расскажите о себе"

**❌ Плохо:**
> "Я программист, работаю 3 года, знаю Python..."

**✅ Хорошо:**
> "Я Senior Python разработчик с 3 годами опыта в ML-проектах. Последний год работаю над rap-scraper - это система анализа текстов песен с использованием QWEN LLM.
>
> Мои ключевые достижения:
> - Оптимизировал систему кэширования, снизив costs на 60%
> - Внедрил Google Style Guide, повысил quality score с 3.4 до 9.6
> - Покрыл критичный код тестами, снизил production bugs на 80%
>
> Сейчас ищу позицию где смогу применить опыт в distributed systems и ML."

---

### Ловушка 2: "Какие у вас слабости?"

**❌ Плохо:**
> "Я перфекционист" (избитая фраза)

**✅ Хорошо:**
> "Раньше я не уделял достаточно внимания тестированию - писал код быстро, но без тестов. Это привело к production багу, который стоил команде 2 дня работы.
>
> После этого я:
> - Изучил pytest и TDD подход
> - Теперь пишу тесты ПЕРЕД кодом для критичных функций
> - Настроил CI/CD с обязательным coverage >70%
>
> Сейчас мой код более надёжный, и я могу рефакторить без страха сломать что-то."

---

# 📊 Примеры кода: До/После

## Пример 1: Обработка ошибок

### ❌ БЫЛО (Junior уровень):
```python
def analyze_lyrics(lyrics):
    try:
        result = api.call(lyrics)
        return result
    except:
        return None
```

**Проблемы:**
- Нет типов
- Bare except
- Теряется информация об ошибке
- Непонятно что вернулось

---

### ✅ СТАЛО (Senior уровень):
```python
from typing import TypedDict
from openai import APIConnectionError, AuthenticationError
import logging

logger = logging.getLogger(__name__)

class AnalysisResult(TypedDict):
    """Structured analysis result."""
    data: dict
    success: bool
    error: str | None

def analyze_lyrics(lyrics: str) -> AnalysisResult:
    """Analyze lyrics with proper error handling.

    Args:
        lyrics: Rap lyrics text to analyze.

    Returns:
        AnalysisResult with data or error information.

    Raises:
        ValueError: If lyrics is empty.
        AuthenticationError: If API key is invalid.
    """
    if not lyrics or not lyrics.strip():
        raise ValueError("Lyrics cannot be empty")

    try:
        result = api.call(lyrics)
        return {
            "data": result,
            "success": True,
            "error": None
        }

    except APIConnectionError as e:
        logger.warning(f"API connection failed: {e}")
        return {
            "data": {},
            "success": False,
            "error": "Service temporarily unavailable"
        }

    except AuthenticationError as e:
        logger.error(f"Authentication failed: {e}")
        raise  # Re-raise - это критичная ошибка
```

**Улучшения:**
- ✅ Полная типизация
- ✅ Специфичные исключения
- ✅ Docstring в Google Style
- ✅ Логирование с уровнями
- ✅ Структурированный результат

---

## Пример 2: Retry Logic

### ❌ БЫЛО:
```python
def call_api(data):
    for i in range(3):
        try:
            return api.call(data)
        except:
            time.sleep(1)
    return {"error": "failed"}
```

**Проблемы:**
- Magic number (3)
- Линейный backoff
- Все ошибки ретраятся
- Нет логирования

---

### ✅ СТАЛО:
```python
import time
import logging
from typing import TypeVar, Callable
from openai import APIConnectionError, APITimeoutError, AuthenticationError

logger = logging.getLogger(__name__)

T = TypeVar('T')

MAX_RETRIES = 3
BACKOFF_BASE = 2
MAX_BACKOFF = 32

def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = MAX_RETRIES
) -> T:
    """Execute function with exponential backoff retry.

    Args:
        func: Function to execute.
        max_retries: Maximum retry attempts.

    Returns:
        Function result.

    Raises:
        AuthenticationError: If auth fails (no retry).
        Exception: If all retries exhausted.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retries}")
            return func()

        except AuthenticationError as e:
            # Fatal - don't retry
            logger.error(f"Auth failed: {e}")
            raise

        except (APIConnectionError, APITimeoutError) as e:
            # Retryable
            last_error = e
            logger.warning(
                f"Attempt {attempt} failed (retryable): {e}"
            )

            if attempt < max_retries:
                wait = min(BACKOFF_BASE ** attempt, MAX_BACKOFF)
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)

    # All retries failed
    logger.error(f"All {max_retries} attempts failed")
    raise last_error

# Использование:
result = retry_with_backoff(
    lambda: api.call(data)
)
```

**Улучшения:**
- ✅ Константы вместо magic numbers
- ✅ Exponential backoff (2^n)
- ✅ Различение retryable/non-retryable
- ✅ Детальное логирование
- ✅ Generic функция (работает с любым типом)

---

# 🎓 Чек-лист перед собеседованием

## За 1 день до собеса

- [ ] Перечитать код своих проектов
- [ ] Вспомнить STAR истории (3-5 штук)
- [ ] Повторить основы Python (decorators, generators, async)
- [ ] Просмотреть эту тетрадь

## Во время собеса

- [ ] Задавать уточняющие вопросы перед ответом
- [ ] Использовать STAR формат
- [ ] Приводить конкретные примеры из кода
- [ ] Упоминать метрики (% улучшения, $ сэкономлено)
- [ ] Признавать что чего-то не знаю (но готов изучить)

## После собеса

- [ ] Записать вопросы которые задавали
- [ ] Проанализировать что ответил хорошо/плохо
- [ ] Дополнить эту тетрадь новыми кейсами

---

# 📚 Дополнительные ресурсы

## Книги (must read):
1. **"Clean Code"** - Robert Martin
2. **"The Pragmatic Programmer"** - Hunt & Thomas
3. **"Designing Data-Intensive Applications"** - Martin Kleppmann

## Статьи:
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [AWS Architecture Blog](https://aws.amazon.com/blogs/architecture/)
- [Postmortems от крупных компаний](https://github.com/danluu/post-mortems)

## Практика:
- LeetCode (алгоритмы)
- System Design Primer (архитектура)
- Real Python (best practices)

---

# 🎯 Итоговые советы

## Что ценят FAANG компании:

1. **Security First** - думай о безопасности сразу
2. **Test Coverage** - код без тестов = broken code
3. **Observability** - логи, метрики, alerting
4. **Simple Solutions** - не overengineering
5. **Documentation** - код пишется 1 раз, читается 100 раз

## Главное правило:

> **"Code is read more often than it is written"**
> - Guido van Rossum (создатель Python)

Пиши код так, будто его будет поддерживать психопат, который знает где ты живёшь 😄

---

**Удачи на собесах, босс! 🚀**

*Эта тетрадь основана на реальном code review qwen_analyzer.py и опыте FAANG компаний.*
