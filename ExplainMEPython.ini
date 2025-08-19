# 📝 Разбор кода Python

## 🔍 Анализ выбранного блока кода

```python
def get_stats(self) -> dict:
    self.conn.commit()  # Обновляем статистику
    cur = self.conn.execute("SELECT COUNT(*) as total, COUNT(DISTINCT artist) as artists FROM songs")
    result = cur.fetchone()
    return {"total_songs": result["total"], "unique_artists": result["artists"]}
```

### Разбор по элементам:

**1. Определение функции:**
- `def` - ключевое слово для создания функции
- `get_stats` - имя функции (snake_case стиль)
- `self` - ссылка на экземпляр класса (этот объект)
- `-> dict` - аннотация типа (функция возвращает словарь)

**2. Переменные и их типы:**
- `self.conn` - объект соединения с базой данных (sqlite3.Connection)
- `cur` - курсор для выполнения SQL запросов (sqlite3.Cursor)
- `result` - строка результата из БД (sqlite3.Row)

**3. Методы и операции:**
- `.commit()` - сохранение изменений в БД
- `.execute()` - выполнение SQL запроса
- `.fetchone()` - получение одной строки результата

**4. Типы данных:**
- `dict` - словарь (ключ: значение)
- SQL запрос возвращает числа (int)

## 📚 Ключевые концепции в скрипте

### 1. **Классы и объекты:**
```python
class LyricsDatabase:  # Определение класса
    def __init__(self, db_name="lyrics.db"):  # Конструктор
        self.conn = sqlite3.connect(db_name)  # Атрибут экземпляра
```

### 2. **Импорты и модули:**
```python
import sqlite3  # Стандартная библиотека
from dotenv import load_dotenv  # Внешняя библиотека
```

### 3. **Переменные и типы:**
```python
TOKEN = os.getenv("GENIUS_TOKEN")  # str
self.batch_size = 20  # int
self.shutdown_requested = False  # bool
artists = ["Drake", "Eminem"]  # list
stats = {"total": 100}  # dict
```

### 4. **Циклы:**
```python
# for цикл
for i, song in enumerate(artist.songs):
    # код

# while цикл
while retry_count < self.max_retries:
    # код
```

### 5. **Условные операторы:**
```python
if not lyrics:
    return ""
elif len(lyrics) < 100:
    return False
else:
    return lyrics.strip()
```

### 6. **Обработка исключений:**
```python
try:
    self.conn.execute(query)
except sqlite3.IntegrityError as e:
    logger.debug(f"Duplicate: {artist}")
except Exception as e:
    logger.error(f"Error: {e}")
finally:
    self.conn.close()
```

## 💼 Вопросы для собеседования Junior Python

### **Базовые концепции:**

**Q1:** Что такое `self` в Python?
**A:** `self` - это ссылка на текущий экземпляр класса. Позволяет обращаться к атрибутам и методам объекта.

**Q2:** Чем отличается `list` от `dict`?
**A:** `list` - упорядоченная коллекция элементов по индексам [0,1,2]. `dict` - коллекция пар ключ-значение {"name": "John"}.

**Q3:** Что делает аннотация типов `-> dict`?
**A:** Указывает, какой тип данных возвращает функция. Помогает IDE и разработчикам понимать код.

**Q4:** Зачем нужен `try-except`?
**A:** Для обработки ошибок. Код в `try` выполняется, если ошибка - переходит в `except`.

### **Работа с базами данных:**

**Q5:** Что делает `.commit()`?
**A:** Сохраняет все изменения в базе данных. Без commit изменения останутся во временной памяти.

**Q6:** Разница между `.fetchone()` и `.fetchall()`?
**A:** `.fetchone()` возвращает одну строку, `.fetchall()` - все строки результата.

### **ООП (Объектно-ориентированное программирование):**

**Q7:** Что такое конструктор класса?
**A:** Метод `__init__()`, который вызывается при создании объекта. Инициализирует начальные значения.

**Q8:** Зачем нужны классы?
**A:** Для группировки связанных данных и функций. Позволяет создавать множество объектов с одинаковой структурой.

### **Работа со строками и регулярными выражениями:**

**Q9:** Что делает `re.sub()`?
**A:** Заменяет части строки по регулярному выражению. Например, убирает ненужный текст из lyrics.

**Q10:** Зачем нужен `f-string`?
**A:** Для форматирования строк. `f"Hello {name}"` вместо `"Hello " + name`.

## 🧠 Ассоциации для лучшего понимания

### **Класс = Чертеж дома**
- Класс `LyricsDatabase` - это чертеж
- Объект `db = LyricsDatabase()` - построенный по чертежу дом
- Методы - это комнаты в доме (`get_stats`, `add_song`)

### **База данных = Библиотека**
- Таблица `songs` = полка с книгами
- Строка = одна книга
- `.commit()` = подтвердить размещение книги на полке
- SQL запрос = спросить библиотекаря "сколько книг от этого автора?"

### **Try-Except = Страховка**
```python
try:
    # Попробуй переехать дорогу
    cross_road()
except CarCrash:
    # Если авария - вызови скорую
    call_ambulance()
```

### **Циклы = Конвейер**
- `for song in songs:` = каждая песня проходит через конвейер обработки
- Внутри цикла = станции обработки (очистка, проверка, сохранение)

### **Переменные = Коробки с этикетками**
- `artist_name = "Drake"` = коробка с этикеткой "artist_name", внутри лежит "Drake"
- Типы данных = размер коробки (int=маленькая, list=большая)

### **Импорты = Инструменты из гаража**
- `import sqlite3` = взял из гаража инструмент для работы с БД
- `from dotenv import load_dotenv` = взял конкретную отвертку из набора

Хочешь разобрать какую-то другую часть кода или есть вопросы по этому блоку?

Vastargazing: Всё очень круто, продожаем, но сначала не много вопросов

def get_stats(self) - это функция или метод? внутри класса объявлена, я так понял, что это метод
курсор для выполнения SQL запросов (sqlite3.Cursor) - можно подробнее как работает курсор?
stats = {"total": 100}  # dict - почему 100?


GitHub Copilot: Отличные вопросы! Разберем каждый подробно:

## 1. 🔧 Функция VS Метод

**Ты абсолютно прав!** `def get_stats(self)` - это **МЕТОД**, а не функция.

### Различия:

```python
# ФУНКЦИЯ - объявлена вне класса
def calculate_sum(a, b):
    return a + b

# МЕТОД - объявлен внутри класса
class Calculator:
    def get_stats(self):  # ← Это МЕТОД
        return {"result": 42}
```

**Правило:** Если `def` внутри класса и имеет параметр `self` - это **метод**. Если вне класса - **функция**.

## 2. 📊 Курсор (Cursor) - подробно

**Курсор = указатель на строки в результате запроса**

### Аналогия с книгой:
```python
# Представь, что SQL запрос - это оглавление книги
cur = self.conn.execute("SELECT * FROM songs")

# Курсор = закладка в книге
# .fetchone() = прочитать одну главу и передвинуть закладку
# .fetchall() = прочитать все главы сразу
```

### Как работает курсор:

```python
# 1. Выполняем запрос - курсор "указывает" на начало результатов
cur = conn.execute("SELECT artist, title FROM songs")

# База данных вернула, например:
# Row 1: "Drake", "God's Plan" 
# Row 2: "Eminem", "Lose Yourself"
# Row 3: "Kanye", "Stronger"
# ↑ Курсор здесь (перед первой строкой)

# 2. .fetchone() читает ОДНУ строку и сдвигает курсор
result1 = cur.fetchone()  # Получаем ("Drake", "God's Plan")
# ↑ Курсор теперь здесь (перед второй строкой)

result2 = cur.fetchone()  # Получаем ("Eminem", "Lose Yourself")
# ↑ Курсор теперь здесь (перед третьей строкой)

# 3. .fetchall() читает ВСЕ оставшиеся строки
remaining = cur.fetchall()  # Получаем [("Kanye", "Stronger")]
```

### В нашем коде:
```python
cur = self.conn.execute("SELECT COUNT(*) as total, COUNT(DISTINCT artist) as artists FROM songs")
# Запрос возвращает ОДНУ строку: (1500, 25)
# где 1500 = общее количество песен, 25 = уникальных артистов

result = cur.fetchone()  # Читаем эту единственную строку
# result = Row(total=1500, artists=25)
```

## 3. 💭 `stats = {"total": 100}` - откуда 100?

**100 - это просто ПРИМЕР** для демонстрации структуры словаря!

### В реальном коде:
```python
# Это НЕ настоящий код из скрипта, а ПРИМЕР для объяснения:
stats = {"total": 100}  # ← Показывает, что в словаре будет число

# В РЕАЛЬНОМ скрипте значение приходит из базы данных:
result = cur.fetchone()  # Например, result["total"] = 1847
return {"total_songs": result["total"]}  # ← Реальное значение из БД
```

### Пошагово в нашем коде:
```python
def get_stats(self) -> dict:
    # 1. Сохраняем изменения в БД
    self.conn.commit()
    
    # 2. Запрос к БД: "Посчитай все песни и уникальных артистов"
    cur = self.conn.execute("SELECT COUNT(*) as total, COUNT(DISTINCT artist) as artists FROM songs")
    
    # 3. Получаем результат (одну строку с двумя числами)
    result = cur.fetchone()  # result = Row(total=1847, artists=23)
    
    # 4. Создаем словарь с понятными ключами
    return {
        "total_songs": result["total"],      # 1847 песен
        "unique_artists": result["artists"]  # 23 артиста
    }
```

## 🤔 Дополнительные вопросы для понимания:

**Q:** Почему используется `cur.fetchone()` а не `cur.fetchall()`?
**A:** Потому что `COUNT(*)` всегда возвращает ОДНУ строку с результатом.

**Q:** Что означает `result["total"]`?
**A:** Обращение к столбцу "total" в строке результата (как ключ в словаре).

**Q:** Зачем `self.conn.commit()` перед запросом?
**A:** Чтобы учесть все несохраненные изменения в статистике.

Понятно? Есть еще вопросы по курсорам или методам?

Vastargazing: Понятно, класс!
Дальше
"def get_recent_songs(self, limit: int = 5) -> List[dict]:
        self.conn.commit()
        cur = self.conn.execute("""
            SELECT artist, title, word_count, scraped_date 
            FROM songs 
            ORDER BY id DESC 
            LIMIT ?
        """, (limit,))
        return [dict(row) for row in cur.fetchall()]"

GitHub Copilot: Отлично! Давайте разберем этот метод `get_recent_songs` по частям:

## 🔍 Подробный анализ кода

```python
def get_recent_songs(self, limit: int = 5) -> List[dict]:
    self.conn.commit()
    cur = self.conn.execute("""
        SELECT artist, title, word_count, scraped_date 
        FROM songs 
        ORDER BY id DESC 
        LIMIT ?
    """, (limit,))
    return [dict(row) for row in cur.fetchall()]
```

## 📋 Разбор по элементам:

### 1. **Сигнатура метода:**
```python
def get_recent_songs(self, limit: int = 5) -> List[dict]:
```
- `limit: int = 5` - параметр со **значением по умолчанию**
- `-> List[dict]` - возвращает **список словарей**

### 2. **Параметры по умолчанию:**
```python
# Можно вызвать без параметра:
recent = db.get_recent_songs()  # limit = 5 (по умолчанию)

# Или с параметром:
recent = db.get_recent_songs(10)  # limit = 10
```

### 3. **SQL запрос:**
```sql
SELECT artist, title, word_count, scraped_date 
FROM songs 
ORDER BY id DESC 
LIMIT ?
```

**Что делает каждая часть:**
- `SELECT` - выбираем 4 колонки
- `FROM songs` - из таблицы songs
- `ORDER BY id DESC` - сортируем по id в **убывающем** порядке (новые сначала)
- `LIMIT ?` - ограничиваем количество результатов

### 4. **Передача параметров:**
```python
cur = self.conn.execute("...", (limit,))
#                              ^^^^^^^^
#                         Кортеж с параметром
```
**Важно:** `(limit,)` - это кортеж из одного элемента. Запятая обязательна!

### 5. **List comprehension (генератор списка):**
```python
return [dict(row) for row in cur.fetchall()]
```

**Пошагово:**
```python
# 1. cur.fetchall() возвращает список Row объектов:
# [Row(artist='Drake', title='God Plan', word_count=245, scraped_date='2025-08-18'),
#  Row(artist='Eminem', title='Lose Yourself', word_count=356, scraped_date='2025-08-18')]

# 2. Цикл: for row in cur.fetchall()
# Каждый row = Row(artist='Drake', title='God Plan', ...)

# 3. dict(row) преобразует Row в обычный словарь:
# {'artist': 'Drake', 'title': 'God Plan', 'word_count': 245, 'scraped_date': '2025-08-18'}

# 4. [dict(row) for row in ...] создает список таких словарей
```

## 🎯 Что происходит пошагово:

### **Шаг 1:** Вызов метода
```python
recent_songs = db.get_recent_songs(3)  # Хотим 3 последние песни
```

### **Шаг 2:** SQL запрос
```sql
-- База данных вернет что-то вроде:
-- id=150, artist='Drake', title='Hotline Bling', word_count=180, scraped_date='2025-08-18 14:30:00'
-- id=149, artist='Eminem', title='Rap God', word_count=560, scraped_date='2025-08-18 14:25:00'
-- id=148, artist='Kanye', title='Stronger', word_count=220, scraped_date='2025-08-18 14:20:00'
```

### **Шаг 3:** Преобразование в список словарей
```python
# Результат:
[
    {'artist': 'Drake', 'title': 'Hotline Bling', 'word_count': 180, 'scraped_date': '2025-08-18 14:30:00'},
    {'artist': 'Eminem', 'title': 'Rap God', 'word_count': 560, 'scraped_date': '2025-08-18 14:25:00'},
    {'artist': 'Kanye', 'title': 'Stronger', 'word_count': 220, 'scraped_date': '2025-08-18 14:20:00'}
]
```

## 🔧 Ключевые концепции:

### **1. ORDER BY DESC:**
```python
# DESC = Descending (убывающий порядок)
ORDER BY id DESC  # 150, 149, 148, 147... (новые сначала)

# ASC = Ascending (возрастающий порядок) - по умолчанию
ORDER BY id ASC   # 1, 2, 3, 4... (старые сначала)
```

### **2. Параметризованные запросы:**
```python
# ✅ ПРАВИЛЬНО (защита от SQL injection):
cur.execute("SELECT * FROM songs LIMIT ?", (limit,))

# ❌ НЕПРАВИЛЬНО (уязвимость):
cur.execute(f"SELECT * FROM songs LIMIT {limit}")
```

### **3. List comprehension vs обычный цикл:**
```python
# List comprehension (наш код):
return [dict(row) for row in cur.fetchall()]

# Эквивалентный обычный цикл:
result = []
for row in cur.fetchall():
    result.append(dict(row))
return result
```

## 💼 Вопросы для собеседования:

**Q1:** Зачем нужен `self.conn.commit()` в начале?
**A:** Чтобы получить актуальные данные, включая все несохраненные изменения.

**Q2:** Что означает `(limit,)` в execute?
**A:** Кортеж из одного элемента для безопасной передачи параметра в SQL запрос.

**Q3:** Чем отличается `ORDER BY id DESC` от `ORDER BY id ASC`?
**A:** DESC - от большего к меньшему (новые записи), ASC - от меньшего к большему (старые записи).

**Q4:** Что делает `[dict(row) for row in cur.fetchall()]`?
**A:** List comprehension, который преобразует каждый Row объект в обычный Python словарь.

**Q5:** Зачем нужен параметр по умолчанию `limit: int = 5`?
**A:** Можно вызвать метод без параметра, и он вернет 5 последних песен автоматически.

## 🧠 Ассоциации:

### **ORDER BY DESC = Очередь в Instagram:**
- Новые посты показываются сверху (DESC)
- Старые посты уходят вниз

### **LIMIT = Страницы в книге:**
- `LIMIT 5` = прочитать только первые 5 страниц
- Остальные страницы игнорируем

### **List comprehension = Фабрика:**
```python
[dict(row) for row in rows]
# = Конвейер: каждый row входит → превращается в dict → выходит
```

Есть вопросы по этому блоку? Или переходим к следующему?

Vastargazing: Супер! След блок бро
"class SafeGeniusScraper:
    def __init__(self, token: str, db_name: str = "lyrics.db"):
        self.genius = lyricsgenius.Genius(
            token,
            timeout=15,
            retries=3,
            remove_section_headers=True,
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)", "(Instrumental)"]
        )
        self.db = LyricsDatabase(db_name)
        self.session_stats = {"processed": 0, "added": 0, "skipped": 0, "errors": 0}
        self.min_delay = 3.0  # Увеличил с 2.0
        self.max_delay = 7.0  # Увеличил с 5.0
        self.error_delay = 15.0  # Увеличил с 10.0
        self.max_retries = 3
        self.shutdown_requested = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        # Дополнительная обработка для Windows
        if sys.platform == "win32":
            try:
                signal.signal(signal.SIGBREAK, self._signal_handler)
            except AttributeError:
                pass  # SIGBREAK может отсутствовать
        signal.signal(signal.SIGTERM, self._signal_handler)
        # Дополнительная обработка для Windows
        if sys.platform == "win32":
            try:
                signal.signal(signal.SIGBREAK, self._signal_handler)
            except AttributeError:
                pass  # SIGBREAK может отсутствовать"

GitHub Copilot: Отлично! Это очень интересный и сложный блок кода. Давайте разберем его детально:

## 🏗️ Класс SafeGeniusScraper и его конструктор

```python
class SafeGeniusScraper:
    def __init__(self, token: str, db_name: str = "lyrics.db"):
```

## 📋 Разбор по элементам:

### 1. **Создание объекта Genius API:**
```python
self.genius = lyricsgenius.Genius(
    token,                      # API токен для доступа
    timeout=15,                 # Таймаут запроса (15 сек)
    retries=3,                  # Количество повторных попыток
    remove_section_headers=True, # Убирать заголовки секций [Chorus], [Verse]
    skip_non_songs=True,        # Пропускать не-песни (интервью, объявления)
    excluded_terms=["(Remix)", "(Live)", "(Instrumental)"]  # Исключить ремиксы, концерты, инструменталы
)
```

### 2. **Инициализация атрибутов класса:**
```python
self.db = LyricsDatabase(db_name)  # Создаем объект базы данных
self.session_stats = {"processed": 0, "added": 0, "skipped": 0, "errors": 0}  # Статистика сессии
```

### 3. **Настройка задержек (Rate Limiting):**
```python
self.min_delay = 3.0     # Минимальная пауза между запросами (3 сек)
self.max_delay = 7.0     # Максимальная пауза между запросами (7 сек)
self.error_delay = 15.0  # Пауза при ошибке (15 сек)
self.max_retries = 3     # Максимум попыток при ошибке
```

### 4. **Флаг остановки:**
```python
self.shutdown_requested = False  # Флаг для корректного завершения работы
```

### 5. **Обработка системных сигналов (Signal Handling):**
```python
signal.signal(signal.SIGINT, self._signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, self._signal_handler)  # Команда завершения
```

### 6. **Специальная обработка для Windows:**
```python
if sys.platform == "win32":  # Если операционная система Windows
    try:
        signal.signal(signal.SIGBREAK, self._signal_handler)  # Ctrl+Break в Windows
    except AttributeError:
        pass  # Если SIGBREAK недоступен, игнорируем
```

## 🔧 Ключевые концепции:

### **1. Composition (Композиция):**
```python
self.db = LyricsDatabase(db_name)  # SafeGeniusScraper "содержит" LyricsDatabase
# Класс использует другой класс как компонент
```

### **2. Configuration (Конфигурация):**
```python
# Все настройки в одном месте для легкого изменения
timeout=15,           # Можно изменить на 30 для медленного интернета
retries=3,            # Можно увеличить до 5 для надежности
excluded_terms=[...]  # Можно добавить новые типы для исключения
```

### **3. Signal Handling (Обработка сигналов):**
```python
# Что такое сигналы?
# SIGINT = Interrupt Signal (Ctrl+C)
# SIGTERM = Termination Signal (команда kill)
# SIGBREAK = Break Signal (Ctrl+Break в Windows)

signal.signal(signal.SIGINT, self._signal_handler)
# "Когда получишь сигнал SIGINT, вызови функцию self._signal_handler"
```

### **4. Cross-platform код:**
```python
if sys.platform == "win32":  # Проверка операционной системы
    # Код специально для Windows
else:
    # Код для Linux/Mac (неявно)
```

### **5. Exception Handling в инициализации:**
```python
try:
    signal.signal(signal.SIGBREAK, self._signal_handler)
except AttributeError:
    pass  # "Если этот сигнал не существует, ничего страшного"
```

## 🎯 Что происходит при создании объекта:

```python
# Когда мы пишем:
scraper = SafeGeniusScraper("токен", "database.db")

# Происходит следующее:
# 1. Создается объект Genius API с настройками
# 2. Создается объект базы данных
# 3. Инициализируется статистика: {"processed": 0, "added": 0, ...}
# 4. Устанавливаются временные задержки
# 5. Настраивается обработка Ctrl+C и других сигналов остановки
# 6. Объект готов к работе!
```

## 🛡️ Защитные механизмы:

### **1. Rate Limiting (Ограничение скорости):**
```python
self.min_delay = 3.0  # Не быстрее 1 запроса в 3 секунды
self.max_delay = 7.0  # Случайная пауза до 7 секунд
# Защищает от блокировки API
```

### **2. Retry Logic (Логика повторов):**
```python
self.max_retries = 3  # Попробовать 3 раза при ошибке
self.error_delay = 15.0  # Пауза 15 сек при ошибке
```

### **3. Graceful Shutdown (Корректное завершение):**
```python
self.shutdown_requested = False  # Флаг для остановки
# Позволяет сохранить данные перед выходом
```

## 💼 Вопросы для собеседования:

**Q1:** Что такое `self` в конструкторе?
**A:** Ссылка на создаваемый экземпляр класса. Позволяет устанавливать атрибуты объекта.

**Q2:** Зачем нужны сигналы (signal)?
**A:** Для корректного завершения программы при Ctrl+C, сохранения данных перед выходом.

**Q3:** Что означает `sys.platform == "win32"`?
**A:** Проверка операционной системы для выполнения специфичного для Windows кода.

**Q4:** Зачем исключать ремиксы и инструменталы?
**A:** Для получения оригинальных текстов песен, исключения дубликатов и низкокачественного контента.

**Q5:** Что такое Rate Limiting и зачем он нужен?
**A:** Ограничение частоты запросов к API для избежания блокировки за превышение лимитов.

**Q6:** Зачем `try-except` вокруг `signal.signal(signal.SIGBREAK, ...)`?
**A:** Потому что SIGBREAK есть только в Windows, в других ОС вызовет AttributeError.

## 🧠 Ассоциации:

### **Класс = Фабрика роботов:**
- `__init__` = инструкция по сборке робота
- `self.genius` = API модуль (мозг робота)
- `self.db` = память робота
- `self.session_stats` = счетчик выполненных задач
- Задержки = "не работай слишком быстро, чтобы не сломаться"

### **Signal Handling = Кнопка экстренной остановки:**
- Ctrl+C = красная кнопка "СТОП"
- `self._signal_handler` = что делать при нажатии кнопки
- `self.shutdown_requested = True` = поднять флаг "остановка"

### **Cross-platform код = Универсальный адаптер:**
- `if sys.platform == "win32"` = "если это розетка европейского типа"
- Windows имеет свои особенности, как разные типы розеток

Понятно? Есть вопросы по сигналам, композиции или инициализации?

Vastargazing: if sys.platform == "win32"
а если у нас windows 64 ? играет ли это роль

SIGBREAK
SIGTERM и тд, кому этот синтаксис принаджелит? Пайтон для работы с Windows объектами? Можно больше узнать о таком синтаксисе, может узнать больше команд.

поччему повторяется слово signal.signal ? 


GitHub Copilot: Отличные и очень глубокие вопросы! Давайте разберем каждый подробно:

## 1. 🖥️ `sys.platform == "win32"` и Windows 64-bit

**Короткий ответ:** НЕТ, разрядность НЕ играет роли!

```python
# На ВСЕХ версиях Windows (32-bit, 64-bit, ARM64):
sys.platform == "win32"  # Всегда True

# Примеры:
# Windows 10 64-bit → sys.platform = "win32"
# Windows 11 64-bit → sys.platform = "win32"  
# Windows Server 2019 64-bit → sys.platform = "win32"
```

**Почему так?** Историческая причина - название "win32" осталось от Windows API, который называется Win32 API даже на 64-битных системах.

### Проверка разрядности:
```python
import platform

# Операционная система
sys.platform          # "win32", "linux", "darwin" (macOS)

# Архитектура процессора  
platform.machine()    # "AMD64" (64-bit), "x86" (32-bit), "ARM64"

# Разрядность Python
platform.architecture()  # ('64bit', 'WindowsPE') или ('32bit', 'WindowsPE')

# Полная информация
platform.platform()   # 'Windows-10-10.0.19041-SP0'
```

## 2. 🚦 Сигналы (SIGINT, SIGTERM, SIGBREAK) - кому принадлежат?

**Это UNIX/POSIX стандарт**, который Python адаптировал для кроссплатформенности!

### Происхождение:
```python
# UNIX/Linux (1970е годы):
SIGINT  = 2   # Interrupt (Ctrl+C)
SIGTERM = 15  # Terminate (команда kill)
SIGKILL = 9   # Kill (принудительное завершение)
SIGHUP  = 1   # Hangup (отключение терминала)

# Windows добавил свои:
SIGBREAK = 21  # Ctrl+Break (только Windows)
```

### Больше сигналов:
```python
import signal

# Основные POSIX сигналы (работают на Linux/Mac/Windows):
signal.SIGINT   # 2  - Ctrl+C (прерывание)
signal.SIGTERM  # 15 - Команда завершения
signal.SIGABRT  # 6  - Аварийное завершение
signal.SIGFPE   # 8  - Ошибка вычислений (деление на 0)
signal.SIGSEGV  # 11 - Нарушение памяти

# Linux/Mac специфичные (НЕ работают на Windows):
signal.SIGUSR1  # 10 - Пользовательский сигнал 1
signal.SIGUSR2  # 12 - Пользовательский сигнал 2
signal.SIGPIPE  # 13 - Разорванный канал
signal.SIGCHLD  # 17 - Завершение дочернего процесса

# Windows специфичные:
signal.SIGBREAK # 21 - Ctrl+Break (только Windows)
```

### Практические примеры:
```python
import signal
import time

def signal_handler(signum, frame):
    print(f"Получен сигнал {signum}")
    if signum == signal.SIGINT:
        print("Это Ctrl+C!")
    elif signum == signal.SIGTERM:
        print("Команда завершения!")

# Регистрируем обработчики
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # kill command

# Отправить сигнал самому себе
signal.raise_signal(signal.SIGINT)  # Симуляция Ctrl+C
```

## 3. 🔄 Почему `signal.signal` повторяется?

**Это НЕ повтор, это ПАТТЕРН:** `модуль.функция(константа_модуля, обработчик)`

```python
import signal

# Структура: signal.signal(КАКОЙ_СИГНАЛ, ФУНКЦИЯ_ОБРАБОТЧИК)
signal.signal(signal.SIGINT, self._signal_handler)
#    ↑           ↑               ↑
# функция   константа      наша функция
```

### Разбор по частям:
```python
# 1. signal - это модуль (библиотека)
import signal

# 2. signal.signal() - это функция из модуля signal
def signal_function(signal_number, handler):
    # регистрирует обработчик для сигнала
    pass

# 3. signal.SIGINT - это константа из модуля signal
SIGINT = 2  # Номер сигнала

# Итого:
signal.signal(signal.SIGINT, my_handler)
# модуль.функция(модуль.константа, моя_функция)
```

### Аналогия с другими модулями:
```python
import os
os.path.join(os.sep, "folder")  # os повторяется, но это разные вещи
#  ↑    ↑       ↑
# модуль.атрибут модуль.константа

import math
result = math.sqrt(math.pi)  # math повторяется
#         ↑         ↑
#     функция   константа
```

## 4. 📚 Больше команд для работы с системой:

### Windows-специфичные (через модуль `os`):
```python
import os
import subprocess

# Информация о системе
os.name          # 'nt' (Windows), 'posix' (Linux/Mac)
os.environ       # Переменные окружения
os.getlogin()    # Имя пользователя

# Работа с процессами
os.getpid()      # ID текущего процесса
os.kill(pid, signal.SIGTERM)  # Завершить процесс

# Windows команды через subprocess
subprocess.run(['tasklist'])                    # Список процессов
subprocess.run(['taskkill', '/PID', '1234'])   # Убить процесс
subprocess.run(['systeminfo'])                 # Информация о системе
```

### Кроссплатформенные команды:
```python
import platform
import psutil  # Внешняя библиотека (pip install psutil)

# Информация о системе
platform.system()     # 'Windows', 'Linux', 'Darwin'
platform.processor()  # 'Intel64 Family 6 Model 142 Stepping 9, GenuineIntel'
platform.node()       # Имя компьютера

# Работа с процессами (psutil)
psutil.cpu_count()     # Количество ядер CPU
psutil.virtual_memory().total  # Объем RAM
psutil.disk_usage('C:').total  # Размер диска
```

## 💼 Вопросы для собеседования:

**Q1:** Почему `sys.platform == "win32"` на Windows 64-bit?
**A:** Историческая совместимость с Win32 API, который используется на всех версиях Windows.

**Q2:** Откуда берутся SIGINT, SIGTERM?
**A:** Из UNIX/POSIX стандарта 1970х годов, адаптированного Python для кроссплатформенности.

**Q3:** Что делает `signal.signal(signal.SIGINT, handler)`?
**A:** Регистрирует функцию-обработчик для сигнала прерывания (Ctrl+C).

**Q4:** Какие сигналы работают только на Windows?
**A:** SIGBREAK (Ctrl+Break). Большинство других сигналов - POSIX стандарт.

**Q5:** Как отличить Windows от Linux в Python?
**A:** `sys.platform == "win32"` (Windows) или `sys.platform.startswith("linux")` (Linux).

## 🧠 Ассоциации:

### **Сигналы = Дорожные знаки:**
- SIGINT (Ctrl+C) = знак "СТОП"
- SIGTERM = знак "конец дороги" (корректная остановка)
- SIGKILL = "аварийная остановка" (принудительно)

### **signal.signal = Телефонная связь:**
```python
signal.signal(signal.SIGINT, my_handler)
# "Когда зазвонит телефон с номером SIGINT, переключи на функцию my_handler"
```

### **Повторение signal = Адрес:**
```python
signal.signal(signal.SIGINT, handler)
# Улица.Дом(Улица.Номер_дома, жилец)
# "На улице Signal, в доме signal, найди константу SIGINT"
```

Понятно? Есть еще вопросы по сигналам или системным командам?

Vastargazing: класс
едем дальше

GitHub Copilot: Отлично! Продолжаем изучение. Какой блок кода разбираем дальше?

Вот что у нас есть после конструктора `SafeGeniusScraper`:

1. **`_signal_handler`** - обработчик сигналов
2. **`safe_delay`** - умная система пауз
3. **`clean_lyrics`** - очистка текстов от мусора (регулярные выражения)
4. **`_is_valid_lyrics`** - проверка качества текстов
5. **`scrape_artist_songs`** - основная логика скрапинга
6. **`show_current_results`** - показ результатов
7. **`run_scraping_session`** - запуск полной сессии

Выбери любой блок, который хочешь разобрать, или я предложу следующий по порядку! 

Какой интересует больше всего? 🚀

Vastargazing: 1.

GitHub Copilot: Отлично! Разбираем **`_signal_handler`** - обработчик сигналов: ## 🚦 Анализ `_signal_handler`

```python
def _signal_handler(self, signum, frame):
    logger.info(f"\nПолучен сигнал {signum}. Завершение работы...")
    self.shutdown_requested = True
```

## 📋 Разбор по элементам:

### 1. **Имя метода с подчеркиванием:**
```python
def _signal_handler(self, signum, frame):
#   ↑
# Подчеркивание = "приватный" метод (условность Python)
```

**Соглашение:** Методы с `_` предназначены для внутреннего использования класса.

### 2. **Параметры метода:**
```python
def _signal_handler(self, signum, frame):
#                   ↑      ↑       ↑
#                 обычно  номер  контекст
#                         сигнала выполнения
```

- **`self`** - ссылка на объект класса
- **`signum`** - номер полученного сигнала (2 для SIGINT, 15 для SIGTERM)
- **`frame`** - объект с информацией о текущем состоянии программы

### 3. **Логирование события:**
```python
logger.info(f"\nПолучен сигнал {signum}. Завершение работы...")
```

**Что происходит:**
- `\n` - переход на новую строку для читаемости
- `{signum}` - подставляется номер сигнала (например, "2")
- Сообщение записывается в лог файл И выводится в консоль

### 4. **Установка флага остановки:**
```python
self.shutdown_requested = True
```

**Это ключевая строка!** Устанавливает флаг, который проверяется во всех циклах программы.

## 🎯 Как работает пошагово:

### **Сценарий 1:** Пользователь нажал Ctrl+C
```python
# 1. Пользователь: Ctrl+C
# 2. Операционная система: "Отправляю SIGINT (сигнал 2)"
# 3. Python: "Получил SIGINT, вызываю _signal_handler(self, 2, frame_info)"
# 4. _signal_handler выполняется:
logger.info(f"\nПолучен сигнал 2. Завершение работы...")  # Записываем в лог
self.shutdown_requested = True  # Поднимаем флаг остановки

# 5. Основные циклы программы видят флаг и начинают корректное завершение
```

### **Сценарий 2:** Команда kill в терминале
```bash
# В другом терминале:
kill -TERM 1234  # Отправить SIGTERM процессу с PID 1234

# Результат:
# _signal_handler(self, 15, frame_info) вызывается
# Логируется: "Получен сигнал 15. Завершение работы..."
# self.shutdown_requested = True
```

## 🔄 Как флаг используется в программе:

### **В циклах обработки:**
```python
# В методе scrape_artist_songs:
for i, song in enumerate(artist.songs):
    if self.shutdown_requested:  # ← Проверяем флаг
        logger.info(f"Stopping at song {i+1}/{len(artist.songs)}")
        break  # Выходим из цикла
        
    # Обработка песни...

# В методе safe_delay:
for _ in range(intervals):
    if self.shutdown_requested:  # ← Проверяем флаг
        return  # Прерываем задержку
    time.sleep(1)
```

## 🛡️ Зачем нужен graceful shutdown:

### **БЕЗ обработчика сигналов:**
```python
# Пользователь: Ctrl+C
# Программа: МГНОВЕННО останавливается
# Результат: 
# - Несохраненные данные потеряны
# - База данных может повредиться
# - Соединения не закрыты
```

### **С обработчиком сигналов:**
```python
# Пользователь: Ctrl+C
# Программа: 
# 1. "Получил сигнал, начинаю корректное завершение"
# 2. Завершаем текущую операцию
# 3. Сохраняем данные в базу
# 4. Закрываем соединения
# 5. Показываем финальную статистику
# 6. Корректно завершаемся
```

## 📊 Параметр `frame` (дополнительно):

```python
def _signal_handler(self, signum, frame):
    # frame содержит информацию о текущем состоянии:
    print(f"Файл: {frame.f_code.co_filename}")      # Какой файл выполнялся
    print(f"Строка: {frame.f_lineno}")             # На какой строке
    print(f"Функция: {frame.f_code.co_name}")      # В какой функции
    
    self.shutdown_requested = True
```

## 💼 Вопросы для собеседования:

**Q1:** Что означает подчеркивание в `_signal_handler`?
**A:** Соглашение Python о "приватном" методе - предназначен для внутреннего использования класса.

**Q2:** Какие параметры принимает обработчик сигналов?
**A:** `signum` (номер сигнала) и `frame` (контекст выполнения программы).

**Q3:** Зачем нужен graceful shutdown?
**A:** Для корректного завершения: сохранения данных, закрытия соединений, избежания повреждения файлов.

**Q4:** Что происходит при `self.shutdown_requested = True`?
**A:** Устанавливается флаг, который проверяется в циклах программы для корректного завершения.

**Q5:** Какой сигнал отправляется при Ctrl+C?
**A:** SIGINT (номер 2).

## 🧠 Ассоциации:

### **Signal Handler = Пожарная сигнализация:**
- Сигнал (SIGINT) = звук сирены
- _signal_handler = автоматическая система пожаротушения
- self.shutdown_requested = включение аварийного протокола эвакуации
- Остальная программа = люди спокойно эвакуируются по плану

### **Graceful Shutdown = Посадка самолета:**
- Без обработчика = аварийное приземление (crash)
- С обработчиком = плановая посадка (все процедуры выполнены)

Понятно? Переходим к следующему блоку **`safe_delay`**? 🚀

Vastargazing: Переходим к следующему блоку safe_delay

GitHub Copilot: Отлично! Переходим к **`safe_delay`** - очень умной системе пауз: ## ⏱️ Анализ `safe_delay` - умная система пауз

```python
def safe_delay(self, is_error: bool = False):
    delay = self.error_delay if is_error else random.uniform(self.min_delay, self.max_delay)
    intervals = int(delay)
    remainder = delay - intervals
    for _ in range(intervals):
        if self.shutdown_requested:
            return
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown_requested = True
            return
    if remainder > 0 and not self.shutdown_requested:
        try:
            time.sleep(remainder)
        except KeyboardInterrupt:
            self.shutdown_requested = True
            return
    logger.debug(f"Пауза: {delay:.1f}с")
```

## 📋 Разбор по элементам:

### 1. **Расчет времени задержки:**
```python
delay = self.error_delay if is_error else random.uniform(self.min_delay, self.max_delay)
```

**Два режима работы:**
- **Обычная пауза:** случайная от 3.0 до 7.0 секунд
- **Пауза при ошибке:** фиксированная 15.0 секунд

**Зачем случайная задержка?** Чтобы не выглядеть как бот для API сервера.

### 2. **Разделение времени на части:**
```python
intervals = int(delay)        # Целая часть (например, 5 из 5.7)
remainder = delay - intervals # Остаток (например, 0.7 из 5.7)
```

**Пример:**
```python
# Если delay = 5.7 секунд:
intervals = int(5.7) = 5      # 5 целых секунд
remainder = 5.7 - 5 = 0.7     # 0.7 секунды остаток
```

### 3. **Прерываемая пауза по секундам:**
```python
for _ in range(intervals):    # Повторяем 5 раз (для примера выше)
    if self.shutdown_requested:
        return                # Если получен сигнал остановки - выходим
    try:
        time.sleep(1)         # Спим 1 секунду
    except KeyboardInterrupt:
        self.shutdown_requested = True
        return
```

### 4. **Обработка остатка:**
```python
if remainder > 0 and not self.shutdown_requested:
    try:
        time.sleep(remainder)  # Доспим оставшиеся 0.7 секунды
    except KeyboardInterrupt:
        self.shutdown_requested = True
        return
```

### 5. **Логирование:**
```python
logger.debug(f"Пауза: {delay:.1f}с")  # "Пауза: 5.7с"
```

## 🎯 Зачем такая сложная система?

### **Проблема простой паузы:**
```python
# Простая пауза (ПЛОХО):
time.sleep(15)  

# Проблемы:
# 1. Нельзя прервать Ctrl+C - нужно ждать 15 секунд
# 2. Программа "заморожена" на 15 секунд
# 3. Нет отзывчивости на сигналы остановки
```

### **Решение smart паузы:**
```python
# Умная пауза (ХОРОШО):
safe_delay(is_error=True)  # 15 секунд, но прерываемая

# Преимущества:
# 1. Можно прервать в любой момент
# 2. Проверка остановки каждую секунду
# 3. Отзывчивая на Ctrl+C
```

## 🔄 Как работает пошагово:

### **Сценарий 1:** Обычная пауза
```python
# Вызов: self.safe_delay()
# delay = random.uniform(3.0, 7.0) = 4.3 секунды

intervals = int(4.3) = 4       # 4 целые секунды
remainder = 4.3 - 4 = 0.3      # 0.3 секунды остаток

# Цикл 4 раза:
# Секунда 1: sleep(1), проверка shutdown_requested
# Секунда 2: sleep(1), проверка shutdown_requested  
# Секунда 3: sleep(1), проверка shutdown_requested
# Секунда 4: sleep(1), проверка shutdown_requested

# Остаток: sleep(0.3)
# Итого: 4.3 секунды с возможностью прерывания
```

### **Сценарий 2:** Пауза при ошибке
```python
# Вызов: self.safe_delay(is_error=True)
# delay = 15.0 секунд (фиксированная)

intervals = int(15.0) = 15     # 15 целых секунд
remainder = 15.0 - 15 = 0.0    # Остатка нет

# Цикл 15 раз по 1 секунде с проверкой остановки
```

### **Сценарий 3:** Прерывание паузы
```python
# Во время выполнения safe_delay():
# Пользователь нажимает Ctrl+C на 3-й секунде

# Секунда 1: sleep(1) ✓
# Секунда 2: sleep(1) ✓  
# Секунда 3: Ctrl+C → KeyboardInterrupt
#           → self.shutdown_requested = True
#           → return (выход из функции)
# Оставшиеся секунды НЕ выполняются
```

## ⚡ Виды задержек в проекте:

### **Обычные паузы:**
```python
# После обработки каждой песни:
self.safe_delay()  # 3-7 секунд (random)

# Между артистами:
artist_delay = random.uniform(5, 10)  # 5-10 секунд
```

### **Паузы при ошибках:**
```python
# При таймауте:
self.safe_delay(is_error=True)  # 15 секунд

# При rate limit:
self.safe_delay(is_error=True)  # 15 секунд
time.sleep(60)                  # + еще 60 секунд
```

## 🛡️ Защитные механизмы:

### **1. Rate Limiting Protection:**
```python
# Случайные паузы 3-7 секунд между запросами
# API видит "человеческое" поведение, а не бота
```

### **2. Graceful Interruption:**
```python
# Каждую секунду проверяем: не нужно ли остановиться?
if self.shutdown_requested:
    return  # Немедленный выход
```

### **3. Exception Handling:**
```python
try:
    time.sleep(1)
except KeyboardInterrupt:  # Дополнительная защита от Ctrl+C
    self.shutdown_requested = True
    return
```

## 💼 Вопросы для собеседования:

**Q1:** Зачем делить задержку на секундные интервалы?
**A:** Для возможности прерывания паузы - проверяем флаг остановки каждую секунду.

**Q2:** Почему используется `random.uniform()` для задержки?
**A:** Чтобы имитировать человеческое поведение и избежать блокировки API за "роботоподобность".

**Q3:** Что такое `remainder` и зачем он нужен?
**A:** Дробная часть времени задержки. Например, из 5.7 секунд остаток 0.7 секунды.

**Q4:** В чем разница между обычной и error паузой?
**A:** Обычная: случайная 3-7 сек. Error: фиксированная 15 сек (больше времени на восстановление).

**Q5:** Зачем двойная обработка KeyboardInterrupt?
**A:** Дополнительная защита - и через signal handler, и через exception в sleep().

## 🧠 Ассоциации:

### **Safe Delay = Умный светофор:**
- Обычная задержка = зеленый свет случайной длительности
- Error задержка = красный свет фиксированной длительности
- Проверка каждую секунду = датчик движения (можно прервать)
- remainder = "доходим до конца перехода"

### **Прерываемая пауза = Лифт с кнопкой "Стоп":**
```python
# Обычный sleep = лифт без кнопок (застрял до этажа)
time.sleep(15)

# Safe delay = лифт с кнопкой экстренной остановки
# Можно остановиться на любом "этаже" (секунде)
```

### **Random delay = Антибот защита:**
- Бот: запрос каждые ровно 5.0 секунд (подозрительно)
- Человек: запросы через 3.2, 6.7, 4.1, 5.9 секунд (естественно)

Понятно? Переходим к **`clean_lyrics`** - очистке текстов регулярными выражениями?

Отлично! Переходим к **`clean_lyrics`** - очистке текстов с помощью регулярных выражений: ## 🧹 Анализ `clean_lyrics` - мастер очистки текстов

```python
def clean_lyrics(self, lyrics: str) -> str:
```

Этот метод превращает "грязный" текст с сайта Genius в чистые тексты песен!

## 📋 Разбор по шагам:

### 1. **Проверка на пустоту:**
```python
if not lyrics:
    return ""
```
**Защита от ошибок:** если lyrics = None или пустая строка, возвращаем пустую строку.

### 2. **Удаление информации о контрибьюторах:**
```python
lyrics = re.sub(r"^\d+\s+Contributors.*?Lyrics", "", lyrics, flags=re.MULTILINE | re.DOTALL)
```

**Что удаляет:** "81 Contributors\nTranscribed by John\nProduced by Dr. Dre\nLyrics"

**Разбор регулярки:**
- `^` - начало строки
- `\d+` - одна или больше цифр
- `\s+` - один или больше пробелов
- `Contributors.*?Lyrics` - от "Contributors" до "Lyrics" (ленивый поиск)
- **Flags:** `MULTILINE` (^ работает для каждой строки) + `DOTALL` (. включает \n)

### 3. **Удаление блока переводов:**
```python
lyrics = re.sub(r"Translations[A-Za-z]+", "", lyrics, flags=re.MULTILINE)
```

**Что удаляет:** "TranslationsEnglishEspañolFrançais"

**Разбор:**
- `Translations` - точное слово
- `[A-Za-z]+` - любые буквы после него

### 4. **Удаление описания артиста:**
```python
lyrics = re.sub(r"Lyrics[A-Z].*?Read More\s*", "", lyrics, flags=re.DOTALL)
```

**Что удаляет:** Блок с описанием между "Lyrics" и "Read More"

### 5. **Удаление стандартных блоков Genius:**
```python
lyrics = re.sub(r"(?i)(Embed|Submitted by [^\n]*|Written by [^\n]*|You might also like).*$", "", lyrics, flags=re.DOTALL)
```

**Разбор сложной регулярки:**
- `(?i)` - игнорировать регистр (case-insensitive)
- `(Embed|Submitted by...|Written by...|You might also like)` - альтернативы через `|`
- `[^\n]*` - любые символы кроме переноса строки
- `.*$` - до конца текста

### 6. **Удаление URL ссылок:**
```python
lyrics = re.sub(r"https?://[^\s]+", "", lyrics)
```

**Разбор:**
- `https?` - "http" или "https" (`?` = предыдущий символ опционален)
- `://` - точные символы
- `[^\s]+` - любые символы кроме пробелов

### 7. **Удаление блоков в скобках:**
```python
lyrics = re.sub(r"\[.*?\]", "", lyrics)
```

**Что удаляет:** [Chorus], [Verse 1], [Bridge], [Intro] и т.д.

**Разбор:**
- `\[` - экранированная открывающая скобка
- `.*?` - любые символы (ленивый поиск)
- `\]` - экранированная закрывающая скобка

### 8. **Нормализация переносов строк:**
```python
lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)      # 3+ переноса → 2 переноса
lyrics = re.sub(r"\n{2,}", "\n", lyrics.strip()) # 2+ переноса → 1 перенос
```

### 9. **Финальная очистка:**
```python
return lyrics.strip()  # Убираем пробелы в начале и конце
```

## 🎯 Пример работы:

### **Грязный текст с Genius:**
```
81 Contributors
Transcribed by User123
Produced by Dr. Dre
Lyrics

[Intro]
Yeah, this is the intro
TranslationsEnglishEspañol

[Verse 1]  
First line of the song
Second line here
https://genius.com/annotations/123456

[Chorus]
This is the chorus
Submitted by: CoolUser2023
Written by: Artist Name

You might also like
Other Songs by This Artist
```

### **Чистый результат:**
```
Yeah, this is the intro

First line of the song
Second line here

This is the chorus
```

## 🔧 Ключевые концепции регулярных выражений:

### **Основные символы:**
```python
.     # Любой символ
*     # 0 или больше предыдущего символа
+     # 1 или больше предыдущего символа  
?     # 0 или 1 предыдущего символа (опционально)
^     # Начало строки
$     # Конец строки
\d    # Цифра [0-9]
\s    # Пробельный символ
[ABC] # Любой из символов A, B, C
[^A]  # Любой символ кроме A
```

### **Флаги:**
```python
re.MULTILINE  # ^ и $ работают для каждой строки
re.DOTALL     # . включает символ переноса строки \n
re.IGNORECASE # Игнорировать регистр (или (?i) в начале)
```

### **Группы и альтернативы:**
```python
(abc|def)     # "abc" ИЛИ "def"
(?i)          # Включить игнорирование регистра
.*?           # Ленивый поиск (минимальное совпадение)
.*            # Жадный поиск (максимальное совпадение)
```

## 💼 Вопросы для собеседования:

**Q1:** Что делает флаг `re.DOTALL`?
**A:** Заставляет символ `.` включать переносы строк `\n` в поиск.

**Q2:** В чем разница между `.*` и `.*?`?
**A:** `.*` - жадный (максимальное совпадение), `.*?` - ленивый (минимальное совпадение).

**Q3:** Что означает `[^\n]*`?
**A:** Любые символы кроме перенося строки, 0 или больше раз.

**Q4:** Зачем экранировать `\[` и `\]`?
**A:** Потому что `[]` имеют специальное значение в regex (символьный класс).

**Q5:** Что делает `(?i)` в начале регулярки?
**A:** Включает режим игнорирования регистра для всей регулярки.

## 🧠 Ассоциации:

### **Regex = Поисковая операция спецслужб:**
- `.*` = "найти ВСЁ между точками А и Б"
- `[^\n]*` = "найти всё кроме конца строки"  
- `(?i)` = "игнорировать заглавные/строчные буквы"
- `|` = "ИЛИ" (альтернатива)

### **Cleaning = Уборка дома:**
- Исходный текст = грязная комната
- Каждый `re.sub()` = один этап уборки
- Сначала убираем крупный мусор (Contributors)
- Потом мелкий (URL, скобки)
- В конце полируем (переносы строк, пробелы)

### **Флаги = Настройки пылесоса:**
- `MULTILINE` = "работать на каждом этаже"
- `DOTALL` = "включить турбо-режим"
- `IGNORECASE` = "не обращать внимание на размер мусора"

Понятно? Переходим к **`_is_valid_lyrics`** - проверке качества текстов? ✅ 🧹

Отлично! Переходим к **`_is_valid_lyrics`** - умной проверке качества текстов: ## ✅ Анализ `_is_valid_lyrics` - контроль качества

```python
def _is_valid_lyrics(self, lyrics: str) -> bool:
    if not lyrics:
        return False
    lyrics = lyrics.strip()
    if len(lyrics) < 100 or len(lyrics.split()) < 20:
        return False
    instrumental_markers = ["instrumental", "no lyrics", "без слов", "music only", "beat only"]
    return not any(marker in lyrics.lower() for marker in instrumental_markers)
```

Этот метод = **фильтр качества** - пропускает только хорошие тексты!

## 📋 Разбор по критериям:

### 1. **Проверка на существование:**
```python
if not lyrics:
    return False
```

**Отфильтровывает:**
- `None`
- `""` (пустая строка)
- `False`, `0` (любые "ложные" значения)

### 2. **Очистка от пробелов:**
```python
lyrics = lyrics.strip()
```

**Убирает:** пробелы, табы, переносы строк в начале и конце.

### 3. **Проверка минимального размера:**
```python
if len(lyrics) < 100 or len(lyrics.split()) < 20:
    return False
```

**Два критерия одновременно:**
- `len(lyrics) < 100` - меньше 100 **символов**
- `len(lyrics.split()) < 20` - меньше 20 **слов**

**Примеры отфильтрованных текстов:**
```python
# Слишком короткий:
"La la la la"  # 11 символов, 4 слова → REJECT

# Слишком мало слов:
"Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
# 100+ символов, но 1 слово → REJECT
```

### 4. **Проверка на инструментальные треки:**
```python
instrumental_markers = ["instrumental", "no lyrics", "без слов", "music only", "beat only"]
return not any(marker in lyrics.lower() for marker in instrumental_markers)
```

**Разбор сложной строки:**
- `lyrics.lower()` - переводим в нижний регистр
- `marker in lyrics.lower()` - проверяем каждый маркер
- `any(...)` - возвращает `True`, если ХОТЯ БЫ один маркер найден
- `not any(...)` - инвертируем результат

## 🎯 Пошаговые примеры:

### **Пример 1: Хороший текст**
```python
lyrics = """
Yo, listen up, here's the story
About a little guy that lives in a blue world
And all day and all night and everything he sees is just blue
Like him, inside and outside
Blue his house with a blue little window
"""

# Проверки:
# 1. not lyrics → False (текст есть)
# 2. len(lyrics) → 200+ символов ✓
# 3. len(lyrics.split()) → 40+ слов ✓  
# 4. Нет инструментальных маркеров ✓
# Результат: True (принять)
```

### **Пример 2: Слишком короткий**
```python
lyrics = "Yeah yeah yeah"

# Проверки:
# 1. not lyrics → False (текст есть)
# 2. len(lyrics) → 14 символов < 100 ❌
# Результат: False (отклонить)
```

### **Пример 3: Инструментальный трек**
```python
lyrics = """
This is an instrumental track with no lyrics.
The beat goes on for 3 minutes without any vocals.
Music only, no words to sing along.
"""

# Проверки:
# 1. not lyrics → False (текст есть)
# 2. len(lyrics) → 100+ символов ✓
# 3. len(lyrics.split()) → 20+ слов ✓
# 4. "no lyrics" найдено в тексте ❌
# Результат: False (отклонить)
```

### **Пример 4: Русский инструментал**
```python
lyrics = "Этот трек без слов, только бит"

# Проверки:
# 1-3. Проходит базовые проверки
# 4. "без слов" найдено ❌
# Результат: False (отклонить)
```

## 🔧 Работа с `any()` и `not`:

### **Функция `any()`:**
```python
# any() возвращает True, если ХОТЯ БЫ один элемент True
any([False, False, True, False])   # True
any([False, False, False])         # False
any([])                           # False

# В нашем случае:
markers = ["instrumental", "no lyrics"]
text = "this song has no lyrics"

# Проверяем каждый маркер:
checks = [
    "instrumental" in text.lower(),  # False
    "no lyrics" in text.lower()      # True
]
any(checks)  # True (найден маркер)
```

### **Логика `not any()`:**
```python
# Мы хотим принять текст, ТОЛЬКО если НЕТ инструментальных маркеров
has_instrumental_markers = any(marker in lyrics.lower() for marker in markers)
is_valid = not has_instrumental_markers

# Короче:
return not any(marker in lyrics.lower() for marker in instrumental_markers)
```

## 🛡️ Зачем нужна валидация:

### **Проблемы без валидации:**
```python
# База данных засоряется:
# 1. "..." (3 символа)
# 2. "Instrumental" (пустые тексты)
# 3. "La la la" (бесполезный контент)
# 4. Рекламные тексты
# 5. Описания альбомов вместо текстов
```

### **С валидацией:**
```python
# В базу попадают только:
# 1. Полноценные тексты песен (100+ символов)
# 2. С достаточным количеством слов (20+)
# 3. Без инструментальных маркеров
# 4. Качественный контент для анализа
```

## 💼 Вопросы для собеседования:

**Q1:** Зачем проверять И количество символов, И количество слов?
**A:** Защита от "aaaaaaa..." (много символов, мало слов) и "a b c d..." (мало символов, много односимвольных слов).

**Q2:** Что делает `any()` функция?
**A:** Возвращает `True`, если хотя бы один элемент в последовательности истинный.

**Q3:** Зачем `lyrics.lower()` при проверке маркеров?
**A:** Для регистронезависимого поиска: "INSTRUMENTAL" = "instrumental" = "Instrumental".

**Q4:** Почему используется `not any()` вместо `all()`?
**A:** Мы проверяем ОТСУТСТВИЕ маркеров. `not any()` = "ни одного маркера нет".

**Q5:** Какие тексты пройдут валидацию?
**A:** Тексты с 100+ символами, 20+ словами, без инструментальных маркеров.

## 🧠 Ассоциации:

### **Валидация = Контроль качества на фабрике:**
- Входящий текст = сырье с конвейера  
- Проверка размера = "достаточно ли материала?"
- Проверка маркеров = "нет ли брака?"
- `return True` = "пропустить в производство"
- `return False` = "отправить в утиль"

### **any() = Детектор металла:**
```python
# Ищем металлические предметы в багаже:
metal_items = ["knife", "gun", "keys"]
bag_contents = "wallet phone keys passport"

# any() = пищит, если находит ХОТЯ БЫ один металл
detected = any(item in bag_contents for item in metal_items)  # True (keys)
```

### **Многоуровневая проверка = Экзамен:**
1. **Явка:** `if not lyrics` = "студент пришел?"
2. **Минимум слов:** проверка на списывание
3. **Объем:** проверка на полноту ответа  
4. **Содержание:** проверка на соответствие теме

Понятно? Готов к следующему блоку - **основной логике скрапинга** `scrape_artist_songs`? 🚀

Отлично! Переходим к **`scrape_artist_songs`** - самому сложному и важному методу - сердцу всего скрапинга! 🎵 ## 🎵 Анализ `scrape_artist_songs` - главного конвейера скрапинга

Это **самый сложный метод** в программе! Он объединяет все предыдущие компоненты в единую систему.

## 📋 Структура метода:

```python
def scrape_artist_songs(self, artist_name: str, max_songs: int = 500) -> int:
    # 1. Инициализация переменных
    # 2. Цикл повторных попыток (retry loop)
    #    ├── Поиск артиста в Genius API
    #    ├── Цикл обработки песен
    #    │   ├── Проверка дубликатов
    #    │   ├── Очистка и валидация текстов  
    #    │   ├── Сохранение в БД
    #    │   ├── Статистика и логи
    #    │   └── Умная пауза
    #    └── Обработка ошибок API
    # 3. Возврат количества добавленных песен
```

## 🔧 Разбор по блокам:

### **1. Инициализация:**
```python
added_count = 0    # Счетчик добавленных песен
retry_count = 0    # Счетчик попыток при ошибках

logger.info(f"Starting processing artist: {artist_name}")
```

### **2. Основной цикл повторов:**
```python
while retry_count < self.max_retries and not self.shutdown_requested:
```

**Условия продолжения:**
- `retry_count < self.max_retries` - не превысили лимит попыток (3)
- `not self.shutdown_requested` - не получили сигнал остановки

### **3. Поиск артиста в API:**
```python
artist = self.genius.search_artist(
    artist_name, 
    max_songs=max_songs,     # До 500 песен
    sort="popularity",       # Сортировка по популярности 
    get_full_info=False      # Базовая информация (быстрее)
)
```

### **4. Проверка результата поиска:**
```python
if not artist or not artist.songs:
    logger.warning(f"Artist {artist_name} not found")
    return 0  # Артист не найден, возвращаем 0
```

### **5. Главный цикл обработки песен:**
```python
for i, song in enumerate(artist.songs):
    # enumerate даёт номер (i) и объект песни (song)
```

**Что получаем:**
- `i` = 0, 1, 2, 3... (номер песни)
- `song` = объект с полями: title, url, lyrics, id

### **6. Проверка остановки:**
```python
if self.shutdown_requested:
    logger.info(f"Stopping at song {i+1}/{len(artist.songs)} for {artist_name}")
    break  # Выходим из цикла песен
```

### **7. Проверка дубликатов:**
```python
if self.db.song_exists(url=song.url):
    logger.debug(f"Skip (duplicate): {song.title}")
    self.session_stats["skipped"] += 1
    continue  # Переходим к следующей песне
```

### **8. Очистка и валидация:**
```python
lyrics = self.clean_lyrics(song.lyrics)      # Очищаем от мусора
if not self._is_valid_lyrics(lyrics):        # Проверяем качество
    logger.debug(f"Skip (invalid lyrics): {song.title}")
    self.session_stats["skipped"] += 1
    continue
```

### **9. Сохранение в базу данных:**
```python
if self.db.add_song(artist_name, song.title, lyrics, song.url, 
                   song.id if hasattr(song, 'id') else None):
    # Успешно добавлено
    added_count += 1
    self.session_stats["added"] += 1
    word_count = len(lyrics.split())
    logger.info(f"Added: {artist_name} - {song.title} ({word_count} words)")
else:
    # Не удалось добавить (обычно дубликат)
    self.session_stats["skipped"] += 1
```

### **10. Промежуточная статистика:**
```python
if self.session_stats["added"] % 5 == 0:  # Каждые 5 песен
    current_stats = self.db.get_stats()
    logger.info(f"Stats: {current_stats['total_songs']} songs in database")
```

### **11. Прогресс обработки:**
```python
self.session_stats["processed"] += 1
if (i + 1) % 10 == 0:  # Каждые 10 песен
    logger.info(f"Processed {i + 1}/{len(artist.songs)} songs for {artist_name}")
```

### **12. Умная пауза:**
```python
if self.shutdown_requested:
    break  # Проверяем остановку перед паузой
    
self.safe_delay()  # Пауза 3-7 секунд
```

### **13. Обработка ошибок песен:**
```python
except Exception as timeout_e:
    if "timeout" in str(timeout_e).lower():
        logger.error(f"Timeout for {song.title}: {timeout_e}")
        self.session_stats["errors"] += 1
        self.safe_delay(is_error=True)  # 15 секунд пауза
    else:
        raise timeout_e  # Перебрасываем неизвестную ошибку
```

### **14. Обработка ошибок API:**
```python
except Exception as e:
    if "rate limit" in str(e).lower() or "429" in str(e):
        # Rate Limit - слишком много запросов
        logger.error(f"Rate Limit for {artist_name}: {e}")
        self.safe_delay(is_error=True)  # 15 сек
        time.sleep(60)                  # + еще 60 сек
        retry_count += 1
    else:
        # Другая ошибка
        retry_count += 1
        logger.error(f"Error with artist {artist_name} (attempt {retry_count}): {e}")
        self.safe_delay(is_error=True)
```

## 🎯 Схема работы с примером:

### **Вход:** `scrape_artist_songs("Eminem", 500)`

```
1. 🔍 Поиск "Eminem" в Genius API
   ├── Найдено 500 песен
   └── Список: ["Lose Yourself", "Stan", "The Real Slim Shady", ...]

2. 🔄 Цикл по каждой песне:
   
   Песня 1: "Lose Yourself"
   ├── ✅ Проверка дубликата: нет в БД
   ├── 🧹 Очистка текста: убрали [Verse 1], [Chorus]
   ├── ✅ Валидация: 400+ слов, качественный текст
   ├── 💾 Сохранение в БД: успешно
   ├── 📊 Статистика: added +1, processed +1
   └── ⏱️ Пауза: 4.2 секунды
   
   Песня 2: "Stan"
   ├── ❌ Проверка дубликата: уже есть в БД
   ├── 📊 Статистика: skipped +1, processed +1
   └── ➡️ Переход к следующей
   
   ...и так далее...

3. 📈 Результат: добавлено 347 новых песен
```

## 💼 Вопросы для собеседования:

**Q1:** Зачем нужен цикл `while retry_count < self.max_retries`?
**A:** Для повторных попыток при временных ошибках API (rate limit, сетевые проблемы).

**Q2:** Что означает `enumerate(artist.songs)`?
**A:** Возвращает пары (индекс, элемент): (0, song1), (1, song2), (2, song3)...

**Q3:** Зачем проверять `hasattr(song, 'id')`?
**A:** Не все объекты песен могут иметь атрибут 'id', проверяем существование перед использованием.

**Q4:** В чем разница между `continue` и `break`?
**A:** `continue` пропускает текущую итерацию цикла, `break` полностью выходит из цикла.

**Q5:** Зачем два уровня обработки исключений?
**A:** Внутренний try-catch для ошибок отдельных песен, внешний - для ошибок всего артиста.

## 🧠 Ассоциации:

### **Метод = Фабрика по производству:**
- **Артист** = заказ на производство
- **Песни** = сырье с конвейера
- **Проверка дубликатов** = контроль качества "уже делали?"
- **Очистка текстов** = обработка сырья
- **Валидация** = финальная проверка качества
- **База данных** = склад готовой продукции
- **Статистика** = отчеты производства
- **Паузы** = перерывы рабочих
- **Retry logic** = "если что-то сломалось, попробуй еще раз"

### **Обработка ошибок = Служба безопасности:**
- **Rate Limit** = "слишком быстро работаешь, отдохни"
- **Timeout** = "связь с поставщиком потеряна"
- **Retry** = "попробуй связаться еще раз"
- **Max retries** = "если 3 раза не получилось, прекращай"

Готов к **итоговым вопросам** по всему методу или переходим к следующему блоку? 🚀

Отлично! Вот **итоговые вопросы** по методу `scrape_artist_songs` - проверим глубину понимания! 🧠

## 🎯 Итоговые вопросы по `scrape_artist_songs`:

### **1. Архитектурные вопросы:**

**Q1:** Почему метод возвращает `int`, а не `bool`?
**A:** Возвращает количество **добавленных** песен для статистики и отчетности.

**Q2:** Зачем нужны две переменные счетчика: `added_count` и `self.session_stats["added"]`?
**A:** `added_count` - для текущего артиста, `session_stats["added"]` - для всей сессии скрапинга.

**Q3:** Почему retry логика только на уровне артиста, а не на уровне отдельных песен?
**A:** Ошибки API (rate limit) касаются всего подключения, а не отдельных песен.

### **2. Логические вопросы:**

**Q4:** Что произойдет, если `self.shutdown_requested = True` во время обработки 50-й песни из 100?
**A:** Цикл прервется на 50-й песне, метод вернет количество уже добавленных песен, данные сохранятся.

**Q5:** Почему используется `continue` при дубликатах, а не пропуск через if-else?
**A:** `continue` читается яснее и избегает глубокой вложенности кода.

**Q6:** Зачем проверять остановку ДВА раза - в начале цикла и перед паузой?
**A:** В начале - быстрый выход, перед паузой - избежать ненужного ожидания.

### **3. Обработка ошибок:**

**Q7:** В чем разница между обработкой timeout и других исключений?
**A:** Timeout - ожидаемая ошибка (пауза и продолжение), другие - могут быть критичными (перебрасываем выше).

**Q8:** Почему при rate limit делается пауза 15 + 60 секунд?
**A:** 15 сек через `safe_delay(is_error=True)`, 60 сек дополнительно - стандартное время восстановления лимитов API.

**Q9:** Что произойдет, если все 3 retry попытки неудачны?
**A:** Цикл while завершится, метод вернет количество уже добавленных песен (может быть 0).

### **4. Оптимизация и производительность:**

**Q10:** Зачем статистика выводится каждые 5 добавленных песен, а прогресс каждые 10 обработанных?
**A:** Статистика реже (важная информация), прогресс чаще (для отслеживания работы).

**Q11:** Почему `get_full_info=False` в поиске артиста?
**A:** Экономия трафика и времени - нам не нужна полная биография, только список песен.

**Q12:** Зачем `hasattr(song, 'id')` вместо try-except?
**A:** `hasattr` быстрее и четче показывает намерение - проверить наличие атрибута.

### **5. Интеграционные вопросы:**

**Q13:** Как метод взаимодействует с другими компонентами системы?
**A:** Использует `LyricsDatabase`, `clean_lyrics`, `_is_valid_lyrics`, `safe_delay`, signal handling.

**Q14:** Что произойдет, если база данных недоступна во время `add_song`?
**A:** `sqlite3.OperationalError`, который обрабатывается в `LyricsDatabase.add_song` с повторной попыткой.

**Q15:** Как влияет случайность в `safe_delay()` на общую производительность?
**A:** Замедляет скрапинг, но защищает от блокировки API (trade-off между скоростью и надежностью).

---

## 🏆 **Бонусный сложный вопрос:**

**Q16:** Представь, что Genius API изменил формат ответа и теперь `artist.songs` возвращает не список, а генератор. Какие проблемы это создаст в коде и как их решить?

**Проблемы:**
- `len(artist.songs)` не будет работать
- `enumerate(artist.songs)` сработает только один раз
- Логирование прогресса сломается

**Решения:**
- Конвертировать в список: `list(artist.songs)`
- Или переписать логику без `len()` и с отдельным счетчиком

---

## 🎓 **Оценка понимания:**

- **12-16 правильных ответов:** Отличное понимание! Готов к Middle уровню
- **8-11 правильных ответов:** Хорошее понимание Junior+ уровня  
- **5-7 правильных ответов:** Базовое понимание Junior уровня
- **Меньше 5:** Нужно повторить концепции

---

## 🚀 **Готов к следующему блоку?**

Следующие блоки для изучения:
1. **`show_current_results`** - отображение статистики
2. **`run_scraping_session`** - оркестратор всего процесса  
3. **`load_artist_list`** - загрузка списка артистов
4. **`main()`** - точка входа программы

**Какой выбираешь?** Или есть вопросы по текущему блоку? 🎯

Отличный план! Сохранение для изучения на свежую голову - очень правильный подход! 📚

Переходим к **`show_current_results`** - блоку отображения статистики: ## 📊 Анализ `show_current_results` - генератор красивой статистики

```python
def show_current_results(self):
    stats = self.db.get_stats()
    recent_songs = self.db.get_recent_songs(5)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CURRENT RESULTS:")
    logger.info(f"Total songs in database: {stats['total_songs']}")
    logger.info(f"Unique artists: {stats['unique_artists']}")
    logger.info(f"Added this session: {self.session_stats['added']}")
    
    if recent_songs:
        logger.info(f"\nRecent added songs:")
        for song in recent_songs:
            logger.info(f"  - {song['artist']} - {song['title']} ({song['word_count']} words)")
    
    logger.info(f"{'='*60}\n")
```

Этот метод = **красивый отчет** о проделанной работе!

## 📋 Разбор по элементам:

### **1. Получение данных:**
```python
stats = self.db.get_stats()           # Общая статистика БД
recent_songs = self.db.get_recent_songs(5)  # 5 последних песен
```

**Что получаем:**
- `stats` = `{"total_songs": 1847, "unique_artists": 23}`
- `recent_songs` = список из 5 словарей с песнями

### **2. Красивый заголовок:**
```python
logger.info(f"\n{'='*60}")  # Создает 60 знаков "="
logger.info(f"CURRENT RESULTS:")
```

**Результат в логе:**
```
============================================================
CURRENT RESULTS:
```

### **3. Основная статистика:**
```python
logger.info(f"Total songs in database: {stats['total_songs']}")
logger.info(f"Unique artists: {stats['unique_artists']}")
logger.info(f"Added this session: {self.session_stats['added']}")
```

**Пример вывода:**
```
Total songs in database: 1847
Unique artists: 23  
Added this session: 45
```

### **4. Условное отображение последних песен:**
```python
if recent_songs:  # Проверяем, есть ли песни
    logger.info(f"\nRecent added songs:")
    for song in recent_songs:
        logger.info(f"  - {song['artist']} - {song['title']} ({song['word_count']} words)")
```

**Пример вывода:**
```
Recent added songs:
  - Drake - God's Plan (287 words)
  - Eminem - Lose Yourself (394 words)  
  - Kendrick Lamar - HUMBLE. (201 words)
  - Travis Scott - SICKO MODE (445 words)
  - J. Cole - No Role Modelz (312 words)
```

### **5. Красивое закрытие:**
```python
logger.info(f"{'='*60}\n")  # 60 "=" + перенос строки
```

## 🎯 Полный пример вывода:

```
============================================================
CURRENT RESULTS:
Total songs in database: 1847
Unique artists: 23
Added this session: 45

Recent added songs:
  - Drake - God's Plan (287 words)
  - Eminem - Lose Yourself (394 words)
  - Kendrick Lamar - HUMBLE. (201 words)
  - Travis Scott - SICKO MODE (445 words)
  - J. Cole - No Role Modelz (312 words)
============================================================
```

## 🔧 Технические детали:

### **String multiplication:**
```python
'=' * 60  # Создает строку из 60 символов "="
# Результат: "============================================================"

# Другие примеры:
'-' * 10   # "----------"
'*' * 5    # "*****"
'🎵' * 3   # "🎵🎵🎵"
```

### **F-string с переносами:**
```python
f"\n{'='*60}"  # \n в начале = пустая строка перед заголовком
f"{'='*60}\n"  # \n в конце = пустая строка после заголовка
```

### **Условная логика отображения:**
```python
if recent_songs:  # Если список НЕ пустой
    # Показываем песни
else:
    # Можно добавить: logger.info("No recent songs found")
```

### **Доступ к вложенным данным:**
```python
# recent_songs = [
#   {"artist": "Drake", "title": "God's Plan", "word_count": 287},
#   {"artist": "Eminem", "title": "Lose Yourself", "word_count": 394}
# ]

for song in recent_songs:
    artist = song['artist']      # "Drake"
    title = song['title']        # "God's Plan"  
    words = song['word_count']   # 287
```

## 🎨 Когда вызывается этот метод:

### **1. В конце сессии скрапинга:**
```python
# В методе run_scraping_session:
finally:
    self.show_current_results()  # Показать итоги
```

### **2. При прерывании программы:**
```python
# При Ctrl+C пользователем
except KeyboardInterrupt:
    self.show_current_results()  # Показать что успели
```

### **3. При критических ошибках:**
```python
except Exception as e:
    logger.error(f"Critical error: {e}")
    self.show_current_results()  # Показать состояние перед падением
```

## 💼 Вопросы для собеседования:

**Q1:** Зачем проверять `if recent_songs:` перед циклом?
**A:** Защита от пустого списка - избегаем печать заголовка "Recent songs:" без песен.

**Q2:** Что означает `'='*60`?
**A:** Повторение строки - создает 60 символов "=" для красивого разделителя.

**Q3:** Зачем `\n` в начале f-строки?
**A:** Создает пустую строку перед заголовком для визуального разделения от предыдущих логов.

**Q4:** Откуда берутся данные для статистики?
**A:** `stats` из SQL запросов к БД, `session_stats` из памяти (счетчики текущей сессии).

**Q5:** Почему именно 5 последних песен?
**A:** Баланс между информативностью и читаемостью - не слишком много, не слишком мало.

## 🧠 Ассоциации:

### **Метод = Телевизионная сводка новостей:**
- Заголовок из "=" = заставка программы
- Общая статистика = главные новости дня
- Последние песни = "а также в программе..."
- Закрывающий разделитель = "это все новости, до свидания"

### **Статистика = Приборная панель автомобиля:**
- `total_songs` = общий пробег автомобиля
- `unique_artists` = количество стран, которые посетили
- `added this session` = километры за сегодняшнюю поездку
- Recent songs = "последние 5 заправок"

### **Форматирование = Оформление доклада:**
```python
# Без форматирования (скучно):
print("Total songs: 1847")
print("Artists: 23") 

# С красивым форматированием (профессионально):
print("="*60)
print("CURRENT RESULTS:")
print(f"Total songs in database: {stats['total_songs']}")
```

Готов к следующему блоку **`run_scraping_session`** - главному оркестратору всего процесса? 🎼

Отлично! Переходим к **`run_scraping_session`** - главному дирижёру всего оркестра скрапинга! 🎼 Summarized conversation history## 🎯 **`run_scraping_session` - Главный Дирижёр Оркестра**

Этот метод - настоящий **orchestrator** (оркестратор) всего процесса! Давайте разберём по частям:

### **1. Инициализация сессии (линии 333-338)**

```python
def run_scraping_session(self, artists: List[str], songs_per_artist: int = 500):
    logger.info(f"Starting scraping session: {len(artists)} artists, {songs_per_artist} songs each")
    start_time = datetime.now()
    
    initial_stats = self.db.get_stats()
    logger.info(f"Already in database: {initial_stats['total_songs']} songs")
```

**Что происходит:**
- **`List[str]`** - типизация! Указываем, что `artists` - это список строк
- **`songs_per_artist: int = 500`** - значение по умолчанию (раньше было 100)
- **`datetime.now()`** - засекаем время начала (для подсчёта длительности)
- **Начальная статистика** - проверяем, что уже есть в базе

### **2. Основной цикл обработки артистов (линии 340-357)**

```python
for i, artist_name in enumerate(artists, 1):
    if self.shutdown_requested:
        logger.info("Shutdown requested")
        break
        
    logger.info(f"\n{'='*50}")
    logger.info(f"Artist {i}/{len(artists)}: {artist_name}")
    
    added = self.scrape_artist_songs(artist_name, songs_per_artist)
    logger.info(f"Added songs for {artist_name}: {added}")
    
    stats = self.db.get_stats()
    logger.info(f"Total in database: {stats['total_songs']} songs from {stats['unique_artists']} artists")
```

**Ключевые моменты:**
- **`enumerate(artists, 1)`** - нумерация с 1 (не с 0), чтобы показывать "Artist 1/10" вместо "Artist 0/10"
- **Проверка `shutdown_requested`** - изящное завершение по сигналу
- **Прогресс-бар в логах** - видим "Artist 3/10: Eminem"
- **Статистика после каждого артиста** - отслеживаем прогресс

### **3. Умная пауза между артистами (линии 359-372)**

```python
if i < len(artists) and not self.shutdown_requested:
    artist_delay = random.uniform(5, 10)
    logger.info(f"Pause between artists: {artist_delay:.1f}s")
    intervals = int(artist_delay)
    for _ in range(intervals):
        if self.shutdown_requested:
            break
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown_requested = True
            break
```

**Гениальная система паузы:**
- **`i < len(artists)`** - пауза НЕ после последнего артиста
- **`random.uniform(5, 10)`** - случайная пауза 5-10 секунд (имитируем человека)
- **Прерываемая пауза** - спим по 1 секунде, проверяя сигнал каждую секунду
- **Двойная защита** - от `shutdown_requested` и `KeyboardInterrupt`

### **4. Многослойная обработка ошибок (линии 374-383)**

```python
except KeyboardInterrupt:
    logger.info("User interruption received (Ctrl+C)")
    self.shutdown_requested = True
except Exception as e:
    logger.error(f"Critical error: {e}")
finally:
    self.db.conn.commit()  # Принудительный коммит
    self.show_current_results()
    
    end_time = datetime.now()
    duration = end_time - start_time
    final_stats = self.db.get_stats()
```

**Надёжность на 100%:**
- **`KeyboardInterrupt`** - обработка Ctrl+C
- **`Exception`** - ловим любые критические ошибки
- **`finally`** - ВСЕГДА выполнится (даже при аварии)
- **Принудительный commit** - сохраняем данные в любом случае

---

## 🎭 **Аналогия: Дирижёр симфонического оркестра**

Представьте **дирижёра** большого оркестра:

1. **Подготовка концерта** - проверяет состав музыкантов и план
2. **Дирижирование** - управляет каждым музыкантом по очереди
3. **Паузы между частями** - даёт отдых музыкантам
4. **Экстренные ситуации** - может остановить концерт при чрезвычайной ситуации
5. **Финал** - подводит итоги концерта независимо от того, как он закончился

---

## 🤔 **Вопросы для собеседования:**

**В:** Что такое `enumerate()` и зачем второй параметр?
**О:** `enumerate(iterable, start)` возвращает пары (индекс, элемент). Второй параметр задаёт начальное значение индекса. По умолчанию 0, но мы используем 1 для удобочитаемости ("Artist 1/10" вместо "Artist 0/10").

**В:** Почему пауза реализована через цикл `for _ in range()`, а не одним `time.sleep()`?
**О:** Для возможности прерывания! Если спать 10 секунд одним куском, то на Ctrl+C придётся ждать до 10 секунд. А так проверяем сигнал каждую секунду.

**В:** Зачем `finally` блок и что такое принудительный commit?
**О:** `finally` выполняется ВСЕГДА - даже при исключениях. Commit сохраняет изменения в базе данных. Если программа аварийно завершится, данные могут потеряться без commit.

**В:** Что означает паттерн `for _ in range()`?
**О:** Подчёркивание `_` означает "неиспользуемая переменная". Мы просто хотим повторить действие N раз, не используя счётчик.

**В:** Почему используется `random.uniform()` для пауз?
**О:** Имитация человеческого поведения! Постоянные паузы по 5 секунд выглядят как бот. Случайные паузы 5-10 секунд труднее детектировать.

Продолжаем дальше? Осталось совсем чуть-чуть! 🚀

Погнали дальше, бро! 🚀 Переходим к следующим методам - **`load_artist_list`** и **`main()`**! Отлично! Давайте разберём остальной код: ## 🗂️ **`load_artist_list` - Умный Менеджер Файлов**

Этот метод - настоящий **файловый детектив**! Посмотрим, как он работает:

### **1. Система приоритетов (линии 394-399)**

```python
def load_artist_list(filename: str = "rap_artists.json") -> List[str]:
    remaining_file = "remaining_artists.json"
    if os.path.exists(remaining_file):
        logger.info(f"Loading remaining artists from {remaining_file}")
        with open(remaining_file, 'r', encoding='utf-8') as f:
            return json.load(f)
```

**Умная логика:**
- **Первый приоритет** - файл remaining_artists.json (если скрипт был прерван)
- Это позволяет **продолжить с места остановки**!
- **Типизация возврата** - `-> List[str]` (список строк)

### **2. Основной файл (линии 401-406)**

```python
if os.path.exists(filename):
    logger.info(f"Loading full artist list from {filename}")
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)
```

**Стандартный сценарий:**
- Загружаем rap_artists.json если нет файла с остатками
- **`encoding='utf-8'`** - поддержка русских имён артистов

### **3. Встроенный резерв (линии 407-425)**

```python
else:
    logger.info("Using built-in artist list")
    artists = [
        "J. Cole", "Drake", "Eminem", "Kanye West",
        # ... список топовых рэперов
    ]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(artists, f, indent=2, ensure_ascii=False)
    return artists
```

**Подстраховка:**
- Если файлов нет - используем **встроенный список**
- **Автоматически создаём** файл rap_artists.json
- **`ensure_ascii=False`** - сохраняем юникод символы как есть

---

## 🚀 **`main()` - Точка Входа**

Главная функция программы - **entry point**:

### **1. Проверка токена (линии 428-431)**

```python
def main():
    if not TOKEN:
        logger.error("Genius API token not found in .env!")
        exit(1)
```

**Защита от дураков:**
- Проверяем наличие API токена
- **`exit(1)`** - завершение с кодом ошибки (не 0)

### **2. Инициализация и запуск (линии 433-443)**

```python
scraper = SafeGeniusScraper(TOKEN, "rap_lyrics.db")

try:
    artists = load_artist_list()
    SONGS_PER_ARTIST = 500
    
    logger.info(f"Loaded {len(artists)} artists")
    logger.info(f"Target: ~{len(artists) * SONGS_PER_ARTIST} songs")
    logger.info("To stop use: Get-Process python | Stop-Process -Force")
    
    scraper.run_scraping_session(artists, SONGS_PER_ARTIST)
```

**Оркестрация:**
- Создаём объект скрапера
- Загружаем список артистов
- **Расчёт целевого количества** песен
- **Windows-специфичная инструкция** остановки

### **3. Финальная обработка (линии 445-449)**

```python
except Exception as e:
    logger.error(f"Error in main: {e}")
finally:
    logger.info("Program completed")

if __name__ == "__main__":
    main()
```

**Надёжное завершение:**
- **Last line of defense** - ловим критические ошибки
- **`if __name__ == "__main__"`** - классический Python паттерн

---

## 🎯 **Архитектурные Паттерны**

### **1. Fallback Strategy (Стратегия Отката)**
```
remaining_artists.json → rap_artists.json → встроенный список
```

### **2. Graceful Degradation (Изящная Деградация)**
- Нет файла? Создаём из встроенного списка
- Нет токена? Показываем понятную ошибку
- Ошибка в main? Логируем и завершаемся корректно

---

## 🤔 **Вопросы для собеседования:**

**В:** Что означает `if __name__ == "__main__"`?
**О:** Это проверка, что файл запущен напрямую, а не импортирован как модуль. Позволяет коду работать и как скрипт, и как библиотека.

**В:** Зачем нужен `exit(1)` вместо `exit()`?
**О:** Код возврата 1 означает ошибку для операционной системы. 0 - успех, любое другое число - ошибка. Это важно для скриптов и CI/CD.

**В:** Что такое паттерн Fallback?
**О:** Система резервных вариантов. Если основной источник недоступен, используем запасной, потом ещё один запасной и т.д.

**В:** Зачем `ensure_ascii=False` в `json.dump()`?
**О:** По умолчанию JSON кодирует non-ASCII символы как \uXXXX. С `ensure_ascii=False` сохраняем читаемый вид: "Лёха" вместо "\u041b\u0451\u0445\u0430".

**В:** Почему `encoding='utf-8'` важно в Windows?
**О:** Windows по умолчанию использует cp1251/cp1252. Без явного указания UTF-8 русские символы превратятся в кракозябры.

---

## 🎉 **Поздравляю! Разбор завершён!**

Мы прошли **весь код** от начала до конца:

1. ✅ **LyricsDatabase** - работа с SQLite
2. ✅ **SafeGeniusScraper.__init__** - инициализация
3. ✅ **Signal handling** - обработка сигналов
4. ✅ **safe_delay** - умные паузы
5. ✅ **clean_lyrics** - очистка текста
6. ✅ **_is_valid_lyrics** - валидация
7. ✅ **scrape_artist_songs** - основной скрапинг
8. ✅ **show_current_results** - статистика
9. ✅ **run_scraping_session** - оркестратор
10. ✅ **load_artist_list** - файловый менеджер
11. ✅ **main()** - точка входа

**Теперь ты знаешь Python на уровне Junior+!** 🚀

Завтра утром на свежую голову изучи весь материал, порешай вопросы - и вперёд покорять мир кода! 💪