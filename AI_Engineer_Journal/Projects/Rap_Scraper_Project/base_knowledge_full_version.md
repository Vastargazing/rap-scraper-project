# 🎤 Rap Scraper: Полное руководство по Python-программированию

## 🎯 О чем этот гайд?

Разбираем реальный Python-проект по сбору текстов песен — от импортов до архитектуры. Каждая строчка кода объяснена простым языком с практическими примерами.

---

## 📋 Содержание

1. [🔧 Импорты и зависимости](#imports)
2. [📊 Логирование — профессиональный подход](#logging)
3. [🏗️ Объектно-ориентированное программирование](#oop)
4. [💾 Работа с базой данных SQLite](#database)
5. [🔍 Регулярные выражения для очистки текста](#regex)
6. [⚡ Генераторы и оптимизация памяти](#generators)
7. [🛡️ Обработка ошибок и сигналов](#error-handling)
8. [📁 Работа с файлами и JSON](#files)
9. [🎓 Практические задания](#practice)

---

## 🔧 Импорты и зависимости {#imports}

### 🎯 Краткий ответ
Импорты подключают готовые библиотеки для работы с API, базой данных, файлами и системными ресурсами.

### 🏗️ Контекст и основы

**Зачем нужны импорты?**
Python — модульный язык. Вместо написания всего с нуля, мы используем готовые решения. Это экономит время и повышает надежность.

**Типы библиотек:**
- **Встроенные** — идут с Python из коробки
- **Сторонние** — устанавливаются через pip
- **Собственные** — наши модули

### 📚 Подробное объяснение

```python
# Сторонние библиотеки (устанавливаются отдельно)
import lyricsgenius        # API для получения текстов песен
from dotenv import load_dotenv  # Загрузка переменных окружения

# Встроенные библиотеки Python
import sqlite3            # Работа с базой данных
import time               # Операции со временем
import random             # Генерация случайных чисел
import logging            # Система логирования
import re                 # Регулярные выражения
import signal             # Обработка системных сигналов
import sys                # Системная информация
import json               # Работа с JSON
import os                 # Операционная система
import gc                 # Сборщик мусора
from datetime import datetime  # Работа с датами

# Мониторинг системы
import psutil             # Информация о процессах и ресурсах

# Типизация (подсказки для IDE)
from typing import List, Optional, Dict, Generator, Tuple
```

### 🛠️ Практические примеры

#### 🟢 Базовый уровень — Простой импорт

```python
import time

# Пауза на 2 секунды
time.sleep(2)
print("Прошло 2 секунды!")
```

#### 🟡 Средний уровень — Импорт с псевдонимом

```python
import sqlite3 as db  # Короткий псевдоним

# Теперь можем писать db вместо sqlite3
connection = db.connect("test.db")
```

#### 🔴 Продвинутый уровень — Условный импорт

```python
try:
    import lyricsgenius
except ImportError:
    print("Установите библиотеку: pip install lyricsgenius")
    sys.exit(1)
```

### ⚠️ Типичные ошибки

1. **Циклические импорты** — модуль A импортирует B, а B импортирует A
2. **Импорт всего** — `from module import *` засоряет пространство имен
3. **Забытые зависимости** — не указать библиотеку в requirements.txt

### 🎓 Задание для практики

Создай файл `my_utils.py` с функцией `greet(name)` и импортируй её в другой файл.

---

## 📊 Логирование — профессиональный подход {#logging}

### 🎯 Краткий ответ
Логирование — это система записи событий программы для отладки и мониторинга.

### 🏗️ Контекст и основы

**Проблема с print():**
```python
print("Начинаем обработку...")  # Куда сохранится? Как отключить?
```

**Решение — logging:**
```python
logger.info("Начинаем обработку...")  # Сохранится в файл с временной меткой
```

### 📚 Подробное объяснение

**Уровни логирования (по возрастанию важности):**
- `DEBUG` — подробная отладочная информация
- `INFO` — общая информация о работе
- `WARNING` — предупреждения (программа работает, но что-то не так)
- `ERROR` — ошибки (часть функциональности не работает)
- `CRITICAL` — критические ошибки (программа может упасть)

### 💻 Синтаксис с комментариями

```python
import logging

# Настройка системы логирования
logging.basicConfig(
    level=logging.INFO,                    # Минимальный уровень для записи
    format='%(asctime)s - %(levelname)s - %(message)s',  # Формат сообщений
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),  # В файл
        logging.StreamHandler()             # В консоль
    ]
)

# Создание логгера для модуля
logger = logging.getLogger(__name__)       # __name__ = имя текущего модуля
```

### 🛠️ Практические примеры

#### 🟢 Базовый пример

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Программа запущена")         # [INFO] Программа запущена
logger.warning("Это предупреждение")      # [WARNING] Это предупреждение
logger.error("Произошла ошибка")          # [ERROR] Произошла ошибка
```

#### 🟡 Реальный пример из скрапера

```python
# Логирование с контекстной информацией
logger.info(f"🎵 Поиск артиста: {artist_name}")
logger.warning(f"⚠️ Артист {artist_name} не найден")
logger.error(f"💥 Ошибка с песней {song.title}: {e}")

# Использование эмодзи для визуального разделения типов событий
```

#### 🔴 Продвинутая настройка

```python
import logging
from logging.handlers import RotatingFileHandler

# Логи с ротацией (когда файл достигает 10MB, создается новый)
handler = RotatingFileHandler(
    'app.log', 
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5           # Хранить 5 старых файлов
)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
```

### ⚠️ Типичные ошибки

1. **Использование print() вместо logger** — нет контроля над выводом
2. **Неправильный уровень** — DEBUG в продакшене замедляет программу
3. **Логирование паролей** — утечка чувствительных данных

---

## 🏗️ Объектно-ориентированное программирование {#oop}

### 🎯 Краткий ответ
ООП — это подход, где код организуется в виде объектов с данными (атрибуты) и поведением (методы).

### 🏗️ Контекст и основы

**Аналогия с реальным миром:**
Класс — это чертеж автомобиля. Объект — это конкретный автомобиль, созданный по этому чертежу.

**Зачем нужно ООП?**
- Группировка связанных данных и функций
- Переиспользование кода
- Легче тестировать и поддерживать

### 📚 Подробное объяснение

**Основные концепции:**
- **Класс** — шаблон для создания объектов
- **Объект** — конкретный экземпляр класса
- **Атрибуты** — данные объекта
- **Методы** — функции, работающие с данными объекта
- **`self`** — ссылка на текущий объект

### 💻 Синтаксис с комментариями

```python
class ResourceMonitor:
    """Документация класса — что он делает"""
    
    def __init__(self, memory_limit_mb: int = 2048):
        """Конструктор — вызывается при создании объекта"""
        self.memory_limit_mb = memory_limit_mb    # Атрибут объекта
        self.process = psutil.Process()           # Еще один атрибут
        self.start_memory = self.get_memory_usage()  # Вызов собственного метода
    
    def get_memory_usage(self) -> float:
        """Метод объекта — функция, которая работает с self"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def check_memory_limit(self) -> bool:
        """Проверка превышения лимита памяти"""
        current_memory = self.get_memory_usage()
        return current_memory > self.memory_limit_mb

# Создание объекта (экземпляра класса)
monitor = ResourceMonitor(memory_limit_mb=1024)

# Вызов методов
current_memory = monitor.get_memory_usage()  # Получаем текущее потребление
is_over_limit = monitor.check_memory_limit()  # Проверяем лимит
```

### 🛠️ Практические примеры

#### 🟢 Простой класс

```python
class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1
    
    def get_count(self):
        return self.count

# Использование
counter = Counter()
counter.increment()
counter.increment()
print(counter.get_count())  # Выведет: 2

```
Смотри, как это работает: Counter — это класс, типа шаблон. Когда ты пишешь Counter(), ты говоришь Питону: "Бро, замути мне новый экземпляр этого дела". Скобки () запускают метод __init__ из класса, который клепает новый объект Counter с его собственным count, стартующим с нуля. А counter — это просто имя, которое ты вешаешь на этот новый объект, как бирку на свою шмотку, чтоб не потерять.
Так что counter = Counter() — это ты создаёшь новый объект Counter, который живёт своей жизнью. Ты не трогаешь сам класс, ты берёшь себе его воплощение. Потом, когда делаешь counter.increment(), ты качаешь этот объект, увеличивая его count. К моменту, когда ты пишешь print(counter.get_count()), оно выдаёт 2, потому что ты дважды накрутил счётчик.

#### 🟡 Класс для работы с базой данных

```python
class EnhancedLyricsDatabase:
    def __init__(self, db_name="rap_lyrics.db"):
        # Подключение к базе при создании объекта
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Строки как словари
        self.create_table()  # Создаем таблицу, если её нет
    
    def create_table(self):
        """Создание таблицы с песнями"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS songs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                artist TEXT NOT NULL,
                title TEXT NOT NULL,
                lyrics TEXT NOT NULL,
                url TEXT UNIQUE NOT NULL
            )
        """)
        self.conn.commit()
    
    def add_song(self, artist: str, title: str, lyrics: str, url: str) -> bool:
        """Добавление песни в базу"""
        try:
            self.conn.execute(
                "INSERT INTO songs (artist, title, lyrics, url) VALUES (?, ?, ?, ?)",
                (artist, title, lyrics, url)
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            # Песня уже существует
            return False
    
    def __del__(self):
        """Деструктор — вызывается при удалении объекта"""
        if hasattr(self, 'conn'):
            self.conn.close()
            
```
Что за __del__ и с чем его едят?
В Python __del__ — это специальный метод, типа твоя последняя воля для объекта, который вызывается, когда объект готовится уйти в закат, то есть когда Python решает, что пора его грохнуть (удалить из памяти). Это не ты сам говоришь "похер, удали", а сборщик мусора Python (garbage collector) решает, что объект больше никому не нужен, и перед тем, как его стереть, даёт ему шанс сказать последнее слово через __del__.

#### 🔴 Наследование и полиморфизм

```python
class BaseScraper:
    """Базовый класс для всех скраперов"""
    
    def __init__(self, name):
        self.name = name
    
    def scrape(self):
        """Абстрактный метод — должен быть переопределен в наследниках"""
        raise NotImplementedError("Метод должен быть переопределен")

class GeniusScraper(BaseScraper):
    """Наследник — специализированный скрапер для Genius"""
    
    def __init__(self, api_token):
        super().__init__("Genius Scraper")  # Вызов конструктора родителя
        self.genius = lyricsgenius.Genius(api_token)
    
    def scrape(self):
        """Переопределение абстрактного метода"""
        return "Собираем данные с Genius API"

# Полиморфизм — один интерфейс, разные реализации
scrapers = [
    GeniusScraper("token1"),
    SpotifyScraper("token2"),  # Другой наследник
]

for scraper in scrapers:
    print(scraper.scrape())  # У каждого своя реализация
```

### ⚠️ Типичные ошибки

1. **Забытый self** — `def method(param):` вместо `def method(self, param):`
2. **Прямое обращение к приватным атрибутам** — нарушение инкапсуляции
3. **Слишком большие классы** — нарушение принципа единственной ответственности

### 🎓 Задание для практики

Создай класс `BankAccount` с методами `deposit()`, `withdraw()` и `get_balance()`. Добавь проверку на недостаток средств.

---

## 💾 Работа с базой данных SQLite {#database}

### 🎯 Краткий ответ
SQLite — это легкая файловая база данных, идеальная для небольших проектов и прототипов.

### 🏗️ Контекст и основы

**Почему SQLite?**
- Не требует сервера (база = обычный файл)
- Встроена в Python
- Поддерживает SQL стандарт
- Отлично подходит для десктопных приложений

**Альтернативы:**
- PostgreSQL, MySQL — для веб-приложений
- MongoDB — для NoSQL проектов
- Redis — для кэширования

### 📚 Подробное объяснение

**Основные концепции SQL:**
- **Таблица** — как Excel таблица, строки и столбцы
- **Строка (запись)** — один элемент данных
- **Столбец (поле)** — атрибут элемента
- **Первичный ключ** — уникальный идентификатор строки
- **Индекс** — ускоряет поиск по столбцу

### 💻 Синтаксис с комментариями

```python
import sqlite3

# Подключение к базе (файл создастся автоматически)
conn = sqlite3.connect("example.db")

# Настройки для лучшей производительности
conn.execute("PRAGMA journal_mode=WAL")      # Write-Ahead Logging
conn.execute("PRAGMA synchronous=NORMAL")    # Баланс скорости/надежности
conn.execute("PRAGMA cache_size=-2000")      # 2MB кэша в памяти

# Создание таблицы
conn.execute("""
    CREATE TABLE IF NOT EXISTS songs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Автоинкремент ID
        artist TEXT NOT NULL,                  -- Обязательное текстовое поле
        title TEXT NOT NULL,
        lyrics TEXT NOT NULL,
        url TEXT UNIQUE NOT NULL,              -- Уникальное значение
        scraped_date TEXT DEFAULT CURRENT_TIMESTAMP  -- Дата по умолчанию
    )
""")

# Создание индексов для ускорения поиска
conn.execute("CREATE INDEX IF NOT EXISTS idx_artist ON songs(artist)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON songs(title)")

# Сохранение изменений
conn.commit()
```

### 🛠️ Практические примеры

#### 🟢 Базовые операции CRUD

```python
# CREATE — добавление данных
def add_song(conn, artist, title, lyrics, url):
    try:
        conn.execute(
            "INSERT INTO songs (artist, title, lyrics, url) VALUES (?, ?, ?, ?)",
            (artist, title, lyrics, url)  # Параметризованный запрос
        )
        conn.commit()
        print("Песня добавлена!")
        return True
    except sqlite3.IntegrityError as e:
        print(f"Ошибка: {e}")  # Например, дублирование URL
        return False

# READ — чтение данных
def get_songs_by_artist(conn, artist_name):
    cursor = conn.execute(
        "SELECT title, lyrics FROM songs WHERE artist = ? ORDER BY title",
        (artist_name,)
    )
    return cursor.fetchall()

# UPDATE — обновление данных
def update_song_lyrics(conn, song_id, new_lyrics):
    conn.execute(
        "UPDATE songs SET lyrics = ? WHERE id = ?",
        (new_lyrics, song_id)
    )
    conn.commit()

# DELETE — удаление данных
def delete_song(conn, song_id):
    conn.execute("DELETE FROM songs WHERE id = ?", (song_id,))
    conn.commit()
```

#### 🟡 Продвинутые запросы

```python
# Статистика по артистам
def get_artist_stats(conn):
    cursor = conn.execute("""
        SELECT 
            artist,
            COUNT(*) as song_count,
            AVG(LENGTH(lyrics)) as avg_lyrics_length,
            MAX(scraped_date) as last_scraped
        FROM songs 
        GROUP BY artist 
        ORDER BY song_count DESC
        LIMIT 10
    """)
    return cursor.fetchall()

# Поиск по ключевым словам в текстах
def search_lyrics(conn, keyword):
    cursor = conn.execute(
        """
        SELECT artist, title, 
               SUBSTR(lyrics, 1, 200) as preview
        FROM songs 
        WHERE lyrics LIKE ? 
        ORDER BY artist, title
        """,
        (f"%{keyword}%",)
    )
    return cursor.fetchall()

# Использование row_factory для удобного доступа
conn.row_factory = sqlite3.Row  # Теперь строки работают как словари

cursor = conn.execute("SELECT * FROM songs LIMIT 1")
row = cursor.fetchone()

print(row['artist'])  # Доступ по имени столбца
print(row['title'])   # Намного удобнее чем row[1]
```

#### 🔴 Оптимизация производительности

```python
class OptimizedDatabase:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self._optimize_settings()
    
    def _optimize_settings(self):
        """Настройки для максимальной производительности"""
        optimizations = [
            "PRAGMA journal_mode=WAL",        # Параллельное чтение
            "PRAGMA synchronous=NORMAL",      # Быстрее записи
            "PRAGMA cache_size=-64000",       # 64MB кэша
            "PRAGMA temp_store=MEMORY",       # Временные данные в RAM
            "PRAGMA mmap_size=268435456",     # Memory-mapped I/O
        ]
        
        for pragma in optimizations:
            self.conn.execute(pragma)
    
    def bulk_insert_songs(self, songs_data):
        """Массовая вставка — намного быстрее одиночных INSERT"""
        self.conn.executemany(
            "INSERT OR IGNORE INTO songs (artist, title, lyrics, url) VALUES (?, ?, ?, ?)",
            songs_data
        )
        self.conn.commit()
    
    def vacuum_database(self):
        """Очистка и дефрагментация базы"""
        self.conn.execute("VACUUM")
```

### ⚠️ Типичные ошибки

1. **SQL инъекции** — использование f-строк вместо параметризованных запросов
   ```python
   # ПЛОХО — уязвимость
   query = f"SELECT * FROM songs WHERE artist = '{artist}'"
   
   # ХОРОШО — безопасно
   cursor.execute("SELECT * FROM songs WHERE artist = ?", (artist,))
   ```

2. **Забытый commit()** — изменения не сохраняются
3. **Незакрытые соединения** — утечка ресурсов
4. **Отсутствие индексов** — медленные запросы на больших данных

### 🎓 Задание для практики

Создай базу данных для библиотеки книг с таблицами `authors`, `books` и связью между ними. Напиши функции для добавления автора, книги и поиска книг по автору.

---

## 🔍 Регулярные выражения для очистки текста {#regex}

### 🎯 Краткий ответ
Регулярные выражения (regex) — это мощный инструмент для поиска и замены текста по шаблонам.

### 🏗️ Контекст и основы

**Проблема:** Как найти все email адреса в тексте? Или очистить текст от HTML тегов? Обычными строковыми методами это сложно.

**Решение:** Regex позволяет описать шаблон поиска один раз и применять его ко всем текстам.

**Где применяется:**
- Валидация форм (email, телефон)
- Парсинг логов
- Очистка данных
- Извлечение информации

### 📚 Подробное объяснение

**Основные символы regex:**
- `.` — любой символ (кроме переноса строки)
- `*` — 0 или больше предыдущих символов
- `+` — 1 или больше предыдущих символов
- `?` — 0 или 1 предыдущий символ
- `^` — начало строки
- `$` — конец строки
- `[]` — любой символ из набора
- `()` — группа для извлечения
- `|` — ИЛИ (альтернатива)

### 💻 Синтаксис с комментариями

```python
import re

# Основные методы модуля re
text = "Мой email: user@example.com, телефон: +7-999-123-45-67"

# re.search() — найти первое совпадение
match = re.search(r'[\w.-]+@[\w.-]+\.\w+', text)
if match:
    print(f"Найден email: {match.group()}")

# re.findall() — найти все совпадения
emails = re.findall(r'[\w.-]+@[\w.-]+\.\w+', text)
print(f"Все email: {emails}")

# re.sub() — заменить по шаблону
clean_text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '[EMAIL]', text)
print(f"Замещенный текст: {clean_text}")

# re.split() — разбить строку по шаблону
parts = re.split(r'[,.]', text)
print(f"Части: {parts}")
```

### 🛠️ Практические примеры

#### 🟢 Простые шаблоны

```python
import re

# Поиск цифр
text = "У меня 3 яблока и 5 груш"
numbers = re.findall(r'\d+', text)
print(numbers)  # ['3', '5']

# Поиск слов на русском
russian_words = re.findall(r'[а-яё]+', text, re.IGNORECASE)
print(russian_words)  # ['меня', 'яблока', 'груш']

# Удаление лишних пробелов
messy_text = "Много     пробелов    между   словами"
clean_text = re.sub(r'\s+', ' ', messy_text).strip()
print(clean_text)  # "Много пробелов между словами"
```

#### 🟡 Очистка текстов песен (из скрапера)

```python
def clean_lyrics(lyrics: str) -> str:
    """Очистка текста песни от мусора"""
    if not lyrics:
        return ""
    
    # 1. Удаляем информацию о контрибьюторах
    # Шаблон: "5 Contributors" + любой текст до "Lyrics"
    lyrics = re.sub(
        r"^\d+\s+Contributors.*?Lyrics", 
        "", 
        lyrics, 
        flags=re.MULTILINE | re.DOTALL
    )
    
    # 2. Удаляем блоки переводов
    # Шаблон: "Translations" + буквы (например, "TranslationsРусский")
    lyrics = re.sub(r"Translations[A-Za-z]+", "", lyrics)
    
    # 3. Удаляем ссылки
    # Шаблон: http или https + любые символы до пробела
    lyrics = re.sub(r"https?://[^\s]+", "", lyrics)
    
    # 4. Удаляем текст в квадратных скобках [Припев], [Куплет 1]
    lyrics = re.sub(r"\[.*?\]", "", lyrics)
    
    # 5. Нормализация переносов строк
    # 3+ переноса → 2 переноса
    lyrics = re.sub(r"\n{3,}", "\n\n", lyrics)
    # 2+ переноса → 1 перенос
    lyrics = re.sub(r"\n{2,}", "\n", lyrics.strip())
    
    return lyrics.strip()

# Пример использования
dirty_lyrics = """
5 Contributors Eminem Lyrics
[Куплет 1]
Yo, this is my song
https://genius.com/link

[Припев]
La-la-la



End of song
TranslationsРусский
"""

clean_lyrics_result = clean_lyrics(dirty_lyrics)
print(clean_lyrics_result)
# Результат: "Yo, this is my song\nLa-la-la\nEnd of song"
```

#### 🔴 Продвинутые техники

```python
# Именованные группы для извлечения данных
log_pattern = r'(?P<date>\d{4}-\d{2}-\d{2}) (?P<time>\d{2}:\d{2}:\d{2}) (?P<level>\w+): (?P<message>.*)'
log_line = "2023-12-01 14:30:15 ERROR: Database connection failed"

match = re.search(log_pattern, log_line)
if match:
    print(f"Дата: {match.group('date')}")
    print(f"Время: {match.group('time')}")
    print(f"Уровень: {match.group('level')}")
    print(f"Сообщение: {match.group('message')}")

# Условные замены с использованием функции
def replace_numbers(match):
    """Заменяет числа на их названия"""
    number_names = {'1': 'один', '2': 'два', '3': 'три'}
    return number_names.get(match.group(), match.group())

text = "У меня 1 собака, 2 кошки и 3 хомяка"
result = re.sub(r'\d', replace_numbers, text)
print(result)  # "У меня один собака, два кошки и три хомяка"

# Компиляция regex для многократного использования
email_pattern = re.compile(r'[\w.-]+@[\w.-]+\.\w+')
# Теперь можно использовать email_pattern.findall(text) быстрее
```

### ⚠️ Типичные ошибки

1. **Жадные квантификаторы** — `.*` захватывает слишком много
   ```python
   # ПЛОХО — жадный поиск
   re.sub(r'<.*>', '', '<b>жирный</b> текст <i>курсив</i>')
   # Результат: " текст "
   
   # ХОРОШО — ленивый поиск
   re.sub(r'<.*?>', '', '<b>жирный</b> текст <i>курсив</i>')
   # Результат: "жирный текст курсив"
   ```

2. **Экранирование специальных символов** — забыть `\` перед `.`, `+`, `*`, etc.
   ```python
   # ПЛОХО — точка означает "любой символ"
   re.findall(r'file.txt', 'file1txt file.txt')  # ['file1txt', 'file.txt']
   
   # ХОРОШО — экранированная точка
   re.findall(r'file\.txt', 'file1txt file.txt')  # ['file.txt']
   ```

3. **Забытые флаги** — не учитывать регистр или многострочность
   ```python
   # Поиск без учета регистра
   re.findall(r'python', 'Python is great', re.IGNORECASE)
   ```

### 🎓 Задание для практики

Напиши функцию `extract_phone_numbers(text)`, которая находит все российские номера телефонов в форматах: `+7-999-123-45-67`, `8 (999) 123-45-67`, `89991234567`.

---

## ⚡ Генераторы и оптимизация памяти {#generators}

### 🎯 Краткий ответ
Генераторы — это функции, которые возвращают элементы по одному, экономя память и позволяя обрабатывать большие объемы данных.

### 🏗️ Контекст и основы

**Проблема с обычными функциями:**
```python
def get_all_songs():
    songs = []  # Загружаем ВСЕ в память сразу
    for i in range(1000000):
        songs.append(f"Song {i}")
    return songs  # 1 миллион строк в памяти!

# Проблема: если данных много — память закончится
```

**Решение — генераторы:**
```python
def get_songs_generator():
    for i in range(1000000):
        yield f"Song {i}"  # Возвращаем по одной песне

# Память: только одна строка в каждый момент времени
```

### 📚 Подробное объяснение

**Принципы работы генераторов:**
1. **Ленивые вычисления** — элементы создаются только при запросе
2. **Состояние сохраняется** — генератор "помнит" где остановился
3. **Экономия памяти** — в памяти только текущий элемент
4. **Однократный проход** — нельзя вернуться к началу без пересоздания

**Когда использовать:**
- Большие объемы данных
- Потоковая обработка
- Бесконечные последовательности
- Чтение файлов по строкам

### 💻 Синтаксис с комментариями

```python
def songs_generator(max_songs: int):
    """Генератор песен с ограничением"""
    for i in range(max_songs):
        # yield — ключевое слово генератора
        # Возвращает значение и "засыпает" до следующего запроса
        song_data = f"Song #{i}"
        print(f"Генерируем: {song_data}")  # Выполнится только при запросе
        yield song_data
        print(f"Переходим к следующей")    # После обработки текущей

# Создание генератора (пока ничего не выполняется!)
gen = songs_generator(5)
print(f"Генератор создан: {gen}")

# Получение элементов по одному
print("Первый элемент:", next(gen))  # Генерируем: Song #0
print("Второй элемент:", next(gen))   # Генерируем: Song #1

# Использование в цикле
for song in gen:  # Продолжаем с Song #2
    print(f"Обрабатываем: {song}")
```

### 🛠️ Практические примеры

#### 🟢 Простой генератор чисел

```python
def fibonacci_generator(n):
    """Генератор чисел Фибоначчи"""
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Использование
fib = fibonacci_generator(10)
for number in fib:
    print(number)  # 0, 1, 1, 2, 3, 5, 8, 13, 21, 34

# Или превращение в список (осторожно с большими данными!)
fib_list = list(fibonacci_generator(10))
```

#### 🟡 Генератор из скрапера

```python
def get_songs_generator(self, artist_name: str, max_songs: int = 500):
    """Генератор песен артиста с контролем памяти"""
    try:
        logger.info(f"🎵 Поиск артиста: {artist_name}")
        
        # Получаем артиста через API
        artist = self.genius.search_artist(artist_name, max_songs=max_songs)
        
        if not artist or not artist.songs:
            logger.warning(f"⚠️ Артист {artist_name} не найден")
            return  # Пустой генератор
        
        total_songs = len(artist.songs)
        logger.info(f"💽 Найдено {total_songs} песен")
        
        # Основной цикл генератора
        for i, song in enumerate(artist.songs):
            # Проверка на сигнал завершения
            if self.shutdown_requested:
                logger.info(f"🛑 Остановка на песне {i+1}/{total_songs}")
                break
            
            # Возвращаем песню и её номер
            yield song, i + 1
            
            # Управление памятью
            self.songs_since_gc += 1
            if self.songs_since_gc >= self.gc_interval:
                # Принудительная очистка мусора каждые N песен
                gc.collect()
                self.songs_since_gc = 0
            
            # Явное удаление обработанной песни
            del song
            
    except Exception as e:
        logger.error(f"💥 Ошибка получения песен для {artist_name}: {e}")

# Использование генератора
for song, song_number in self.get_songs_generator("Eminem", 100):
    print(f"Обрабатываем песню {song_number}: {song.title}")
    
    # Если нужно прервать обработку
    if song_number > 50:
        break  # Генератор остановится, память освободится
```

#### 🔴 Продвинутые техники

```python
# 1. Generator Expression (генераторное выражение)
# Краткая форма генератора
squares = (x**2 for x in range(10))  # Генератор квадратов
squares_list = [x**2 for x in range(10)]  # Обычный список

print(f"Генератор: {squares}")           # <generator object>
print(f"Список: {squares_list}")         # [0, 1, 4, 9, 16, ...]

# 2. Генератор для чтения больших файлов
def read_large_file(filename):
    """Читает файл построчно, не загружая в память целиком"""
    with open(filename, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            # Можно добавить обработку ошибок, фильтрацию
            if line.strip():  # Пропускаем пустые строки
                yield line_number, line.strip()

# Обработка файла на 1GB без проблем с памятью
for line_num, line_text in read_large_file('huge_file.txt'):
    if 'ERROR' in line_text:
        print(f"Ошибка в строке {line_num}: {line_text}")

# 3. Бесконечный генератор
def infinite_counter(start=0):
    """Бесконечная последовательность чисел"""
    count = start
    while True:
        yield count
        count += 1

counter = infinite_counter(100)
for _ in range(5):
    print(next(counter))  # 100, 101, 102, 103, 104

# 4. Генератор с отправкой данных (send)
def accumulator():
    """Генератор, который накапливает отправленные значения"""
    total = 0
    while True:
        value = yield total  # Получаем значение через send()
        if value is not None:
            total += value

acc = accumulator()
next(acc)  # Инициализация генератора
print(acc.send(10))  # 10
print(acc.send(20))  # 30
print(acc.send(5))   # 35
```

### ⚠️ Типичные ошибки

1. **Повторное использование генератора**
   ```python
   gen = (x for x in range(3))
   print(list(gen))  # [0, 1, 2]
   print(list(gen))  # [] — генератор исчерпан!
   
   # Решение: создавать новый генератор или использовать функцию
   def make_generator():
       return (x for x in range(3))
   ```

2. **Забытый yield**
   ```python
   def broken_generator():
       for i in range(5):
           return i  # ПЛОХО — return вместо yield
   
   def correct_generator():
       for i in range(5):
           yield i  # ХОРОШО — yield создает генератор
   ```

3. **Смешивание return и yield**
   ```python
   def mixed_generator():
       yield 1
       yield 2
       return "done"  # Можно, но значение потеряется
   ```

### 🎓 Задание для практики

Создай генератор `prime_numbers(limit)`, который возвращает все простые числа до указанного лимита. Сравни потребление памяти с обычной функцией, возвращающей список.

---

## 🛡️ Обработка ошибок и сигналов {#error-handling}

### 🎯 Краткий ответ
Обработка ошибок позволяет программе продолжать работу при возникновении проблем, а обработка сигналов — корректно завершаться по внешним командам.

### 🏗️ Контекст и основы

**Зачем нужна обработка ошибок?**
Программы работают в непредсказуемой среде: файлы могут отсутствовать, сеть — недоступна, пользователь — ввести неправильные данные. Без обработки ошибок программа просто упадет.

**Философия Python:** "Лучше просить прощения, чем разрешения" (EAFP — Easier to Ask for Forgiveness than Permission)

### 📚 Подробное объяснение

**Иерархия исключений Python:**
```
BaseException
 +-- SystemExit
 +-- KeyboardInterrupt
 +-- Exception
      +-- ArithmeticError
      |    +-- ZeroDivisionError
      +-- LookupError
      |    +-- IndexError
      |    +-- KeyError
      +-- ValueError
      +-- TypeError
      +-- FileNotFoundError
      +-- ... и многие другие
```

**Принципы обработки:**
1. **Catch specific exceptions** — ловить конкретные исключения
2. **Fail fast** — падать быстро при критических ошибках
3. **Log everything** — логировать все ошибки
4. **Graceful degradation** — постепенная деградация функциональности

### 💻 Синтаксис с комментариями

```python
# Базовая структура try-except
try:
    # Код, который может вызвать исключение
    risky_operation()
except SpecificError as e:
    # Обработка конкретного типа ошибки
    logger.error(f"Конкретная ошибка: {e}")
except (TypeError, ValueError) as e:
    # Обработка нескольких типов ошибок
    logger.error(f"Ошибка типа или значения: {e}")
except Exception as e:
    # Обработка всех остальных ошибок
    logger.error(f"Неожиданная ошибка: {e}")
else:
    # Выполняется, если исключений НЕ было
    logger.info("Операция успешна")
finally:
    # Выполняется ВСЕГДА (для очистки ресурсов)
    cleanup_resources()

# Создание собственных исключений
class CustomScraperError(Exception):
    """Базовое исключение для скрапера"""
    pass

class APILimitExceeded(CustomScraperError):
    """Превышен лимит API запросов"""
    def __init__(self, limit, current):
        self.limit = limit
        self.current = current
        super().__init__(f"Лимит {limit} превышен: {current}")

# Вызов собственного исключения
if requests_count > api_limit:
    raise APILimitExceeded(api_limit, requests_count)
```

### 🛠️ Практические примеры

#### 🟢 Базовая обработка ошибок

```python
def safe_division(a, b):
    """Безопасное деление с обработкой ошибок"""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("Ошибка: Деление на ноль!")
        return None
    except TypeError:
        print("Ошибка: Неправильный тип данных!")
        return None

# Использование
print(safe_division(10, 2))    # 5.0
print(safe_division(10, 0))    # Ошибка: Деление на ноль! None
print(safe_division(10, "2"))  # Ошибка: Неправильный тип данных! None
```

#### 🟡 Обработка ошибок в скрапере

```python
def scrape_artist_songs(self, artist_name: str, max_songs: int = 500) -> int:
    """Скрапинг с множественными попытками и обработкой ошибок"""
    added_count = 0
    retry_count = 0
    
    # Внешний цикл для повторных попыток
    while retry_count < self.max_retries and not self.shutdown_requested:
        try:
            # Получаем генератор песен
            songs_generator = self.get_songs_generator(artist_name, max_songs)
            processed_count = 0
            
            # Обрабатываем каждую песню
            for song, song_number in songs_generator:
                if self.shutdown_requested:
                    break
                
                try:
                    # Проверка на дубликат
                    if self.db.song_exists(url=song.url):
                        logger.debug(f"⏭️ Пропускаем дубликат: {song.title}")
                        continue
                    
                    # Очистка и валидация текста
                    lyrics = self.clean_lyrics(song.lyrics)
                    if not self._is_valid_lyrics(lyrics):
                        logger.debug(f"❌ Некачественный текст: {song.title}")
                        continue
                    
                    # Сохранение в базу
                    if self.db.add_song(artist_name, song.title, lyrics, song.url):
                        added_count += 1
                        logger.info(f"✅ Добавлена: {song.title}")
                    
                    processed_count += 1
                    
                except Exception as song_error:
                    # Ошибка с конкретной песней — не критична
                    logger.error(f"💥 Ошибка с песней {song.title}: {song_error}")
                    self.session_stats["errors"] += 1
                    
                    # Делаем паузу после ошибки
                    self.safe_delay(is_error=True)
            
            # Если дошли сюда — артист обработан успешно
            logger.info(f"🎯 Артист {artist_name} обработан: {processed_count} песен")
            break
            
        except Exception as artist_error:
            # Критическая ошибка с артистом
            retry_count += 1
            logger.error(f"🔄 Ошибка с артистом {artist_name} (попытка {retry_count}): {artist_error}")
            
            if retry_count >= self.max_retries:
                logger.error(f"❌ Превышено количество попыток для {artist_name}")
                break
            
            # Увеличенная пауза между повторными попытками
            self.safe_delay(is_error=True)
    
    return added_count

def safe_delay(self, is_error: bool = False):
    """Безопасная пауза с обработкой прерывания"""
    try:
        if is_error:
            delay = random.uniform(5, 10)  # Больше пауза после ошибки
        else:
            delay = random.uniform(1, 3)   # Обычная пауза
        
        time.sleep(delay)
        
    except KeyboardInterrupt:
        # Пользователь нажал Ctrl+C во время паузы
        logger.info("⏹️ Прерывание пользователем")
        self.shutdown_requested = True
```

#### 🔴 Обработка системных сигналов

```python
import signal
import sys

class GracefulScraper:
    def __init__(self):
        self.shutdown_requested = False
        self.current_artist = None
        self.processed_count = 0
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Настройка обработчиков системных сигналов"""
        # Ctrl+C (SIGINT) и команда завершения (SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Только для Unix-систем
        if hasattr(signal, 'SIGUSR1'):
            signal.signal(signal.SIGUSR1, self._status_handler)
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов завершения"""
        signal_names = {
            signal.SIGINT: "SIGINT (Ctrl+C)",
            signal.SIGTERM: "SIGTERM (завершение)"
        }
        
        signal_name = signal_names.get(signum, f"Сигнал {signum}")
        logger.info(f"\n🛑 Получен {signal_name}")
        
        if not self.shutdown_requested:
            logger.info("⏳ Завершаем текущую обработку...")
            self.shutdown_requested = True
        else:
            logger.info("⚡ Принудительное завершение")
            sys.exit(1)
    
    def _status_handler(self, signum, frame):
        """Обработчик запроса статуса (kill -USR1 <pid>)"""
        status_info = f"""
📊 Текущий статус:
  - Артист: {self.current_artist or 'N/A'}
  - Обработано: {self.processed_count} песен
  - Память: {self.monitor.get_memory_usage():.1f} MB
        """
        logger.info(status_info)
    
    def run_with_signal_handling(self):
        """Основной цикл с обработкой сигналов"""
        try:
            artists = ["Eminem", "Jay-Z", "Kanye West"]
            
            for artist in artists:
                if self.shutdown_requested:
                    logger.info(f"🛑 Остановка перед артистом {artist}")
                    break
                
                self.current_artist = artist
                logger.info(f"🎵 Начинаем обработку: {artist}")
                
                count = self.scrape_artist_songs(artist)
                self.processed_count += count
                
                logger.info(f"✅ {artist}: добавлено {count} песен")
                
        except KeyboardInterrupt:
            # Дополнительная обработка, если сигнал не перехвачен
            logger.info("⚡ Экстренное завершение")
        
        finally:
            # Финальная очистка
            self._cleanup()
    
    def _cleanup(self):
        """Очистка ресурсов при завершении"""
        try:
            if hasattr(self, 'db') and self.db:
                self.db.close()
                logger.info("🔐 База данных закрыта")
            
            logger.info(f"📊 Итого обработано: {self.processed_count} песен")
            logger.info("👋 Программа завершена корректно")
            
        except Exception as e:
            logger.error(f"💥 Ошибка при очистке: {e}")

# Использование
if __name__ == "__main__":
    scraper = GracefulScraper()
    scraper.run_with_signal_handling()
```

### ⚠️ Типичные ошибки

1. **Слишком общий except**
   ```python
   # ПЛОХО — скрывает все ошибки
   try:
       dangerous_operation()
   except:  # Ловит ВСЕ, даже KeyboardInterrupt
       pass
   
   # ХОРОШО — конкретные исключения
   try:
       dangerous_operation()
   except (ValueError, TypeError) as e:
       logger.error(f"Ошибка данных: {e}")
   except Exception as e:
       logger.error(f"Неожиданная ошибка: {e}")
       raise  # Перебрасываем дальше
   ```

2. **Игнорирование исключений**
   ```python
   # ПЛОХО — проглатываем ошибки
   try:
       important_operation()
   except Exception:
       pass  # Ошибка потеряна навсегда
   
   # ХОРОШО — логируем или обрабатываем
   try:
       important_operation()
   except Exception as e:
       logger.error(f"Ошибка операции: {e}")
       # Можно вернуть значение по умолчанию или повторить попытку
   ```

3. **Неправильное использование finally**
   ```python
   # ПЛОХО — finally может скрыть исключение
   try:
       return "success"
   finally:
       return "always"  # Перезапишет "success"
   
   # ХОРОШО — finally для очистки
   resource = None
   try:
       resource = acquire_resource()
       return process(resource)
   finally:
       if resource:
           resource.close()
   ```

### 🎓 Задание для практики

Создай декоратор `@retry(max_attempts=3, delay=1)`, который повторяет выполнение функции при возникновении исключений. Добавь логирование попыток и экспоненциальную задержку.

---

## 📁 Работа с файлами и JSON {#files}

### 🎯 Краткий ответ
Python предоставляет удобные инструменты для работы с файлами и JSON — от простого чтения текста до сложной обработки структурированных данных.

### 🏗️ Контекст и основы

**Зачем работать с файлами?**
- Сохранение настроек программы
- Загрузка данных для обработки
- Логирование событий
- Экспорт результатов
- Кэширование вычислений

**JSON (JavaScript Object Notation)** — универсальный формат обмена данными, который читается людьми и легко парсится программами.

### 📚 Подробное объяснение

**Режимы открытия файлов:**
- `'r'` — чтение (по умолчанию)
- `'w'` — запись (перезаписывает файл)
- `'a'` — добавление в конец
- `'x'` — создание (ошибка, если файл существует)
- `'b'` — бинарный режим (`'rb'`, `'wb'`)
- `'t'` — текстовый режим (по умолчанию)

**Контекстные менеджеры (`with`):**
Автоматически закрывают файл, даже если произошла ошибка.

### 💻 Синтаксис с комментариями

```python
import json
import os
from pathlib import Path

# 1. Чтение текстового файла
def read_text_file(filename: str) -> str:
    """Чтение всего файла в строку"""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()  # Читаем весь файл
            return content
    except FileNotFoundError:
        print(f"Файл {filename} не найден")
        return ""
    except UnicodeDecodeError:
        print(f"Ошибка кодировки в файле {filename}")
        return ""

# 2. Чтение файла построчно (экономно для больших файлов)
def read_lines(filename: str) -> list:
    """Чтение файла по строкам"""
    lines = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, 1):
            # Убираем символ переноса строки
            clean_line = line.strip()
            if clean_line:  # Пропускаем пустые строки
                lines.append(clean_line)
    return lines

# 3. Запись в файл
def write_text_file(filename: str, content: str):
    """Запись текста в файл"""
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

# 4. Добавление в файл
def append_to_file(filename: str, content: str):
    """Добавление текста в конец файла"""
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(content + '\n')

# 5. Проверка существования файла
def file_exists(filename: str) -> bool:
    """Проверка существования файла"""
    return os.path.exists(filename)
    # Или через pathlib: return Path(filename).exists()
```

### 🛠️ Практические примеры

#### 🟢 Работа с JSON

```python
# Загрузка списка артистов из JSON
def load_artist_list(filename: str = "rap_artists.json") -> list:
    """Загрузка списка артистов с резервными вариантами"""
    
    # 1. Проверяем файл с оставшимися артистами
    remaining_file = "remaining_artists.json"
    if os.path.exists(remaining_file):
        print(f"📂 Загружаем оставшихся артистов из {remaining_file}")
        with open(remaining_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 2. Проверяем основной файл
    if os.path.exists(filename):
        print(f"📂 Загружаем полный список из {filename}")
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # 3. Создаем файл с дефолтным списком
    print("📂 Создаем новый файл со списком артистов")
    default_artists = [
        "Eminem", "Jay-Z", "Kanye West", "Drake", "Kendrick Lamar",
        "J. Cole", "Travis Scott", "Lil Wayne", "Nicki Minaj", "Cardi B"
    ]
    
    # Сохраняем дефолтный список
    save_artist_list(filename, default_artists)
    return default_artists

def save_artist_list(filename: str, artists: list):
    """Сохранение списка артистов в JSON"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(
            artists, 
            f, 
            indent=2,              # Красивое форматирование
            ensure_ascii=False     # Сохраняем русские символы как есть
        )
    print(f"💾 Список из {len(artists)} артистов сохранен в {filename}")

# Обновление списка оставшихся артистов
def save_remaining_artists(remaining: list):
    """Сохранение списка необработанных артистов"""
    filename = "remaining_artists.json"
    
    if remaining:
        save_artist_list(filename, remaining)
        print(f"📝 Осталось обработать: {len(remaining)} артистов")
    else:
        # Удаляем файл, если артистов не осталось
        if os.path.exists(filename):
            os.remove(filename)
            print("🎉 Все артисты обработаны!")
```

#### 🟡 Работа с конфигурационными файлами

```python
# Загрузка настроек из JSON
def load_config(config_file: str = "config.json") -> dict:
    """Загрузка конфигурации с дефолтными значениями"""
    
    default_config = {
        "api": {
            "genius_token": "",
            "request_delay": 2,
            "max_retries": 3
        },
        "database": {
            "name": "rap_lyrics.db",
            "memory_limit_mb": 2048
        },
        "scraping": {
            "songs_per_artist": 500,
            "max_artists_per_session": 50
        },
        "logging": {
            "level": "INFO",
            "file": "scraping.log"
        }
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                
            # Объединяем дефолтную и пользовательскую конфигурацию
            config = deep_merge(default_config, user_config)
            print(f"⚙️ Конфигурация загружена из {config_file}")
            
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка в JSON конфигурации: {e}")
            print("🔧 Используем настройки по умолчанию")
            config = default_config
            
    else:
        # Создаем файл конфигурации
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        print(f"📝 Создан файл конфигурации: {config_file}")
        config = default_config
    
    return config

def deep_merge(default: dict, user: dict) -> dict:
    """Глубокое слияние словарей конфигурации"""
    result = default.copy()
    
    for key, value in user.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

# Использование конфигурации
config = load_config()
api_token = config["api"]["genius_token"]
db_name = config["database"]["name"]
songs_limit = config["scraping"]["songs_per_artist"]
```

#### 🔴 Продвинутые техники работы с файлами

```python
import csv
import pickle
from pathlib import Path
import tempfile
import shutil

class AdvancedFileManager:
    """Продвинутый менеджер файлов для скрапера"""
    
    def __init__(self, base_dir: str = "scraper_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)  # Создаем директорию, если нет
        
        # Поддиректории
        self.config_dir = self.base_dir / "config"
        self.logs_dir = self.base_dir / "logs"
        self.data_dir = self.base_dir / "data"
        self.backup_dir = self.base_dir / "backups"
        
        # Создаем все директории
        for directory in [self.config_dir, self.logs_dir, self.data_dir, self.backup_dir]:
            directory.mkdir(exist_ok=True)
    
    def export_to_csv(self, data: list, filename: str):
        """Экспорт данных в CSV"""
        csv_path = self.data_dir / filename
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            if not data:
                return
                
            # Автоматически определяем заголовки из первой записи
            fieldnames = data[0].keys() if isinstance(data[0], dict) else ['value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        
        print(f"📊 Экспортировано {len(data)} записей в {csv_path}")
    
    def save_cache(self, data: any, cache_name: str):
        """Сохранение данных в кэш (pickle)"""
        cache_path = self.data_dir / f"{cache_name}.pkl"
        
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Кэш сохранен: {cache_path}")
    
    def load_cache(self, cache_name: str, max_age_hours: int = 24):
        """Загрузка данных из кэша с проверкой возраста"""
        cache_path = self.data_dir / f"{cache_name}.pkl"
        
        if not cache_path.exists():
            return None
        
        # Проверяем возраст файла
        file_age = time.time() - cache_path.stat().st_mtime
        if file_age > max_age_hours * 3600:
            print(f"⏰ Кэш {cache_name} устарел ({file_age/3600:.1f}ч)")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"📂 Кэш загружен: {cache_name}")
            return data
        except (pickle.UnpicklingError, EOFError):
            print(f"❌ Поврежденный кэш: {cache_name}")
            cache_path.unlink()  # Удаляем поврежденный файл
            return None
    
    def create_backup(self, source_file: str, keep_backups: int = 5):
        """Создание резервной копии файла"""
        source_path = Path(source_file)
        if not source_path.exists():
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{source_path.stem}_{timestamp}{source_path.suffix}"
        backup_path = self.backup_dir / backup_name
        
        shutil.copy2(source_path, backup_path)
        print(f"💿 Создана резервная копия: {backup_path}")
        
        # Удаляем старые бэкапы
        self._cleanup_old_backups(source_path.stem, keep_backups)
    
    def _cleanup_old_backups(self, file_stem: str, keep_count: int):
        """Удаление старых резервных копий"""
        pattern = f"{file_stem}_*"
        backup_files = sorted(self.backup_dir.glob(pattern), key=lambda x: x.stat().st_mtime)
        
        if len(backup_files) > keep_count:
            for old_backup in backup_files[:-keep_count]:
                old_backup.unlink()
                print(f"🗑️ Удален старый бэкап: {old_backup.name}")
    
    def safe_write_json(self, data: any, filename: str):
        """Безопасная запись JSON с временным файлом"""
        json_path = self.data_dir / filename
        
        # Записываем во временный файл
        with tempfile.NamedTemporaryFile(
            mode='w', 
            encoding='utf-8', 
            suffix='.json',
            dir=self.data_dir,
            delete=False
        ) as temp_file:
            json.dump(data, temp_file, indent=2, ensure_ascii=False)
            temp_path = temp_file.name
        
        # Атомарно перемещаем временный файл на место целевого
        shutil.move(temp_path, json_path)
        print(f"✅ Безопасно сохранен JSON: {json_path}")
    
    def get_file_info(self, filename: str) -> dict:
        """Получение информации о файле"""
        file_path = Path(filename)
        
        if not file_path.exists():
            return {"exists": False}
        
        stat = file_path.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "created": datetime.fromtimestamp(stat.st_ctime),
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "readable": os.access(file_path, os.R_OK),
            "writable": os.access(file_path, os.W_OK)
        }

# Пример использования
file_manager = AdvancedFileManager()

# Экспорт данных
songs_data = [
    {"artist": "Eminem", "title": "Lose Yourself", "year": 2002},
    {"artist": "Jay-Z", "title": "99 Problems", "year": 2003}
]
file_manager.export_to_csv(songs_data, "songs_export.csv")

# Кэширование
artists_list = ["Eminem", "Jay-Z", "Drake"]
file_manager.save_cache(artists_list, "processed_artists")

# Загрузка кэша
cached_artists = file_manager.load_cache("processed_artists", max_age_hours=12)

# Создание бэкапа
file_manager.create_backup("important_data.json")
```

### ⚠️ Типичные ошибки

1. **Забытое закрытие файлов**
   ```python
   # ПЛОХО — файл может остаться открытым
   file = open("data.txt", "r")
   content = file.read()
   # Если тут произойдет ошибка, файл не закроется
   
   # ХОРОШО — автоматическое закрытие
   with open("data.txt", "r") as file:
       content = file.read()
   # Файл закрывается автоматически
   ```

2. **Проблемы с кодировкой**
   ```python
   # ПЛОХО — может быть ошибка с русскими символами
   with open("russian.txt", "r") as file:
       content = file.read()
   
   # ХОРОШО — явно указываем кодировку
   with open("russian.txt", "r", encoding="utf-8") as file:
       content = file.read()
   ```

3. **Неправильная работа с JSON**
   ```python
   # ПЛОХО — не обрабатываем ошибки JSON
   data = json.load(file)
   
   # ХОРОШО — проверяем валидность
   try:
       data = json.load(file)
   except json.JSONDecodeError as e:
       print(f"Ошибка в JSON: {e}")
       data = {}  # Значение по умолчанию
   ```

### 🎓 Задание для практики

Создай класс `ConfigManager`, который:
1. Загружает настройки из JSON файла
2. Позволяет изменять настройки в runtime
3. Автоматически сохраняет изменения
4. Создает резервные копии при каждом изменении
5. Имеет метод `reset_to_defaults()`

---

## 🎓 Практические задания {#practice}

### 🎯 Задание 1: Базовый скрапер (Уровень 🟢)

Создай упрощенную версию скрапера, которая:

```python
class SimpleTextScraper:
    def __init__(self):
        self.data = []
    
    def add_text(self, title: str, content: str):
        """Добавляет текст в коллекцию"""
        # TODO: Реализуй добавление с проверкой дубликатов
        pass
    
    def search(self, keyword: str) -> list:
        """Ищет тексты, содержащие ключевое слово"""
        # TODO: Реализуй поиск (регистронезависимый)
        pass
    
    def export_to_file(self, filename: str):
        """Экспортирует данные в JSON файл"""
        # TODO: Реализуй экспорт с обработкой ошибок
        pass
    
    def get_stats(self) -> dict:
        """Возвращает статистику коллекции"""
        # TODO: Верни количество текстов, среднюю длину, самый длинный текст
        pass

# Тест твоего кода:
scraper = SimpleTextScraper()
scraper.add_text("Песня 1", "Это текст первой песни про любовь")
scraper.add_text("Песня 2", "Это текст второй песни про дружбу")
print(scraper.search("песни"))  # Должен найти обе
print(scraper.get_stats())      # Статистика
```

### 🎯 Задание 2: Генератор с обработкой ошибок (Уровень 🟡)

Создай генератор для чтения больших файлов:

```python
def safe_file_reader(filename: str, chunk_size: int = 1024):
    """
    Генератор для чтения файла порциями с обработкой ошибок
    
    Args:
        filename: путь к файлу
        chunk_size: размер порции в байтах
    
    Yields:
        str: порция текста из файла
        
    Raises:
        FileNotFoundError: если файл не найден
    """
    # TODO: Реализуй генератор, который:
    # 1. Открывает файл безопасно
    # 2. Читает по chunk_size байт
    # 3. Обрабатывает ошибки кодировки
    # 4. Логирует процесс чтения
    # 5. Автоматически закрывает файл
    pass

# Дополнительно: создай декоратор @retry для повтора при ошибках
def retry(max_attempts: int = 3, delay: float = 1.0):
    """Декоратор для повтора функции при ошибках"""
    # TODO: Реализуй декоратор с экспоненциальной задержкой
    pass
```

### 🎯 Задание 3: Система мониторинга (Уровень 🔴)

Создай продвинутую систему мониторинга:

```python
class AdvancedMonitor:
    """
    Система мониторинга с алертами и метриками
    """
    
    def __init__(self, config: dict):
        # TODO: Инициализация с конфигурацией
        # - Лимиты памяти и CPU
        # - Настройки алертов
        # - Интервалы проверки
        pass
    
    def start_monitoring(self):
        """Запуск мониторинга в фоновом потоке"""
        # TODO: Реализуй фоновый мониторинг
        pass
    
    def add_metric(self, name: str, value: float, tags: dict = None):
        """Добавление кастомной метрики"""
        # TODO: Сохранение метрик с временными метками
        pass
    
    def get_report(self, hours: int = 1) -> dict:
        """Генерация отчета за период"""
        # TODO: Агрегация данных: мин, макс, среднее
        pass
    
    def setup_alerts(self, rules: list):
        """Настройка правил алертов"""
        # TODO: Правила вида: "memory > 80% for 5 minutes"
        pass
    
    @contextmanager
    def measure_time(self, operation_name: str):
        """Контекстный менеджер для измерения времени"""
        # TODO: Реализуй через yield
        pass

# Использование:
# with monitor.measure_time("database_query"):
#     result = db.execute(query)
```

### 🎯 Мини-проекты для портфолио

#### 1. **Log Analyzer** 📊
Анализатор логов веб-сервера:
- Парсинг Apache/Nginx логов регулярками
- Статистика по IP, страницам, кодам ответа
- Экспорт в CSV/JSON
- Графики через matplotlib

#### 2. **Configuration Manager** ⚙️
Система управления конфигурациями:
- Иерархические настройки (dev/prod)
- Валидация схемы настроек
- Горячая перезагрузка конфигурации
- Шифрование секретных параметров

#### 3. **Data Pipeline** 🔄
Пайплайн обработки данных:
- Генераторы для больших файлов
- Цепочка трансформаций
- Параллельная обработка
- Мониторинг прогресса

---

## 💡 Ключевые выводы и следующие шаги

### 🧠 Что мы изучили

1. **Импорты и зависимости** — как организовать подключение библиотек
2. **Логирование** — профессиональный подход к отладке и мониторингу
3. **ООП** — структурирование кода через классы и объекты
4. **База данных** — эффективная работа с SQLite
5. **Регулярные выражения** — мощный инструмент обработки текста
6. **Генераторы** — экономия памяти и элегантный код
7. **Обработка ошибок** — надежность и отказоустойчивость
8. **Файлы и JSON** — постоянное хранение данных

### 🚀 Следующие темы для изучения

#### 🟢 Начинающий → Средний
- **Декораторы** — изменение поведения функций
- **Контекстные менеджеры** — управление ресурсами
- **Многопоточность** — threading и concurrent.futures
- **Тестирование** — pytest и unittest
- **Виртуальные окружения** — venv и poetry

#### 🟡 Средний → Продвинутый
- **Асинхронное программирование** — asyncio и aiohttp
- **Дескрипторы** — продвинутое ООП
- **Метаклассы** — создание классов программно
- **Профилирование** — оптимизация производительности
- **Паттерны проектирования** — Singleton, Factory, Observer

#### 🔴 Продвинутый → Эксперт
- **Расширения на C** — cython и ctypes
- **Memory mapping** — работа с большими файлами
- **GIL и производительность** — глубокое понимание Python
- **Создание библиотек** — packaging и distribution

### 📚 Рекомендуемая литература

1. **"Изучаем Python" Марка Лутца** — подробный учебник
2. **"Эффективный Python" Бретта Слаткина** — лучшие практики
3. **"Архитектура приложений на Python" ** — системный дизайн
4. **"Python Tricks" Дэна Бейдера** — продвинутые техники

### 🏆 Заключение

Разбор rap scraper показал, что даже простая задача сбора данных требует знания многих аспектов Python:

- **Архитектурное мышление** — разделение ответственности между классами
- **Работа с внешними API** — обработка ошибок и лимитов
- **Управление памятью** — генераторы и сборка мусора
- **Персистентность данных** — базы данных и файлы
- **Мониторинг и логирование** — наблюдаемость системы

**Главный урок:** Программирование — это не только знание синтаксиса, но и понимание паттернов, архитектуры и экосистемы языка.

**Твой следующий шаг:** Выбери одно из практических заданий и реализуй его полностью. Затем покажи код ментору или сообществу для получения обратной связи.

---

## 📞 Дополнительные ресурсы

### 🌐 Полезные ссылки
- [Python.org](https://python.org) — официальная документация
- [Real Python](https://realpython.com) — качественные туториалы
- [Python Module of the Week](https://pymotw.com) — разбор стандартной библиотеки
- [GitHub Python Topics](https://github.com/topics/python) — открытые проекты для изучения

### 🛠️ Инструменты разработки
- **PyCharm/VS Code** — IDE с автодополнением
- **Black** — автоформатирование кода
- **Flake8** — проверка стиля кода
- **mypy** — проверка типов
- **pytest** — тестирование

### 💬 Сообщества
- **Telegram:** @ru_python, @pydjango
- **Reddit:** r/Python, r/learnpython
- **Stack Overflow:** тег [python]
- **YouTube:** каналы Corey Schafer, Real Python

**Удачи в изучении Python! 🐍✨**