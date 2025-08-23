## 🎯 Вопросы 1-3: import psutil, SIGBREAK и POSIX

### 1. 🎯 Что делает `import psutil`

**Краткий ответ:** `psutil` — это библиотека для мониторинга системных ресурсов (процессы, память, диски, сеть).

### 2. 🏗️ Контекст и основы

**Зачем нужна?** Для получения информации о системе и управления процессами кроссплатформенно.

**Где применяется?**
- Системный мониторинг
- Управление процессами
- Анализ производительности
- DevOps инструменты

### 3. 💻 Практические примеры

```python
import psutil

# Информация о процессах
for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
    print(f"PID: {proc.info['pid']}, Name: {proc.info['name']}")

# Системные ресурсы
print(f"CPU: {psutil.cpu_percent()}%")
print(f"Memory: {psutil.virtual_memory().percent}%")
print(f"Disk: {psutil.disk_usage('/').percent}%")

# Убить процесс
pid = 1234
p = psutil.Process(pid)
p.terminate()  # Мягкое завершение
p.kill()       # Принудительное завершение
```

---

## 🎯 Вопрос 2: SIGBREAK и POSIX

### 1. 🎯 Краткий ответ
**POSIX** — это стандарт операционных систем, а **SIGBREAK** работает только на Windows.

### 2. 🏗️ Контекст и основы

**POSIX (Portable Operating System Interface)** — стандарт, который определяет:
- Системные вызовы
- Интерфейсы командной строки  
- Утилиты
- Сигналы

**Поддерживают POSIX:** Linux, macOS, Unix, FreeBSD
**Не поддерживает полностью:** Windows

### 3. 💻 Сигналы в разных ОС

```python
import signal
import sys

def setup_signals():
    # POSIX сигналы (работают везде кроме Windows)
    signal.signal(signal.SIGTERM, signal_handler)  # Завершение
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    
    # Windows-специфичные
    if sys.platform == "win32":
        try:
            signal.signal(signal.SIGBREAK, signal_handler)  # Ctrl+Break
        except AttributeError:
            print("SIGBREAK не поддерживается")

def signal_handler(signum, frame):
    print(f"Получен сигнал: {signum}")
    
    # Расшифровка сигналов
    signals_map = {
        2: "SIGINT (Ctrl+C)",
        15: "SIGTERM (завершение процесса)",
        21: "SIGBREAK (Ctrl+Break, только Windows)"
    }
    print(f"Это: {signals_map.get(signum, 'Неизвестный сигнал')}")
```

### 4. ⚠️ Различия платформ

```python
# В твоем коде:
if sys.platform == "win32":
    try:
        signal.signal(signal.SIGBREAK, self._signal_handler)
    except AttributeError:
        pass  # SIGBREAK может отсутствовать даже на Windows
```

**Почему так?** Windows не полностью следует POSIX стандарту и имеет свои специфичные сигналы.

---

## 🎯 Вопрос 3: PID 1234

### 1. 🎯 Краткий ответ
**PID (Process IDentifier)** — уникальный номер процесса в операционной системе.

### 2. 🏗️ Как это работает

```bash
# Посмотреть все процессы Python
ps aux | grep python
# Результат:
# user  1234  0.5  2.1  python3 rap_scraper.py
#       ↑
#     это PID

# Отправить сигнал процессу
kill -TERM 1234  # Мягкое завершение
kill -KILL 1234  # Принудительное завершение (осторожно!)
```

### 3. 💻 Работа с PID в Python

```python
import os
import signal

# Получить собственный PID
my_pid = os.getpid()
print(f"Мой PID: {my_pid}")

# Отправить сигнал самому себе (для тестирования)
os.kill(my_pid, signal.SIGTERM)

# В твоем коде это вызовет:
def _signal_handler(self, signum, frame):
    logger.info(f"Получен сигнал {signum}. Завершение работы...")
    self.shutdown_requested = True
    #                        ↑
    # Это флаг для корректного завершения всех операций
```

### 4. 🔗 Связь с твоим кодом

Когда ты отправляешь `kill -TERM 1234`, происходит:
1. ОС отправляет сигнал SIGTERM процессу 1234
2. Python вызывает `_signal_handler(15, frame_info)`
3. Устанавливается `shutdown_requested = True`
4. Все циклы начинают проверять этот флаг и корректно завершаются

---

### 4. 🎯 Приватные методы в Python

**Краткий ответ:** Приватные методы — это методы, которые не предназначены для использования вне класса.

### 🏗️ Контекст и основы

**Зачем нужны?**
- Скрыть внутреннюю логику класса
- Предотвратить случайное использование
- Показать намерения разработчика
- Облегчить рефакторинг

**В твоем коде:**
```python
def _signal_handler(self, signum, frame):  # Приватный метод
    logger.info(f"Получен сигнал {signum}. Завершение работы...")
    self.shutdown_requested = True

def _is_valid_lyrics(self, lyrics: str) -> bool:  # Приватный метод
    # Внутренняя проверка валидности текста
    if not lyrics or len(lyrics) < 100:
        return False
```

### 💻 Уровни приватности в Python

```python
class MyClass:
    def public_method(self):
        """Публичный метод - используй свободно"""
        return "Доступен всем"
    
    def _protected_method(self):
        """Защищенный метод - для внутреннего использования класса"""
        return "Не используй вне класса"
    
    def __private_method(self):
        """Приватный метод - сильно скрыт"""
        return "Очень сложно получить доступ"

# Использование:
obj = MyClass()
print(obj.public_method())        # ✅ Нормально
print(obj._protected_method())    # ⚠️  Работает, но не рекомендуется
print(obj.__private_method())     # ❌ AttributeError
```

### ⚠️ Важное различие

```python
# Python НЕ блокирует доступ к _методам - это просто конвенция!
obj = MyClass()
obj._protected_method()  # Работает, но показывает "я знаю что делаю"

# Только __ методы действительно скрыты (name mangling)
# obj._MyClass__private_method()  # Так можно получить доступ, но зачем?
```

### 6. 🎯 Underscore `_` в циклах

**Краткий ответ:** `_` означает "переменная-пустышка", когда значение не используется.

### 🏗️ Концепция

```python
# В твоем коде:
for _ in range(intervals):
    if self.shutdown_requested:
        return
    time.sleep(1)
```

**Смысл:** "Мне нужно выполнить цикл N раз, но индекс не важен"

### 💻 Сравнение подходов

**❌ Плохо (переменная не используется):**
```python
for i in range(5):
    print("Hello")  # i не используется, но объявлена
```

**✅ Хорошо (показываем намерения):**
```python
for _ in range(5):
    print("Hello")  # Ясно, что индекс не нужен
```

**✅ Хорошо (когда индекс нужен):**
```python
for i in range(5):
    print(f"Iteration {i}")  # i используется
```

### 🛠️ Другие применения `_`

```python
# 1. Игнорирование возвращаемых значений
result, _ = divmod(10, 3)  # Нужно только частное, остаток игнорируем
print(result)  # 3

# 2. Множественное присваивание
first, _, last = "John Middle Doe".split()
print(f"{first} {last}")  # John Doe

# 3. В циклах с кортежами
users = [("John", 25, "Engineer"), ("Jane", 30, "Designer")]
for name, _, job in users:  # Возраст не нужен
    print(f"{name} - {job}")

# 4. В обработке исключений (иногда)
try:
    risky_operation()
except SpecificError as _:  # Ошибка поймана, но детали не нужны
    handle_error()
```

### ⚠️ Важные нюансы

```python
# _ это обычная переменная! Можно использовать:
_ = "test"
print(_)  # test

# Но в интерпретаторе _ означает последний результат:
>>> 2 + 3
5
>>> _
5
>>> _ * 2  
10

# В gettext для интернационализации:
from gettext import gettext as _
print(_("Hello World"))  # Перевод текста
```

### 🎓 В контексте твоего кода

```python
def safe_delay(self, is_error: bool = False):
    delay = 5.7  # Например
    intervals = int(delay)        # 5
    remainder = delay - intervals # 0.7
    
    # Нужно подождать 5 полных секунд, проверяя shutdown каждую секунду
    for _ in range(intervals):    # range(5) -> [0,1,2,3,4]
        if self.shutdown_requested:
            return  # Прерываем ожидание
        time.sleep(1)  # Ждем 1 секунду
    
    # Потом доспать остаток (0.7 сек)
    if remainder > 0:
        time.sleep(remainder)
```

**Почему `_`?** Потому что нам не важно, это 0-я, 1-я или 4-я итерация — важно только количество.


### 7. 🎯 Разбиение времени на целую часть и остаток

**Краткий ответ:** Разбиваем время, чтобы проверять `shutdown_requested` каждую секунду, а не ждать всю задержку целиком.

### 🏗️ Проблема и решение

**Проблема без разбиения:**
```python
# ❌ Плохо - нельзя прервать
def bad_delay(self, delay_time):
    time.sleep(delay_time)  # Ждем 5.7 секунд БЕЗ возможности прерывания
    # Если пользователь нажмет Ctrl+C на 2-й секунде - придется ждать до конца!
```

**Решение с разбиением:**
```python
# ✅ Хорошо - можно прервать в любой момент
def good_delay(self, delay_time):
    intervals = int(delay_time)      # 5 (целая часть)
    remainder = delay_time - intervals # 0.7 (остаток)
    
    # Ждем по 1 секунде и проверяем флаг
    for _ in range(intervals):  # 5 итераций по 1 сек
        if self.shutdown_requested:
            return  # Прерываем МГНОВЕННО
        time.sleep(1)
    
    # Доспать остаток, если нужно
    if remainder > 0:
        if not self.shutdown_requested:
            time.sleep(remainder)  # 0.7 сек
```

### 💻 Визуальное сравнение

```python
# Пример: delay_time = 5.7 секунд

# ❌ Без разбиения:
# [████████████████████████████] 5.7 сек - НЕ ПРЕРЫВАЕТСЯ
time.sleep(5.7)

# ✅ С разбиением:
# [█] [█] [█] [█] [█] [██] - можно прервать на любом этапе
# 1с  1с  1с  1с  1с  0.7с
for _ in range(5):          # 5 проверок
    if shutdown: return     # Выход на любой секунде
    time.sleep(1)
if remainder > 0:
    time.sleep(0.7)         # Остаток
```

### 🛠️ Практический пример

```python
import time
import signal
import sys

class InterruptibleSleep:
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        print("\nПолучен сигнал прерывания!")
        self.interrupted = True
    
    def smart_sleep(self, total_seconds):
        print(f"Засыпаю на {total_seconds} секунд...")
        
        intervals = int(total_seconds)
        remainder = total_seconds - intervals
        
        # Спим по секундам
        for i in range(intervals):
            if self.interrupted:
                print(f"Прервано на {i+1}-й секунде!")
                return False
            print(f"Секунда {i+1}...")
            time.sleep(1)
        
        # Доспать остаток
        if remainder > 0 and not self.interrupted:
            print(f"Досыпаю {remainder:.1f} сек...")
            time.sleep(remainder)
        
        return not self.interrupted

# Тест:
sleeper = InterruptibleSleep()
sleeper.smart_sleep(5.7)  # Попробуй нажать Ctrl+C на 3-й секунде!
```

---

### 8. 🎯 Дополнительная защита от Ctrl+C

**Краткий ответ:** `KeyboardInterrupt` - это исключение от Ctrl+C, которое может произойти даже внутри `time.sleep()`.

### 🏗️ Двойная защита

```python
# У тебя в коде есть ДВА механизма защиты:

# 1. Обработчик сигналов (глобальный)
signal.signal(signal.SIGINT, self._signal_handler)  # Ловит Ctrl+C

# 2. Обработка исключений (локальная)
try:
    time.sleep(1)
except KeyboardInterrupt:  # Дополнительная сетка безопасности
    self.shutdown_requested = True
    return
```

### 💻 Почему нужны ОБА способа?

```python
import time
import signal

class SingleProtection:
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGINT, self._handler)
    
    def _handler(self, sig, frame):
        print("Сигнал получен!")
        self.shutdown = True
    
    def unsafe_sleep(self):
        # ❌ ПРОБЛЕМА: time.sleep может быть прерван ДО проверки флага
        for _ in range(10):
            if self.shutdown:  # Проверка
                return
            time.sleep(1)      # ← Ctrl+C здесь может вызвать KeyboardInterrupt
            # ↑ Исключение "перепрыгнет" через проверку флага!

class DoubleProtection:
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGINT, self._handler)
    
    def _handler(self, sig, frame):
        self.shutdown = True
    
    def safe_sleep(self):
        # ✅ РЕШЕНИЕ: ловим исключение + проверяем флаг
        for _ in range(10):
            if self.shutdown:
                return
            try:
                time.sleep(1)
            except KeyboardInterrupt:  # Ловим исключение от Ctrl+C
                self.shutdown = True   # Устанавливаем флаг
                return                 # И выходим
```

### ⚠️ Разные ситуации прерывания

```python
# Ситуация 1: Ctrl+C между проверками
for _ in range(5):
    if shutdown: return        # ← Ctrl+C здесь: обработчик сработает
    time.sleep(1)

# Ситуация 2: Ctrl+C во время sleep
for _ in range(5):
    if shutdown: return
    time.sleep(1)              # ← Ctrl+C здесь: KeyboardInterrupt исключение

# Ситуация 3: Быстрый двойной Ctrl+C
# Первый Ctrl+C → сигнал обработчик
# Второй Ctrl+C → KeyboardInterrupt исключение
```

### 🔗 В контексте твоего кода

Это особенно важно для длительных операций:
```python
# Твой скрапер может прерваться в ЛЮБОЙ момент:
artist = self.genius.search_artist(...)  # ← Ctrl+C здесь
for song in artist.songs:                 # ← или здесь
    lyrics = song.lyrics                   # ← или здесь
    self.safe_delay()                      # ← или внутри задержки
```

---

### 9. 🎯 Почему `max_songs=max_songs`?

**Краткий ответ:** Передаем значение параметра функции в метод API.

### 🏗️ Поток данных

```python
def scrape_artist_songs(self, artist_name: str, max_songs: int = 500):
    #                                            ↑
    #                                  параметр функции
    
    artist = self.genius.search_artist(
        artist_name, 
        max_songs=max_songs,  # ← передаем значение параметра в API
        sort="popularity"
    )
```

### 💻 Развернутый пример

```python
# Вызов из main:
scraper.scrape_artist_songs("Eminem", max_songs=100)
                                      ↓
# Внутри функции:
def scrape_artist_songs(self, artist_name, max_songs=500):
    #                                      ↑
    #                              max_songs теперь равен 100
    
    # Передаем в API Genius:
    artist = self.genius.search_artist(
        "Eminem",
        max_songs=100,    # ← используем полученное значение
        sort="popularity"
    )
```

### ⚠️ Альтернативные варианты (хуже)

```python
# ❌ Плохо - захардкодено:
def scrape_artist_songs(self, artist_name):
    artist = self.genius.search_artist(
        artist_name,
        max_songs=500,  # ← всегда 500, нельзя изменить
    )

# ❌ Плохо - обращение к self.max_songs:
def scrape_artist_songs(self, artist_name):
    artist = self.genius.search_artist(
        artist_name,
        max_songs=self.max_songs,  # ← нужно создавать атрибут класса
    )

# ✅ Хорошо - гибкий параметр:
def scrape_artist_songs(self, artist_name, max_songs=500):
    artist = self.genius.search_artist(
        artist_name,
        max_songs=max_songs,  # ← можно менять для каждого вызова
    )

# Примеры использования:
scraper.scrape_artist_songs("Eminem", 100)     # Только 100 песен
scraper.scrape_artist_songs("Drake", 1000)     # Аж 1000 песен  
scraper.scrape_artist_songs("Jay-Z")           # По умолчанию 500
```

### 🎓 Принцип параметризации

```python
# Общий принцип: делай функции настраиваемыми
def process_data(data, chunk_size=1000, timeout=30):
    #                    ↑               ↑
    #              значения по умолчанию, но можно изменить
    
    api_client.send(data, 
                   chunk_size=chunk_size,  # передаем дальше
                   timeout=timeout)

# Использование:
process_data(my_data)                          # стандартно
process_data(my_data, chunk_size=500)          # маленькими порциями  
process_data(my_data, timeout=60)             # больше времени ожидания
process_data(my_data, chunk_size=100, timeout=10)  # кастомно
```

---

### 10. 🎯 Почему `['total_songs']` в квадратных, а не круглых скобках?

**Краткий ответ:** Квадратные скобки `[]` — это доступ к элементу, круглые `()` — это вызов функции.

### 🏗️ Различия синтаксиса Python

```python
# В твоем коде:
current_stats = self.db.get_stats()  # ← возвращает словарь
# current_stats = {"total_songs": 42, "unique_artists": 5}

print(current_stats['total_songs'])  # ← доступ к ключу словаря
#                   ↑
#            квадратные скобки для индексации
```

### 💻 Сравнение синтаксиса

```python
# 🟡 СЛОВАРИ - квадратные скобки []
my_dict = {"name": "John", "age": 25}
print(my_dict['name'])        # ✅ "John"
print(my_dict('name'))        # ❌ TypeError: 'dict' object is not callable

# 🟡 СПИСКИ - тоже квадратные скобки []
my_list = ["apple", "banana", "cherry"]
print(my_list[0])             # ✅ "apple"
print(my_list(0))             # ❌ TypeError: 'list' object is not callable

# 🟡 ФУНКЦИИ - круглые скобки ()
def my_function(param):
    return param * 2

result = my_function(5)       # ✅ 10
result = my_function[5]       # ❌ TypeError: 'function' object is not subscriptable

# 🟡 КОРТЕЖИ - создание круглые (), доступ квадратные []
my_tuple = ("red", "green", "blue")
print(my_tuple[1])            # ✅ "green"
print(my_tuple(1))            # ❌ TypeError: 'tuple' object is not callable
```

### 🛠️ Практические примеры из твоего кода

```python
class LyricsDatabase:
    def get_stats(self) -> dict:
        cur = self.conn.execute("SELECT COUNT(*) as total, ...")
        result = cur.fetchone()  # ← это sqlite3.Row (как словарь)
        return {
            "total_songs": result["total"],      # ← доступ к колонке
            "unique_artists": result["artists"]
        }

# Использование:
stats = db.get_stats()           # ← вызов функции ()
total = stats['total_songs']     # ← доступ к ключу []
print(f"Songs: {total}")

# Альтернативы доступа к данным:
stats.get('total_songs')         # ← метод get() с ()
stats['total_songs']             # ← прямой доступ с []
```

### ⚠️ Типичные ошибки

```python
# ❌ Путаница скобок:
data = {"key": "value"}
print(data('key'))              # TypeError!

# ❌ Забыли кавычки:
print(data[key])               # NameError: name 'key' is not defined

# ✅ Правильно:
print(data['key'])             # "value"

# ❌ Функция как словарь:
def get_name():
    return "John"

print(get_name['name'])        # TypeError!

# ✅ Правильно:
print(get_name())              # "John"
```

---

### 11. 🎯 Модульная арифметика `(i + 1) % 10 == 0`

**Краткий ответ:** Операция `%` (модуль) дает остаток от деления. Формула проверяет "каждые 10 элементов".

### 🏗️ Как работает модуль (%)

```python
# Модуль = остаток от деления
print(10 % 3)  # 1 (10 ÷ 3 = 3 остаток 1)
print(15 % 4)  # 3 (15 ÷ 4 = 3 остаток 3)
print(20 % 5)  # 0 (20 ÷ 5 = 4 остаток 0)

# Когда остаток = 0? Когда число нацело делится!
print(10 % 10)  # 0
print(20 % 10)  # 0  
print(30 % 10)  # 0
```

### 💻 Разбор твоей формулы

```python
# В твоем коде:
for i, song in enumerate(artist.songs):  # i = 0, 1, 2, 3, ...
    # ... обработка песни ...
    
    if (i + 1) % 10 == 0:  # Проверяем каждые 10 песен
        logger.info(f"Processed {i + 1}/{len(artist.songs)} songs")

# Пошаговая работа:
# i=0:  (0+1) % 10 = 1 % 10 = 1 ≠ 0  → не выводим
# i=1:  (1+1) % 10 = 2 % 10 = 2 ≠ 0  → не выводим  
# i=2:  (2+1) % 10 = 3 % 10 = 3 ≠ 0  → не выводим
# ...
# i=9:  (9+1) % 10 = 10 % 10 = 0 = 0 → выводим "Processed 10/100"
# i=10: (10+1) % 10 = 11 % 10 = 1 ≠ 0 → не выводим
# ...
# i=19: (19+1) % 10 = 20 % 10 = 0 = 0 → выводим "Processed 20/100"
```

### 🛠️ Визуализация

```python
def demo_modulo():
    for i in range(25):  # Имитируем 25 песен
        processed_count = i + 1
        
        if processed_count % 10 == 0:  # Каждые 10
            print(f"✅ Milestone: {processed_count} songs processed")
        else:
            print(f"   Processing song {processed_count}...")

# Результат:
#    Processing song 1...
#    Processing song 2...
#    ...
#    Processing song 9...
# ✅ Milestone: 10 songs processed
#    Processing song 11...
#    ...
# ✅ Milestone: 20 songs processed
```

### 🧮 Важные формулы для ML/Data Science

```python
# 1. БАТЧИ - обработка по частям
batch_size = 32
for i in range(len(dataset)):
    if (i + 1) % batch_size == 0:  # Каждые 32 элемента
        process_batch()
        print(f"Batch {(i+1)//batch_size} completed")

# 2. ЭПОХИ - полные проходы по данным  
for epoch in range(100):
    if (epoch + 1) % 10 == 0:  # Каждые 10 эпох
        save_model()
        evaluate_model()

# 3. ЛОГИРОВАНИЕ - не засорять логи
for step in range(10000):
    train_step()
    if step % 100 == 0:  # Каждые 100 шагов
        log_metrics()

# 4. ЧЕКПОИНТЫ - сохранение промежуточных результатов
for iteration in range(1000000):
    if iteration % 5000 == 0:  # Каждые 5000 итераций
        save_checkpoint()

# 5. ВАЛИДАЦИЯ - проверка на тестовых данных
for batch_idx in range(num_batches):
    if batch_idx % validation_frequency == 0:
        validate_model()

# 6. ЦИКЛИЧЕСКИЙ LEARNING RATE
learning_rate = base_lr * (0.5 ** (epoch // 30))  # Каждые 30 эпох уменьшаем в 2 раза
```

### 🎓 Почему `(i + 1)` а не просто `i`?

```python
# С i (индексы 0-based):
for i in range(10):  # i = 0, 1, 2, ..., 9
    if i % 3 == 0:
        print(f"Step {i}")
# Результат: Step 0, Step 3, Step 6, Step 9

# С (i + 1) (счетчик 1-based):  
for i in range(10):  # i = 0, 1, 2, ..., 9
    if (i + 1) % 3 == 0:
        print(f"Step {i + 1}")
# Результат: Step 3, Step 6, Step 9

# В твоем случае логичнее "обработано 10 песен", а не "обработано 9 песен"
```

---

### 12. 🎯 Что такое `timeout_e`?

**Краткий ответ:** `timeout_e` — это переменная, которая содержит объект исключения (Exception), пойманный в блоке `except`.

### 🏗️ Анатомия обработки исключений

```python
# В твоем коде:
except Exception as timeout_e:
#      ↑               ↑
#   тип исключения   имя переменной
```

**Это означает:** "Поймай любое исключение и сохрани его в переменную `timeout_e`"

### 💻 Подробный разбор

```python
try:
    # Опасная операция
    song_lyrics = genius_api.get_lyrics(song_id)
    
except Exception as timeout_e:
    #              ↑
    # timeout_e теперь содержит объект исключения
    
    print(type(timeout_e))        # <class 'requests.exceptions.Timeout'>
    print(str(timeout_e))         # "HTTPSConnectionPool(...): Read timed out"
    print(timeout_e.args)         # Аргументы исключения
    
    # Проверяем тип ошибки:
    if "timeout" in str(timeout_e).lower():
        print("Это действительно timeout!")
        handle_timeout()
    else:
        print("Это другая ошибка!")
        raise timeout_e  # Перебрасываем дальше
```

### 🛠️ Типы исключений в API

```python
import requests
import time

def demo_exceptions():
    try:
        # Имитируем разные ошибки
        response = requests.get("https://api.genius.com/...", timeout=1)
        
    except requests.exceptions.Timeout as e:
        print(f"Timeout error: {e}")
        # Объект исключения содержит детали
        
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        print(f"Status code: {e.response.status_code}")
        
    except Exception as e:  # Ловим все остальное
        print(f"Unknown error: {type(e).__name__}: {e}")
```

### ⚠️ Почему именно такая логика в коде?

```python
# Твоя логика:
except Exception as timeout_e:
    if "timeout" in str(timeout_e).lower():
        # Timeout - ожидаемая проблема, просто ждем и продолжаем
        logger.error(f"Timeout for {song.title}: {timeout_e}")
        self.session_stats["errors"] += 1
        self.safe_delay(is_error=True)  # Пауза 15 сек
    else:
        # Неизвестная ошибка - может быть критичная
        raise timeout_e  # Перебрасываем выше для обработки
```

**Смысл:** Timeout — это "мягкая" ошибка (сервер медленный), можно продолжить. Другие ошибки могут быть критичными (нет интернета, неправильный API ключ).

### 🔗 Альтернативные подходы

```python
# 🟡 Более специфичный подход:
try:
    lyrics = get_lyrics(song_id)
except (requests.Timeout, socket.timeout) as e:
    handle_timeout(e)
except requests.HTTPError as e:
    if e.response.status_code == 429:  # Rate limit
        handle_rate_limit(e)
    else:
        raise e
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise e

# 🟡 Еще более детальный:
try:
    lyrics = get_lyrics(song_id)
except requests.exceptions.Timeout:
    retry_with_longer_timeout()
except requests.exceptions.ConnectionError:
    check_internet_connection()  
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        log_song_not_found()
    elif e.response.status_code == 429:
        handle_rate_limit()
    else:
        raise e
```

### 🎓 Имя переменной исключения

```python
# Можно назвать как угодно:
except Exception as e:           # Стандартное имя
except Exception as error:       # Описательное
except Exception as timeout_e:   # Специфичное (твой случай)
except Exception as ex:          # Короткое

# В твоем коде timeout_e намекает, что ожидаешь в основном timeout ошибки
```

---


### 13. 🎯 Подробный разбор `except Exception as e`

**Краткий ответ:** `except` ловит исключения (ошибки), `Exception` — базовый класс всех ошибок, `as e` — сохраняет ошибку в переменную.

### 🏗️ Иерархия исключений Python

```python
# Упрощенная иерархия:
BaseException
 +-- SystemExit          # exit()
 +-- KeyboardInterrupt   # Ctrl+C
 +-- Exception           # ← Все "нормальные" ошибки
      +-- ArithmeticError
      |    +-- ZeroDivisionError
      +-- LookupError
      |    +-- KeyError
      |    +-- IndexError  
      +-- OSError
      |    +-- FileNotFoundError
      +-- ValueError
      +-- TypeError
      +-- RuntimeError
           +-- RecursionError
```

### 💻 Как работает try-except

```python
def demo_exceptions():
    try:
        # Блок "попробуй выполнить это"
        risky_code()
        print("Все прошло хорошо!")
        
    except SpecificError as e:
        # "Если произошла конкретная ошибка"
        print(f"Известная проблема: {e}")
        
    except Exception as e:
        # "Если произошла любая другая ошибка"
        print(f"Неожиданная проблема: {e}")
        
    else:
        # "Выполнить, если НЕ было ошибок"
        print("Try блок выполнился без ошибок")
        
    finally:
        # "Выполнить ВСЕГДА (даже при ошибке)"
        print("Очистка ресурсов")
```

### 🛠️ Разбор твоего кода по частям

```python
# В твоем коде:
except Exception as e:
    if "rate limit" in str(e).lower() or "429" in str(e):
        # Rate Limit - слишком много запросов
        logger.error(f"Rate Limit for {artist_name}: {e}")
        logger.info(f"Waiting 60 seconds before retry...")
        self.safe_delay(is_error=True)  # 15 сек
        if not self.shutdown_requested:
            time.sleep(60)              # + еще 60 сек
        retry_count += 1
    else:
        # Другая ошибка
        retry_count += 1
        logger.error(f"Error with artist {artist_name} (attempt {retry_count}): {e}")
        logger.error(f"Error type: {type(e).__name__}")
        if retry_count >= self.max_retries:
            logger.error(f"Max retries reached for {artist_name}")
            self.session_stats["errors"] += 1
            break
        logger.info(f"Pause before retry...")
        self.safe_delay(is_error=True)
```

### 🔍 Пошаговый анализ

**1. Ловим исключение:**
```python
except Exception as e:
# e содержит объект ошибки, например:
# requests.exceptions.HTTPError: 429 Client Error: Too Many Requests
```

**2. Анализируем тип ошибки:**
```python
if "rate limit" in str(e).lower() or "429" in str(e):
# str(e) = "429 Client Error: Too Many Requests for url: ..."
# str(e).lower() = "429 client error: too many requests for url: ..."
# "rate limit" in ... = False
# "429" in ... = True ✓
```

**3. Специальная обработка Rate Limit:**
```python
# Rate Limit = API говорит "слишком много запросов"
logger.error(f"Rate Limit for {artist_name}: {e}")
self.safe_delay(is_error=True)  # Ждем 15 секунд
time.sleep(60)                  # + еще 60 секунд = итого 75 сек
retry_count += 1                # Увеличиваем счетчик попыток
```

**4. Обработка других ошибок:**
```python
else:
    # Неизвестная ошибка - может быть что угодно
    retry_count += 1
    logger.error(f"Error with artist {artist_name} (attempt {retry_count}): {e}")
    logger.error(f"Error type: {type(e).__name__}")  # Показываем класс ошибки
```

### ⚠️ Типичные ошибки в API

```python
# Примеры реальных исключений:

# 1. Timeout (медленный сервер)
# requests.exceptions.Timeout: HTTPSConnectionPool: Read timed out

# 2. Rate Limit (слишком быстро)
# requests.exceptions.HTTPError: 429 Client Error: Too Many Requests

# 3. Нет интернета
# requests.exceptions.ConnectionError: Failed to establish a new connection

# 4. Неправильный API ключ  
# requests.exceptions.HTTPError: 401 Client Error: Unauthorized

# 5. Сервер недоступен
# requests.exceptions.HTTPError: 500 Server Error: Internal Server Error
```

---

### 14. 🎯 Retry логика: почему только на уровне артиста?

**Краткий ответ:** Ошибки API (rate limit, неправильный ключ) влияют на ВСЕ запросы, а не на конкретную песню.

### 🏗️ Архитектура API и ошибок

```python
# 🟡 УРОВЕНЬ АРТИСТА (где retry):
try:
    artist = genius.search_artist("Eminem")  # ← API запрос
    # Если здесь 429 Rate Limit - это проблема со всем подключением
    
    for song in artist.songs:  # ← Уже НЕ API запрос
        # song.lyrics уже загружены вместе с артистом
        
except Exception as e:
    # Retry имеет смысл - может помочь

# 🟡 УРОВЕНЬ ПЕСНИ (где retry НЕ нужен):
for song in artist.songs:  # artist уже загружен успешно
    try:
        lyrics = clean_lyrics(song.lyrics)  # ← Обработка текста
        # Если здесь ошибка - это проблема конкретной песни
        
    except Exception as e:
        # Retry бессмысленен - ошибка в данных этой песни
        continue  # Просто пропускаем песню
```

### 💻 Визуализация разных типов ошибок

```python
# 🔴 СИСТЕМНЫЕ ОШИБКИ (влияют на ВСЕ):
# - Rate Limit (429)         → retry поможет
# - Неправильный API ключ    → retry НЕ поможет, но можно попробовать
# - Нет интернета           → retry может помочь
# - Сервер недоступен       → retry может помочь

# 🟡 ОШИБКИ ДАННЫХ (влияют на ОДНУ песню):
# - Битые символы в тексте  → retry НЕ поможет
# - Пустой текст песни      → retry НЕ поможет  
# - Некорректный формат     → retry НЕ поможет

def demonstrate_error_levels():
    # УРОВЕНЬ 1: Поиск артиста (системные ошибки)
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            artist = genius.search_artist("Drake")  # API call
            break  # Успех - выходим из retry цикла
            
        except requests.HTTPError as e:
            if "429" in str(e):  # Rate limit
                print(f"Rate limit, retry {retry_count + 1}/{max_retries}")
                time.sleep(60)  # Ждем и повторяем
                retry_count += 1
            else:
                raise e  # Другая HTTP ошибка
                
        except requests.ConnectionError:
            print(f"Connection error, retry {retry_count + 1}/{max_retries}")
            time.sleep(10)
            retry_count += 1
    
    # УРОВЕНЬ 2: Обработка песен (ошибки данных)
    for song in artist.songs:
        try:
            # Здесь НЕ нужен retry - если данные битые, 
            # повторный запрос вернет те же битые данные
            lyrics = process_lyrics(song.lyrics)
            save_to_db(lyrics)
            
        except UnicodeDecodeError:
            print(f"Skip song {song.title} - bad encoding")
            continue  # Просто пропускаем
            
        except ValueError as e:
            print(f"Skip song {song.title} - invalid data: {e}")
            continue  # Просто пропускаем
```

### 🎓 Принципы retry логики

```python
# ✅ ХОРОШИЕ кандидаты для retry:
# - Временные сетевые проблемы
# - Rate limiting
# - Server overload (503)
# - Timeout errors

# ❌ ПЛОХИЕ кандидаты для retry:
# - Неправильные параметры (400 Bad Request)
# - Отсутствующие данные (404 Not Found)  
# - Проблемы авторизации (401 Unauthorized)
# - Ошибки валидации данных

def smart_retry_logic(func, *args, **kwargs):
    """Умная retry логика"""
    retryable_errors = [
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    ]
    
    retryable_status_codes = [429, 500, 502, 503, 504]
    
    for attempt in range(3):
        try:
            return func(*args, **kwargs)
            
        except tuple(retryable_errors):
            if attempt < 2:  # Не последняя попытка
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            raise
            
        except requests.HTTPError as e:
            if e.response.status_code in retryable_status_codes:
                if attempt < 2:
                    time.sleep(60 if e.response.status_code == 429 else 10)
                    continue
            raise
```

---

### 15. 🎯 `continue` vs `if-else`: сравнение подходов

**Краткий ответ:** `continue` делает код более плоским и читаемым, избегая глубокой вложенности.

### 💻 Сравнение плохого и хорошего кода

**❌ ПЛОХО - глубокая вложенность:**
```python
for song in artist.songs:
    if not self.db.song_exists(url=song.url):  # Проверка дубликата
        lyrics = self.clean_lyrics(song.lyrics)
        if self._is_valid_lyrics(lyrics):  # Проверка валидности
            if self.db.add_song(artist_name, song.title, lyrics, song.url):  # Добавление
                added_count += 1
                logger.info(f"Added: {song.title}")
                
                if added_count % 5 == 0:  # Статистика
                    stats = self.db.get_stats()
                    logger.info(f"Stats: {stats['total_songs']} songs")
                    
            else:  # Ошибка добавления
                logger.error(f"Failed to add: {song.title}")
                
        else:  # Невалидный текст
            logger.debug(f"Skip (invalid): {song.title}")
            
    else:  # Дубликат
        logger.debug(f"Skip (duplicate): {song.title}")
        
    # Задержка в любом случае
    self.safe_delay()
```

**✅ ХОРОШО - с continue:**
```python
for song in artist.songs:
    # Ранние выходы (early returns)
    if self.db.song_exists(url=song.url):
        logger.debug(f"Skip (duplicate): {song.title}")
        continue  # Переходим к следующей песне
    
    lyrics = self.clean_lyrics(song.lyrics)
    if not self._is_valid_lyrics(lyrics):
        logger.debug(f"Skip (invalid): {song.title}")
        continue  # Переходим к следующей песне
    
    # Основная логика - без вложенности
    if self.db.add_song(artist_name, song.title, lyrics, song.url):
        added_count += 1
        logger.info(f"Added: {song.title}")
        
        if added_count % 5 == 0:
            stats = self.db.get_stats()
            logger.info(f"Stats: {stats['total_songs']} songs")
    else:
        logger.error(f"Failed to add: {song.title}")
    
    self.safe_delay()
```

### 🛠️ Принцип "Guard Clauses" (защитные условия)

```python
def process_user_data(user_data):
    # ❌ Плохо - пирамида doom
    if user_data is not None:
        if 'email' in user_data:
            if '@' in user_data['email']:
                if len(user_data['email']) > 5:
                    # Основная логика где-то в глубине
                    return send_email(user_data['email'])
                else:
                    return "Email too short"
            else:
                return "Invalid email format"
        else:
            return "Email missing"
    else:
        return "No user data"

def process_user_data_better(user_data):
    # ✅ Хорошо - guard clauses
    if user_data is None:
        return "No user data"
    
    if 'email' not in user_data:
        return "Email missing"
    
    if '@' not in user_data['email']:
        return "Invalid email format"
    
    if len(user_data['email']) <= 5:
        return "Email too short"
    
    # Основная логика на верхнем уровне!
    return send_email(user_data['email'])
```

### 🎓 Когда использовать каждый подход

```python
# ✅ ИСПОЛЬЗУЙ continue КОГДА:
# - Много условий для пропуска
# - Хочешь избежать глубокой вложенности
# - Логика "если что-то не так - пропусти"

for item in items:
    if item.is_deleted:
        continue
    if item.is_expired:
        continue  
    if not item.is_valid:
        continue
    
    # Главная логика здесь
    process(item)

# ✅ ИСПОЛЬЗУЙ if-else КОГДА:
# - Простое ветвление
# - Нужно обработать ОБА случая
# - Логика "если это - то это, иначе - то"

for item in items:
    if item.is_premium:
        apply_premium_discount(item)
    else:
        apply_regular_discount(item)
    
    # В любом случае продолжаем
    process(item)

# ❌ НЕ СМЕШИВАЙ continue и сложную логику после него:
for item in items:
    if should_skip(item):
        continue
    
    # Этот код может быть пропущен - неочевидно!
    if complex_condition(item):
        do_something()
    else:
        do_something_else()
```

### ⚠️ Подводные камни

```python
# ❌ ОСТОРОЖНО - continue пропускает ВСЕ что после:
for i in range(10):
    if i % 2 == 0:
        continue
    
    print(f"Odd number: {i}")
    important_cleanup()  # ← Выполнится только для нечетных!

# ✅ ЛУЧШЕ - явно показать что делаем:
for i in range(10):
    if i % 2 == 0:
        print(f"Skip even number: {i}")
        continue
        
    print(f"Process odd number: {i}")
    important_cleanup()  # Теперь ясно когда выполняется
```

---

### 16. 🎯 Зачем проверять остановку ДВА раза?

**Краткий ответ:** Первая проверка — быстрый выход из цикла, вторая — избежание ненужного ожидания перед паузой.

### 🏗️ Анализ двойной проверки

```python
# В твоем коде:
for i, song in enumerate(artist.songs):
    if self.shutdown_requested:  # ← ПРОВЕРКА №1: в начале цикла
        logger.info(f"Stopping at song {i+1}/{len(artist.songs)}")
        break
    
    # ... обработка песни ...
    
    if self.shutdown_requested:  # ← ПРОВЕРКА №2: перед паузой
        break
    
    self.safe_delay()  # ← Пауза 3-7 секунд
```

### 💻 Сценарии работы

**Сценарий 1: Ctrl+C в начале итерации**
```python
# Пользователь нажал Ctrl+C между песнями
for i, song in enumerate(artist.songs):
    if self.shutdown_requested:  # ← Сработает здесь
        break  # Мгновенный выход, без обработки песни
    
    process_song(song)  # ← НЕ выполнится
    self.safe_delay()   # ← НЕ выполнится
```

**Сценарий 2: Ctrl+C после обработки песни**
```python
for i, song in enumerate(artist.songs):
    if self.shutdown_requested:  # ← НЕ сработает
        break
    
    process_song(song)  # ← Выполнится (песня обработается)
    
    if self.shutdown_requested:  # ← Сработает здесь
        break  # Выход БЕЗ ненужной паузы
    
    self.safe_delay()  # ← НЕ выполнится (избегаем 3-7 сек ожидания)
```

**Сценарий 3: Ctrl+C во время паузы**
```python
for i, song in enumerate(artist.songs):
    if self.shutdown_requested:  # НЕ сработает
        break
    
    process_song(song)  # Выполнится
    
    if self.shutdown_requested:  # НЕ сработает  
        break
    
    self.safe_delay()  # ← Ctrl+C здесь прервет задержку изнутри
```

### 🛠️ Демонстрация без двойной проверки

```python
# ❌ БЕЗ второй проверки:
def bad_loop():
    for i in range(100):
        if self.shutdown_requested:
            break
        
        process_item(i)  # 1 секунда работы
        # Нет проверки здесь!
        time.sleep(5)    # 5 секунд ненужного ожидания после Ctrl+C

# ✅ С двойной проверкой:
def good_loop():
    for i in range(100):
        if self.shutdown_requested:
            break
            
        process_item(i)  # 1 секунда работы
        
        if self.shutdown_requested:  # Дополнительная проверка
            break  # Избегаем 5-секундной задержки
            
        time.sleep(5)
```

### 🎯 Реальный пример с таймингом

```python
import time
import signal

class TaskProcessor:
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        print("\n🛑 Shutdown signal received!")
        self.shutdown_requested = True
    
    def single_check_loop(self):
        """❌ Только одна проверка - плохо"""
        print("Single check loop started (press Ctrl+C)")
        
        for i in range(10):
            if self.shutdown_requested:
                print(f"Exiting at iteration {i}")
                break
            
            print(f"Processing item {i}...")
            time.sleep(1)  # Имитация работы
            
            # НЕТ проверки здесь!
            print(f"Starting 5-second delay after item {i}...")
            time.sleep(5)  # ← Ненужное ожидание после Ctrl+C
    
    def double_check_loop(self):
        """✅ Двойная проверка - хорошо"""  
        print("Double check loop started (press Ctrl+C)")
        
        for i in range(10):
            if self.shutdown_requested:
                print(f"Quick exit at start of iteration {i}")
                break
                
            print(f"Processing item {i}...")
            time.sleep(1)  # Имитация работы
            
            if self.shutdown_requested:  # ← Дополнительная проверка
                print(f"Smart exit after processing item {i}")
                break
            
            print(f"Starting 5-second delay after item {i}...")
            time.sleep(5)

# Тест:
processor = TaskProcessor()
processor.double_check_loop()  # Попробуй нажать Ctrl+C в разные моменты
```

---

### 17. 🎯 Timeout vs другие исключения

**Краткий ответ:** Timeout — ожидаемая "мягкая" ошибка (можно продолжить), другие ошибки могут быть критичными (нужно выбросить выше).

### 🏗️ Классификация ошибок

```python
# 🟢 МЯГКИЕ ОШИБКИ (можно игнорировать/повторить):
# - Timeout (сервер медленный)
# - Temporary network issues  
# - Server temporarily overloaded

# 🔴 ЖЕСТКИЕ ОШИБКИ (критичные):
# - Invalid API key (401)
# - Rate limit exceeded (требует долгого ожидания)
# - Programming errors (bugs in code)
# - Data corruption
```

### 💻 Разбор логики в твоем коде

```python
except Exception as timeout_e:
    if "timeout" in str(timeout_e).lower():
        # 🟢 МЯГКАЯ ОШИБКА: timeout
        logger.error(f"Timeout for {song.title}: {timeout_e}")
        self.session_stats["errors"] += 1
        self.safe_delay(is_error=True)  # Пауза и продолжаем
        # НЕ выбрасываем ошибку дальше - продолжаем со следующей песней
        
    else:
        # 🔴 НЕИЗВЕСТНАЯ ОШИБКА: может быть критичной
        raise timeout_e  # Выбрасываем выше для обработки на уровне артиста
```

### 🛠️ Подробная демонстрация

```python
def demonstrate_error_handling():
    songs = ["Song1", "Song2", "Song3", "Song4", "Song5"]
    
    for song in songs:
        try:
            result = fetch_song_lyrics(song)  # Может упасть с разными ошибками
            print(f"✅ {song}: {result}")
            
        except Exception as e:
            error_message = str(e).lower()
            
            # МЯГКИЕ ОШИБКИ - логируем и продолжаем
            if "timeout" in error_message:
                print(f"⏰ {song}: Timeout, skipping...")
                continue  # Переходим к следующей песне
                
            elif "connection" in error_message:
                print(f"🌐 {song}: Connection issue, skipping...")
                continue
                
            elif "temporarily unavailable" in error_message:
                print(f"⏸️ {song}: Temporary issue, skipping...")
                continue
            
            # ЖЕСТКИЕ ОШИБКИ - останавливаем все
            else:
                print(f"💥 {song}: Critical error: {e}")
                print("🛑 Stopping entire process due to critical error")
                raise e  # Выбрасываем наверх - весь процесс остановится

def fetch_song_lyrics(song_name):
    """Имитация API запроса с разными ошибками"""
    import random
    
    error_types = [
        "Success",
        "Connection timeout occurred",
        "Temporary server overload", 
        "Invalid API key provided",  # ← Критичная
        "Database connection failed"  # ← Критичная
    ]
    
    result = random.choice(error_types)
    if result == "Success":
        return f"Lyrics for {song_name}"
    else:
        raise Exception(result)
```

### ⚠️ Реальные примеры ошибок

```python
# 🟢 МЯГКИЕ (продолжаем):
try:
    response = requests.get(url, timeout=10)
except requests.exceptions.Timeout:
    # Сервер медленный - не критично
    logger.warning("Timeout, skipping this request")
    continue

except requests.exceptions.ConnectionError as e:
    if "temporary failure" in str(e).lower():
        # Временная проблема сети
        logger.warning("Temporary connection issue")
        continue

# 🔴 ЖЕСТКИЕ (останавливаем):
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        # Неправильный API ключ - дальше работать нельзя
        logger.error("Invalid API credentials!")
        raise e
    
    elif e.response.status_code == 403:
        # Доступ запрещен - возможно забанили
        logger.error("API access forbidden!")
        raise e

except Exception as e:
    # Неожиданная ошибка - может быть баг в коде
    logger.error(f"Unexpected error: {type(e).__name__}: {e}")
    raise e
```

### 🎓 Стратегии обработки

```python
# 📋 СТРАТЕГИЯ 1: Graceful Degradation
def graceful_approach():
    for item in items:
        try:
            result = risky_operation(item)
            use_result(result)
        except MildError:
            use_fallback()  # Запасной вариант
        except CriticalError:
            raise  # Прекращаем все

# 📋 СТРАТЕГИЯ 2: Fail Fast  
def fail_fast_approach():
    for item in items:
        try:
            result = risky_operation(item)
            use_result(result)
        except Exception:
            raise  # При любой ошибке останавливаемся

# 📋 СТРАТЕГИЯ 3: Error Accumulation
def error_accumulation_approach():
    errors = []
    results = []
    
    for item in items:
        try:
            result = risky_operation(item)
            results.append(result)
        except Exception as e:
            errors.append((item, e))  # Накапливаем ошибки
    
    # Анализируем в конце
    if len(errors) > len(results):
        raise Exception("Too many failures")
```

---

### 18. 🎯 Статистика каждые 5 vs прогресс каждые 10

**Краткий ответ:** Текущий подход разумен, но можно сделать еще лучше с адаптивной частотой.

### 🏗️ Анализ текущей логики

```python
# В твоем коде:

# СТАТИСТИКА каждые 5 добавленных песен:
if self.session_stats["added"] % 5 == 0:
    current_stats = self.db.get_stats()  # ← SQL запрос к БД
    logger.info(f"Stats: {current_stats['total_songs']} songs in database")

# ПРОГРЕСС каждые 10 обработанных песен:
if (i + 1) % 10 == 0:
    logger.info(f"Processed {i + 1}/{len(artist.songs)} songs for {artist_name}")
```

**Логика:**
- **Статистика реже** (важная информация о росте БД)
- **Прогресс чаще** (обратная связь о текущей работе)

### 💻 Проблемы текущего подхода

```python
# 🔴 ПРОБЛЕМА 1: Статистика привязана к успешным добавлениям
# Если много дубликатов - статистика показывается редко
duplicates_in_row = 20  # 20 дубликатов подряд
# Статистика НЕ покажется ни разу!

# 🔴 ПРОБЛЕМА 2: SQL запрос при каждой статистике
current_stats = self.db.get_stats()  # COUNT(*) запрос - медленный на больших БД

# 🔴 ПРОБЛЕМА 3: Фиксированные интервалы не учитывают скорость работы
# Быстрая обработка - много логов
# Медленная обработка - мало обратной связи
```

### 🛠️ Улучшенная версия

```python
class ImprovedProgressTracker:
    def __init__(self):
        self.last_progress_time = time.time()
        self.last_stats_time = time.time()
        self.progress_interval = 30  # Каждые 30 секунд
        self.stats_interval = 120    # Каждые 2 минуты
        
        # Кэшируем статистику БД
        self.cached_db_stats = {"total_songs": 0, "unique_artists": 0}
        self.cache_update_time = 0
    
    def should_show_progress(self):
        """Показывать прогресс по времени, а не по количеству"""
        now = time.time()
        if now - self.last_progress_time > self.progress_interval:
            self.last_progress_time = now
            return True
        return False
    
    def should_show_stats(self):
        """Показывать статистику по времени"""
        now = time.time()
        if now - self.last_stats_time > self.stats_interval:
            self.last_stats_time = now
            return True
        return False
    
    def get_cached_stats(self):
        """Кэшированная статистика - избегаем частых SQL запросов"""
        now = time.time()
        if now - self.cache_update_time > 60:  # Обновляем кэш раз в минуту
            self.cached_db_stats = self.db.get_stats()
            self.cache_update_time = now
        return self.cached_db_stats
    
    def update_progress(self, current, total, artist_name):
        """Умное логирование прогресса"""
        if self.should_show_progress():
            speed = self.calculate_speed()
            eta = self.calculate_eta(current, total, speed)
            
            logger.info(f"Progress: {current}/{total} songs for {artist_name} "
                       f"({current/total*100:.1f}%) - {speed:.1f} songs/min - ETA: {eta}")
    
    def update_stats(self, added_this_session):
        """Умное логирование статистики"""
        if self.should_show_stats():
            stats = self.get_cached_stats()  # Из кэша, не из БД
            logger.info(f"DB Stats: {stats['total_songs']} total songs, "
                       f"{added_this_session} added this session")

# Использование:
def improved_scraping_loop(self):
    tracker = ImprovedProgressTracker()
    
    for i, song in enumerate(artist.songs):
        # ... обработка песни ...
        
        # Прогресс по времени (каждые 30 сек)
        tracker.update_progress(i + 1, len(artist.songs), artist_name)
        
        # Статистика по времени (каждые 2 мин)  
        tracker.update_stats(self.session_stats["added"])
```

### 🎓 Адаптивные интервалы

```python
class AdaptiveLogger:
    def __init__(self):
        self.processing_speed = 1.0  # песен в секунду
        
    def adaptive_intervals(self):
        """Адаптируем частоту к скорости обработки"""
        if self.processing_speed > 2.0:
            # Быстрая обработка - логируем реже
            return {"progress": 50, "stats": 200}
        elif self.processing_speed < 0.5:
            # Медленная обработка - логируем чаще (больше обратной связи)
            return {"progress": 15, "stats": 60}
        else:
            # Средняя скорость
            return {"progress": 30, "stats": 120}

    def smart_logging(self, processed, added, total):
        """Логирование с учетом контекста"""
        intervals = self.adaptive_intervals()
        
        # Показываем детали если много ошибок
        error_rate = (processed - added) / processed if processed > 0 else 0
        
        if error_rate > 0.5:  # Много ошибок - логируем чаще
            intervals["progress"] //= 2
            logger.warning(f"High error rate: {error_rate:.1%}")
        
        # Показываем прогресс на важных этапах
        progress_pct = processed / total if total > 0 else 0
        
        if progress_pct in [0.25, 0.5, 0.75, 0.9]:  # 25%, 50%, 75%, 90%
            logger.info(f"🎯 Milestone: {progress_pct:.0%} complete")
```

### ⚠️ Рекомендации

```python
# ✅ ЛУЧШИЕ ПРАКТИКИ:

# 1. Логируй по времени, не по количеству
if time.time() - last_log > 30:  # Каждые 30 сек

# 2. Кэшируй тяжелые операции  
if time.time() - last_db_check > 60:  # Раз в минуту

# 3. Показывай контекстную информацию
logger.info(f"Progress: {current}/{total} ({pct:.1f}%) - {speed} items/sec")

# 4. Адаптируйся к скорости работы
log_interval = max(10, min(60, 100 / items_per_second))

# 5. Важные события логируй всегда
logger.info(f"🎉 Completed artist {artist_name}: {added} songs added")
```

---

Готов к вопросам 19-21? Там будет про `hasattr` vs `try-except`, блокировки базы данных и концепцию trade-off!

## 19. 🎯 hasattr vs try-except

### 🎯 **Краткий ответ**
`hasattr()` проверяет наличие атрибута без вызова исключений, что быстрее и читаемее чем try-except для простых проверок.

### 🏗️ **Контекст и основы**
**Зачем нужно?** Безопасно проверить, есть ли у объекта определённый атрибут, прежде чем его использовать.

**Где применяется?**
- API с непостоянной структурой данных
- Проверка совместимости версий библиотек
- Обработка объектов с динамическими атрибутами

### 📚 **Подробное объяснение**

`hasattr(obj, 'attribute')` внутренне использует `getattr()` и перехватывает `AttributeError`, но делает это на уровне C, что быстрее Python try-except.

### 💻 **Синтаксис с комментариями**

```python
# hasattr - встроенная функция
hasattr(object, 'attribute_name')  # Возвращает True/False

# Эквивалент через try-except
try:
    object.attribute_name
    result = True
except AttributeError:
    result = False
```

### 🛠️ **Практические примеры**

**❌ Плохой код (медленно и громоздко):**
```python
def get_song_id(song):
    try:
        song_id = song.id  # Может вызвать AttributeError
        return song_id
    except AttributeError:
        return None
```

**✅ Хороший код (быстро и читаемо):**
```python
def get_song_id(song):
    if hasattr(song, 'id'):
        return song.id
    return None

# Или ещё лучше - с getattr()
def get_song_id(song):
    return getattr(song, 'id', None)  # None как значение по умолчанию
```

**Реальный пример из скрипта:**
```python
# В коде скрапера
if self.db.add_song(
    artist_name, 
    song.title, 
    lyrics, 
    song.url, 
    song.id if hasattr(song, 'id') else None  # ← Безопасная проверка
):
    # Обработка успешного добавления
```

**Продвинутый пример:**
```python
class APIResponse:
    def __init__(self, data):
        # Динамически создаём атрибуты из словаря
        for key, value in data.items():
            setattr(self, key, value)

def process_response(response):
    # Проверяем наличие разных полей
    if hasattr(response, 'error'):
        handle_error(response.error)
    elif hasattr(response, 'data'):
        return response.data
    else:
        raise ValueError("Unknown response format")
```

### ⚠️ **Типичные ошибки и подводные камни**

1. **Путаница с методами vs свойствами:**
```python
# Неправильно - hasattr найдёт метод, но не проверит его работоспособность
if hasattr(obj, 'broken_method'):
    obj.broken_method()  # Может упасть внутри метода

# Правильно - для методов лучше try-except
try:
    result = obj.possibly_broken_method()
except Exception as e:
    handle_error(e)
```

2. **Производительность при частых проверках:**
```python
# Медленно - проверяем дважды
if hasattr(obj, 'expensive_property'):
    value = obj.expensive_property  # Вычисляется второй раз!

# Быстрее - используем getattr
value = getattr(obj, 'expensive_property', None)
if value is not None:
    # Используем value
```

3. **hasattr может скрывать другие ошибки:**
```python
class BadClass:
    @property
    def broken_prop(self):
        raise RuntimeError("Something went wrong")

# hasattr вернёт False, хотя атрибут существует!
print(hasattr(BadClass(), 'broken_prop'))  # False
```

### 🎓 **Задание для практики**

Создай функцию `safe_extract()`, которая извлекает данные из объекта API:

```python
def safe_extract(api_object, fields):
    """
    Извлекает поля из объекта, возвращает словарь
    fields = ['name', 'id', 'description', 'optional_field']
    """
    # Твой код здесь
    pass

# Тест
class MockAPI:
    def __init__(self):
        self.name = "Test"
        self.id = 123

api = MockAPI()
result = safe_extract(api, ['name', 'id', 'missing_field'])
print(result)  # {'name': 'Test', 'id': 123, 'missing_field': None}
```

### 🔗 **Связанные темы**
- `getattr()` и `setattr()` - работа с атрибутами
- `__dict__` и `vars()` - получение всех атрибутов
- Дескрипторы и свойства (`@property`)
- Рефлексия в Python

### 📝 **Итог**

`hasattr()` идеален для простых проверок существования атрибутов - он быстрый, читаемый и pythonic. Используй try-except только когда нужно обработать специфические ошибки или когда `hasattr()` может скрыть важные исключения.

---

## 20. 🎯 Обработка недоступности базы данных

### 🎯 **Краткий ответ**
При недоступности БД возникает `sqlite3.OperationalError`, который обрабатывается повторной попыткой через 2 секунды в методе `add_song()`.

### 🏗️ **Контекст и основы**
**Зачем нужно?** БД может быть временно заблокирована другим процессом, заполнен диск, или произошла сетевая ошибка.

**Где применяется?**
- Многопоточные приложения
- Веб-серверы с высокой нагрузкой
- Системы с ненадёжным хранилищем

### 📚 **Подробное объяснение**

SQLite использует файловые блокировки. Если два процесса пытаются писать одновременно, второй получит `database is locked`. Стратегия retry с delay - стандартное решение.

### 💻 **Синтаксис с комментариями**

```python
def add_song(self, artist, title, lyrics, url, genius_id=None):
    try:
        # Первая попытка записи
        self.conn.execute("INSERT INTO songs (...) VALUES (...)")
        return True
        
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):  # Специфичная ошибка блокировки
            logger.warning("База заблокирована, повторная попытка...")
            time.sleep(2)  # Ждём освобождения
            try:
                # Вторая попытка
                self.conn.execute("INSERT INTO songs (...) VALUES (...)")
                return True
            except Exception:  # Любая ошибка при повторе
                logger.error("Повторная попытка не удалась")
                return False
        else:
            raise e  # Перебрасываем другие OperationalError
```

### 🛠️ **Практические примеры**

**Простой пример - базовая обработка:**
```python
import sqlite3
import time

def safe_insert(conn, query, data):
    """Безопасная вставка с повтором"""
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            conn.execute(query, data)
            conn.commit()
            return True
            
        except sqlite3.OperationalError as e:
            if "locked" in str(e) and attempt < max_attempts - 1:
                print(f"Попытка {attempt + 1}: БД заблокирована, ждём...")
                time.sleep(2 ** attempt)  # Экспоненциальная задержка
            else:
                raise  # Последняя попытка или другая ошибка
                
    return False
```

**Реальный пример - контекстный менеджер:**
```python
import sqlite3
import time
from contextlib import contextmanager

@contextmanager
def resilient_connection(db_path, max_retries=3):
    """Устойчивое подключение к БД с повторами"""
    conn = None
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(db_path, timeout=30)  # 30 сек таймаут
            yield conn
            conn.commit()  # Коммитим изменения
            break
            
        except sqlite3.OperationalError as e:
            if conn:
                conn.rollback()  # Откатываем транзакцию
                conn.close()
                
            if "locked" in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"БД недоступна, ждём {wait_time}с...")
                time.sleep(wait_time)
            else:
                raise
                
        except Exception as e:
            if conn:
                conn.rollback()
                conn.close()
            raise
    else:
        if conn:
            conn.close()

# Использование
try:
    with resilient_connection('data.db') as conn:
        conn.execute("INSERT INTO songs VALUES (?, ?)", ("Artist", "Title"))
except Exception as e:
    print(f"Не удалось записать в БД: {e}")
```

**Продвинутый пример - пул соединений:**
```python
import sqlite3
import threading
import time
from queue import Queue, Empty

class ConnectionPool:
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool = Queue(maxsize=pool_size)
        self.lock = threading.Lock()
        
        # Создаём пул соединений
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path, check_same_thread=False)
            self.pool.put(conn)
    
    def get_connection(self, timeout=10):
        """Получить соединение из пула"""
        try:
            return self.pool.get(timeout=timeout)
        except Empty:
            raise RuntimeError("Все соединения заняты")
    
    def return_connection(self, conn):
        """Вернуть соединение в пул"""
        self.pool.put(conn)
    
    def execute_with_retry(self, query, data, max_retries=3):
        """Выполнить запрос с повторами"""
        conn = None
        
        for attempt in range(max_retries):
            try:
                conn = self.get_connection()
                conn.execute(query, data)
                conn.commit()
                return True
                
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Быстрая экспоненциальная задержка
                    continue
                else:
                    raise
                    
            finally:
                if conn:
                    self.return_connection(conn)
                    
        return False
```

### ⚠️ **Типичные ошибки и подводные камни**

1. **Бесконечные повторы:**
```python
# Плохо - может зависнуть навсегда
while True:
    try:
        conn.execute(query, data)
        break
    except sqlite3.OperationalError:
        time.sleep(1)  # Бесконечное ожидание
```

2. **Игнорирование типов ошибок:**
```python
# Плохо - перехватываем все ошибки
try:
    conn.execute(query, data)
except Exception:  # Слишком широкий перехват
    time.sleep(2)
    conn.execute(query, data)  # Может снова упасть
```

3. **Отсутствие логирования:**
```python
# Плохо - молчаливые ошибки
try:
    conn.execute(query, data)
except sqlite3.OperationalError:
    pass  # Теряем информацию об ошибке
```

### 🎓 **Задание для практики**

Создай класс `ResilientDatabase`, который автоматически повторяет операции:

```python
class ResilientDatabase:
    def __init__(self, db_path, max_retries=3, base_delay=0.1):
        self.db_path = db_path
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.conn = sqlite3.connect(db_path)
    
    def execute_with_retry(self, query, params=None):
        """
        Выполни запрос с автоматическими повторами
        Используй экспоненциальную задержку
        Логируй все попытки
        """
        # Твой код здесь
        pass

# Тест
db = ResilientDatabase('test.db')
db.execute_with_retry("CREATE TABLE IF NOT EXISTS test (id INTEGER, name TEXT)")
db.execute_with_retry("INSERT INTO test VALUES (?, ?)", (1, "Test"))
```

### 🔗 **Связанные темы**
- Обработка исключений (`try-except-finally`)
- Контекстные менеджеры (`with` statement)
- Многопоточность и блокировки
- Паттерн Retry с экспоненциальным backoff

### 📝 **Итог**

Обработка недоступности БД - критически важная часть надёжного приложения. Используй специфичные исключения, ограничивай количество повторов, добавляй задержки и обязательно логируй ошибки для отладки.

---

## 21. 🎯 Trade-off концепция

### 🎯 **Краткий ответ**
Trade-off (компромисс) - это ситуация, где улучшение одного параметра неизбежно ухудшает другой. В программировании постоянно выбираем между скоростью, памятью, читаемостью, надёжностью.

### 🏗️ **Контекст и основы**
**Зачем понимать?** Нет идеальных решений - всегда есть цена. Понимание trade-off помогает принимать осознанные решения в разработке.

**Где встречается?**
- Архитектура приложений
- Выбор алгоритмов и структур данных
- Оптимизация производительности
- API дизайн

### 📚 **Подробное объяснение**

Trade-off происходит из экономики - "не бывает бесплатного обеда". В программировании каждое решение имеет скрытые или явные издержки. Искусство разработки - найти оптимальный баланс для конкретной задачи.

### 💻 **Основные типы trade-off в программировании**

```python
# 1. Время vs Память (Time vs Space)
# Быстро, но много памяти
cache = {}  # Кэшируем результаты
def fibonacci_cached(n):
    if n in cache:
        return cache[n]  # O(1) время, O(n) память
    result = fibonacci_cached(n-1) + fibonacci_cached(n-2)
    cache[n] = result
    return result

# Медленно, но мало памяти
def fibonacci_simple(n):
    if n <= 1:
        return n
    return fibonacci_simple(n-1) + fibonacci_simple(n-2)  # O(2^n) время, O(n) память стека
```

### 🛠️ **Практические примеры**

**1. Скорость vs Надёжность (из вашего скрапера):**
```python
# Быстрый вариант - без задержек
def fast_scraping():
    for song in songs:
        download_lyrics(song)  # Максимальная скорость
        # Риск: блокировка IP, потеря всех данных

# Медленный, но надёжный вариант
def safe_scraping():
    for song in songs:
        download_lyrics(song)
        time.sleep(random.uniform(3, 7))  # 5-10x медленнее
        # Плюс: стабильная работа, сохранность данных
```

**2. Читаемость vs Производительность:**
```python
# Читаемо, но медленно
def process_data_readable(items):
    result = []
    for item in items:
        if item.is_valid():
            processed = item.transform()
            if processed.meets_criteria():
                result.append(processed)
    return result

# Быстро, но менее понятно
def process_data_fast(items):
    return [item.transform() for item in items 
            if item.is_valid() and item.transform().meets_criteria()]
    # Проблема: transform() вызывается дважды!

# Оптимальный компромисс
def process_data_optimal(items):
    return [processed for item in items 
            if item.is_valid() 
            for processed in [item.transform()] 
            if processed.meets_criteria()]
```

**3. Гибкость vs Простота:**
```python
# Простой, но негибкий
class SimpleLogger:
    def log(self, message):
        print(f"[LOG] {message}")

# Гибкий, но сложный
class FlexibleLogger:
    def __init__(self, level='INFO', format='[{level}] {time}: {message}', 
                 outputs=None, filters=None):
        self.level = level
        self.format = format
        self.outputs = outputs or [ConsoleOutput()]
        self.filters = filters or []
    
    def log(self, message, level='INFO'):
        if self._should_log(level):
            formatted = self._format_message(message, level)
            for output in self.outputs:
                if self._passes_filters(formatted):
                    output.write(formatted)
```

**4. Консистентность vs Доступность (CAP теорема):**
```python
# Строгая консистентность - медленнее
class StrictDatabase:
    def write(self, key, value):
        # Ждём подтверждения от всех реплик
        for replica in self.replicas:
            replica.write(key, value)
            replica.confirm_write()  # Блокируем до подтверждения
        return True

# Eventual consistency - быстрее  
class EventualDatabase:
    def write(self, key, value):
        # Пишем в одну реплику, остальные синхронизируются позже
        primary = self.replicas[0]
        primary.write(key, value)
        self._schedule_replication(key, value)  # Асинхронно
        return True  # Возвращаем сразу
```

**5. DRY vs YAGNI (Don't Repeat Yourself vs You Aren't Gonna Need It):**
```python
# DRY - избегаем дублирования
class AbstractDataProcessor:
    def process(self, data):
        validated = self.validate(data)
        transformed = self.transform(validated)
        return self.save(transformed)
    
    def validate(self, data): raise NotImplementedError
    def transform(self, data): raise NotImplementedError  
    def save(self, data): raise NotImplementedError

# YAGNI - пишем только то, что нужно сейчас
def process_user_data(user_data):
    if user_data.get('email'):
        user_data['email'] = user_data['email'].lower()
    save_to_database(user_data)
    return user_data
```

### ⚠️ **Как принимать решения о trade-off**

1. **Определи приоритеты:**
```python
# Для скрапера: надёжность > скорость
priorities = {
    'reliability': 10,    # Критично - потеря данных недопустима
    'speed': 6,          # Важно, но не критично
    'memory_usage': 3,   # Не критично для данной задачи  
    'code_simplicity': 7  # Важно для поддержки
}
```

2. **Измеряй impact:**
```python
# Конкретные цифры для решения
slow_but_safe = {
    'requests_per_hour': 500,
    'success_rate': 99.5,
    'blocks_per_day': 0.1
}

fast_but_risky = {
    'requests_per_hour': 2000, 
    'success_rate': 85,
    'blocks_per_day': 2.5
}
```

### 🎓 **Задание для практики**

Проанализируй trade-off для кэширования результатов API:

```python
import time
import requests
from functools import lru_cache

# Вариант 1: Без кэша
def get_user_data_no_cache(user_id):
    response = requests.get(f'https://api.example.com/users/{user_id}')
    return response.json()

# Вариант 2: С кэшем
@lru_cache(maxsize=1000)  
def get_user_data_cached(user_id):
    response = requests.get(f'https://api.example.com/users/{user_id}')
    return response.json()

# Задание: опиши trade-off между этими подходами
# - Скорость
# - Память  
# - Актуальность данных
# - Простота кода
# Когда какой подход выбрать?
```

### 🔗 **Связанные темы**
- Big O нотация и анализ сложности
- Паттерны проектирования
- CAP теорема (Consistency, Availability, Partition tolerance)
- Принципы SOLID

### 📝 **Итог**

Trade-off неизбежны в программировании. Вместо поиска "лучшего" решения, ищи "подходящее" для конкретной ситуации. Всегда четко формулируй, что ты готов принести в жертву и что получаешь взамен. Документируй принятые решения для будущих разработчиков.