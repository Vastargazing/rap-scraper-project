# 🎯 Интервью на позицию Junior ML Engineer

## 📋 Критерии оценки:
- **Проходной балл:** 15/20 правильных ответов
- **Отлично (Senior potential):** 18-20 ответов  
- **Хорошо (Strong Junior):** 15-17 ответов
- **Слабо (Отказ):** <15 ответов

---

## 🔥 БЛОК 1: Python Fundamentals (5 вопросов)

### Вопрос 1 ⭐⭐⭐ 
**В вашем коде есть строка `for _ in range(intervals):`. Объясните:**
- Что означает символ `_`?
- Когда его использовать?
- Приведите 2 альтернативных способа написать этот же цикл

<details>
<summary>Правильный ответ</summary>

`_` - это соглашение Python для "неиспользуемой переменной":
- Показывает, что значение итератора нас не интересует
- Используется когда нужно количество итераций, но не сами значения
- Некоторые IDE/линтеры не выдают warning на неиспользуемую `_`

Альтернативы:
```python
# 1. С обычной переменной (хуже)
for i in range(intervals):
    # i не используется

# 2. While цикл (сложнее)
counter = 0
while counter < intervals:
    counter += 1
```
</details>

---

### Вопрос 2 ⭐⭐⭐
**В коде используется `current_stats['total_songs']` вместо `current_stats.total_songs`. Почему квадратные скобки?**

<details>
<summary>Правильный ответ</summary>

`current_stats` - это словарь (dict), возвращаемый из SQL запроса:
- `dict['key']` - доступ к элементу словаря
- `obj.attribute` - доступ к атрибуту объекта
- `sqlite3.Row` ведет себя как словарь, поэтому используем `[]`
- `current_stats.total_songs` вызвал бы `AttributeError`
</details>

---

### Вопрос 3 ⭐⭐⭐⭐
**Объясните разницу между этими подходами обработки ошибок:**
```python
# Подход 1
if hasattr(song, 'id'):
    genius_id = song.id
else:
    genius_id = None

# Подход 2  
try:
    genius_id = song.id
except AttributeError:
    genius_id = None

# Подход 3
genius_id = getattr(song, 'id', None)
```

<details>
<summary>Правильный ответ</summary>

- **Подход 1 (hasattr):** Быстрый, читаемый, для простых проверок атрибутов
- **Подход 2 (try-except):** Лучше когда нужно различать типы ошибок или обработать сложную логику
- **Подход 3 (getattr):** Самый pythonic и быстрый для случая "атрибут или значение по умолчанию"

Лучший выбор: **Подход 3** для данной задачи.
</details>

---

### Вопрос 4 ⭐⭐⭐
**В чём разница между `exit()` и `exit(1)`? Зачем передавать число?**

<details>
<summary>Правильный ответ</summary>

- `exit()` или `exit(0)` - успешное завершение программы
- `exit(1)` - завершение с ошибкой (код возврата 1)
- Коды возврата важны для:
  - Скриптов оболочки (bash проверяет `$?`)
  - CI/CD пайплайнов
  - Системного администрирования
- Стандарт: 0 = успех, 1-255 = различные типы ошибок
</details>

---

### Вопрос 5 ⭐⭐⭐
**Что произойдёт с этим кодом и как исправить?**
```python
def process_songs(songs):
    for song in songs:
        if song.invalid:
            continue
        process_song(song)
        break  # ← Проблема здесь
    return "completed"
```

<details>
<summary>Правильный ответ</summary>

**Проблема:** `break` выходит из цикла после обработки первой валидной песни.

**Исправление:**
```python
def process_songs(songs):
    for song in songs:
        if song.invalid:
            continue  # Пропускаем невалидную
        process_song(song)    # Обрабатываем валидную
        # Убираем break, чтобы обработать все
    return "completed"
```
</details>

---

## 🔥 БЛОК 2: Обработка данных и API (5 вопросов)

### Вопрос 6 ⭐⭐⭐⭐
**В скрипте есть retry логика только на уровне артиста, а не песен. Почему такая архитектура? Какие плюсы/минусы?**

<details>
<summary>Правильный ответ</summary>

**Причины:**
- **Rate limiting** действует на уровне API connection, не на отдельные запросы
- Если API заблокировал - все последующие запросы будут падать
- Экономия времени: не пытаемся повторять каждую песню

**Плюсы:**
- Быстрее обнаруживает проблемы с API
- Меньше ненужных запросов
- Проще логика

**Минусы:**
- Теряем отдельные песни при временных сбоях
- Менее гранулярный контроль
</details>

---

### Вопрос 7 ⭐⭐⭐⭐⭐
**Объясните эту формулу: `if (i + 1) % 10 == 0:`. Напишите 3 других полезных формулы с модулем для ML задач.**

<details>
<summary>Правильный ответ</summary>

`(i + 1) % 10 == 0` - проверяет, является ли (i+1) кратным 10.
- При i=9: (9+1) % 10 = 0 ✅
- При i=19: (19+1) % 10 = 0 ✅

**ML формулы с модулем:**
```python
# 1. Логирование каждые N эпох
if epoch % log_frequency == 0:
    print(f"Epoch {epoch}, Loss: {loss}")

# 2. Сохранение модели каждые N батчей  
if batch_idx % save_every == 0:
    torch.save(model.state_dict(), f'model_{batch_idx}.pth')

# 3. Валидация каждые N шагов
if step % validation_interval == 0:
    evaluate_model(model, val_loader)
```
</details>

---

### Вопрос 8 ⭐⭐⭐
**В коде используется `random.uniform(3.0, 7.0)` для задержек. Объясните:**
- Зачем случайные задержки?
- Какие альтернативы существуют?
- Как выбрать диапазон?

<details>
<summary>Правильный ответ</summary>

**Зачем случайные задержки:**
- Имитация человеческого поведения
- Избежание синхронизации множественных скраперов
- Защита от pattern detection на сервере

**Альтернативы:**
- **Exponential backoff:** `delay = base_delay * (2 ** attempt)`
- **Fixed delay:** постоянная задержка
- **Jittered backoff:** экспоненциальная + случайность

**Выбор диапазона:**
- Анализ ToS API (часто указывают лимиты)
- A/B тестирование разных диапазонов
- Мониторинг rate limit ответов
</details>

---

### Вопрос 9 ⭐⭐⭐⭐
**Что такое "graceful shutdown" в контексте вашего скрипта? Реализуйте простую версию для ML тренировки.**

<details>
<summary>Правильный ответ</summary>

**Graceful shutdown** - корректное завершение работы при получении сигнала остановки:

```python
import signal
import torch

class GracefulTrainer:
    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"Получен сигнал {signum}, завершаем эпоху...")
        self.should_stop = True
    
    def train(self, model, dataloader, optimizer):
        for epoch in range(1000):
            if self.should_stop:
                print("Сохраняем модель перед выходом...")
                torch.save(model.state_dict(), 'checkpoint_emergency.pth')
                break
                
            for batch in dataloader:
                if self.should_stop:  # Проверяем и внутри батчей
                    break
                # Обычная тренировка
                loss = train_step(model, batch, optimizer)
```
</details>

---

### Вопрос 10 ⭐⭐⭐⭐
**В базе данных используется `UNIQUE(artist, title)`. Объясните:**
- Зачем этот constraint?
- Что произойдёт при попытке вставить дубликат?
- Как это влияет на производительность?

<details>
<summary>Правильный ответ</summary>

**Зачем UNIQUE constraint:**
- Предотвращает дублирование песен одного артиста
- Обеспечивает целостность данных на уровне БД
- Защита от багов в коде приложения

**При вставке дубликата:**
- Возникает `sqlite3.IntegrityError`
- Транзакция откатывается
- В коде обрабатывается как "уже существует"

**Влияние на производительность:**
- **Плюс:** Быстрая проверка существования через индекс
- **Минус:** Дополнительная проверка при каждой вставке
- **Решение:** Создание composite index на (artist, title)
</details>

---

## 🔥 БЛОК 3: ML Engineering & Best Practices (5 вопросов)

### Вопрос 11 ⭐⭐⭐⭐⭐
**Этот скрипт собирает данные для NLP модели. Какие проблемы качества данных вы видите? Предложите 3 улучшения.**

<details>
<summary>Правильный ответ</summary>

**Проблемы:**
1. **Нет нормализации текста** - разные кодировки, спецсимволы
2. **Нет фильтрации по качеству** - могут попасть инструментальные треки
3. **Нет балансировки датасета** - перекос в сторону популярных артистов

**Улучшения:**
```python
# 1. Нормализация текста
def normalize_lyrics(text):
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    return text.lower().strip()

# 2. Фильтрация качества
def is_valid_lyrics(lyrics):
    words = lyrics.split()
    return (len(words) >= 50 and 
            len(set(words)) / len(words) > 0.3 and  # Lexical diversity
            not any(marker in lyrics.lower() for marker in 
                   ['instrumental', 'no lyrics', 'beat only']))

# 3. Балансировка
def sample_songs_balanced(artist_songs, max_per_artist=100):
    return random.sample(artist_songs, min(len(artist_songs), max_per_artist))
```
</details>

---

### Вопрос 12 ⭐⭐⭐⭐
**Как бы вы добавили логирование метрик для мониторинга процесса скрапинга? Какие метрики важны?**

<details>
<summary>Правильный ответ</summary>

**Важные метрики:**
```python
import time
from collections import defaultdict

class ScrapingMetrics:
    def __init__(self):
        self.metrics = defaultdict(int)
        self.timings = []
        self.start_time = time.time()
    
    def log_metrics(self):
        duration = time.time() - self.start_time
        
        # Ключевые метрики
        success_rate = self.metrics['success'] / max(self.metrics['total'], 1)
        avg_processing_time = sum(self.timings) / max(len(self.timings), 1)
        throughput = self.metrics['success'] / duration * 3600  # песен/час
        
        # Логирование
        logger.info(f"""
        Success Rate: {success_rate:.2%}
        Average Processing Time: {avg_processing_time:.2f}s  
        Throughput: {throughput:.1f} songs/hour
        Errors: {self.metrics['errors']}
        Rate Limits: {self.metrics['rate_limits']}
        """)
        
        # Отправка в мониторинг (Prometheus, etc.)
        send_to_monitoring({
            'scraping_success_rate': success_rate,
            'scraping_throughput': throughput,
            'scraping_error_rate': self.metrics['errors'] / self.metrics['total']
        })
```
</details>

---

### Вопрос 13 ⭐⭐⭐⭐⭐
**Представьте, что этот скрипт нужно масштабировать на 10,000 артистов. Какие узкие места и как решать?**

<details>
<summary>Правильный ответ</summary>

**Узкие места:**
1. **Последовательная обработка** - один артист за раз
2. **Единая база данных** - блокировки при записи  
3. **Rate limiting** - один IP, один поток запросов

**Решения:**
```python
# 1. Многопоточность с очередью
import threading
from queue import Queue
import concurrent.futures

class ScalableScrapser:
    def __init__(self, num_workers=5):
        self.artist_queue = Queue()
        self.num_workers = num_workers
        
    def process_artists_parallel(self, artists):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Разделяем артистов между воркерами
            chunks = [artists[i::self.num_workers] for i in range(self.num_workers)]
            futures = [executor.submit(self.process_chunk, chunk) for chunk in chunks]
            concurrent.futures.wait(futures)

# 2. Шардинг БД
class ShardedDatabase:
    def __init__(self, num_shards=4):
        self.shards = [sqlite3.connect(f'lyrics_shard_{i}.db') for i in range(num_shards)]
    
    def get_shard(self, artist_name):
        shard_id = hash(artist_name) % len(self.shards)
        return self.shards[shard_id]

# 3. Пул прокси/IP адресов
class ProxyManager:
    def __init__(self, proxy_list):
        self.proxies = itertools.cycle(proxy_list)
        
    def get_session(self):
        proxy = next(self.proxies)
        session = requests.Session()
        session.proxies = {'http': proxy, 'https': proxy}
        return session
```
</details>

---

### Вопрос 14 ⭐⭐⭐
**Какие потенциальные bias (предвзятости) может содержать такой датасет? Как их минимизировать?**

<details>
<summary>Правильный ответ</summary>

**Типы bias:**
1. **Popularity bias** - больше песен у популярных артистов
2. **Temporal bias** - только современные треки
3. **Genre bias** - перекос в сторону хип-хопа
4. **Language bias** - преимущественно английский язык
5. **Platform bias** - только то, что есть на Genius

**Методы минимизации:**
```python
# 1. Стратификация по жанрам
def balanced_sampling(artists_by_genre, samples_per_genre=100):
    balanced_dataset = []
    for genre, artists in artists_by_genre.items():
        sampled = random.sample(artists, min(len(artists), samples_per_genre))
        balanced_dataset.extend(sampled)
    return balanced_dataset

# 2. Временная балансировка
def sample_by_decade(songs):
    by_decade = defaultdict(list)
    for song in songs:
        decade = (song.year // 10) * 10
        by_decade[decade].append(song)
    
    # Равное количество из каждого десятилетия
    balanced = []
    min_count = min(len(songs) for songs in by_decade.values())
    for decade_songs in by_decade.values():
        balanced.extend(random.sample(decade_songs, min_count))
    return balanced
```
</details>

---

### Вопрос 15 ⭐⭐⭐⭐
**Как бы вы реализовали data versioning для этого датасета? Зачем это нужно в ML проектах?**

<details>
<summary>Правильный ответ</summary>

**Зачем data versioning:**
- **Reproducibility** - можно воспроизвести эксперименты
- **Debugging** - найти, на каких данных модель работала плохо
- **Compliance** - отследить происхождение данных
- **Collaboration** - команда работает с одной версией данных

**Реализация:**
```python
import hashlib
import json
from datetime import datetime

class DatasetVersioning:
    def __init__(self, base_path):
        self.base_path = base_path
        self.metadata_file = f"{base_path}/dataset_metadata.json"
        
    def create_version(self, dataset_path, description=""):
        # Хэш для проверки целостности
        dataset_hash = self._calculate_hash(dataset_path)
        
        version_info = {
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'hash': dataset_hash,
            'description': description,
            'size': os.path.getsize(dataset_path),
            'created_at': datetime.now().isoformat(),
            'scraped_artists': self._get_artist_list(),
            'total_songs': self._count_songs(),
            'filters_applied': self._get_filters()
        }
        
        # Сохраняем метаданные
        self._save_metadata(version_info)
        
        # Копируем датасет с версионированием
        versioned_path = f"{self.base_path}/dataset_v{version_info['version']}.db"
        shutil.copy2(dataset_path, versioned_path)
        
        return version_info
    
    def _calculate_hash(self, file_path):
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
```
</details>

---

## 🔥 БЛОК 4: Проблемы и Отладка (5 вопросов)

### Вопрос 16 ⭐⭐⭐⭐
**Скрипт внезапно стал получать много timeout ошибок. Опишите step-by-step процедуру диагностики.**

<details>
<summary>Правильный ответ</summary>

**Пошаговая диагностика:**

```python
# 1. Добавить детальное логирование
import time
import psutil

def diagnose_timeouts():
    start_time = time.time()
    
    # Проверяем сетевое подключение
    try:
        response = requests.get('https://genius.com', timeout=5)
        logger.info(f"Genius доступен: {response.status_code}")
    except Exception as e:
        logger.error(f"Genius недоступен: {e}")
    
    # Проверяем системные ресурсы
    logger.info(f"CPU: {psutil.cpu_percent()}%")
    logger.info(f"Memory: {psutil.virtual_memory().percent}%")
    logger.info(f"Disk: {psutil.disk_usage('/').percent}%")
    
    # Проверяем производительность БД
    db_start = time.time()
    conn.execute("SELECT COUNT(*) FROM songs")
    db_time = time.time() - db_start
    logger.info(f"DB query time: {db_time:.3f}s")

# 2. Мониторинг паттернов ошибок
class TimeoutMonitor:
    def __init__(self):
        self.timeout_times = []
        self.success_times = []
    
    def log_request(self, success, duration):
        if success:
            self.success_times.append(duration)
        else:
            self.timeout_times.append(time.time())
    
    def analyze_pattern(self):
        # Есть ли временные паттерны в timeout?
        if len(self.timeout_times) >= 5:
            intervals = [self.timeout_times[i] - self.timeout_times[i-1] 
                        for i in range(1, len(self.timeout_times))]
            avg_interval = sum(intervals) / len(intervals)
            logger.info(f"Average timeout interval: {avg_interval:.1f}s")
```

**Возможные причины и решения:**
1. **Rate limiting** → увеличить задержки
2. **Сетевые проблемы** → добавить retry с backoff
3. **Перегрузка API** → использовать несколько ключей/прокси
4. **Проблемы с БД** → оптимизировать запросы, добавить индексы
</details>

---

### Вопрос 17 ⭐⭐⭐
**В логах видно: "Database locked" каждые 30 секунд. В чём причина и как исправить?**

<details>
<summary>Правильный ответ</summary>

**Возможные причины:**
1. **Длинные незакрытые транзакции** 
2. **Другой процесс держит соединение**
3. **Незакрытые курсоры**
4. **WAL mode не включён**

**Диагностика и решения:**
```python
# 1. Включить WAL mode для лучшей конкурентности
def optimize_sqlite():
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL") 
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=1073741824")  # 1GB mmap
    conn.commit()

# 2. Контекстные менеджеры для транзакций
@contextmanager
def db_transaction(conn):
    try:
        conn.execute("BEGIN")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

# 3. Проверка заблокированных процессов  
def check_db_locks():
    try:
        # Проверяем, кто держит файл БД
        import lsof  # pip install lsof
        processes = lsof.lsof('+D', 'lyrics.db')
        for proc in processes:
            logger.info(f"Process holding DB: PID {proc.pid}, {proc.command}")
    except:
        logger.info("Could not check DB locks")

# 4. Batch commits вместо частых коммитов
class BatchDatabase:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.pending_operations = []
    
    def add_song(self, *args):
        self.pending_operations.append(('INSERT', args))
        if len(self.pending_operations) >= self.batch_size:
            self.flush()
    
    def flush(self):
        with db_transaction(self.conn):
            for op_type, args in self.pending_operations:
                if op_type == 'INSERT':
                    self.conn.execute("INSERT INTO songs VALUES (...)", args)
        self.pending_operations.clear()
```
</details>

---

### Вопрос 18 ⭐⭐⭐⭐
**Программа работала 3 дня, собрала 50k песен, затем крашнулась с MemoryError. Что случилось и как предотвратить?**

<details>
<summary>Правильный ответ</summary>

**Возможные причины:**
1. **Memory leak** в lyricsgenius библиотеке
2. **Накопление объектов** в session_stats или кэшах
3. **Рост WAL файла** SQLite
4. **Незакрытые соединения/курсоры**

**Решения:**
```python
# 1. Мониторинг памяти
import psutil
import gc

class MemoryMonitor:
    def __init__(self, threshold_mb=1000):
        self.threshold_mb = threshold_mb
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
    def check_memory(self):
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        growth = current_memory - self.initial_memory
        
        logger.info(f"Memory usage: {current_memory:.1f}MB (+{growth:.1f}MB)")
        
        if growth > self.threshold_mb:
            logger.warning("High memory usage detected!")
            self.cleanup()
            
    def cleanup(self):
        # Принудительная сборка мусора
        collected = gc.collect()
        logger.info(f"Garbage collected: {collected} objects")
        
        # Очистка кэшей
        if hasattr(self, 'genius'):
            self.genius._session.close()
            self.genius = lyricsgenius.Genius(TOKEN)  # Пересоздаем

# 2. Переодическая очистка БД
def optimize_database_periodically(conn, interval_songs=10000):
    if self.session_stats['added'] % interval_songs == 0:
        logger.info("Optimizing database...")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # Очищаем WAL
        conn.execute("VACUUM")  # Дефрагментируем БД
        conn.execute("ANALYZE")  # Обновляем статистику

# 3. Ограничение размера сессии
def restart_session_periodically(self, max_songs_per_session=5000):
    if self.session_stats['processed'] >= max_songs_per_session:
        logger.info("Restarting session to prevent memory leaks...")
        self.close()
        self.__init__(TOKEN, self.db_path)  # Пересоздаём объект

# 4. Streaming обработка вместо загрузки всего в память
def stream_process_artists(self, artists):
    for artist in artists:
        # Обрабатываем по одному артисту, не держим всех в памяти
        songs = self.get_artist_songs_generator(artist)  # Generator!
        for song in songs:
            self.process_song(song)
            del song  # Явно удаляем
```
</details>

---

### Вопрос 19 ⭐⭐⭐⭐⭐
**После обновления библиотеки lyricsgenius структура song объекта изменилась. Как сделать код устойчивым к таким изменениям?**

<details>
<summary>Правильный ответ</summary>

**Стратегии устойчивости к изменениям API:**

```python
# 1. Adapter pattern для изоляции внешних зависимостей
class SongAdapter:
    def __init__(self, raw_song):
        self.raw_song = raw_song
        
    @property
    def title(self):
        # Пробуем разные варианты названий атрибутов
        for attr in ['title', 'name', 'song_title', 'track_name']:
            if hasattr(self.raw_song, attr):
                return getattr(self.raw_song, attr)
        return "Unknown Title"
    
    @property  
    def lyrics(self):
        for attr in ['lyrics', 'text', 'content', 'lyric_text']:
            if hasattr(self.raw_song, attr):
                lyrics = getattr(self.raw_song, attr)
                if lyrics and len(lyrics.strip()) > 0:
                    return lyrics
        return ""
    
    @property
    def url(self):
        return getattr(self.raw_song, 'url', 
                      getattr(self.raw_song, 'song_url', ''))

# 2. Версионирование совместимости
class GeniusClient:
    def __init__(self,