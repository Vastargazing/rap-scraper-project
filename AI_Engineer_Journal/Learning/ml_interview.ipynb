# 🎯 Практические задачи для Junior ML Engineer

*Основано на анализе вашего скрипта скрапинга*

---

## 🔥 ЗАДАЧА 1: Базовая обработка данных

### 📋 Условие задачи:
```
У вас есть список текстов песен. Напишите функцию, которая:
1. Удаляет тексты короче 50 слов
2. Приводит к нижнему регистру
3. Удаляет специальные символы (кроме букв и пробелов)
4. Возвращает статистику обработки
```

### ✅ Ожидаемое решение:

```python
import re
from typing import List, Dict, Tuple

def clean_lyrics_dataset(lyrics_list: List[str]) -> Tuple[List[str], Dict[str, int]]:
    """
    Очищает датасет текстов песен
    
    Args:
        lyrics_list: Список исходных текстов
        
    Returns:
        Tuple: (очищенные_тексты, статистика)
    """
    cleaned_lyrics = []
    stats = {
        'original_count': len(lyrics_list),
        'filtered_out_short': 0,
        'cleaned_count': 0,
        'total_words_before': 0,
        'total_words_after': 0
    }
    
    for lyrics in lyrics_list:
        # Подсчет слов до обработки
        original_words = len(lyrics.split())
        stats['total_words_before'] += original_words
        
        # Фильтрация коротких текстов
        if original_words < 50:
            stats['filtered_out_short'] += 1
            continue
        
        # Очистка текста
        # 1. Приведение к нижнему регистру
        cleaned = lyrics.lower()
        
        # 2. Удаление специальных символов (оставляем только буквы, цифры, пробелы)
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', cleaned)
        
        # 3. Нормализация пробелов (удаление множественных пробелов)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Подсчет слов после обработки
        cleaned_words = len(cleaned.split())
        stats['total_words_after'] += cleaned_words
        
        cleaned_lyrics.append(cleaned)
        stats['cleaned_count'] += 1
    
    return cleaned_lyrics, stats

# Пример использования:
if __name__ == "__main__":
    sample_lyrics = [
        "Hello World! This is a very short text.",  # Будет отфильтрован
        "This is a much longer text with more than fifty words. " * 10,
        "Another song with special chars: @#$%^&*()! But still long enough. " * 8
    ]
    
    cleaned, stats = clean_lyrics_dataset(sample_lyrics)
    
    print(f"Обработано: {stats['cleaned_count']}/{stats['original_count']}")
    print(f"Отфильтровано коротких: {stats['filtered_out_short']}")
    print(f"Слов до/после: {stats['total_words_before']}/{stats['total_words_after']}")
```

### 💡 Объяснение решения:
**Что проверяю на собеседовании:**
1. **Типизация** - использует ли type hints
2. **Обработка edge cases** - что если список пустой?
3. **Статистика** - важно для ML мониторинга  
4. **Regex понимание** - базовый навык для NLP
5. **Документация** - docstring обязателен

---

## 🔥 ЗАДАЧА 2: Retry механизм с экспоненциальным backoff

### 📋 Условие задачи:
```
Реализуйте декоратор @retry_with_backoff, который:
1. Повторяет функцию до N раз при ошибке
2. Использует экспоненциальную задержку: 1с, 2с, 4с, 8с...
3. Добавляет случайный jitter (±20%)
4. Логирует каждую попытку
5. Пробрасывает исключение после всех неудачных попыток
```

### ✅ Ожидаемое решение:

```python
import time
import random
import logging
from functools import wraps
from typing import Callable, Any

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_with_backoff(max_attempts: int = 3, base_delay: float = 1.0, 
                      jitter: bool = True, exceptions: tuple = (Exception,)):
    """
    Декоратор для повторных попыток с экспоненциальным backoff
    
    Args:
        max_attempts: Максимальное количество попыток
        base_delay: Базовая задержка в секундах
        jitter: Добавлять ли случайность к задержке
        exceptions: Типы исключений для retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)  # Сохраняет метаданные оригинальной функции
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    logger.info(f"Попытка {attempt + 1}/{max_attempts} для {func.__name__}")
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:  # Логируем успех после неудач
                        logger.info(f"{func.__name__} выполнена успешно с {attempt + 1} попытки")
                    
                    return result
                    
                except exceptions as e:
                    last_exception = e
                    logger.warning(f"Попытка {attempt + 1} неудачна: {str(e)}")
                    
                    # Если это последняя попытка - не делаем задержку
                    if attempt == max_attempts - 1:
                        break
                    
                    # Вычисляем задержку с экспоненциальным ростом
                    delay = base_delay * (2 ** attempt)
                    
                    # Добавляем jitter (случайность ±20%)
                    if jitter:
                        jitter_factor = 1 + random.uniform(-0.2, 0.2)
                        delay *= jitter_factor
                    
                    logger.info(f"Ожидание {delay:.2f}с перед следующей попыткой...")
                    time.sleep(delay)
            
            # Если все попытки неудачны
            logger.error(f"Все {max_attempts} попыток для {func.__name__} неудачны")
            raise last_exception
        
        return wrapper
    return decorator

# Примеры использования:

@retry_with_backoff(max_attempts=4, base_delay=0.5, exceptions=(ConnectionError, TimeoutError))
def unreliable_api_call(url: str) -> dict:
    """Симуляция ненадежного API вызова"""
    if random.random() < 0.7:  # 70% шанс неудачи
        raise ConnectionError("API недоступно")
    return {"status": "success", "data": f"Data from {url}"}

@retry_with_backoff(max_attempts=3, exceptions=(ValueError,))
def process_data(data: str) -> str:
    """Обработка данных с возможными ошибками"""
    if len(data) < 5:
        raise ValueError("Данные слишком короткие")
    return data.upper()

# Тестирование:
if __name__ == "__main__":
    try:
        result = unreliable_api_call("https://api.example.com/data")
        print(f"Успех: {result}")
    except ConnectionError as e:
        print(f"Финальная ошибка: {e}")
```

### 💡 Объяснение решения:
**Что проверяю на собеседовании:**
1. **Понимание декораторов** - базовая Python концепция
2. **@wraps использование** - сохранение метаданных функции
3. **Экспоненциальный backoff** - стандартная практика в distributed systems
4. **Jitter концепция** - предотвращение "thundering herd"
5. **Proper exception handling** - не глотать исключения
6. **Логирование** - критично для production систем

---

## 🔥 ЗАДАЧА 3: Batch обработка с progress tracking

### 📋 Условие задачи:
```
Напишите класс BatchProcessor, который:
1. Обрабатывает данные порциями (batches)
2. Показывает прогресс обработки
3. Сохраняет промежуточные результаты каждые N батчей
4. Может восстанавливать работу с места остановки
5. Обрабатывает ошибки на уровне батчей
```

### ✅ Ожидаемое решение:

```python
import json
import os
import time
from typing import List, Any, Callable, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Обработчик данных батчами с поддержкой восстановления и прогресса
    """
    
    def __init__(self, batch_size: int = 100, save_every: int = 10, 
                 checkpoint_file: str = "checkpoint.json"):
        self.batch_size = batch_size
        self.save_every = save_every
        self.checkpoint_file = checkpoint_file
        self.processed_count = 0
        self.failed_count = 0
        self.results = []
        self.failed_items = []
        
    def process_data(self, data: List[Any], processor_func: Callable[[List[Any]], List[Any]], 
                    output_file: str = "results.json") -> dict:
        """
        Основной метод обработки данных
        
        Args:
            data: Список данных для обработки
            processor_func: Функция обработки батча
            output_file: Файл для сохранения результатов
            
        Returns:
            Статистика обработки
        """
        total_items = len(data)
        start_time = datetime.now()
        
        # Попытка восстановления с checkpoint
        start_index = self._load_checkpoint()
        if start_index > 0:
            logger.info(f"Восстановление с позиции {start_index}/{total_items}")
        
        logger.info(f"Начало обработки {total_items} элементов батчами по {self.batch_size}")
        
        # Обработка батчами
        for i in range(start_index, total_items, self.batch_size):
            batch_start = time.time()
            
            # Создание текущего батча
            end_index = min(i + self.batch_size, total_items)
            current_batch = data[i:end_index]
            batch_num = (i // self.batch_size) + 1
            total_batches = (total_items + self.batch_size - 1) // self.batch_size
            
            try:
                # Обработка батча
                logger.info(f"Обработка батча {batch_num}/{total_batches} "
                           f"(элементы {i+1}-{end_index})")
                
                batch_results = processor_func(current_batch)
                self.results.extend(batch_results)
                self.processed_count += len(current_batch)
                
                batch_time = time.time() - batch_start
                self._log_progress(batch_num, total_batches, batch_time, 
                                 self.processed_count, total_items)
                
            except Exception as e:
                logger.error(f"Ошибка в батче {batch_num}: {str(e)}")
                self.failed_count += len(current_batch)
                self.failed_items.extend([
                    {"batch": batch_num, "item": item, "error": str(e)} 
                    for item in current_batch
                ])
            
            # Сохранение checkpoint
            if batch_num % self.save_every == 0:
                self._save_checkpoint(end_index, output_file)
                logger.info(f"Checkpoint сохранен на позиции {end_index}")
        
        # Финальное сохранение
        self._save_final_results(output_file, start_time)
        self._cleanup_checkpoint()
        
        return self._get_stats(start_time)
    
    def _load_checkpoint(self) -> int:
        """Загрузка checkpoint для восстановления"""
        if not os.path.exists(self.checkpoint_file):
            return 0
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                self.processed_count = checkpoint.get('processed_count', 0)
                self.failed_count = checkpoint.get('failed_count', 0)
                self.results = checkpoint.get('results', [])
                self.failed_items = checkpoint.get('failed_items', [])
                return checkpoint.get('last_index', 0)
        except Exception as e:
            logger.warning(f"Не удалось загрузить checkpoint: {e}")
            return 0
    
    def _save_checkpoint(self, current_index: int, output_file: str):
        """Сохранение промежуточного состояния"""
        checkpoint = {
            'last_index': current_index,
            'processed_count': self.processed_count,
            'failed_count': self.failed_count,
            'results': self.results,
            'failed_items': self.failed_items,
            'timestamp': datetime.now().isoformat(),
            'output_file': output_file
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _save_final_results(self, output_file: str, start_time: datetime):
        """Сохранение финальных результатов"""
        final_data = {
            'results': self.results,
            'failed_items': self.failed_items,
            'metadata': {
                'processed_count': self.processed_count,
                'failed_count': self.failed_count,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'batch_size': self.batch_size
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_data, f, indent=2)
        
        logger.info(f"Результаты сохранены в {output_file}")
    
    def _log_progress(self, current_batch: int, total_batches: int, 
                     batch_time: float, processed: int, total: int):
        """Логирование прогресса"""
        progress_pct = (processed / total) * 100
        avg_time_per_item = batch_time / self.batch_size
        
        # Оценка времени до завершения
        remaining_items = total - processed
        eta_seconds = remaining_items * avg_time_per_item
        eta_minutes = eta_seconds / 60
        
        logger.info(f"Прогресс: {progress_pct:.1f}% "
                   f"({processed}/{total}), "
                   f"Батч: {batch_time:.2f}с, "
                   f"ETA: {eta_minutes:.1f}мин")
    
    def _get_stats(self, start_time: datetime) -> dict:
        """Получение итоговой статистики"""
        end_time = datetime.now()
        duration = end_time - start_time
        
        return {
            'total_processed': self.processed_count,
            'total_failed': self.failed_count,
            'success_rate': (self.processed_count / (self.processed_count + self.failed_count)) * 100,
            'duration_seconds': duration.total_seconds(),
            'items_per_second': self.processed_count / duration.total_seconds(),
            'batches_processed': (self.processed_count + self.batch_size - 1) // self.batch_size
        }
    
    def _cleanup_checkpoint(self):
        """Удаление checkpoint после успешного завершения"""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)

# Пример использования:
def sample_processor(batch: List[str]) -> List[dict]:
    """Пример функции обработки батча"""
    results = []
    for item in batch:
        # Симуляция обработки
        time.sleep(0.1)  # Имитация работы
        if len(item) < 3:  # Некоторые элементы "ломаются"
            raise ValueError(f"Слишком короткий элемент: {item}")
        
        results.append({
            'original': item,
            'processed': item.upper(),
            'length': len(item),
            'processed_at': datetime.now().isoformat()
        })
    return results

if __name__ == "__main__":
    # Тестовые данные
    test_data = [f"item_{i:03d}" for i in range(500)] + ["a", "b"]  # Добавляем плохие элементы
    
    processor = BatchProcessor(batch_size=50, save_every=3)
    stats = processor.process_data(test_data, sample_processor, "output.json")
    
    print("Статистика обработки:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
```

### 💡 Объяснение решения:
**Что проверяю на собеседовании:**
1. **Checkpoint/Recovery паттерн** - критично для долгих ML задач
2. **Progress tracking** - UX для ML инженеров  
3. **Error handling на уровне батчей** - не падать на одной плохой записи
4. **JSON сериализация** - работа с персистентностью
5. **Logging best practices** - структурированные логи
6. **ETA calculation** - понимание математики прогресса

---

## 🔥 ЗАДАЧА 4: Конфигурационный менеджер

### 📋 Условие задачи:
```
Создайте класс ConfigManager, который:
1. Загружает конфигурацию из JSON, YAML, переменных окружения
2. Поддерживает вложенные конфигурации (database.host, database.port)
3. Валидирует обязательные параметры
4. Поддерживает значения по умолчанию
5. Может перезагружать конфигурацию без перезапуска
```

### ✅ Ожидаемое решение:

```python
import json
import yaml
import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ConfigField:
    """Описание поля конфигурации"""
    name: str
    required: bool = False
    default: Any = None
    validator: Optional[callable] = None
    description: str = ""

class ConfigManager:
    """
    Менеджер конфигурации с поддержкой множественных источников
    """
    
    def __init__(self, config_files: List[str] = None, env_prefix: str = "APP_"):
        self.config_files = config_files or []
        self.env_prefix = env_prefix
        self.config_data = {}
        self.field_definitions = {}
        self._watchers = []  # Для callback при изменении конфига
        
    def define_field(self, field_def: ConfigField):
        """Определение поля конфигурации с валидацией"""
        self.field_definitions[field_def.name] = field_def
        
    def load_config(self) -> Dict[str, Any]:
        """
        Загрузка конфигурации из всех источников в порядке приоритета:
        1. Переменные окружения (высший приоритет)
        2. YAML файлы  
        3. JSON файлы
        4. Значения по умолчанию (низший приоритет)
        """
        self.config_data = {}
        
        # 1. Загружаем значения по умолчанию
        self._load_defaults()
        
        # 2. Загружаем из файлов конфигурации
        for config_file in self.config_files:
            if Path(config_file).exists():
                file_config = self._load_from_file(config_file)
                self._deep_merge(self.config_data, file_config)
                logger.info(f"Загружена конфигурация из {config_file}")
            else:
                logger.warning(f"Файл конфигурации не найден: {config_file}")
        
        # 3. Переопределяем переменными окружения
        env_config = self._load_from_env()
        self._deep_merge(self.config_data, env_config)
        
        # 4. Валидируем конфигурацию
        self._validate_config()
        
        logger.info(f"Конфигурация загружена успешно: {len(self.config_data)} параметров")
        return self.config_data.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Получение значения с поддержкой вложенных ключей
        Пример: get('database.host') для {"database": {"host": "localhost"}}
        """
        keys = key.split('.')
        current = self.config_data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
                
        return current
    
    def set(self, key: str, value: Any):
        """Установка значения с поддержкой вложенных ключей"""
        keys = key.split('.')
        current = self.config_data
        
        # Создаем вложенную структуру если нужно
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
        
        # Уведомляем watchers об изменении
        self._notify_watchers(key, value)
    
    def watch(self, callback: callable):
        """Добавление callback для отслеживания изменений конфига"""
        self._watchers.append(callback)
    
    def reload(self) -> bool:
        """Перезагрузка конфигурации"""
        try:
            old_config = self.config_data.copy()
            self.load_config()
            
            # Проверяем, что изменилось
            changes = self._detect_changes(old_config, self.config_data)
            if changes:
                logger.info(f"Конфигурация изменена: {len(changes)} параметров")
                for key, (old_val, new_val) in changes.items():
                    logger.info(f"  {key}: {old_val} -> {new_val}")
                    self._notify_watchers(key, new_val)
            
            return True
        except Exception as e:
            logger.error(f"Ошибка перезагрузки конфигурации: {e}")
            return False
    
    def _load_defaults(self):
        """Загрузка значений по умолчанию"""
        for field_name, field_def in self.field_definitions.items():
            if field_def.default is not None:
                self._set_nested_value(field_name, field_def.default)
    
    def _load_from_file(self, file_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации из файла"""
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    raise ValueError(f"Неподдерживаемый формат файла: {file_path.suffix}")
        except Exception as e:
            logger.error(f"Ошибка загрузки {file_path}: {e}")
            return {}
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Загрузка конфигурации из переменных окружения"""
        env_config = {}
        
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # Удаляем префикс и конвертируем в nested ключ
                config_key = key[len(self.env_prefix):].lower()
                config_key = config_key.replace('__', '.')  # APP__DB__HOST -> db.host
                
                # Попытка автоматического приведения типов
                converted_value = self._convert_env_value(value)
                self._set_nested_value_in_dict(env_config, config_key, converted_value)
        
        return env_config
    
    def _convert_env_value(self, value: str) -> Any:
        """Автоматическое приведение типов для переменных окружения"""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        
        # Number
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # JSON
        if value.startswith(('{', '[')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        
        # String (default)
        return value
    
    def _set_nested_value(self, key: str, value: Any):
        """Установка значения в config_data по вложенному ключу"""
        self._set_nested_value_in_dict(self.config_data, key, value)
    
    def _set_nested_value_in_dict(self, target_dict: Dict, key: str, value: Any):
        """Установка значения в словаре по вложенному ключу"""
        keys = key.split('.')
        current = target_dict
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _deep_merge(self, target: Dict, source: Dict):
        """Глубокое слияние словарей"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _validate_config(self):
        """Валидация конфигурации согласно определениям полей"""
        errors = []
        
        for field_name, field_def in self.field_definitions.items():
            value = self.get(field_name)
            
            # Проверка обязательных полей
            if field_def.required and value is None:
                errors.append(f"Обязательное поле отсутствует: {field_name}")
                continue
            
            # Кастомная валидация
            if value is not None and field_def.validator:
                try:
                    if not field_def.validator(value):
                        errors.append(f"Валидация не пройдена для {field_name}: {value}")
                except Exception as e:
                    errors.append(f"Ошибка валидации {field_name}: {e}")
        
        if errors:
            raise ValueError("Ошибки конфигурации:\n" + "\n".join(errors))
    
    def _detect_changes(self, old: Dict, new: Dict, prefix: str = "") -> Dict[str, tuple]:
        """Определение изменений между конфигурациями"""
        changes = {}
        
        all_keys = set(old.keys()) | set(new.keys())
        
        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key
            
            old_val = old.get(key)
            new_val = new.get(key)
            
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                # Рекурсивно проверяем вложенные объекты
                nested_changes = self._detect_changes(old_val, new_val, full_key)
                changes.update(nested_changes)
            elif old_val != new_val:
                changes[full_key] = (old_val, new_val)
        
        return changes
    
    def _notify_watchers(self, key: str, value: Any):
        """Уведомление watchers об изменении"""
        for callback in self._watchers:
            try:
                callback(key, value)
            except Exception as e:
                logger.error(f"Ошибка в watcher callback: {e}")

# Пример использования:
if __name__ == "__main__":
    # Определяем схему конфигурации
    config = ConfigManager(
        config_files=['config.yaml', 'config.json'], 
        env_prefix='MYAPP_'
    )
    
    # Определяем поля
    config.define_field(ConfigField(
        name="database.host",
        required=True,
        default="localhost",
        validator=lambda x: isinstance(x, str) and len(x) > 0,
        description="Хост базы данных"
    ))
    
```python
    config.define_field(ConfigField(
        name="database.port", 
        required=True,
        default=5432,
        validator=lambda x: isinstance(x, int) and 1 <= x <= 65535,
        description="Порт базы данных"
    ))
    
    config.define_field(ConfigField(
        name="api.rate_limit",
        default=100,
        validator=lambda x: isinstance(x, int) and x > 0,
        description="Лимит запросов в час"
    ))
    
    config.define_field(ConfigField(
        name="ml.model_path",
        required=True,
        validator=lambda x: Path(x).exists() if isinstance(x, str) else False,
        description="Путь к ML модели"
    ))
    
    # Добавляем watcher для логирования изменений
    def config_changed(key: str, value: Any):
        logger.info(f"Конфигурация изменена: {key} = {value}")
    
    config.watch(config_changed)
    
    # Загружаем конфигурацию
    try:
        config.load_config()
        
        # Примеры использования
        db_host = config.get('database.host')
        db_port = config.get('database.port') 
        
        print(f"Database: {db_host}:{db_port}")
        print(f"Rate limit: {config.get('api.rate_limit')}")
        
        # Динамическое изменение
        config.set('api.rate_limit', 200)
        
        # Перезагрузка конфигурации
        config.reload()
        
    except ValueError as e:
        print(f"Ошибка конфигурации: {e}")
```

### 💡 Объяснение решения:
**Что проверяю на собеседовании:**
1. **Nested configuration handling** - работа со сложными структурами
2. **Multiple data sources** - приоритеты и слияние конфигураций
3. **Runtime configuration changes** - hot reload без перезапуска
4. **Validation patterns** - проверка конфигурации перед использованием
5. **Environment variable parsing** - автоматическое приведение типов
6. **Observer pattern** - watchers для реакции на изменения
7. **Error handling** - graceful degradation при проблемах с конфигом

---

## 🔥 ЗАДАЧА 5: Простой data pipeline

### 📋 Условие задачи:
```
Создайте класс DataPipeline, который:
1. Принимает список этапов обработки (steps)
2. Каждый step - это функция, которая принимает и возвращает данные  
3. Поддерживает parallel выполнение steps где возможно
4. Логирует время выполнения каждого step
5. Может пропускать steps по условию
6. Сохраняет промежуточные результаты для отладки
```

### ✅ Ожидаемое решение:

```python
import time
import json
from typing import Any, List, Callable, Dict, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PipelineStep:
    """Определение шага пайплайна"""
    name: str
    function: Callable[[Any], Any]
    parallel: bool = False
    condition: Optional[Callable[[Any], bool]] = None
    save_intermediate: bool = False
    timeout: Optional[float] = None

class DataPipeline:
    """
    Пайплайн обработки данных с поддержкой параллельности и мониторинга
    """
    
    def __init__(self, steps: List[PipelineStep], debug_mode: bool = False,
                 intermediate_dir: str = "pipeline_debug"):
        self.steps = steps
        self.debug_mode = debug_mode
        self.intermediate_dir = Path(intermediate_dir)
        self.execution_stats = {}
        
        if self.debug_mode:
            self.intermediate_dir.mkdir(exist_ok=True)
    
    def run(self, input_data: Any) -> Dict[str, Any]:
        """
        Выполнение пайплайна
        
        Returns:
            Dict с результатами и статистикой выполнения
        """
        start_time = time.time()
        current_data = input_data
        
        logger.info(f"Запуск пайплайна с {len(self.steps)} шагами")
        
        # Сохраняем входные данные если включен debug
        if self.debug_mode:
            self._save_intermediate_data("00_input", input_data)
        
        try:
            for i, step in enumerate(self.steps, 1):
                step_start = time.time()
                
                # Проверяем условие выполнения шага
                if step.condition and not step.condition(current_data):
                    logger.info(f"Шаг {i} '{step.name}' пропущен (условие не выполнено)")
                    self.execution_stats[step.name] = {
                        'status': 'skipped',
                        'duration': 0,
                        'input_size': self._get_data_size(current_data)
                    }
                    continue
                
                logger.info(f"Выполнение шага {i}/{len(self.steps)}: '{step.name}'")
                
                # Выполнение шага
                if step.parallel and self._is_parallelizable(current_data):
                    current_data = self._execute_parallel_step(step, current_data)
                else:
                    current_data = self._execute_step(step, current_data)
                
                step_duration = time.time() - step_start
                
                # Сохранение статистики
                self.execution_stats[step.name] = {
                    'status': 'completed',
                    'duration': step_duration,
                    'input_size': self._get_data_size(current_data),
                    'output_size': self._get_data_size(current_data)
                }
                
                # Сохранение промежуточных результатов
                if step.save_intermediate or self.debug_mode:
                    self._save_intermediate_data(f"{i:02d}_{step.name}", current_data)
                
                logger.info(f"Шаг '{step.name}' завершен за {step_duration:.2f}с")
        
        except Exception as e:
            logger.error(f"Ошибка в пайплайне: {e}")
            total_time = time.time() - start_time
            return {
                'success': False,
                'error': str(e),
                'execution_stats': self.execution_stats,
                'total_time': total_time
            }
        
        total_time = time.time() - start_time
        
        # Финальная статистика
        logger.info(f"Пайплайн завершен за {total_time:.2f}с")
        self._log_pipeline_stats()
        
        return {
            'success': True,
            'result': current_data,
            'execution_stats': self.execution_stats,
            'total_time': total_time
        }
    
    def _execute_step(self, step: PipelineStep, data: Any) -> Any:
        """Выполнение одного шага"""
        try:
            if step.timeout:
                # Здесь можно добавить timeout logic через threading
                return step.function(data)
            else:
                return step.function(data)
        except Exception as e:
            logger.error(f"Ошибка в шаге '{step.name}': {e}")
            raise
    
    def _execute_parallel_step(self, step: PipelineStep, data: List[Any]) -> List[Any]:
        """Параллельное выполнение шага для списка данных"""
        if not isinstance(data, list):
            logger.warning(f"Шаг '{step.name}' помечен как parallel, но данные не список")
            return self._execute_step(step, data)
        
        results = []
        max_workers = min(len(data), 4)  # Ограничиваем количество потоков
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Запускаем задачи
            future_to_index = {
                executor.submit(step.function, item): i 
                for i, item in enumerate(data)
            }
            
            # Собираем результаты в правильном порядке
            indexed_results = {}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result(timeout=step.timeout)
                    indexed_results[index] = result
                except Exception as e:
                    logger.error(f"Ошибка в параллельном выполнении шага '{step.name}', элемент {index}: {e}")
                    indexed_results[index] = None
            
            # Восстанавливаем порядок
            results = [indexed_results[i] for i in range(len(data))]
        
        # Фильтруем None результаты (ошибки)
        return [r for r in results if r is not None]
    
    def _is_parallelizable(self, data: Any) -> bool:
        """Проверка, можно ли данные обрабатывать параллельно"""
        return isinstance(data, list) and len(data) > 1
    
    def _get_data_size(self, data: Any) -> int:
        """Получение размера данных для статистики"""
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            return len(data)
        elif isinstance(data, str):
            return len(data)
        else:
            return 1
    
    def _save_intermediate_data(self, step_name: str, data: Any):
        """Сохранение промежуточных данных для отладки"""
        try:
            file_path = self.intermediate_dir / f"{step_name}.json"
            
            # Подготавливаем данные для JSON сериализации
            serializable_data = self._make_json_serializable(data)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'step': step_name,
                    'timestamp': time.time(),
                    'data_type': type(data).__name__,
                    'data_size': self._get_data_size(data),
                    'data': serializable_data
                }, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.warning(f"Не удалось сохранить промежуточные данные для {step_name}: {e}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Конвертация объекта в JSON-сериализуемый формат"""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj[:100]]  # Ограничиваем размер
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        else:
            return str(obj)
    
    def _log_pipeline_stats(self):
        """Логирование статистики выполнения пайплайна"""
        total_duration = sum(stats.get('duration', 0) for stats in self.execution_stats.values())
        completed_steps = sum(1 for stats in self.execution_stats.values() if stats['status'] == 'completed')
        skipped_steps = sum(1 for stats in self.execution_stats.values() if stats['status'] == 'skipped')
        
        logger.info(f"Статистика пайплайна:")
        logger.info(f"  Выполнено шагов: {completed_steps}")
        logger.info(f"  Пропущено шагов: {skipped_steps}")
        logger.info(f"  Общее время выполнения шагов: {total_duration:.2f}с")
        
        # Детальная статистика по шагам
        for step_name, stats in self.execution_stats.items():
            if stats['status'] == 'completed':
                logger.info(f"  {step_name}: {stats['duration']:.2f}с")

# Примеры функций для шагов пайплайна
def load_data(file_path: str) -> List[dict]:
    """Загрузка данных"""
    logger.info(f"Загрузка данных из {file_path}")
    # Симуляция загрузки
    time.sleep(0.5)
    return [{'id': i, 'text': f'sample text {i}'} for i in range(100)]

def clean_text(items: List[dict]) -> List[dict]:
    """Очистка текста"""
    for item in items:
        item['text'] = item['text'].upper().strip()
    return items

def extract_features(item: dict) -> dict:
    """Извлечение признаков (выполняется параллельно)"""
    time.sleep(0.01)  # Симуляция обработки
    item['features'] = {
        'length': len(item['text']),
        'word_count': len(item['text'].split()),
        'has_numbers': any(c.isdigit() for c in item['text'])
    }
    return item

def filter_valid(items: List[dict]) -> List[dict]:
    """Фильтрация валидных элементов"""
    return [item for item in items if item['features']['word_count'] > 1]

def save_results(items: List[dict]) -> str:
    """Сохранение результатов"""
    logger.info(f"Сохранение {len(items)} элементов")
    return f"Saved {len(items)} items to database"

# Пример использования
if __name__ == "__main__":
    # Определяем пайплайн
    steps = [
        PipelineStep(
            name="load_data",
            function=lambda _: load_data("input.json"),
            save_intermediate=True
        ),
        PipelineStep(
            name="clean_text", 
            function=clean_text,
            condition=lambda data: len(data) > 0
        ),
        PipelineStep(
            name="extract_features",
            function=extract_features,
            parallel=True  # Будет выполняться параллельно
        ),
        PipelineStep(
            name="filter_valid",
            function=filter_valid,
            save_intermediate=True
        ),
        PipelineStep(
            name="save_results",
            function=save_results
        )
    ]
    
    # Создаем и запускаем пайплайн
    pipeline = DataPipeline(steps, debug_mode=True)
    result = pipeline.run("dummy_input")
    
    if result['success']:
        print(f"Пайплайн завершен успешно за {result['total_time']:.2f}с")
        print(f"Результат: {result['result']}")
    else:
        print(f"Ошибка в пайплайне: {result['error']}")
    
    # Вывод детальной статистики
    print("\nДетальная статистика:")
    for step_name, stats in result['execution_stats'].items():
        print(f"  {step_name}: {stats}")
```

### 💡 Объяснение решения:
**Что проверяю на собеседовании:**
1. **Pipeline pattern** - стандартная архитектура для ML
2. **Parallel processing** - понимание многопоточности в Python
3. **Conditional execution** - гибкость пайплайна
4. **Intermediate results** - отладка сложных пайплайнов  
5. **Error handling** - graceful failure recovery
6. **Performance monitoring** - метрики времени выполнения
7. **JSON serialization** - работа с данными
8. **Threading concepts** - ThreadPoolExecutor usage

---

## 🎯 Итоговая оценка кандидата

### 📊 Критерии прохождения интервью:

| Уровень | Задач решено | Качество кода | Объяснения | Решение |
|---------|--------------|---------------|-------------|----------|
| **Отказ** | 0-2/5 | Много ошибок | Путается в базовых концепциях | ❌ Не принимаем |
| **Слабый Junior** | 3/5 | Работает, но неоптимально | Понимает с подсказками | ⚠️ Испытательный срок |
| **Средний Junior** | 4/5 | Хороший код | Четкие объяснения | ✅ Принимаем |
| **Сильный Junior** | 5/5 | Отличный код + улучшения | Глубокое понимание | ⭐ Принимаем с повышенной зарплатой |

### 🔍 На что особенно обращаю внимание:

1. **Type hints** - использует ли их кандидат
2. **Error handling** - думает ли о edge cases
3. **Logging** - понимает ли важность логирования
4. **Code organization** - структурирует ли код логично
5. **Performance awareness** - думает ли о производительности
6. **Testing mindset** - как бы протестировал код
7. **Documentation** - пишет ли docstrings

### 💬 Финальные вопросы после задач:

1. **"Как бы вы протестировали эти функции?"**
2. **"Какие метрики собирали бы для мониторинга?"**  
3. **"Что будет узким местом при масштабировании?"**
4. **"Как бы организовали CI/CD для такого кода?"**

Эти задачи покрывают 80% навыков, необходимых Junior ML Engineer для работы с реальными проектами!


Нет, для позиции Junior ML Engineer не требуется писать весь этот код из головы наизусть. Однако ожидается, что вы понимаете ключевые концепции, можете объяснить логику кода и предложить решения для типичных задач. Давайте разберём, что действительно важно для успешного прохождения интервью на такую позицию, основываясь на предоставленном документе.

### Что ожидают от Junior ML Engineer
1. **Понимание основ Python** (Блок 1):
   - Умение работать с базовыми конструкциями языка (циклы, словари, обработка ошибок).
   - Знание, когда использовать `try-except`, `getattr`, или `hasattr` (как в вопросе 3).
   - Понимание, как работают такие функции, как `exit()` или `random.uniform()` (вопросы 4 и 8).
   - Способность находить и исправлять простые ошибки в коде (вопрос 5).

   **Пример:** Если вас просят объяснить, что делает `_` в цикле `for _ in range(intervals)`, достаточно сказать, что это неиспользуемая переменная, и предложить альтернативу, например, цикл с `while`. Писать весь код наизусть не нужно, но вы должны уметь написать простой цикл или объяснить его логику.

2. **Работа с данными и API** (Блок 2):
   - Понимание, как обрабатывать данные, взаимодействовать с API, и учитывать ограничения (rate limits, задержки).
   - Знание, зачем нужны такие механизмы, как `UNIQUE` constraint в базе данных (вопрос 10) или случайные задержки (вопрос 8).
   - Умение предложить идеи для улучшения архитектуры, например, почему retry логика на уровне артиста, а не песен (вопрос 6).

   **Пример:** Если спросят про случайные задержки, достаточно объяснить, что они имитируют человеческое поведение и помогают избежать бана API. Не нужно писать полный код для экспоненциального backoff, но упомянуть его как альтернативу будет плюсом.

3. **ML Engineering и лучшие практики** (Блок 3):
   - Базовое понимание проблем качества данных (вопрос 11), таких как нормализация текста или балансировка датасета.
   - Знание, как мониторить процесс (вопрос 12) или масштабировать скрипт (вопрос 13).
   - Понимание концепций, таких как data versioning (вопрос 15) или bias в данных (вопрос 14).

   **Пример:** Для вопроса о bias в датасете достаточно назвать 2-3 типа предвзятости (например, popularity bias или genre bias) и предложить простое решение, вроде стратификации по жанрам. Писать сложный код для балансировки не требуется, но понимание идеи важно.

4. **Отладка и решение проблем** (Блок 4):
   - Умение диагностировать типичные проблемы, такие как таймауты (вопрос 16), блокировки базы данных (вопрос 17), или утечки памяти (вопрос 18).
   - Знание, как сделать код устойчивым к изменениям внешних библиотек (вопрос 19).

   **Пример:** Если спросят про MemoryError, достаточно объяснить, что это может быть из-за накопления объектов в памяти, и предложить решения, такие как использование генераторов или периодическая очистка памяти через `gc.collect()`. Полный код писать не нужно, но понимание процесса отладки важно.

### Нужно ли писать весь код из головы?
- **Короткий ответ:** Нет, не нужно. На интервью для Junior позиции редко требуют писать сложный код с нуля без подсказок. Обычно просят:
  - Объяснить, как работает код.
  - Найти и исправить ошибку в предоставленном коде.
  - Написать небольшой фрагмент кода для решения конкретной задачи (например, нормализация текста или обработка ошибок).
  - Предложить идеи или улучшения для архитектуры.

- **Что важнее кода:** 
  - Умение объяснить свои действия.
  - Понимание, почему используется тот или иной подход.
  - Способность рассуждать о плюсах и минусах решений.

Например, в вопросе 9 про "graceful shutdown" достаточно объяснить концепцию (корректное завершение работы при сигнале остановки) и написать псевдокод или описать, как бы вы это реализовали (например, сохранить модель перед выходом). Точный синтаксис `signal.signal(signal.SIGINT, self._signal_handler)` знать необязательно, но понимание, что это обработка сигналов, будет плюсом.

### Как подготовиться
1. **Основы Python**:
   - Пройдитесь по вопросам из Блока 1. Попробуйте объяснить их устно или написать простые решения (например, цикл без `break` или обработку ошибок с `getattr`).
   - Практикуйтесь в чтении кода и поиске ошибок.

2. **Работа с данными**:
   - Изучите основы работы с API (rate limits, задержки, retry).
   - Поймите, как работают базы данных (например, SQLite) и что такое индексы, транзакции, и `UNIQUE` constraint.

3. **ML Engineering**:
   - Прочитайте про типичные проблемы качества данных (bias, нормализация).
   - Поймите, зачем нужны логирование, мониторинг, и data versioning.
   - Практикуйтесь в объяснении, как масштабировать скрипты (многопоточность, шардинг).

4. **Отладка**:
   - Изучите типичные ошибки (MemoryError, Database locked, таймауты) и их причины.
   - Поймите, как диагностировать проблемы (логирование, мониторинг ресурсов).

5. **Практика**:
   - Попробуйте решить задачи из документа устно или на бумаге, не заучивая код.
   - Напишите несколько простых функций (например, нормализация текста, как в вопросе 11, или обработка ошибок, как в вопросе 3).
   - Практикуйтесь на платформах вроде LeetCode или HackerRank, но фокусируйтесь на задачах, связанных с обработкой данных и отладкой.

### Что будет плюсом
- **Понимание контекста ML**: Знание, как данные влияют на обучение модели (например, bias или качество датасета).
- **Чистота кода**: Если вас просят написать код, используйте понятные имена переменных и следуйте PEP 8.
- **Вопросы к интервьюеру**: Спрашивайте уточнения, например, "Какой тип данных ожидается?" или "Нужно ли учитывать масштабирование?" Это показывает, что вы думаете о задаче.

### Итог
Для Junior ML Engineer достаточно уверенно отвечать на вопросы уровня ⭐⭐⭐ (3 звезды) и показывать базовое понимание вопросов ⭐⭐⭐⭐ (4 звезды). Вопросы с ⭐⭐⭐⭐⭐ (5 звёзд) часто рассчитаны на демонстрацию потенциала роста или на более опытных кандидатов. Если вы можете объяснить концепции, предложить решения и написать простой код для базовых задач, этого хватит для проходного балла (15/20). Чтобы выделиться, постарайтесь показать, что вы понимаете, как применять знания в реальных ML-проектах (например, нормализация данных или мониторинг).

Если хотите, могу помочь с конкретным вопросом из документа или разобрать какую-то тему подробнее!

Отличный и очень важный вопрос — **нет, тебя не будут ожидать, что ты напишешь весь этот код из головы**, чтобы получить позицию **Junior ML Engineer**.

Разберёмся подробно.

---

### ❌ Что **не требуется** от Junior ML Engineer:

- ✖️ Зубрить синтаксис `PRAGMA journal_mode=WAL` или `signal.SIGTERM`
- ✖️ Помнить наизусть все параметры `requests.Session()` или `sqlite3`
- ✖️ Держать в голове полный код `Adapter pattern` или `Graceful shutdown`
- ✖️ Уметь с нуля написать `BatchDatabase` или `DatasetVersioning` без подсказок

---

### ✅ Что **ожидают** от Junior ML Engineer:

| Что важно | Почему |
|--------|--------|
| **Понимание логики** | Ты должен **объяснить**, зачем нужен `UNIQUE(artist, title)` или `try-except`, даже если не помнишь синтаксис |
| **Знание принципов** | Например: «Я не помню, как именно включить WAL в SQLite, но знаю, что это улучшает параллельный доступ» |
| **Умение читать и анализировать код** | Видеть проблему в `break` внутри цикла и понимать, что он останавливает обработку всех песен |
| **Поиск решений** | Гуглить, читать документацию, смотреть примеры — это **нормально и ожидается** |
| **Базовое понимание пайплайна** | От сбора данных → до логирования и обработки ошибок |

---

### 💡 Как проходит реальное интервью?

1. **Тебе дают задачу или кусок кода** (например, цикл с `break`).
2. **Спрашивают: "Что делает этот код? Есть ли проблема?"**
3. Ты **объясняешь**, что видишь.
4. Возможно, предлагаешь **идею исправления**.
5. Если забыл синтаксис — говоришь:  
   > *"Я бы использовал `getattr` с значением по умолчанию, но точно не помню синтаксис — могу заглянуть в документацию."*  
   → Это **абсолютно нормально** для Junior.

---

### 🎯 Что означает "15/20 правильных ответов"?

Смотри на критерии из файла:

> **Проходной балл: 15/20** — это **Strong Junior**

Это **не значит**, что ты должен идеально ответить на 15 вопросов.  
На практике:

- ✅ Правильно объяснил 10 вопросов
- 🟡 На 5 дал частичный ответ ("Я бы проверил ошибку, но не помню, как именно")
- ✅ Признал, что не знает чего-то: *"Про data versioning слышал, но не делал — читал про DVC, но не пробовал"*

→ Это может быть **15 баллов**, потому что ты **понимаешь контекст** и **мыслить правильно**.

---

### 🔁 Пример: вопрос про `for _ in range(intervals)`

**Ожидают:**
- Понимание, что `_` — это переменная, которую мы намеренно игнорируем
- Объяснение, зачем это нужно (чтобы не видеть ворнинги в IDE)
- Пример альтернативы: `for i in range(...)` — даже если напишешь `i`, это не ошибка

**Не ожидается:**
- Знать, что `hasattr` медленнее `getattr`
- Писать `while` цикл без ошибок с первого раза

---

### 🧠 Что стоит выучить (на память):

| Что полезно помнить |
|---------------------|
| Базовый синтаксис Python: `if`, `for`, `try-except`, `def`, `class` |
| Разница между `list`, `dict`, `tuple` |
| Как читать из файла / базы данных (в общих чертах) |
| Простые конструкции: `getattr(obj, 'attr', default)`, `os.path.exists()` |
| Основы ООП: `self`, `__init__`, `@property` |
| Самые частые ошибки: `AttributeError`, `KeyError`, `IndexError` |

---

### 🛠 А что писать с подсказками?

Всё, что связано с:
- Работой с API (retry, backoff)
- Базами данных (SQLite, индексы)
- Многопоточностью
- Логированием и мониторингом
- Версионированием данных

→ Это **senior-level практики**. От тебя **не ждут идеального кода**, но **желательно понимать концепцию**.

---

### 🎯 Итог

> ❓ *"Я должен уметь писать весь этот код из головы?"*  
> **Нет.**

> ✅ Что нужно:
- **Понимать логику**
- **Объяснять, что делает код**
- **Видеть ошибки и предлагать решения**
- **Уметь находить информацию**
- **Показать, что ты можешь учиться и расти**

---

### 💬 Совет от практикующих ML-инженеров:

> «Мы берём Junior не за знания, а за **мышление**.  
> Лучше сказать: *«Я не знаю, но попробую догадаться»* — чем молчать.  
> А ещё лучше: *«Я бы заглянул в документацию, но, кажется, тут нужен try-except»*.  
> — Это уже почти Senior mindset.»

---

# 🎯 Интервью на позицию Junior ML Engineer

## 📋 Критерии оценки:
- **Проходной балл:** 15/20 правильных ответов — это значит, что если вы правильно ответите на 15 вопросов из 20, вы проходите. Для junior-позиции это базовый уровень, где важно показать понимание основ, а не идеальное знание всего.
- **Отлично (Senior potential):** 18-20 ответов — здесь вы демонстрируете не только знания, но и потенциал роста, например, предлагая улучшения или альтернативы.
- **Хорошо (Strong Junior):** 15-17 ответов — вы сильный junior, понимаете ключевые концепции, но можете упускать нюансы.
- **Слабо (Отказ):** <15 ответов — нужно подтянуть базу, особенно Python fundamentals и обработку данных.

**Дополнительный комментарий от меня (как учителя по Python ML):** Этот файл — отличный способ подготовиться к интервью. Я доработал его, добавив:
- **Подробные комментарии:** Объяснения простым языком, почему это важно для ML Engineer, связи с реальными задачами (например, сбор данных для NLP-модели).
- **Сухой алгоритм действий:** Для каждого вопроса, где есть код или логика, я добавил список шагов — как компьютер выполняет код по порядку.
- **Разбор кода:** Подробно: какие библиотеки подключаем (и зачем), переменные (что они хранят), циклы (как итерации работают), функции (вход/выход). Если код сложный, добавил "визуализацию" — текстовые диаграммы или пошаговые симуляции (как код работает на примере).
- **Для новичка (4-й день изучения):** Я объясняю с нуля, без предположений. Если что-то неясно, подумай: "Что делает эта строка? Что в переменной сейчас?" Практикуйся, копируя код в Python-редактор (например, VS Code) и запуская по частям с print().

---

## 🔥 БЛОК 1: Python Fundamentals (5 вопросов)

### Вопрос 1 ⭐⭐⭐ 
**В вашем коде есть строка `for _ in range(intervals):`. Объясните:**
- Что означает символ `_`?
- Когда его использовать?
- Приведите 2 альтернативных способа написать этот же цикл

**Подробный комментарий:** В Python циклы — основа для повторяющихся задач, как обработка данных в ML (например, итерации по датасету). `_` — это конвенция (не ошибка), показывающая, что переменная не нужна. В ML это полезно для обучения модели: если нужно повторить эпоху N раз, но индекс не важен. Почему важно: помогает писать чистый код, избегая предупреждений от IDE (как PyCharm).

**Сухой алгоритм действий:**
1. Вызвать `range(intervals)` — создать последовательность чисел от 0 до intervals-1.
2. Для каждой итерации: присвоить число переменной `_` (но игнорировать его).
3. Выполнить тело цикла (что внутри for).
4. Повторить intervals раз.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** Здесь нужна только встроенная функция `range()` из Python (не нужно импортировать).
- **Переменные:** `intervals` — целое число (например, 5), задаёт количество повторений. `_` — временная переменная, хранит индекс (0,1,2...), но мы её не используем.
- **Как работает цикл:** Представь intervals=3. Цикл выполнится 3 раза:
  - Итерация 1: _ = 0 → выполни тело.
  - Итерация 2: _ = 1 → выполни тело.
  - Итерация 3: _ = 2 → выполни тело.
- Визуализация (текстовый флоу):
  ```
  Start → range(3) → [0,1,2]
  ├── Итерация 1: _=0 → [тело цикла] → next
  ├── Итерация 2: _=1 → [тело цикла] → next
  └── Итерация 3: _=2 → [тело цикла] → End
  ```

<details>
<summary>Правильный ответ</summary>

`_` - это соглашение Python для "неиспользуемой переменной":
- Показывает, что значение итератора нас не интересует (мы не используем индекс в коде).
- Используется когда нужно количество итераций, но не сами значения — например, в ML для N повторений эксперимента без зависимости от индекса.
- Некоторые IDE/линтеры (как pylint) не выдают warning на неиспользуемую `_`, но ругаются на другие переменные (типа `i`).

Альтернативы:
```python
# 1. С обычной переменной (хуже, потому что i не используется и IDE может предупредить)
for i in range(intervals):  # i будет 0,1,2... но игнорируется
    # i не используется

# 2. While цикл (сложнее, но полезен для ручного контроля)
counter = 0  # Переменная-счётчик, инициализируем 0
while counter < intervals:  # Пока counter меньше intervals, повторяем
    # Тело цикла здесь
    counter += 1  # Увеличиваем счётчик на 1 каждый раз
```
**Разбор альтернативы 2:** 
- Библиотеки: Никаких, чистый Python.
- Переменные: `counter` — хранит текущее значение (начинает с 0).
- Как работает: Проверяет условие → выполняет тело → увеличивает counter. Если intervals=3: counter=0 (да) → тело → counter=1; counter=1 (да) → тело → counter=2; counter=2 (да) → тело → counter=3; counter=3 (нет) → конец.
</details>

---

### Вопрос 2 ⭐⭐⭐
**В коде используется `current_stats['total_songs']` вместо `current_stats.total_songs`. Почему квадратные скобки?**

**Подробный комментарий:** Это про типы данных в Python. В ML часто работаем с данными из баз (SQL), которые возвращают словари (dict) — гибкие структуры для хранения "ключ-значение". Квадратные скобки [] для dict, точка . для объектов. Важно: если перепутать, код упадёт с ошибкой — в ML это может сломать пайплайн обработки данных.

**Сухой алгоритм действий:**
1. Проверить, что `current_stats` — словарь (dict).
2. Доступиться к ключу 'total_songs' через [].
3. Если ключа нет — ошибка KeyError (нужно обработать).

**Разбор кода (визуализация):**
- **Библиотеки/модули:** Если из SQL — нужен `sqlite3` (импорт: `import sqlite3` — для работы с базами данных SQLite).
- **Переменные:** `current_stats` — dict, например {'total_songs': 100}. Ключ 'total_songs' — строка, значение — число.
- Визуализация: Представь dict как коробку:
  ```
  current_stats = {'total_songs': 100, 'other': 'data'}
  Access: current_stats['total_songs'] → 100
  Если .total_songs → Ошибка: dict не имеет атрибутов как объект!
  ```

<details>
<summary>Правильный ответ</summary>

`current_stats` - это словарь (dict), возвращаемый из SQL запроса:
- `dict['key']` - доступ к элементу словаря (ключ — строка, как индекс в списке).
- `obj.attribute` - доступ к атрибуту объекта (класса).
- `sqlite3.Row` ведет себя как словарь, поэтому используем [] — это стандарт для результатов запросов в библиотеке sqlite3.
- `current_stats.total_songs` вызвал бы `AttributeError` (Python подумает, что это объект, а не dict).
**Почему в ML:** В pandas (библиотека для данных) DataFrame тоже использует [] для колонок.
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

**Подробный комментарий:** Обработка ошибок — ключ в ML, где данные "грязные" (например, API вернул объект без поля). Эти подходы проверяют, есть ли атрибут 'id' у объекта song. Для junior: начни с try-except, оно универсальное. В ML это спасает от краха скрипта при парсинге данных.

**Сухой алгоритм действий для каждого:**
- Подход 1: 1. Проверить hasattr (вернёт True/False). 2. Если да — взять значение. 3. Иначе — None.
- Подход 2: 1. Попробовать взять song.id. 2. Если ошибка AttributeError — поймать и присвоить None.
- Подход 3: 1. Вызвать getattr с дефолтом None. 2. Если атрибут есть — вернёт значение, иначе None.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** Встроенные: `hasattr` и `getattr` — из Python (не импорт). try-except — синтаксис языка.
- **Переменные:** `song` — объект (например, из API). `genius_id` — хранит ID или None.
- Визуализация на примере song без 'id':
  ```
  Подход 1: hasattr(song, 'id') → False → genius_id = None
  Подход 2: try song.id → AttributeError → except → genius_id = None
  Подход 3: getattr(song, 'id', None) → атрибут нет → genius_id = None
  ```
  Если 'id' есть: Все вернут значение.

<details>
<summary>Правильный ответ</summary>

- **Подход 1 (hasattr):** Быстрый, читаемый, для простых проверок атрибутов. Минус: Не ловит другие ошибки.
- **Подход 2 (try-except):** Лучше когда нужно различать типы ошибок или обработать сложную логику. В ML: Используй для API, где могут быть разные исключения.
- **Подход 3 (getattr):** Самый pythonic (идиоматичный) и быстрый для случая "атрибут или значение по умолчанию". Рекомендую для junior.

Лучший выбор: **Подход 3** для данной задачи. В ML пример: getattr(model, 'layers', []) — если нет слоёв, пустой список.
</details>

---

### Вопрос 4 ⭐⭐⭐
**В чём разница между `exit()` и `exit(1)`? Зачем передавать число?**

**Подробный комментарий:** `exit()` — из модуля sys (импорт: import sys), завершает программу. В ML скриптах важно указывать код: 0 — всё OK, >0 — ошибка, чтобы в пайплайнах (CI/CD) знать, сломалось ли.

**Сухой алгоритм действий:**
1. Вызвать sys.exit(code).
2. Если code=0 (или без) — успех.
3. Если code=1 — ошибка, ОС получит 1.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** `import sys` — системный модуль Python для взаимодействия с ОС.
- **Переменные:** Нет, это функция.
- Визуализация:
  ```
  exit() → sys.exit(0) → Программа конец, статус 0 (успех)
  exit(1) → sys.exit(1) → Программа конец, статус 1 (ошибка)
  В bash: echo $? → покажет 1
  ```

<details>
<summary>Правильный ответ</summary>

- `exit()` или `exit(0)` - успешное завершение программы.
- `exit(1)` - завершение с ошибкой (код возврата 1).
- Коды возврата важны для:
  - Скриптов оболочки (bash проверяет `$?` — переменная с кодом).
  - CI/CD пайплайнов (Jenkins/GitHub Actions: если >0, билд failed).
  - Системного администрирования (мониторинг ошибок).
- Стандарт: 0 = успех, 1-255 = различные типы ошибок. В ML: exit(1) если данные не загрузились.
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

**Подробный комментарий:** Циклы с continue/break — контроль потока. В ML это для фильтрации данных (пропустить invalid). Ошибка: break выходит после первой, а нужно все обработать. Для новичка: всегда проверяй, сколько итераций ожидается.

**Сухой алгоритм действий (проблемный код):**
1. Для каждой song в songs:
2. Если invalid — continue (пропустить остаток итерации).
3. process_song(song).
4. break — выйти из всего цикла.
5. Return "completed" (но только после первой валидной song).

**Разбор кода (визуализация):**
- **Библиотеки/модули:** Нет, чистый Python.
- **Переменные:** `songs` — список объектов. `song` — текущий элемент.
- Визуализация на songs = [invalid1, valid1, valid2]:
  ```
  song=invalid1 → if invalid → continue → next
  song=valid1 → process_song(valid1) → break → End цикл → return
  valid2 — не обработана!
  ```

<details>
<summary>Правильный ответ</summary>

**Проблема:** `break` выходит из цикла после обработки первой валидной песни. Результат: Не все песни обработаны.

**Исправление:**
```python
def process_songs(songs):  # Функция, вход: список songs
    for song in songs:  # Цикл по каждому song
        if song.invalid:  # Проверка атрибута invalid (предполагаем bool)
            continue  # Пропускаем невалидную, переходим к следующей
        process_song(song)    # Обрабатываем валидную (вызов другой функции)
        # Убираем break, чтобы обработать все
    return "completed"  # После всего цикла
```
**Разбор исправления:** Цикл теперь полный. На примере: invalid1 → continue; valid1 → process; valid2 → process; return.
</details>

---

## 🔥 БЛОК 2: Обработка данных и API (5 вопросов)

### Вопрос 6 ⭐⭐⭐⭐
**В скрипте есть retry логика только на уровне артиста, а не песен. Почему такая архитектура? Какие плюсы/минусы?**

**Подробный комментарий:** Retry — повтор попытки при ошибке (например, API перегруз). В ML для скрапинга данных (как lyrics для NLP). Архитектура: Retry на артисте, потому что ошибки глобальные (rate limit на весь API). Для junior: Думай о эффективности — не трать время на заведомо провальные запросы.

**Сухой алгоритм действий (retry на артисте):**
1. Для каждого артиста: Попробовать получить песни.
2. Если ошибка (rate limit) — подождать и retry весь артист.
3. Если OK — обработать песни без retry на каждой.

**Разбор кода (визуализация):** Предполагаем код с retry:
- **Библиотеки/модули:** Возможно `time` для sleep (import time — для пауз).
- **Переменные:** attempt — счётчик попыток.
- Визуализация:
  ```
  Артист1 → Запрос → Ошибка → Retry (sleep) → Запрос OK → Песни
  Артист2 → Запрос OK → Песни (без retry)
  ```

<details>
<summary>Правильный ответ</summary>

**Причины:**
- **Rate limiting** действует на уровне API connection, не на отдельные запросы (если заблокировали — все песни упадут).
- Если API заблокировал - все последующие запросы будут падать.
- Экономия времени: не пытаемся повторять каждую песню отдельно.

**Плюсы:**
- Быстрее обнаруживает проблемы с API (не тратит время на песни).
- Меньше ненужных запросов (экономит квоту API).
- Проще логика (один retry-цикл на артиста).

**Минусы:**
- Теряем отдельные песни при временных сбоях (если ошибка на одной песне).
- Менее гранулярный контроль (нельзя retry только неудачную песню).
**В ML:** Для больших датасетов — плюс, чтобы не ждать часы.
</details>

---

### Вопрос 7 ⭐⭐⭐⭐⭐
**Объясните эту формулу: `if (i + 1) % 10 == 0:`. Напишите 3 других полезных формулы с модулем для ML задач.**

**Подробный комментарий:** % — модуль (остаток от деления). В ML для периодических действий (лог каждые 10 шагов). Для новичка: (i+1) делает проверку на 10,20... (i начинается с 0).

**Сухой алгоритм действий:**
1. Взять i (индекс, от 0).
2. Добавить 1.
3. % 10 — остаток от деления на 10.
4. Если ==0 — да (кратно 10).

**Разбор кода (визуализация):**
- **Библиотеки/модули:** Нет.
- **Переменные:** i — int (0,1,2...).
- Визуализация: i=9 → (9+1)%10=10%10=0 → да; i=10 →11%10=1 → нет.

<details>
<summary>Правильный ответ</summary>

`(i + 1) % 10 == 0` - проверяет, является ли (i+1) кратным 10.
- При i=9: (9+1) % 10 = 0 ✅ (10-я итерация).
- При i=19: (19+1) % 10 = 0 ✅ (20-я).

**ML формулы с модулем:**
```python
# 1. Логирование каждые N эпох (в обучении модели)
if epoch % log_frequency == 0:  # epoch=0,10,20... (если log_frequency=10)
    print(f"Epoch {epoch}, Loss: {loss}")  # Вывод лосса

# 2. Сохранение модели каждые N батчей (в PyTorch)  
if batch_idx % save_every == 0:  # batch_idx=0,100,200...
    torch.save(model.state_dict(), f'model_{batch_idx}.pth')  # Сохранить состояние

# 3. Валидация каждые N шагов (проверка модели)
if step % validation_interval == 0:  # step=0,50,100...
    evaluate_model(model, val_loader)  # Вызов функции оценки
```
**Разбор 1:** import torch (если ML). Переменные: epoch — текущая эпоха. % проверяет кратность.
</details>

---

### Вопрос 8 ⭐⭐⭐
**В коде используется `random.uniform(3.0, 7.0)` для задержек. Объясните:**
- Зачем случайные задержки?
- Какие альтернативы существуют?
- Как выбрать диапазон?

**Подробный комментарий:** random — для случайности. В API-скрапинге задержки маскируют бота под человека. В ML: При сборе данных, чтобы не бан.

**Сухой алгоритм действий:**
1. Вызвать random.uniform(a,b) — число от a до b.
2. time.sleep(то число) — пауза.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** `import random` (случайные числа), `import time` (sleep).
- **Переменные:** Нет, функция возвращает float (например, 4.2).
- Визуализация: uniform(3,7) → может 3.1, 4.5, 6.9 (случайно).

<details>
<summary>Правильный ответ</summary>

**Зачем случайные задержки:**
- Имитация человеческого поведения (люди не кликают ровно каждые 5 сек).
- Избежание синхронизации множественных скраперов (если все одинаково — сервер заметит).
- Защита от pattern detection на сервере (анти-бот системы).

**Альтернативы:**
- **Exponential backoff:** `delay = base_delay * (2 ** attempt)` — растёт при ошибках (import math или **).
- **Fixed delay:** постоянная задержка, time.sleep(5) — просто, но легко детектить.
- **Jittered backoff:** экспоненциальная + случайность, delay + random.uniform(-1,1).

**Выбор диапазона:**
- Анализ ToS API (часто указывают лимиты, например, 100 запросов/минуту).
- A/B тестирование разных диапазонов (запустить скрипт, увидеть, когда бан).
- Мониторинг rate limit ответов (в headers API).
**В ML:** Для датасетов — тестируй на маленьком сете.
</details>

---

### Вопрос 9 ⭐⭐⭐⭐
**Что такое "graceful shutdown" в контексте вашего скрипта? Реализуйте простую версию для ML тренировки.**

**Подробный комментарий:** Graceful shutdown — аккуратный стоп (с сохранением). В ML: Если обучение длинное, сохрани модель при Ctrl+C. Для junior: Используй сигналы ОС.

**Сухой алгоритм действий:**
1. Установить обработчики сигналов (SIGINT, SIGTERM).
2. При сигнале — установить флаг stop.
3. В цикле проверять флаг и сохранить.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** `import signal` (сигналы ОС), `import torch` (для ML сохранения).
- **Переменные:** self.should_stop — bool (False → True при сигнале).
- Визуализация:
  ```
  Тренировка → Ctrl+C → _signal_handler → should_stop=True
  Цикл: if should_stop → save → break
  ```

<details>
<summary>Правильный ответ</summary>

**Graceful shutdown** - корректное завершение работы при получении сигнала остановки (Ctrl+C или kill).

```python
import signal  # Для обработки сигналов ОС (SIGINT=Ctrl+C, SIGTERM=kill)
import torch  # Библиотека ML, для моделей (pip install torch, но в коде assume установлен)

class GracefulTrainer:  # Класс для тренера модели
    def __init__(self):  # Инициализация
        self.should_stop = False  # Флаг: False — продолжаем
        signal.signal(signal.SIGINT, self._signal_handler)  # Установить обработчик для Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler)  # Для kill
    
    def _signal_handler(self, signum, frame):  # Обработчик: signum — номер сигнала
        print(f"Получен сигнал {signum}, завершаем эпоху...")  # Лог
        self.should_stop = True  # Установить флаг
    
    def train(self, model, dataloader, optimizer):  # Метод тренировки: model — модель, dataloader — данные, optimizer — оптимизатор
        for epoch in range(1000):  # Цикл по 1000 эпохам
            if self.should_stop:  # Проверка флага
                print("Сохраняем модель перед выходом...")  # Лог
                torch.save(model.state_dict(), 'checkpoint_emergency.pth')  # Сохранить состояние модели
                break  # Выход из цикла
                
            for batch in dataloader:  # Внутренний цикл по батчам
                if self.should_stop:  # Проверка внутри
                    break  # Выход из батча
                # Обычная тренировка: loss = ... (упрощено)
                loss = train_step(model, batch, optimizer)  # Предполагаемая функция
```
**Разбор:** Класс изолирует логику. Сигнал → handler → флаг → проверка в цикле. В ML: Сохраняет checkpoint для продолжения позже.
</details>

---

### Вопрос 10 ⭐⭐⭐⭐
**В базе данных используется `UNIQUE(artist, title)`. Объясните:**
- Зачем этот constraint?
- Что произойдёт при попытке вставить дубликат?
- Как это влияет на производительность?

**Подробный комментарий:** UNIQUE — ограничение в SQL. В ML: Для уникальности данных (нет дублей в датасете). База — SQLite, лёгкая для локальных скриптов.

**Сухой алгоритм действий (при вставке):**
1. Пытаться INSERT.
2. БД проверяет UNIQUE.
3. Если дубликат — ошибка.
4. Откат транзакции.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** `import sqlite3` — для БД.
- **Переменные:** Нет, это SQL-команда при создании таблицы.
- Визуализация:
  ```
  INSERT (artist='A', title='B') → UNIQUE check → Уже есть? → IntegrityError
  ```

<details>
<summary>Правильный ответ</summary>

**Зачем UNIQUE constraint:**
- Предотвращает дублирование песен одного артиста (artist+title уникальны вместе).
- Обеспечивает целостность данных на уровне БД (даже если код ошибётся).
- Защита от багов в коде приложения (скрапинг не добавит дубль).

**При вставке дубликата:**
- Возникает `sqlite3.IntegrityError` (исключение).
- Транзакция откатывается (изменения не сохраняются).
- В коде обрабатывается как "уже существует" (try-except).

**Влияние на производительность:**
- **Плюс:** Быстрая проверка существования через индекс (БД создаёт индекс автоматически).
- **Минус:** Дополнительная проверка при каждой вставке (немного замедляет INSERT).
- **Решение:** Создание composite index на (artist, title) — SQL: CREATE INDEX idx_artist_title ON songs(artist, title);
**В ML:** Ускоряет запросы SELECT WHERE artist='X'.
</details>

---

## 🔥 БЛОК 3: ML Engineering & Best Practices (5 вопросов)

### Вопрос 11 ⭐⭐⭐⭐⭐
**Этот скрипт собирает данные для NLP модели. Какие проблемы качества данных вы видите? Предложите 3 улучшения.**

**Подробный комментарий:** Качество данных — 80% успеха в ML. Проблемы: Грязный текст, несбалансированность. Для NLP (тексты): Нормализация обязательна. Для junior: Всегда чисти данные перед моделью.

**Сухой алгоритм действий (улучшения):**
1. Нормализация: Очистить текст.
2. Фильтрация: Проверить длину/качество.
3. Балансировка: Ограничить per artist.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** `import unicodedata` (нормализация unicode), `import re` (регулярки для очистки), `import random` (семплинг).
- **Переменные:** text — строка lyrics.
- Визуализация нормализации: "Café" → "Cafe" (NFKD).

<details>
<summary>Правильный ответ</summary>

**Проблемы:**
1. **Нет нормализации текста** - разные кодировки, спецсимволы (например, é → e, или эмодзи).
2. **Нет фильтрации по качеству** - могут попасть инструментальные треки (без текста).
3. **Нет балансировки датасета** - перекос в сторону популярных артистов (больше песен у Тейлор Свифт, чем у нишевых).

**Улучшения:**
```python
# 1. Нормализация текста (очистка для NLP)
import unicodedata  # Для unicode нормализации
import re  # Для регулярных выражений (удаление символов)

def normalize_lyrics(text):  # Вход: строка text
    text = unicodedata.normalize('NFKD', text)  # Разложить символы (é → e')
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)  # Удалить не-буквы/пробелы
    return text.lower().strip()  # В нижний регистр, убрать пробелы

# 2. Фильтрация качества (проверить, что lyrics валидны)
def is_valid_lyrics(lyrics):  # Вход: строка
    words = lyrics.split()  # Разбить на слова (list)
    return (len(words) >= 50 and  # Минимум 50 слов
            len(set(words)) / len(words) > 0.3 and  # Lexical diversity (уникальные/все >30%)
            not any(marker in lyrics.lower() for marker in  # Нет маркеров
                   ['instrumental', 'no lyrics', 'beat only']))

# 3. Балансировка (ограничить песни per artist)
import random  # Для семплинга

def sample_songs_balanced(artist_songs, max_per_artist=100):  # Вход: list песен, max
    return random.sample(artist_songs, min(len(artist_songs), max_per_artist))  # Семпл из списка
```
**Разбор 1:** re.sub — заменяет паттерн: r'[^\w\s]' — всё кроме букв/пробелов.
</details>

---

### Вопрос 12 ⭐⭐⭐⭐
**Как бы вы добавили логирование метрик для мониторинга процесса скрапинга? Какие метрики важны?**

**Подробный комментарий:** Логи — для отладки. В ML: Мониторинг, чтобы знать, сколько данных собрано. Метрики: Как в аналитике.

**Сухой алгоритм действий:**
1. Инициализировать метрики.
2. В процессе — обновлять.
3. Логировать.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** `import time` (время), `from collections import defaultdict` (dict с дефолтом 0).
- **Переменные:** metrics — dict для счёта.
- Визуализация: success_rate = 0.8 (80%).

<details>
<summary>Правильный ответ</summary>

**Важные метрики:** Success rate (успех/все), время обработки, throughput (песен/час).

```python
import time  # Для измерения времени
from collections import defaultdict  # Dict, где значения по умолчанию 0

class ScrapingMetrics:  # Класс для метрик
    def __init__(self):  # Init
        self.metrics = defaultdict(int)  # {'success':0, ...}
        self.timings = []  # List для времён
        self.start_time = time.time()  # Текущее время
    
    def log_metrics(self):  # Метод логирования
        duration = time.time() - self.start_time  # Прошедшее время
        
        # Ключевые метрики
        success_rate = self.metrics['success'] / max(self.metrics['total'], 1)  # % успеха
        avg_processing_time = sum(self.timings) / max(len(self.timings), 1)  # Среднее время
        throughput = self.metrics['success'] / duration * 3600  # песен/час
        
        # Логирование (предполагаем logger из logging)
        logger.info(f"""
        Success Rate: {success_rate:.2%}  # Формат 80.00%
        Average Processing Time: {avg_processing_time:.2f}s  
        Throughput: {throughput:.1f} songs/hour
        Errors: {self.metrics['errors']}
        Rate Limits: {self.metrics['rate_limits']}
        """)
        
        # Отправка в мониторинг (Prometheus, etc.) — опционально
        send_to_monitoring({  # Функция отправки
            'scraping_success_rate': success_rate,
            'scraping_throughput': throughput,
            'scraping_error_rate': self.metrics['errors'] / self.metrics['total']
        })
```
**Разбор:** defaultdict(int) — при metrics['new'] =1, создаёт 0 сначала.
</details>

---

### Вопрос 13 ⭐⭐⭐⭐⭐
**Представьте, что этот скрипт нужно масштабировать на 10,000 артистов. Какие узкие места и как решать?**

**Подробный комментарий:** Масштаб — ключ в ML engineering. Узкие места: Последовательность, БД, API. Решения: Параллелизм.

**Сухой алгоритм действий:**
1. Разделить артистов на чанки.
2. Запустить в потоках.
3. Использовать шардинг БД.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** `import threading`, `from queue import Queue`, `import concurrent.futures` (параллелизм), `import sqlite3` (БД).
- **Переменные:** queue — очередь задач.
- Визуализация: 5 workers → каждый берёт чанк артистов параллельно.

<details>
<summary>Правильный ответ</summary>

**Узкие места:**
1. **Последовательная обработка** - один артист за раз (медленно на 10k).
2. **Единая база данных** - блокировки при записи (много потоков пишут одновременно).  
3. **Rate limiting** - один IP, один поток запросов (бан от API).

**Решения:**
```python
# 1. Многопоточность с очередью (параллельная обработка)
import threading  # Потоки
from queue import Queue  # Очередь задач
import concurrent.futures  # Executor для пула

class ScalableScraper:  # Класс скрапера
    def __init__(self, num_workers=5):  # 5 потоков
        self.artist_queue = Queue()  # Очередь артистов
        self.num_workers = num_workers
        
    def process_artists_parallel(self, artists):  # Вход: list артистов
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:  # Пул потоков
            # Разделяем артистов между воркерами
            chunks = [artists[i::self.num_workers] for i in range(self.num_workers)]  # Чанки: [0,5,10...], [1,6,11...]
            futures = [executor.submit(self.process_chunk, chunk) for chunk in chunks]  # Запуск задач
            concurrent.futures.wait(futures)  # Ждать завершения

# 2. Шардинг БД (разделение на несколько файлов)
import sqlite3  # БД

class ShardedDatabase:  # Класс шардов
    def __init__(self, num_shards=4):  # 4 шарда
        self.shards = [sqlite3.connect(f'lyrics_shard_{i}.db') for i in range(num_shards)]  # List коннектов
    
    def get_shard(self, artist_name):  # Выбрать шард по хэшу
        shard_id = hash(artist_name) % len(self.shards)  # hash → 0-3
        return self.shards[shard_id]  # Вернуть conn

# 3. Пул прокси/IP адресов (для обхода rate limit)
import itertools  # Для cycle

class ProxyManager:  # Менеджер прокси
    def __init__(self, proxy_list):  # Вход: list прокси ['http://ip1', ...]
        self.proxies = itertools.cycle(proxy_list)  # Бесконечный цикл по list
        
    def get_session(self):  # Получить сессию
        proxy = next(self.proxies)  # Следующий прокси
        session = requests.Session()  # Из requests (import requests — для HTTP)
        session.proxies = {'http': proxy, 'https': proxy}  # Установить
        return session
```
**Разбор 1:** ThreadPoolExecutor — запускает функции параллельно, wait — синхронизирует.
</details>

---

### Вопрос 14 ⭐⭐⭐
**Какие потенциальные bias (предвзятости) может содержать такой датасет? Как их минимизировать?**

**Подробный комментарий:** Bias — предвзятость данных, приводит к плохой модели (например, NLP модель лучше на английском). Минимизация: Баланс.

**Сухой алгоритм действий:**
1. Определить тип bias.
2. Семплировать сбалансировано.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** `import random`, `from collections import defaultdict`.
- **Переменные:** by_decade — dict {1970: [songs], ...}.
- Визуализация: Из 10 жанров — по 100 песен.

<details>
<summary>Правильный ответ</summary>

**Типы bias:**
1. **Popularity bias** - больше песен у популярных артистов (перекос к хитам).
2. **Temporal bias** - только современные треки (мало старых).
3. **Genre bias** - перекос в сторону хип-хопа (Genius популярен там).
4. **Language bias** - преимущественно английский язык.
5. **Platform bias** - только то, что есть на Genius (не все песни).

**Методы минимизации:**
```python
# 1. Стратификация по жанрам (баланс по группам)
def balanced_sampling(artists_by_genre, samples_per_genre=100):  # Вход: dict {genre: [artists]}
    balanced_dataset = []  # Пустой list
    for genre, artists in artists_by_genre.items():  # Цикл по жанрам
        sampled = random.sample(artists, min(len(artists), samples_per_genre))  # Семпл
        balanced_dataset.extend(sampled)  # Добавить
    return balanced_dataset

# 2. Временная балансировка (по десятилетиям)
from collections import defaultdict  # Dict по умолчанию list

def sample_by_decade(songs):  # Вход: list песен с .year
    by_decade = defaultdict(list)  # {decade: [songs]}
    for song in songs:  # Цикл
        decade = (song.year // 10) * 10  # 1995 → 1990
        by_decade[decade].append(song)  # Добавить
    
    # Равное количество из каждого десятилетия
    balanced = []  # Пустой
    min_count = min(len(songs) for songs in by_decade.values())  # Мин размер
    for decade_songs in by_decade.values():  # Цикл
        balanced.extend(random.sample(decade_songs, min_count))  # Семпл и добавить
    return balanced
```
**Разбор 1:** random.sample — берёт случайные без повторений.
</details>

---

### Вопрос 15 ⭐⭐⭐⭐
**Как бы вы реализовали data versioning для этого датасета? Зачем это нужно в ML проектах?**

**Подробный комментарий:** Versioning — как git для данных. В ML: Для reproducibility (воспроизведения экспериментов).

**Сухой алгоритм действий:**
1. Вычислить хэш файла.
2. Создать метаданные.
3. Скопировать с версией.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** `import hashlib` (хэш), `import json` (метаданные), `from datetime import datetime`, `import os`, `import shutil`.
- **Переменные:** version_info — dict.
- Визуализация: dataset.db → dataset_v20250822_120000.db + metadata.json.

<details>
<summary>Правильный ответ</summary>

**Зачем data versioning:**
- **Reproducibility** - можно воспроизвести эксперименты (модель на v1 данных).
- **Debugging** - найти, на каких данных модель работала плохо.
- **Compliance** - отследить происхождение данных (GDPR).
- **Collaboration** - команда работает с одной версией данных.

**Реализация:**
```python
import hashlib  # Для MD5 хэша (проверка целостности)
import json  # Для сохранения dict в файл
from datetime import datetime  # Для дат
import os  # Для файлов (size)
import shutil  # Для копирования

class DatasetVersioning:  # Класс версионирования
    def __init__(self, base_path):  # base_path — папка
        self.base_path = base_path
        self.metadata_file = f"{base_path}/dataset_metadata.json"  # Файл метаданных
        
    def create_version(self, dataset_path, description=""):  # Вход: путь к db, описание
        # Хэш для проверки целостности
        dataset_hash = self._calculate_hash(dataset_path)  # Вызов метода
        
        version_info = {  # Dict метаданных
            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),  # 20250822_120000
            'hash': dataset_hash,
            'description': description,
            'size': os.path.getsize(dataset_path),  # Размер файла
            'created_at': datetime.now().isoformat(),  # ISO дата
            'scraped_artists': self._get_artist_list(),  # Предполагаемые методы
            'total_songs': self._count_songs(),
            'filters_applied': self._get_filters()
        }
        
        # Сохраняем метаданные (в json)
        self._save_metadata(version_info)  # Вызов
        
        # Копируем датасет с версионированием
        versioned_path = f"{self.base_path}/dataset_v{version_info['version']}.db"
        shutil.copy2(dataset_path, versioned_path)  # Копировать с метаданными
        
        return version_info  # Вернуть info
    
    def _calculate_hash(self, file_path):  # Приватный метод хэша
        hash_md5 = hashlib.md5()  # MD5 объект
        with open(file_path, "rb") as f:  # Открыть бинарно
            for chunk in iter(lambda: f.read(4096), b""):  # Чанки по 4KB
                hash_md5.update(chunk)  # Обновить хэш
        return hash_md5.hexdigest()  # Строка хэша
```
**Разбор:** _calculate_hash читает файл чанками (не в память целиком, для больших файлов).
</details>

---

## 🔥 БЛОК 4: Проблемы и Отладка (5 вопросов)

### Вопрос 16 ⭐⭐⭐⭐
**Скрипт внезапно стал получать много timeout ошибок. Опишите step-by-step процедуру диагностики.**

**Подробный комментарий:** Таймауты — сетевые проблемы. Диагностика: Логи + проверки. В ML: Для API с данными.

**Сухой алгоритм действий (диагностика):**
1. Добавить логи.
2. Проверить сеть.
3. Проверить ресурсы.
4. Проверить БД.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** `import time`, `import psutil` (ресурсы, pip install psutil), `import requests` (тест сети).
- **Переменные:** start_time — float.
- Визуализация: CPU 90% → проблема.

<details>
<summary>Правильный ответ</summary>

**Пошаговая диагностика:**

```python
# 1. Добавить детальное логирование (в каждый запрос)
import time  # Время
import psutil  # Системные ресурсы (CPU, memory)

def diagnose_timeouts():  # Функция диагностики
    start_time = time.time()  # Старт
    
    # Проверяем сетевое подключение
    try:
        response = requests.get('https://genius.com', timeout=5)  # Тест запрос, 5 сек таймаут
        logger.info(f"Genius доступен: {response.status_code}")  # 200 OK
    except Exception as e:  # Ловить все
        logger.error(f"Genius недоступен: {e}")  # Лог ошибки
    
    # Проверяем системные ресурсы
    logger.info(f"CPU: {psutil.cpu_percent()}%")  # % CPU
    logger.info(f"Memory: {psutil.virtual_memory().percent}%")  # % памяти
    logger.info(f"Disk: {psutil.disk_usage('/').percent}%")  # % диска
    
    # Проверяем производительность БД
    db_start = time.time()  # Старт
    conn.execute("SELECT COUNT(*) FROM songs")  # Тест запрос
    db_time = time.time() - db_start  # Время
    logger.info(f"DB query time: {db_time:.3f}s")  # Лог

# 2. Мониторинг паттернов ошибок (время таймаутов)
class TimeoutMonitor:  # Класс монитора
    def __init__(self):
        self.timeout_times = []  # List времён таймаутов
        self.success_times = []  # Успехов
    
    def log_request(self, success, duration):  # Лог запроса
        if success:
            self.success_times.append(duration)  # Добавить время
        else:
            self.timeout_times.append(time.time())  # Текущее время
    
    def analyze_pattern(self):  # Анализ
        # Есть ли временные паттерны в timeout?
        if len(self.timeout_times) >= 5:  # Минимум 5
            intervals = [self.timeout_times[i] - self.timeout_times[i-1] 
                        for i in range(1, len(self.timeout_times))]  # Интервалы
            avg_interval = sum(intervals) / len(intervals)  # Среднее
            logger.info(f"Average timeout interval: {avg_interval:.1f}s")  # Лог
```

**Возможные причины и решения:**
1. **Rate limiting** → увеличить задержки (time.sleep больше).
2. **Сетевые проблемы** → добавить retry с backoff (повтор с ростом задержки).
3. **Перегрузка API** → использовать несколько ключей/прокси (как в Q13).
4. **Проблемы с БД** → оптимизировать запросы, добавить индексы (CREATE INDEX).
</details>

---

### Вопрос 17 ⭐⭐⭐
**В логах видно: "Database locked" каждые 30 секунд. В чём причина и как исправить?**

**Подробный комментарий:** Locked — когда БД занята. В SQLite — из-за однопоточности. Решение: WAL mode для concurrency.

**Сухой алгоритм действий:**
1. Включить WAL.
2. Использовать транзакции.
3. Batch вставки.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** `import sqlite3`, `from contextlib import contextmanager` (для @contextmanager).
- **Переменные:** pending_operations — list операций.
- Визуализация: 100 вставок → flush → COMMIT один раз.

<details>
<summary>Правильный ответ</summary>

**Возможные причины:**
1. **Длинные незакрытые транзакции** (BEGIN без COMMIT). 
2. **Другой процесс держит соединение** (два скрипта на одну БД).
3. **Незакрытые курсоры** (не close()).
4. **WAL mode не включён** (по умолчанию journal mode — locks).

**Диагностика и решения:**
```python
# 1. Включить WAL mode для лучшей конкурентности (позволяет читать во время записи)
def optimize_sqlite(conn):  # conn — sqlite3.connect
    conn.execute("PRAGMA journal_mode=WAL")  # WAL — write-ahead log
    conn.execute("PRAGMA synchronous=NORMAL")  # Синхронизация нормальная (быстрее)
    conn.execute("PRAGMA temp_store=MEMORY")  # Temp в памяти
    conn.execute("PRAGMA mmap_size=1073741824")  # 1GB mmap для скорости
    conn.commit()  # Применить

# 2. Контекстные менеджеры для транзакций (авто COMMIT/ROLLBACK)
from contextlib import contextmanager  # Для @contextmanager

@contextmanager
def db_transaction(conn):  # Контекст: with db_transaction(conn):
    try:
        conn.execute("BEGIN")  # Старт транзакции
        yield conn  # Передать conn внутрь with
        conn.execute("COMMIT")  # Успех — коммит
    except Exception:  # Ошибка
        conn.execute("ROLLBACK")  # Откат
        raise  # Перебросить ошибку

# 3. Проверка заблокированных процессов (кто держит файл)  
def check_db_locks():
    try:
        # Проверяем, кто держит файл БД (нужен lsof: pip install lsof)
        import lsof  # Внешний, для ОС
        processes = lsof.lsof('+D', 'lyrics.db')  # Процессы с db
        for proc in processes:
            logger.info(f"Process holding DB: PID {proc.pid}, {proc.command}")
    except:
        logger.info("Could not check DB locks")

# 4. Batch commits вместо частых коммитов (группировать вставки)
class BatchDatabase:  # Класс батчей
    def __init__(self, batch_size=100):  # 100 операций
        self.batch_size = batch_size
        self.pending_operations = []  # List pending
    
    def add_song(self, *args):  # Добавить песню (args — значения)
        self.pending_operations.append(('INSERT', args))  # Tuple в list
        if len(self.pending_operations) >= self.batch_size:  # Если полно
            self.flush()  # Выполнить
    
    def flush(self):  # Выполнить батч
        with db_transaction(self.conn):  # В транзакции
            for op_type, args in self.pending_operations:  # Цикл
                if op_type == 'INSERT':
                    self.conn.execute("INSERT INTO songs VALUES (...)", args)  # Вставка
        self.pending_operations.clear()  # Очистить
```
**Разбор 4:** flush — один COMMIT на 100 вставок, меньше locks.
</details>

---

### Вопрос 18 ⭐⭐⭐⭐
**Программа работала 3 дня, собрала 50k песен, затем крашнулась с MemoryError. Что случилось и как предотвратить?**

**Подробный комментарий:** MemoryError — памяти не хватило. Причины: Leak (не очищается). В ML: Большие датасеты.

**Сухой алгоритм действий:**
1. Мониторить память.
2. Очищать периодически.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** `import psutil` (память), `import gc` (garbage collector).
- **Переменные:** growth — разница памяти.
- Визуализация: Memory 500MB → check → >1000? → cleanup.

<details>
<summary>Правильный ответ</summary>

**Возможные причины:**
1. **Memory leak** в lyricsgenius библиотеке (объекты не удаляются).
2. **Накопление объектов** в session_stats или кэшах (list растёт).
3. **Рост WAL файла** SQLite (не очищается).
4. **Незакрытые соединения/курсоры** (hold memory).

**Решения:**
```python
# 1. Мониторинг памяти (проверять usage)
import psutil  # Ресурсы
import gc  # Garbage collector (ручная очистка)

class MemoryMonitor:  # Монитор
    def __init__(self, threshold_mb=1000):  # Порог 1GB
        self.threshold_mb = threshold_mb
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # Начальная память MB
        
    def check_memory(self):  # Проверка
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # Текущая
        growth = current_memory - self.initial_memory  # Рост
        
        logger.info(f"Memory usage: {current_memory:.1f}MB (+{growth:.1f}MB)")  # Лог
        
        if growth > self.threshold_mb:  # Если превысил
            logger.warning("High memory usage detected!")
            self.cleanup()  # Очистить
            
    def cleanup(self):  # Очистка
        # Принудительная сборка мусора
        collected = gc.collect()  # Собрать, вернуть число объектов
        logger.info(f"Garbage collected: {collected} objects")
        
        # Очистка кэшей (пример для genius)
        if hasattr(self, 'genius'):  # Если есть атрибут
            self.genius._session.close()  # Закрыть сессию
            self.genius = lyricsgenius.Genius(TOKEN)  # Пересоздать (import lyricsgenius)

# 2. Периодическая очистка БД (WAL и vacuum)
def optimize_database_periodically(conn, interval_songs=10000):  # Каждые 10k песен
    if self.session_stats['added'] % interval_songs == 0:  # Проверка
        logger.info("Optimizing database...")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # Очистить WAL
        conn.execute("VACUUM")  # Дефрагментировать БД
        conn.execute("ANALYZE")  # Обновить статистику

# 3. Перезапуск сессии периодически (избежать leaks)
def restart_session_periodically(self, max_songs_per_session=5000):  # Каждые 5k
    if self.session_stats['processed'] >= max_songs_per_session:  # Проверка
        logger.info("Restarting session to prevent memory leaks...")
        self.close()  # Закрыть текущую
        self.__init__(TOKEN, self.db_path)  # Переинициализировать

# 4. Streaming обработка вместо загрузки всего в память (генераторы)
def stream_process_artists(self, artists):  # Вход: list артистов
    for artist in artists:  # По одному
        # Обрабатываем по одному артисту, не держим всех в памяти
        songs = self.get_artist_songs_generator(artist)  # Generator! (yield, не list)
        for song in songs:  # По песням
            self.process_song(song)
            del song  # Явно удалить (помогает GC)
```
**Разбор 4:** Generator — yield даёт по одному, не хранит весь list в памяти.
</details>

---

### Вопрос 19 ⭐⭐⭐⭐⭐
**После обновления библиотеки lyricsgenius структура song объекта изменилась. Как сделать код устойчивым к таким изменениям?**

**Подробный комментарий:** API меняются, код ломается. Решение: Adapter — изоляция.

**Сухой алгоритм действий:**
1. Создать adapter.
2. Пробовать разные атрибуты.

**Разбор кода (визуализация):**
- **Библиотеки/модули:** Нет, чистый Python.
- **Переменные:** raw_song — оригинальный объект.
- Визуализация: attr 'title' нет → 'name' → да → return.

<details>
<summary>Правильный ответ</summary>

**Стратегии устойчивости к изменениям API:**

```python
# 1. Adapter pattern для изоляции внешних зависимостей (обёртка)
class SongAdapter:  # Adapter класс
    def __init__(self, raw_song):  # Вход: оригинальный song от API
        self.raw_song = raw_song  # Сохранить
        
    @property  # Свойство (как атрибут)
    def title(self):  # Getter для title
        # Пробуем разные варианты названий атрибутов (на случай изменений)
        for attr in ['title', 'name', 'song_title', 'track_name']:  # List возможных
            if hasattr(self.raw_song, attr):  # Есть ли
                return getattr(self.raw_song, attr)  # Вернуть
        return "Unknown Title"  # Дефолт
    
    @property  
    def lyrics(self):  # Для lyrics
        for attr in ['lyrics', 'text', 'content', 'lyric_text']:  # Варианты
            if hasattr(self.raw_song, attr):
                lyrics = getattr(self.raw_song, attr)  # Взять
                if lyrics and len(lyrics.strip()) > 0:  # Не пусто
                    return lyrics
        return ""  # Пусто
    
    @property
    def url(self):  # Для url
        return getattr(self.raw_song, 'url', 
                      getattr(self.raw_song, 'song_url', ''))  # Каскад дефолтов

# 2. Версионирование совместимости (если версии библиотеки известны)
class GeniusClient:  # Клиент
    def __init__(self, version='latest'):  # Версия
        self.version = version  # Сохранить
        
    def get_lyrics(self, song):  # Метод
        if self.version == 'old':  # Если старая
            return song.lyrics  # Старый атрибут
        else:
            return song.text  # Новый
```
**Разбор 1:** @property — делает метод как атрибут (adapter.title). for attr — fallback на изменения.
</details>

Конечно! Ниже — **полностью доработанная и расширенная версия твоего файла** `Интервью на позицию Junior ML Engineer.md`, переписанная как **обучающий гайд для новичка**, который только 4-й день учит Python.

Я добавил:
- 🧠 Подробные объяснения каждого понятия
- 🔧 Пошаговый алгоритм
- 💡 Комментарии к коду
- 📦 Объяснение библиотек и модулей
- 🔄 Визуализацию потока выполнения
- 🎯 Практические советы

---

# 🧠 Интервью на позицию Junior ML Engineer — Полный разбор для новичка

> 📌 **Ты только 4 дня учишь Python? Отлично!**  
> Этот документ — твой мост от "я только начал" к "я понимаю, как работает реальный ML-скрипт".

---

## 📋 Критерии оценки: Что значит "пройти интервью"?

| Баллы | Что это значит |
|------|----------------|
| **15–20** | Ты понимаешь логику → проходишь как **Strong Junior** |
| **18–20** | Ты мыслишь как инженер → **Senior потенциал** |
| **<15** | Нужно больше практики |

> ✅ **Важно**: тебя **не просят помнить всё наизусть**.  
> Главное — **понимать, почему** и **как думать**, когда видишь код.

---

# 🔥 БЛОК 1: Python Fundamentals (Основы Python)

## Вопрос 1 ⭐⭐⭐  
### `for _ in range(intervals):` — Что это? Зачем `_`?

### 🧠 Что происходит:
```python
for _ in range(intervals):
    download_song()
```

#### 🔍 Разбор по частям:
| Часть | Что делает |
|------|------------|
| `for` | Начинает цикл |
| `_` | Специальное имя переменной — **"я не буду её использовать"** |
| `in` | Говорит: "для каждого элемента из..." |
| `range(intervals)` | Создаёт список чисел: `0, 1, 2, ..., intervals-1` |

#### 💡 Пример:
Если `intervals = 3`, то `range(3)` → `[0, 1, 2]`  
Цикл выполнится 3 раза.

#### ❓ Зачем `_`?
- Это **соглашение** среди Python-разработчиков.
- Показывает: "Мне важен **счётчик**, но не важно **само число**".
- IDE (например, VS Code) не будет ругаться: "Переменная `i` не используется".

#### ✅ Когда использовать:
- Когда тебе нужно просто **повторить действие N раз**.
- Например: "Сделай 5 попыток подключения".

#### 🔄 Альтернативы:

```python
# 1. Использовать обычную переменную (но это "некрасиво")
for i in range(intervals):
    download_song()  # i не используется → IDE может предупредить

# 2. While-цикл (сложнее, но иногда нужен)
counter = 0
while counter < intervals:
    download_song()
    counter += 1  # Не забудь увеличить!
```

> ⚠️ `while` легко забыть увеличить счётчик → бесконечный цикл!

---

## Вопрос 2 ⭐⭐⭐  
### `current_stats['total_songs']` vs `current_stats.total_songs`

### 🧠 В чём разница?

| Синтаксис | Где используется |
|----------|------------------|
| `obj['key']` | Для **словарей (dict)** и `sqlite3.Row` |
| `obj.key` | Для **объектов (classes)** и атрибутов |

#### 💡 Пример:
```python
# Это словарь
current_stats = {
    'total_songs': 100,
    'errors': 5
}

print(current_stats['total_songs'])  # ✅ Работает
print(current_stats.total_songs)     # ❌ Ошибка: AttributeError
```

#### 🛠 Почему `sqlite3.Row` использует `[]`?
- Когда ты делаешь SQL-запрос, база возвращает **не объект**, а **строку данных**.
- `sqlite3.Row` — это специальный тип, который ведёт себя как **словарь**.
- Поэтому: `row['artist']`, а не `row.artist`.

---

## Вопрос 3 ⭐⭐⭐⭐  
### Как безопасно получить атрибут объекта?

### 🧩 Есть 3 способа:

#### 🔹 Подход 1: `hasattr()`
```python
if hasattr(song, 'id'):
    genius_id = song.id
else:
    genius_id = None
```
- ✅ Читаемо
- ❌ Два обращения к объекту: сначала проверка, потом получение

#### 🔹 Подход 2: `try-except`
```python
try:
    genius_id = song.id
except AttributeError:
    genius_id = None
```
- ✅ Надёжно
- ✅ Работает с любыми ошибками
- ❌ Медленнее, если ошибок много

#### 🔹 Подход 3: `getattr()` ← **лучший выбор**
```python
genius_id = getattr(song, 'id', None)
```
- ✅ Одна строка
- ✅ Быстро
- ✅ Pythonic (так пишут настоящие Python-разработчики)

> 📌 `getattr(obj, 'attr', default)` — **запомни эту функцию**.

---

## Вопрос 4 ⭐⭐⭐  
### `exit()` vs `exit(1)` — в чём разница?

### 🧠 Коды завершения программы:

| Код | Значение |
|-----|---------|
| `exit(0)` | Всё хорошо, программа завершилась успешно |
| `exit(1)` | Ошибка! Что-то пошло не так |

#### 💡 Пример:
```python
if not internet_connected():
    print("Нет интернета!")
    exit(1)  # Скрипт остановится, система узнает: "была ошибка"
```

#### 📦 Где это используется?
- В **автоматических скриптах (CI/CD)**
- В **серверах**: если скрипт упал, система может перезапустить его

---

## Вопрос 5 ⭐⭐⭐  
### Ошибка в цикле: `break` внутри `for`

#### ❌ Проблемный код:
```python
def process_songs(songs):
    for song in songs:
        if song.invalid:
            continue
        process_song(song)
        break  # ← ОШИБКА: выходит после первой песни!
    return "completed"
```

#### 🔍 Как работает:
Пусть `songs = [song1, song2, song3]`  
1. `song1` — валидный → обработали → `break` → **выход из цикла**  
2. `song2`, `song3` — **не обработаны!**

#### ✅ Правильный код:
```python
def process_songs(songs):
    for song in songs:
        if song.invalid:
            continue  # Пропускаем плохую песню
        process_song(song)  # Обрабатываем хорошую
        # Никакого break!
    return "completed"
```

#### 🔄 Визуализация:
```
song1 → валидный → обработан → продолжаем
song2 → невалидный → skip → продолжаем
song3 → валидный → обработан → продолжаем
...
все песни обработаны
```

> 💡 `continue` = "пропусти эту итерацию"  
> `break` = "выйди из цикла полностью"

---

# 🔥 БЛОК 2: Обработка данных и API

## Вопрос 6 ⭐⭐⭐⭐  
### Почему retry на уровне артиста, а не песен?

### 🧠 Логика:
- Если API сказал "ты заблокирован", то **все запросы упадут**.
- Нет смысла пытаться скачать 100 песен артиста, если первая уже дала ошибку.

#### ✅ Плюсы:
- Быстро реагируем на блокировку
- Экономим время и трафик

#### ❌ Минусы:
- Если упала только одна песня — мы теряем её

> 📌 Это **компромисс**: скорость vs надёжность.

---

## Вопрос 7 ⭐⭐⭐⭐⭐  
### `(i + 1) % 10 == 0` — что это?

#### 🧮 Оператор `%` — **остаток от деления**

| Выражение | Результат | Почему |
|---------|----------|--------|
| `10 % 3` | 1 | 10 = 3*3 + 1 |
| `8 % 4` | 0 | 8 делится на 4 без остатка |
| `9 + 1 = 10`, `10 % 10 = 0` | ✅ | Значит, 10 кратно 10 |

#### 💡 Зачем `i + 1`?
- Циклы в Python начинаются с `i = 0`
- Чтобы проверить "каждые 10 песен", нужно `(i + 1) % 10 == 0`

#### ✅ Пример:
```python
for i, song in enumerate(songs):
    download_song(song)
    if (i + 1) % 10 == 0:
        print(f"Загружено {i + 1} песен")
```

#### 📊 Где ещё используется `%` в ML?
```python
# 1. Логирование каждые N эпох
if epoch % 5 == 0:
    print(f"Loss на эпохе {epoch}: {loss}")

# 2. Сохранение модели
if batch_idx % 100 == 0:
    torch.save(model.state_dict(), f"model_step_{batch_idx}.pth")

# 3. Валидация
if step % 500 == 0:
    validate(model)
```

---

## Вопрос 8 ⭐⭐⭐  
### `random.uniform(3.0, 7.0)` — зачем случайные задержки?

#### 📦 Библиотека: `random`
- `random.uniform(a, b)` — случайное число между `a` и `b`

#### 💡 Зачем:
- Если ты качаешь песни **каждые 1 секунду**, сервер может понять: "это бот!"
- Случайные задержки имитируют **человека**, который не идеален.

#### ✅ Пример:
```python
import random
import time

for song in songs:
    download_song(song)
    delay = random.uniform(3.0, 7.0)  # от 3 до 7 секунд
    time.sleep(delay)  # ждём
```

#### 🔄 Альтернативы:
| Тип | Пример |
|-----|--------|
| `time.sleep(5)` | постоянная задержка |
| `exponential backoff` | при ошибке ждать 2, 4, 8, 16 секунд |

---

## Вопрос 9 ⭐⭐⭐⭐  
### Что такое **graceful shutdown**?

### 🧠 Это — **вежливое завершение** программы.

#### 💡 Пример:
- Ты нажал `Ctrl+C` → программа не должна просто умереть.
- Она должна:  
  1. Дописать текущую песню  
  2. Сохранить прогресс  
  3. Закрыть файлы и базу данных

#### ✅ Код:
```python
import signal
import torch

class GracefulTrainer:
    def __init__(self):
        self.should_stop = False
        # Ловим сигналы: Ctrl+C (SIGINT) и команду stop (SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"Получен сигнал {signum}, завершаем...")
        self.should_stop = True
    
    def train(self, model, dataloader):
        for epoch in range(1000):
            if self.should_stop:
                print("Сохраняем модель...")
                torch.save(model.state_dict(), 'checkpoint.pth')
                break  # выходим
            # обучение...
```

---

# 🔥 БЛОК 3: ML Engineering & Best Practices

## Вопрос 10 ⭐⭐⭐⭐  
### `UNIQUE(artist, title)` — зачем?

#### 📦 SQLite — база данных
- Хранит данные в таблицах
- `UNIQUE` — **ограничение**: нельзя вставить одинаковые строки

#### 💡 Пример:
| artist | title |
|--------|-------|
| Taylor Swift | Blank Space |
| Taylor Swift | Blank Space | ← ❌ запрещено!

#### ✅ Плюсы:
- Нет дубликатов
- Защита от ошибок в коде

#### ⚠️ Минус:
- При вставке дубля — `IntegrityError`
- Нужно обрабатывать:
```python
try:
    cursor.execute("INSERT INTO songs VALUES (?, ?)", (artist, title))
except sqlite3.IntegrityError:
    print("Песня уже есть")
```

---

## Вопрос 11 ⭐⭐⭐⭐⭐  
### Проблемы качества данных

### 🧠 Что может быть не так с текстами песен?

| Проблема | Пример |
|---------|--------|
| **Мусор в тексте** | спецсимволы, HTML-теги |
| **Плохие песни** | "Instrumental", "No lyrics" |
| **Перекос** | 1000 песен от Taylor Swift, 1 от новичка |

#### ✅ Решения:

```python
import unicodedata
import re

def normalize_lyrics(text):
    # Убираем лишние символы
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^\w\s]', '', text)  # только буквы и пробелы
    return text.lower().strip()

def is_valid_lyrics(lyrics):
    words = lyrics.split()
    # Достаточно слов?
    if len(words) < 50:
        return False
    # Есть ли повторы?
    diversity = len(set(words)) / len(words)
    if diversity < 0.3:
        return False  # слишком много повторов
    # Проверяем маркеры
    bad_markers = ['instrumental', 'no lyrics']
    if any(marker in lyrics.lower() for marker in bad_markers):
        return False
    return True
```

---

# 🔥 БЛОК 4: Проблемы и Отладка

## Вопрос 17 ⭐⭐⭐  
### "Database locked" — что делать?

### 🧠 Причина:
- Два процесса хотят писать в БД одновременно.
- SQLite не любит параллельные записи.

#### ✅ Решение: включить **WAL mode**

```python
conn = sqlite3.connect('lyrics.db')
conn.execute("PRAGMA journal_mode=WAL")  # разрешает чтение во время записи
conn.execute("PRAGMA synchronous=NORMAL")
conn.commit()
```

> 📌 WAL = Write-Ahead Logging — как "черновик", который потом применяется.

---

## Вопрос 18 ⭐⭐⭐⭐  
### `MemoryError` после 3 дней работы

### 🧠 Причины:
- Утечка памяти в библиотеке
- Кэш растёт и не чистится
- Большой файл WAL

#### ✅ Решение: перезапуск сессии

```python
import gc
import psutil

class MemoryMonitor:
    def check_memory(self):
        current = psutil.Process().memory_info().rss / 1024 / 1024
        if current > 1000:  # больше 1 ГБ
            gc.collect()  # принудительная очистка
            self.restart_session()  # пересоздаём клиент
```

---

## Вопрос 19 ⭐⭐⭐⭐⭐  
### Библиотека обновилась — код сломался

### ✅ Решение: **Adapter Pattern**

```python
class SongAdapter:
    def __init__(self, raw_song):
        self.raw_song = raw_song
    
    @property
    def title(self):
        for attr in ['title', 'name', 'track_name']:
            if hasattr(self.raw_song, attr):
                return getattr(self.raw_song, attr)
        return "Unknown"
    
    @property
    def lyrics(self):
        for attr in ['lyrics', 'text', 'content']:
            val = getattr(self.raw_song, attr, None)
            if val and len(val) > 10:
                return val
        return ""
```

> 📌 Теперь, даже если `lyricsgenius` изменит структуру — твой код **не сломается**.

---

# 🧰 Библиотеки, которые ты должен знать

| Библиотека | Для чего |
|-----------|---------|
| `sqlite3` | Работа с базой данных |
| `requests` | Запросы к API |
| `random` | Случайные числа |
| `time` | Задержки |
| `signal` | Обработка сигналов (Ctrl+C) |
| `os`, `shutil` | Работа с файлами |
| `hashlib` | Хэши (для версионирования) |
| `json` | Работа с JSON |
| `re` | Регулярные выражения |
| `psutil` | Мониторинг памяти, CPU |

---

# 🎯 Что делать дальше?

1. ✅ Установи Python и VS Code
2. ✅ Создай проект: `scraping_lyrics/`
3. ✅ Напиши простой цикл с `for _ in range(5)`
4. ✅ Добавь `print("Hello", _)`
5. ✅ Запусти — убедись, что работает
6. ✅ Попробуй `getattr`, `hasattr`, `try-except`
7. ✅ Сделай `requirements.txt`:
   ```
   requests
   lyricsgenius
   psutil
   ```

---

# 🧑‍🏫 Совет от ментора

> "Ты не должен знать всё.  
> Ты должен знать:
> - **Где искать**
> - **Как читать ошибки**
> - **Как задавать вопросы**  
> Остальное — практика."

---

