# 🚀 Enhanced Performance Monitor

Продвинутая система мониторинга и тестирования производительности анализаторов с Rich UI, профилированием и pytest-benchmark интеграцией.

## 🎯 Возможности

### Основные режимы
- **benchmark** - комплексное измерение производительности с Rich UI
- **compare** - сравнение нескольких анализаторов в красивых таблицах
- **load** - нагрузочное тестирование с concurrent users
- **pyspy** - профилирование с py-spy (если установлен)
- **pytest** - генерация pytest-benchmark тестов

### Продвинутые фичи (опционально)
- **Rich Console** - красивые таблицы и progress bars ✅
- **pytest-benchmark** - автоматические CI/CD тесты ✅  
- **py-spy** - профилирование без остановки (if installed)
- **hyperfine** - CLI бенчмаркинг (if installed)
- **memory_profiler** - анализ использования памяти ✅
- **Prometheus metrics** - сбор метрик (опционально)

## 🛠️ Установка

### Базовые зависимости (обязательные)
```bash
# Основные зависимости уже установлены
pip install -r requirements.txt

# Дополнительные инструменты для performance monitoring
pip install rich tabulate pytest-benchmark memory-profiler
```

### Опциональные внешние инструменты
```bash
# py-spy (профилирование)
pip install py-spy

# hyperfine (CLI бенчмаркинг) 
# Ubuntu/Debian: sudo apt install hyperfine
# macOS: brew install hyperfine
# Windows: scoop install hyperfine
```

## 📊 Быстрый старт

### 1. Базовый бенчмарк
```bash
# Тест одного анализатора (Rich UI)
python src/cli/performance_monitor.py --analyzer advanced_algorithmic --mode benchmark

# Больше текстов для точности
python src/cli/performance_monitor.py --analyzer qwen --texts 50 --timeout 30
```

### 2. Сравнение анализаторов  
```bash
# Сравнение всех доступных анализаторов
python src/cli/performance_monitor.py --all --mode compare

# Сохранение результатов
python src/cli/performance_monitor.py --all --mode compare --output results/comparison.json
```

### 3. Нагрузочное тестирование
```bash
# 10 пользователей на 60 секунд
python src/cli/performance_monitor.py --analyzer advanced_algorithmic --mode load --users 10
```

### 4. Pytest автоматические тесты
```bash
# Быстрые CI/CD тесты
pytest tests/benchmarks/test_quick_benchmarks.py --benchmark-only

# Сравнение с baseline
pytest tests/benchmarks/ --benchmark-compare=.benchmarks/
```

## 🎮 CLI Команды

### Основные режимы
```bash
# Benchmark одного анализатора с Rich UI
python src/cli/performance_monitor.py --analyzer qwen --mode benchmark --texts 20

# Сравнение всех анализаторов
python src/cli/performance_monitor.py --all --mode compare

# Нагрузочное тестирование
python src/cli/performance_monitor.py --analyzer emotion_analyzer --mode load --users 5

# Профилирование (если py-spy установлен)
python src/cli/performance_monitor.py --analyzer qwen --mode pyspy --duration 30

# С Prometheus метриками (опционально)
python src/cli/performance_monitor.py --analyzer qwen --prometheus --mode benchmark
```

### Доступные анализаторы
- `advanced_algorithmic` - быстрый алгоритмический анализ
- `qwen` - AI анализ через Novita API (требует NOVITA_API_KEY)
- `emotion_analyzer` - анализ эмоций через Hugging Face

### Параметры CLI
- `--analyzer` - тип анализатора 
- `--all` - тестировать все анализаторы
- `--mode` - режим: benchmark, compare, load, pyspy, pytest
- `--texts` - количество тестовых текстов (default: 20)
- `--timeout` - таймаут на текст в секундах (default: 30.0)
- `--users` - concurrent users для load test (default: 10)
- `--duration` - длительность для py-spy (default: 30)
- `--prometheus` - включить Prometheus метрики
- `--output` - файл для сохранения результатов

## 📈 Мониторинг

### Prometheus метрики
```bash
# Запуск с Prometheus
make prometheus

# Метрики доступны на http://localhost:8000/metrics
```

Доступные метрики:
- `analyzer_requests_total` - счетчик запросов
- `analyzer_request_duration_seconds` - время выполнения
- `analyzer_memory_usage_mb` - использование памяти
- `analyzer_cpu_usage_percent` - использование CPU

### Grafana дашборды
```bash
# Открываем Grafana
open http://localhost:3000
# admin/admin123

# Импортируем дашборд
# monitoring/grafana/dashboards/analyzer-performance.json
```

### Алерты
Настроенные алерты:
- Высокое время отклика (>5s)
- Высокий уровень ошибок (>10%)
- Высокое потребление памяти (>1GB)
- Низкая пропускная способность (<1 RPS)

## 🔍 Типы профилирования

### 1. cProfile (встроенный)
```python
# Автоматически включается в benchmark mode
python enhanced_monitor.py --analyzer qwen --mode profile
```

### 2. py-spy (sampling profiler)
```bash
# Профилирование живого процесса
make pyspy ANALYZER=qwen DURATION=30

# Результат: pyspy_profile_qwen.svg
```

### 3. Memory profiling
```bash
# Анализ использования памяти
make memory-profile ANALYZER=qwen

# Результат: memory_qwen.png
```

### 4. Line profiling
```python
# Добавить @profile к функции
@profile
def analyze_text(text):
    # ваш код
    pass

# Запуск
kernprof -l -v your_script.py
```

## 🧪 Pytest интеграция

### Структура тестов
```
tests/
├── test_benchmarks.py          # Benchmark тесты
├── conftest.py                 # Конфигурация pytest
└── benchmarks/                 # Результаты бенчмарков
    ├── baseline/               # Baseline для сравнения
    └── results/               # Текущие результаты
```

### Запуск тестов
```bash
# Все бенчмарки
pytest tests/test_benchmarks.py --benchmark-only

# С сохранением baseline
pytest tests/test_benchmarks.py --benchmark-save=baseline

# Сравнение с baseline
pytest tests/test_benchmarks.py --benchmark-compare=baseline

# Гистограммы
pytest tests/test_benchmarks.py --benchmark-histogram
```

### Кастомные метрики
```python
def test_with_custom_metrics(benchmark_with_custom_metrics):
    result = benchmark_with_custom_metrics(my_function, arg1, arg2)
    # Автоматически добавляются memory_growth_mb, cpu_time_user, etc.
```

## 📊 Анализ результатов

### Структура результатов
```
performance_results/
├── benchmarks/                 # Основные бенчмарки
│   ├── algorithmic_benchmark.json
│   └── pytest_results.json
├── profiles/                   # Профили
│   ├── algorithmic_profile.json
│   ├── pyspy_profile_qwen.svg
│   └── memory_qwen.png
├── reports/                    # Отчеты
│   ├── comparison_report.json
│   ├── comprehensive_report.json
│   └── report.html
└── prometheus/                 # Prometheus данные
```

### Ключевые метрики

#### Временные метрики
- `avg_time` - среднее время выполнения
- `min_time/max_time` - минимальное/максимальное время
- `median_time` - медианное время
- `latency_p95/p99` - 95й/99й процентили

#### Системные метрики
- `avg_cpu_percent` - среднее использование CPU
- `avg_memory_mb` - среднее использование памяти
- `peak_memory_mb` - пиковое использование памяти
- `memory_growth_mb` - рост памяти за тест

#### Метрики качества
- `success_rate` - процент успешных запросов
- `error_count` - количество ошибок
- `items_per_second` - пропускная способность
- `cpu_efficiency` - эффективность (items/cpu%)

#### Профилирование
- `hottest_function` - самая нагруженная функция
- `profile_data` - детальные данные профилирования

### Интерпретация результатов

#### 🟢 Хорошие показатели
- `success_rate` > 95%
- `avg_time` < 1s для большинства задач
- `memory_growth_mb` < 50MB
- `cpu_efficiency` > 10

#### 🟡 Требует внимания
- `latency_p95` значительно выше `avg_time`
- `error_count` > 0
- Растущее потребление памяти
- Низкая `items_per_second`

#### 🔴 Проблемы
- `success_rate` < 90%
- `avg_time` > 5s
- `memory_growth_mb` > 100MB
- Высокое CPU использование без пропорционального увеличения производительности

## 🔧 Расширенная конфигурация

### Prometheus конфигурация
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  
scrape_configs:
  - job_name: 'analyzer-performance'
    static_configs:
      - targets: ['host.docker.internal:8000']
    scrape_interval: 5s
```

### Grafana дашборды
- **Analyzer Performance** - основные метрики
- **System Resources** - системные ресурсы
- **Error Analysis** - анализ ошибок
- **Comparative Analysis** - сравнение анализаторов

### AlertManager правила
```yaml
# Критичные алерты
- alert: HighErrorRate
  expr: rate(analyzer_requests_total{status="error"}[5m]) > 0.1
  
# Предупреждения
- alert: HighMemoryUsage
  expr: analyzer_memory_usage_mb > 1000
```

## 🚀 Продвинутое использование

### Непрерывное профилирование
```bash
# Запуск профилирования в фоне
make pyspy ANALYZER=qwen DURATION=3600 &  # 1 час

# Мониторинг в реальном времени
make watch
```

### CI/CD интеграция
```yaml
# .github/workflows/performance.yml
- name: Performance Testing
  run: |
    make ci-test
    make pytest-bench
    
- name: Performance Regression Check
  run: |
    pytest tests/test_benchmarks.py --benchmark-compare-fail=min:10%
```

### Кастомные анализаторы
```python
# Добавление нового анализатора
class MyAnalyzer:
    async def analyze(self, text: str):
        # ваша логика
        return result

# Регистрация в Application
app.register_analyzer("my_analyzer", MyAnalyzer())
```

## 🐛 Отладка и устранение проблем

### Частые проблемы

#### py-spy не работает
```bash
# Установка через pip
pip install py-spy

# Установка через cargo
cargo install py-spy

# Права доступа
sudo py-spy record -p PID
```

#### hyperfine не найден
```bash
# Ubuntu/Debian
sudo apt install hyperfine

# macOS
brew install hyperfine

# Cargo
cargo install hyperfine
```

#### Prometheus метрики не собираются
```bash
# Проверка порта
curl http://localhost:8000/metrics

# Проверка конфигурации
docker-compose logs prometheus
```

### Логирование
```bash
# Включение debug логов
export LOG_LEVEL=DEBUG
python enhanced_monitor.py --analyzer qwen --mode benchmark

# Анализ логов
tail -f performance.log | grep ERROR
```

### Производительность мониторинга
```python
# Настройка интервала мониторинга
monitor = EnhancedPerformanceMonitor(monitoring_interval=0.05)  # 50ms

# Отключение части метрик
monitor = EnhancedPerformanceMonitor(enable_prometheus=False)
```

## 📚 Дополнительные ресурсы

### Документация инструментов
- [py-spy](https://github.com/benfred/py-spy) - Sampling profiler
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) - Pytest plugin
- [hyperfine](https://github.com/sharkdp/hyperfine) - Command-line benchmarking
- [memory_profiler](https://pypi.org/project/memory-profiler/) - Memory usage profiler

### Мониторинг стек
- [Prometheus](https://prometheus.io/docs/) - Monitoring system
- [Grafana](https://grafana.com/docs/) - Visualization platform
- [AlertManager](https://prometheus.io/docs/alerting/latest/alertmanager/) - Alert handling

### Лучшие практики
1. **Baseline тестирование** - всегда сохраняйте baseline для сравнения
2. **Warmup runs** - используйте прогревочные запуски для JIT
3. **Статистическая значимость** - проводите достаточно итераций
4. **Изоляция тестов** - исключайте влияние внешних факторов
5. **Регулярность** - интегрируйте в CI/CD pipeline

## 🤝 Контрибьютинг

### Добавление новых метрик
```python
@dataclass
class EnhancedMetrics:
    # Добавьте новую метрику
    your_metric: float = 0.0
```

### Новые типы профилирования
```python
async def your_profiling_method(self, analyzer_type: str, test_texts: List[str]):
    # Ваша логика профилирования
    pass
```

### Кастомные визуализации
```json
// Добавьте панель в Grafana дашборд
{
  "id": 7,
  "title": "Your Custom Metric",
  "targets": [{"expr": "your_prometheus_metric"}]
}
```

## 🎉 Заключение

Этот инструмент предоставляет полный стек для профессионального мониторинга производительности анализаторов:

- **Разработка**: pytest-benchmark, cProfile, py-spy
- **Testing**: hyperfine, load testing, stress testing  
- **Production**: Prometheus, Grafana, AlertManager
- **Анализ**: comprehensive reporting, visualization

Начните с простых бенчмарков (`make demo`) и постепенно внедряйте продвинутые инструменты в зависимости от ваших потребностей!

---

**Авторы**: AI Assistant + Human  
**Лицензия**: MIT  
**Версия**: 1.0.0