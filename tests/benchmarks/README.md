# 🧪 Pytest-Benchmark Integration

## 🎯 **Назначение**

Автоматизированные performance тесты для анализаторов с интеграцией в CI/CD pipeline.

## 📁 **Структура**

```
tests/benchmarks/
├── __init__.py                 # Описание benchmark системы
├── conftest.py                 # Benchmark-специфичная конфигурация
└── test_quick_benchmarks.py    # Быстрые тесты для CI/CD
```

## 🚀 **Использование**

### Локальное тестирование
```bash
# Запуск всех benchmark тестов
pytest tests/benchmarks/ --benchmark-only

# Только быстрые тесты
pytest tests/benchmarks/ -m quick --benchmark-only

# Сохранение baseline для сравнения
pytest tests/benchmarks/ --benchmark-save=baseline --benchmark-only

# Сравнение с baseline
pytest tests/benchmarks/ --benchmark-compare=.benchmarks/ --benchmark-only

# С красивыми таблицами
pytest tests/benchmarks/ --benchmark-only --benchmark-columns=min,max,mean,stddev
```

### GitHub Actions
- **Автоматический запуск** на каждый push/PR
- **Ежедневный мониторинг** performance drift
- **Комментарии в PR** с результатами
- **Performance regression** detection

## 📊 **Конфигурация**

### pytest.ini
```ini
# Benchmark настройки для CI
benchmark-min-rounds = 3
benchmark-max-time = 10.0
benchmark-min-time = 0.05
```

### GitHub Workflow
- `.github/workflows/benchmarks.yml` - автоматизация CI/CD
- Regression check - fail если анализатор > 5 секунд
- GitHub Pages - история benchmark графиков

## 🔥 **Отличия от Performance Monitor**

| Критерий | pytest-benchmark | performance_monitor.py |
|----------|------------------|------------------------|
| **Цель** | Автоматизация CI/CD | Интерактивная диагностика |
| **Скорость** | Быстрые тесты (3-10s) | Глубокий анализ (30s+) |
| **Фокус** | Regression detection | Rich UI, профилирование |
| **Использование** | GitHub Actions | Ручная диагностика |
| **Метрики** | Базовые (время, память) | Расширенные (CPU, hotspots) |

## 🎯 **Best Practices**

### ✅ **Используй pytest-benchmark для:**
- CI/CD автоматизации
- Regression testing
- Исторических сравнений
- Быстрых проверок

### ✅ **Используй performance_monitor.py для:**
- Глубокой диагностики
- Профилирования hot spots  
- Rich UI и красивых таблиц
- Нагрузочного тестирования

## 🚨 **Troubleshooting**

### Тесты не запускаются
```bash
# Проверка среды
pytest tests/benchmarks/test_quick_benchmarks.py::test_benchmark_environment -v

# Проверка сбора тестов
pytest tests/benchmarks/ --collect-only
```

### Import ошибки
```bash
# Проверка Python path
pytest tests/benchmarks/ -v --tb=short
```

### Performance regression
```bash
# Сравнение с baseline
pytest tests/benchmarks/ --benchmark-compare=.benchmarks/ --benchmark-compare-fail=mean:20%
```

## 📈 **Метрики и мониторинг**

### Основные метрики
- **Mean time** - среднее время выполнения
- **Std deviation** - стабильность
- **Memory growth** - рост памяти
- **CPU efficiency** - items per CPU%

### Алерты
- **> 5 секунд** на анализ - CRITICAL
- **> 50MB** роста памяти - WARNING  
- **> 20%** degradation vs baseline - FAIL

## 🔧 **Настройка для проекта**

1. **Установка зависимостей:**
   ```bash
   pip install pytest-benchmark
   ```

2. **Конфигурация в requirements.txt:**
   ```
   pytest-benchmark>=4.0.0
   ```

3. **GitHub Actions Setup:**
   - Secrets: `GITHUB_TOKEN` (автоматически доступен)
   - Pages: включить для benchmark графиков

---

**Автор:** RapAnalyst AI Assistant 🎤  
**Дата:** 19.09.2025