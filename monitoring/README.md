# 📊 Monitoring System

Система мониторинга для rap-scraper-project с поддержкой PostgreSQL.

## 📁 Структура

```
monitoring/
├── README.md           # Этот файл  
├── logs/              # Директория для логов
├── metrics/           # Метрики производительности
└── scripts/           # Скрипты мониторинга
```

## 🗃️ Архивированные компоненты

Старые SQLite-based мониторинг скрипты перенесены в:
- `archive/legacy_monitoring/monitoring/`

Причина архивации: переход на PostgreSQL архитектуру.

## 🚀 Планируемые компоненты

### 1. PostgreSQL Monitoring
- Мониторинг подключений к БД
- Статистика запросов
- Производительность индексов

### 2. API Monitoring  
- Время отклика API endpoints
- Количество запросов/ошибок
- Rate limiting статистика

### 3. AI Analysis Monitoring
- Прогресс анализа треков
- Статистика использования моделей
- Очередь задач

### 4. System Resources
- CPU/Memory usage
- Disk space
- Network статистика

## 🛠️ Интеграции

- **Prometheus**: Сбор метрик
- **Grafana**: Визуализация дашбордов  
- **Docker Health Checks**: Контроль состояния контейнеров
- **Log Aggregation**: Централизованное логирование

## 📊 Метрики

### База данных
- Количество треков в БД
- Количество проанализированных треков
- Средний размер данных на трек

### API Performance
- Response time percentiles (p50, p95, p99)
- Error rate по endpoint'ам
- Throughput (requests/sec)

### AI Analysis
- Треков обработано в час
- Средний размер анализа
- Ошибки анализа

## 🔧 Конфигурация

Мониторинг настраивается через:
- `config.yaml` - основные параметры
- Environment variables для чувствительных данных
- Docker Compose для оркестрации
