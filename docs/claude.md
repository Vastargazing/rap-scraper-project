# Rap Scraper Project — AI Agent Context (Обновлено: 2025-01-19)

> **Kubernetes-native enterprise ML-pipeline** для анализа рэп-текстов с **PostgreSQL + pgvector**,## 📊 ТЕКУЩИЙ СТАТУС ПРОЕКТА

### Актуальные метрики (2025-01-19)
- 🎵 **Треки**: 57,718 (PostgreSQL)
- 🤖 **Анализ Qwen**: 57,716 (100.0%) | **✅ ЗАВЕРШЕН**
- 🤖 **Анализ Gemma**: 34,320 (59.4%)  
- 🧮 **Алгоритмический анализ**: 57,716 (100.0%) | **✅ ЗАВЕРШЕН**
- 🎯 **Общий анализ**: 57,718/57,718 (100.0%)
- 📊 **Всего анализов**: 269,646
- 🐘 **База**: PostgreSQL 15 + connection pool (20 подключений)
- ☸️ **Kubernetes**: Production-ready инфраструктура с monitoring

### Состояние системы
- ✅ **Phase 1: Kubernetes Migration ЗАВЕРШЕНА** (2025-01-19)
- ✅ **Phase 2: GitOps Integration ЗАВЕРШЕНА** (2025-01-19)
- ✅ **PostgreSQL миграция завершена** (100% целостность данных)
- ✅ **Concurrent processing готов** (20 подключений в пуле)
- ✅ **Полный анализ завершен** (269,646 анализов, 100% coverage)
- ☸️ **Production Infrastructure**: Helm chart, monitoring, auto-scaling
- 🚀 **GitOps Workflow**: ArgoCD, automated deployments, self-healing
- 🎯 **Приоритет**: Phase 2 продолжение - multi-region, Jaeger, securitycontainer orchestration, и comprehensive monitoring stack

## 🎯 ПРИОРИТЕТЫ ДЛЯ AI АГЕНТА

### � **ПРАВИЛА РАБОТЫ С ХРОНОЛОГИЧЕСКИМИ ФАЙЛАМИ**
⚠️ **ВАЖНО:** При обновлении файлов с хронологией (PROGRESS.md, changelog, история):
- ✅ **НОВЫЕ записи ВСЕГДА добавляются В ВЕРХ файла** (после заголовка)
- ❌ **НЕ добавляй в конец файла** - это тратит токены на поиск места
- 🎯 **Структура:** Свежие записи сверху → старые снизу
- 📅 **Формат даты:** `## 📅 YYYY-MM-DD | ЗАГОЛОВОК ИЗМЕНЕНИЯ`

**Пример правильного добавления:**
```markdown
# 📋 Дневник изменений проекта

## 📅 2025-09-18 | НОВОЕ ОБНОВЛЕНИЕ 🚀
### Свежие изменения здесь...

## 📅 2025-09-16 | Предыдущее обновление
### Старые изменения...
```

### �🔥 Критически важные команды
```bash
# ОСНОВНАЯ ДИАГНОСТИКА (используй ПЕРВОЙ)
python scripts/tools/database_diagnostics.py --quick

# ТЕСТИРОВАНИЕ AI АНАЛИЗА
python scripts/mass_qwen_analysis.py --test

# ИНТЕРАКТИВНАЯ РАБОТА С БД
python scripts/db_browser.py

# ПРОВЕРКА CONCURRENT ДОСТУПА
python scripts/tools/database_diagnostics.py --connections
```

---

## 📊 ПОЛНАЯ СХЕМА БАЗЫ ДАННЫХ (PostgreSQL)

### 🎵 Таблица `tracks` (57,718 записей) - ОСНОВНАЯ ТАБЛИЦА
```sql
CREATE TABLE tracks (
    id                      SERIAL PRIMARY KEY,              -- Уникальный ID трека
    title                   VARCHAR NOT NULL,                -- Название трека
    artist                  VARCHAR NOT NULL,                -- Исполнитель
    lyrics                  TEXT,                            -- Текст песни (ОСНОВНОЕ ПОЛЕ)
    url                     TEXT,                            -- URL на Genius.com
    genius_id               INTEGER,                         -- ID в Genius API
    scraped_date            TIMESTAMP,                       -- Дата скрапинга
    word_count              INTEGER,                         -- Количество слов
    genre                   VARCHAR,                         -- Жанр
    release_date            DATE,                            -- Дата релиза
    album                   VARCHAR,                         -- Альбом
    language                VARCHAR,                         -- Язык
    explicit                BOOLEAN,                         -- Explicit контент
    song_art_url            TEXT,                           -- URL обложки
    popularity_score        INTEGER,                        -- Оценка популярности
    lyrics_quality_score    REAL,                           -- Качество текста (0-1)
    created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP, -- Дата добавления
    spotify_data            JSONB                           -- Метаданные Spotify (59% coverage)
);

-- ИНДЕКСЫ:
CREATE INDEX idx_tracks_artist ON tracks(artist);
CREATE INDEX idx_tracks_title ON tracks(title);
CREATE INDEX idx_tracks_lyrics_not_null ON tracks(id) WHERE lyrics IS NOT NULL;
CREATE INDEX idx_tracks_spotify_data ON tracks USING GIN(spotify_data);
```

### 📊 СТАТИСТИКА SPOTIFY ОБОГАЩЕНИЯ
- **Всего треков**: 57,718
- **С Spotify данными**: 34,066 (59.02%)
- **Без Spotify данных**: 23,652 (40.98%)
- **Средняя популярность**: 30.5 (диапазон: 1-94)
- **Топ исполнители**: Gucci Mane (476), Chief Keef (469), Snoop Dogg (469)

### 🤖 Таблица `analysis_results` (269,646 анализов) - ПОЛНЫЙ ОХВАТ
```sql
CREATE TABLE analysis_results (
    id                   SERIAL PRIMARY KEY,        -- Уникальный ID анализа
    track_id             INTEGER REFERENCES tracks(id), -- Связь с треком
    analyzer_type        VARCHAR(50),               -- Тип анализатора
    sentiment            VARCHAR,                   -- Эмоциональный тон
    confidence           NUMERIC,                   -- Уверенность анализа (0-1)
    themes               TEXT,                      -- JSON список тем
    analysis_data        JSONB,                     -- Полные данные анализа
    created_at           TIMESTAMP DEFAULT NOW(),   -- Время анализа
    complexity_score     NUMERIC,                   -- Оценка сложности (0-1)
    processing_time_ms   INTEGER,                   -- Время обработки в мс
    model_version        VARCHAR                    -- Версия модели
);

-- ИНДЕКСЫ:
CREATE INDEX idx_analysis_track_id ON analysis_results(track_id);
CREATE INDEX idx_analysis_analyzer_type ON analysis_results(analyzer_type);
CREATE INDEX idx_analysis_created_at ON analysis_results(created_at);
```

### 📈 СТАТИСТИКА ПО АНАЛИЗАТОРАМ (актуально на 2025-09-26)
```
🤖 ТИПЫ АНАЛИЗАТОРОВ И ИХ ПОКРЫТИЕ:
┌─────────────────────────┬───────────┬───────────┬─────────┐
│ Analyzer Type           │ Анализов  │ Треков    │ Доля    │
├─────────────────────────┼───────────┼───────────┼─────────┤
│ simplified_features     │ 115,434   │ 57,717    │ 42.8%   │
│ qwen-3-4b-fp8          │ 61,933    │ 57,716    │ 23.0%   │
│ simplified_features_v2  │ 57,717    │ 57,717    │ 21.4%   │
│ gemma-3-27b-it         │ 34,320    │ 34,320    │ 12.7%   │
│ emotion_analyzer_v2     │ 207       │ 207       │ 0.1%    │
│ mock_analyzer_v1        │ 27        │ 27        │ 0.0%    │
│ ollama:llama3.2:3b     │ 8         │ 8         │ 0.0%    │
└─────────────────────────┴───────────┴───────────┴─────────┘

📊 ОБЩАЯ СТАТИСТИКА:
• Всего треков: 57,718
• Треков с текстами: 57,718 (100%)  
• Проанализированных треков: 57,718 (100%)
• Всего анализов: 269,646
• Средний анализ на трек: 4.7
• Размер БД: ~420 MB
```

### 🔍 ВАЖНЫЕ SQL-ЗАПРОСЫ ДЛЯ AI АГЕНТА
```sql
-- Найти треки без Qwen анализа
SELECT t.id, t.artist, t.title 
FROM tracks t 
LEFT JOIN analysis_results ar ON t.id = ar.track_id 
  AND ar.analyzer_type = 'qwen-3-4b-fp8'
WHERE ar.id IS NULL AND t.lyrics IS NOT NULL
ORDER BY t.id
LIMIT 100;

-- Статистика анализаторов
SELECT 
    analyzer_type,
    COUNT(*) as total_analyses,
    COUNT(DISTINCT track_id) as unique_tracks,
    AVG(confidence) as avg_confidence,
    AVG(complexity_score) as avg_complexity
FROM analysis_results 
GROUP BY analyzer_type 
ORDER BY total_analyses DESC;

-- Треки конкретного исполнителя с анализом
SELECT t.id, t.title, t.artist, ar.analyzer_type, ar.confidence
FROM tracks t
LEFT JOIN analysis_results ar ON t.id = ar.track_id
WHERE t.artist ILIKE '%eminem%'
ORDER BY t.id;

-- Последние анализы
SELECT t.artist, t.title, ar.analyzer_type, ar.created_at, ar.confidence
FROM analysis_results ar
JOIN tracks t ON ar.track_id = t.id
ORDER BY ar.created_at DESC
LIMIT 20;
```

---

## 📊 ТЕКУЩИЙ СТАТУС ПРОЕКТА

### Актуальные метрики (2025-09-26)
- 🎵 **Треки**: 57,718 (PostgreSQL)
- 🤖 **Анализ Qwen**: 57,716 (100.0%) | **✅ ЗАВЕРШЕН**
- 🤖 **Анализ Gemma**: 34,320 (59.4%)  
- 🧮 **Алгоритмический анализ**: 57,716 (100.0%) | **✅ ЗАВЕРШЕН**
- 🎯 **Общий анализ**: 57,718/57,718 (100.0%)
- � **Всего анализов**: 269,646
- �🐘 **База**: PostgreSQL 15 + connection pool (20 подключений)
- ✅ **Concurrent**: Множественные скрипты работают одновременно

### Состояние системы
- ✅ **PostgreSQL миграция завершена** (100% целостность данных)
- ✅ **Concurrent processing готов** (20 подключений в пуле)
- ✅ **Qwen анализ завершен** (57,716 треков проанализировано)
- ✅ **Алгоритмический анализ завершен** (57,716 треков проанализировано)
- 🎯 **Приоритет**: Внедрение новых функций из NEW_FEATURE.md

---

## 🏗️ АРХИТЕКТУРА (PostgreSQL-центричная)

### Kubernetes-Native Архитектура
```
┌─────────────────── KUBERNETES CLUSTER ───────────────────┐
│                                                          │
│  ┌─── INGRESS CONTROLLER ───┐                           │
│  │  • Load Balancing         │                           │
│  │  • SSL Termination        │                           │
│  │  • Multi-host Routing     │                           │
│  └───────────┬───────────────┘                           │
│              │                                           │
│  ┌─────── FASTAPI SERVICE ────────┐                     │
│  │  • 3-10 Auto-scaling Replicas  │                     │
│  │  • HPA (CPU/Memory based)      │                     │  
│  │  • Health Probes              │                     │
│  │  • Resource Limits            │                     │
│  └───────────┬───────────────────┘                     │
│              │                                           │
│  ┌────── POSTGRESQL + pgvector ──────┐                  │
│  │  • StatefulSet Deployment         │                  │
│  │  • Persistent Volume Claims       │                  │
│  │  • Vector Similarity Search       │                  │
│  │  • Connection Pooling             │                  │
│  └───────────┬────────────────────────┘                  │
│              │                                           │
│  ┌─────── MONITORING STACK ─────────┐                   │
│  │  ┌─── Prometheus ───┐             │                   │
│  │  │  • Metrics Collection │         │                   │
│  │  │  • Custom Alerts      │         │                   │
│  │  └────────────────────────┘         │                   │
│  │  ┌─── Grafana ──────┐             │                   │
│  │  │  • Custom Dashboards  │         │                   │
│  │  │  • Visualization      │         │                   │
│  │  └────────────────────────┘         │                   │
│  └─────────────────────────────────────┘                   │
└──────────────────────────────────────────────────────────┘

### Legacy Development архитектура
```
📦 Основные файлы для AI агента:
├── src/database/postgres_adapter.py     # PostgreSQL подключение (ОСНОВА)
├── scripts/mass_qwen_analysis.py        # Массовый анализ (ГЛАВНЫЙ)
├── scripts/tools/database_diagnostics.py # Диагностика (ПЕРВАЯ ПОМОЩЬ)
├── config.yaml                          # Конфигурация
├── .env                                 # PostgreSQL credentials
└── scripts/db_browser.py               # Интерактивный браузер БД
```

### Database Layer (PostgreSQL)
- **Adapter**: `src/database/postgres_adapter.py` - управление подключениями
- **Pool**: 20 max подключений для concurrent скриптов
- **Drivers**: `asyncpg` (async) + `psycopg2` (sync)
- **Migration**: Полная миграция из SQLite завершена

### AI Analysis Pipeline
- **Qwen API**: `scripts/mass_qwen_analysis.py` - основной анализатор
- **Local Models**: Gemma, Ollama для локального анализа
- **Progress Tracking**: Автоматическое сохранение прогресса
- **Error Recovery**: Robust обработка ошибок API

### 📋 ВСЕ ТАБЛИЦЫ В POSTGRESQL
- **`tracks`** - ОСНОВНАЯ ТАБЛИЦА (57,718 записей)
- **`analysis_results`** - результаты AI анализа (256,021 записей)  
- **`songs`** - LEGACY ТАБЛИЦА (рекомендуется удаление)

### ⚠️ ТАБЛИЦА `songs` - УДАЛИТЬ?
**СТАТУС**: Legacy таблица, дублирует `tracks`
**РЕКОМЕНДАЦИЯ**: ✅ **УДАЛИТЬ** после проверки зависимостей
**ПРИЧИНА**: Все данные мигрированы в `tracks`, избыточность создает путаницу

---

## 🤖 AI АГЕНТ WORKFLOW

### 1. Исследование проблем (ОБНОВЛЕННЫЙ ПРОТОКОЛ)
```python
def investigate_issue(problem_description):
    # ШАГ 1: БАЗА ДАННЫХ (ВСЕГДА ПЕРВЫЙ)
    run_command("python scripts/tools/database_diagnostics.py --quick")
    
    # ШАГ 2: СПЕЦИФИЧЕСКАЯ ДИАГНОСТИКА
    if "analysis" in problem_description.lower():
        run_command("python scripts/mass_qwen_analysis.py --test")
    elif "connection" in problem_description.lower():
        run_command("python scripts/tools/database_diagnostics.py --connections")
    elif "concurrent" in problem_description.lower():
        run_command("python scripts/db_browser.py") # тест интерактивного доступа
    
    # ШАГ 3: КОНФИГУРАЦИЯ
    check_file(".env")  # PostgreSQL credentials
    check_file("config.yaml")  # система конфигурации
    
    # ШАГ 4: КОД АНАЛИЗ (если нужен)
    if requires_code_investigation():
        check_file("src/database/postgres_adapter.py")  # database layer
        check_file("scripts/mass_qwen_analysis.py")     # main script
    
    return solution_with_validation_steps()
```

### 2. Типичные запросы к БД (готовые SQL)
```sql
-- 🔍 ПОИСК НЕАНАЛИЗИРОВАННЫХ ТРЕКОВ QWEN
SELECT COUNT(*) FROM tracks t
LEFT JOIN analysis_results ar ON t.id = ar.track_id 
  AND ar.analyzer_type = 'qwen-3-4b-fp8'
WHERE ar.id IS NULL AND t.lyrics IS NOT NULL;

-- 📊 СТАТИСТИКА ПО ИСПОЛНИТЕЛЯМ
SELECT artist, COUNT(*) as tracks_count
FROM tracks 
WHERE lyrics IS NOT NULL
GROUP BY artist 
ORDER BY tracks_count DESC 
LIMIT 20;

-- 🎯 ПРОГРЕСС QWEN АНАЛИЗА
SELECT 
    (SELECT COUNT(*) FROM tracks WHERE lyrics IS NOT NULL) as total_tracks,
    (SELECT COUNT(DISTINCT track_id) FROM analysis_results 
     WHERE analyzer_type = 'qwen-3-4b-fp8') as analyzed_tracks,
    ROUND(100.0 * (SELECT COUNT(DISTINCT track_id) FROM analysis_results 
     WHERE analyzer_type = 'qwen-3-4b-fp8') / 
     (SELECT COUNT(*) FROM tracks WHERE lyrics IS NOT NULL), 2) as percentage;

-- ⚡ ПРОИЗВОДИТЕЛЬНОСТЬ АНАЛИЗА
SELECT 
    analyzer_type,
    AVG(processing_time_ms) as avg_time_ms,
    MIN(processing_time_ms) as min_time_ms,
    MAX(processing_time_ms) as max_time_ms,
    COUNT(*) as total_analyses
FROM analysis_results 
WHERE processing_time_ms IS NOT NULL
GROUP BY analyzer_type;
```

### 3. Индикаторы PostgreSQL-совместимости
```python
# ✅ ХОРОШИЕ индикаторы (PostgreSQL-ready):
from src.database.postgres_adapter import PostgreSQLManager
import asyncpg, psycopg2
async with db_manager.get_connection() as conn:

# ❌ ПЛОХИЕ индикаторы (SQLite legacy):
import sqlite3
conn = sqlite3.connect("data/rap_lyrics.db")
```

### 4. Шаблон ответа AI агента
```markdown
## 🔍 ДИАГНОСТИКА

**Статус PostgreSQL**: [подключение, пул, производительность]
**Статус скриптов**: [совместимость, concurrent доступ]
**Статус данных**: [целостность, статистика]

## 📋 НАХОДКИ

**Проблема**: [краткое описание]
**Причина**: [root cause с кодом/конфигурацией]
**Воздействие**: [на concurrent processing, данные, производительность]

## 🚀 ПЛАН РЕШЕНИЯ

1. **Немедленные действия**: [что делать прямо сейчас]
2. **Код/Конфигурация**: [изменения в файлах]
3. **Тестирование**: [команды для валидации]
4. **Мониторинг**: [как отслеживать результат]

## ✅ ВАЛИДАЦИЯ

```bash
# Проверка решения
python scripts/tools/database_diagnostics.py --quick
python scripts/mass_qwen_analysis.py --test
```
```

---

## 🔧 КОМАНДЫ ДЛЯ AI АГЕНТА

### Уровень 1: Диагностика (ПЕРВОЕ, что нужно запускать)
```bash
# ОСНОВНАЯ диагностика PostgreSQL
python scripts/tools/database_diagnostics.py --quick

# Детальная диагностика (если нужно)
python scripts/tools/database_diagnostics.py

# Статус AI анализа
python scripts/tools/database_diagnostics.py --analysis

# Поиск неанализированных треков
python scripts/tools/database_diagnostics.py --unanalyzed
```

### Уровень 2: Тестирование
```bash
# AI анализ тест (быстрый)
python scripts/mass_qwen_analysis.py --test

# PostgreSQL подключение
python -c "
from src.database.postgres_adapter import PostgreSQLManager
import asyncio
async def test():
    db = PostgreSQLManager()
    await db.initialize()
    print('✅ PostgreSQL OK')
    await db.close()
asyncio.run(test())
"

# Интерактивный доступ к БД
python scripts/db_browser.py
```

### Уровень 3: Выполнение задач
```bash
# Массовый анализ (production)
python scripts/mass_qwen_analysis.py

# С параметрами
python scripts/mass_qwen_analysis.py --batch 50 --max 1000

# Скрапинг (если нужен)
python scripts/rap_scraper_cli.py scraping --debug
```

### Уровень 4: Администрирование
```bash
# Прямое подключение к PostgreSQL
psql -h localhost -U rap_user -d rap_lyrics -p 5433

# Docker контейнер (если используется)
docker exec rap-analyzer-postgres-vector psql -U rap_user -d rap_lyrics

# Миграция (если нужна)
python scripts/migrate_to_postgresql.py
```

---

## 🎯 КОНКРЕТНЫЕ СЦЕНАРИИ ДЛЯ AI АГЕНТА

### Сценарий 1: "Не работает анализ"
```bash
# Диагностика по шагам
python scripts/tools/database_diagnostics.py --quick
python scripts/mass_qwen_analysis.py --test
cat .env | grep NOVITA  # проверить API ключ
```

### Сценарий 2: "Проблемы с базой данных"
```bash
# PostgreSQL health check
python scripts/tools/database_diagnostics.py --quick
python scripts/db_browser.py  # интерактивная проверка
python -c "from src.utils.config import get_db_config; print(get_db_config())"
```

### Сценарий 3: "Concurrent access не работает"
```bash
# Terminal 1
python scripts/mass_qwen_analysis.py --batch 10 &

# Terminal 2 (одновременно)
python scripts/db_browser.py

# Проверка подключений
python scripts/tools/database_diagnostics.py --connections
```

### Сценарий 4: "Нужна статистика проекта"
```bash
# Полная статистика
python scripts/tools/database_diagnostics.py

# Только цифры
python scripts/tools/database_diagnostics.py --quick

# Overlap анализ (если есть файл)
python check_overlap.py
```

---

## 🚨 TROUBLESHOOTING GUIDE

### Проблема: PostgreSQL не подключается
```bash
# Диагностика
cat .env | grep POSTGRES
python -c "import psycopg2; print('✅ psycopg2 OK')"
docker ps | grep postgres  # если используется Docker

# Решение
# 1. Проверить credentials в .env
# 2. Запустить PostgreSQL сервис
# 3. Проверить порт и firewall
```

### Проблема: AI анализ падает с ошибками
```bash
# Диагностика
python scripts/mass_qwen_analysis.py --test --batch 1
cat .env | grep NOVITA_API_KEY
python -c "import requests; print('✅ requests OK')"

# Решение
# 1. Проверить API ключ
# 2. Проверить rate limits
# 3. Проверить интернет подключение
```

### Проблема: Concurrent access не работает
```bash
# Диагностика
python scripts/tools/database_diagnostics.py --connections
python -c "
from src.database.postgres_adapter import PostgreSQLManager
print('PostgreSQL adapter:', PostgreSQLManager.__file__)
"

# Решение
# 1. Увеличить pool size в конфигурации
# 2. Проверить использование PostgreSQLManager в скриптах
# 3. Проверить async/sync совместимость
```

---

## 📁 СТРУКТУРА ФАЙЛОВ (приоритеты для AI агента)

### 🔥 Критически важные файлы
1. `src/database/postgres_adapter.py` - PostgreSQL connection management
2. `scripts/mass_qwen_analysis.py` - основной анализ скрипт  
3. `scripts/tools/database_diagnostics.py` - главный diagnostic tool
4. `.env` - PostgreSQL credentials и API keys
5. `config.yaml` - система конфигурации

### 📊 Диагностические файлы
6. `scripts/db_browser.py` - интерактивный браузер БД
7. `check_stats.py` - статистика (если существует)
8. `check_overlap.py` - overlap анализ (если существует)

### 🔧 Административные файлы
9. `scripts/migrate_to_postgresql.py` - migration tools
10. `scripts/rap_scraper_cli.py` - scraping interface

### 📦 Legacy файлы (для справки)
11. `scripts/archive/` - SQLite legacy scripts
12. `data/rap_lyrics.db` - SQLite backup (если существует)

### 📅 **Хронологические файлы (НОВЫЕ записи СВЕРХУ!)**
⚠️ **СПЕЦИАЛЬНЫЕ ПРАВИЛА:** Эти файлы требуют добавления записей В ВЕРХ файла:
- `docs/PROGRESS.md` - дневник проекта (✅ новые записи сверху)
- `CHANGELOG.md` - если существует
- Любые файлы с историей изменений или логами

**НЕ ТРАТЬТЕ ТОКЕНЫ** на поиск конца файла - просто добавляйте после заголовка!

---

## 📋 КОНФИГУРАЦИОННЫЕ ФАЙЛЫ

### `.env` (PostgreSQL credentials)
```env
# PostgreSQL Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_USERNAME=rap_user
POSTGRES_PASSWORD=securepassword123
POSTGRES_DATABASE=rap_lyrics

# API Keys
NOVITA_API_KEY=your-novita-api-key-here
GENIUS_ACCESS_TOKEN=your-genius-token
SPOTIFY_CLIENT_ID=your-spotify-client-id
SPOTIFY_CLIENT_SECRET=your-spotify-client-secret
```

### `config.yaml` (система конфигурации)
```yaml
app:
  name: "rap-lyrics-analyzer"
  version: "2.0.0"

database:
  type: "postgresql"
  pool_size: 20
  min_connections: 5
  max_connections: 20

kubernetes:
  enabled: true
  namespace: "rap-analyzer"
  deployment:
    replicas: 3
    autoscaling:
      enabled: true
      min_replicas: 3
      max_replicas: 10
  monitoring:
    prometheus: true
    grafana: true
  timeout: 30

analyzers:
  qwen:
    enabled: true
    model: "qwen/qwen3-4b-fp8"
    max_retries: 3
    timeout: 30
  
  gemma:
    enabled: true
    model: "gemma-3-27b-it"
    local: true

performance:
  batch_size: 50
  max_workers: 4
  concurrent_requests: 3
```

---

## ✅ CHECKLIST ДЛЯ AI АГЕНТА

### Перед началом работы
- [ ] Запустить `python scripts/tools/database_diagnostics.py --quick`
- [ ] Проверить наличие `.env` с PostgreSQL credentials
- [ ] Убедиться в наличии `config.yaml`
- [ ] Проверить `requirements.txt` установлены

### При диагностике проблем
- [ ] Всегда начинать с database diagnostics
- [ ] Проверять PostgreSQL vs SQLite совместимость в коде
- [ ] Тестировать concurrent доступ при необходимости
- [ ] Проверять API ключи для внешних сервисов

### После изменений
- [ ] Запустить тесты: `python scripts/mass_qwen_analysis.py --test`
- [ ] Проверить connection pool: `database_diagnostics.py --connections`
- [ ] Протестировать concurrent access если применимо
- [ ] Обновить документацию если нужно

---

## 🎯 SUCCESS METRICS

### Database Health
- ✅ PostgreSQL подключение < 100ms
- ✅ Query response < 500ms  
- ✅ Connection pool 15+ доступных подключений
- ✅ Data integrity 100%

### Analysis Performance  
- ✅ Qwen API success rate > 90%
- ✅ Processing rate ~2-5 tracks/min
- ✅ Error recovery работает
- ✅ Progress tracking функционирует

### Concurrent Processing
- ✅ Множественные скрипты работают одновременно
- ✅ No database locks
- ✅ Transaction isolation работает
- ✅ Connection pool не исчерпывается

---

## 💡 AI AGENT OPTIMIZATION NOTES

### НЕ нужно запрашивать каждый раз:
- ❌ Схему таблиц (она в этом документе)
- ❌ Статистику БД (актуальная выше)
- ❌ Список колонок (см. CREATE TABLE выше)
- ❌ Типы анализаторов (см. таблицу покрытия)

### НУЖНО использовать готовые:
- ✅ SQL-запросы из раздела "ВАЖНЫЕ SQL-ЗАПРОСЫ"
- ✅ Commands из раздела "КОМАНДЫ ДЛЯ AI АГЕНТА"
- ✅ Troubleshooting scenarios
- ✅ Database diagnostics как первый шаг
- ✅ **ХРОНОЛОГИЧЕСКИЕ ФАЙЛЫ:** Добавлять записи В ВЕРХ (PROGRESS.md, changelog)

### ПЕРВЫЕ КОМАНДЫ при любой проблеме:
1. `python scripts/tools/database_diagnostics.py --quick`
2. `python scripts/mass_qwen_analysis.py --test` (для анализа)
3. `python scripts/db_browser.py` (для интерактивной проверки)

---

## 🚀 KUBERNETES DEPLOYMENT

### Quick Start Commands
```bash
# Deploy complete stack
helm install rap-analyzer ./helm/rap-analyzer --create-namespace --namespace rap-analyzer

# Check deployment status
kubectl get pods -n rap-analyzer
kubectl get svc -n rap-analyzer

# Access applications
kubectl port-forward svc/rap-analyzer-service 8000:8000 -n rap-analyzer
kubectl port-forward svc/grafana-service 3000:3000 -n rap-analyzer
```

### Monitoring URLs (после port-forward)
- **API**: http://localhost:8000/docs
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090

### Helm Configuration
- **Chart Location**: `helm/rap-analyzer/`
- **Values**: `helm/rap-analyzer/values.yaml` (80+ parameters)
- **Templates**: Kubernetes manifests в `helm/rap-analyzer/templates/`

---

**REMEMBER**: Этот проект использует Kubernetes-native архитектуру с PostgreSQL + pgvector для production deployment. Для development - используй Docker Compose. ВСЕГДА используй готовую схему БД из этого документа вместо запросов к базе! Все актуальные метрики уже указаны выше.