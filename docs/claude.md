# Rap Scraper Project — AI Agent Context (Об## 🐳 DOCKER ECOSYSTEM (ОБНОВЛЕНО - 30.09.2025)

### Docker Compose Structure
- **`docker-compose.yml`** - Production (минимальный: API + PostgreSQL + Redis)
- **`docker-compose.dev.yml`** - Development (+ pgAdmin + Grafana + Prometheus)
- **`docker-compose.pgvector.yml`** - Database only (PostgreSQL + Redis для локалки)

### Key Commands
```bash
make docker-up      # Production stack
make docker-dev     # Full development stack
make docker-db      # Only database for local development
make docker-down    # Stop all services
```

### Build Context Optimization
- **Build context size**: 50MB (было 500MB)
- **Build time**: 30-60 seconds (было 2-3 минуты)
- **Optimized .dockerignore**: исключены data/, logs/, tests/, *.db

### 🔥 Критически важные команды
```bash
# QUICK COMMANDS (Start Here)
# 🐳 DOCKER COMMANDS (ОБНОВЛЕНО - 30.09.2025)
make docker-up       # Production stack (API + PostgreSQL + Redis)
make docker-dev      # Development stack (+ pgAdmin + Grafana + Prometheus)
make docker-db       # Database only (PostgreSQL + Redis для локалки)
make docker-down     # Stop all services
make docker-logs     # Show API logs
make docker-ps       # Show running containers

# 🤖 QWEN ML MODEL (ОСНОВНАЯ МОДЕЛЬ)
python models/test_qwen.py --test-api          # Тестирование QWEN API
python models/test_qwen.py --prepare-dataset   # Подготовка dataset (1000 samples)
python models/test_qwen.py --train             # Симуляция обучения
python models/test_qwen.py --all               # Полный ML pipeline

# 🚀 ML API SERVICE (Production Ready)
python src/models/ml_api_service.py --host 127.0.0.1 --port 8001
python test_ml_api.py                          # Тестирование ML API

# MULTI-REGION DEPLOYMENT (Enterprise)
.\multi-region\deploy-multi-region.ps1 -Action deploy      # Deploy all regions
.\multi-region\deploy-multi-region.ps1 -Action status      # Check status
python multi-region/test-multi-region.py                   # Test deployment

# GITOPS DEPLOYMENT 
./gitops/install-argocd.ps1                    # Install ArgoCD
kubectl port-forward svc/argocd-server -n argocd 8080:443  # Access UI
kubectl get applications -n argocd             # Check app status

# KUBERNETES STATUS
kubectl get pods -n rap-analyzer               # Check app pods
helm status rap-analyzer -n rap-analyzer       # Helm status

# DATABASE DIAGNOSTICS (для разработки)
python scripts/tools/database_diagnostics.py --quick
python scripts/mass_qwen_analysis.py --test
python scripts/db_browser.py
```01-19)

> **Kubernetes-native enterprise ML-pipeline** для анализа рэп-текстов с **PostgreSQL + pgvector**,## 📊 ТЕКУЩИЙ СТАТУС ПРОЕКТА

### Актуальные метрики (2025-09-28)
- 🎵 **Треки**: 57,718 (PostgreSQL)
- 🤖 **Анализ Qwen**: 57,716 (100.0%) | **✅ ЗАВЕРШЕН**
- 🤖 **Анализ Gemma**: 34,320 (59.4%)  
- 🧮 **Алгоритмический анализ**: 57,716 (100.0%) | **✅ ЗАВЕРШЕН**
- 🎯 **Общий анализ**: 57,718/57,718 (100.0%)
- 📊 **Всего анализов**: 269,646
- 🐘 **База**: PostgreSQL 15 + connection pool (20 подключений)
- ☸️ **Kubernetes**: Production-ready инфраструктура с monitoring

### 🤖 ML Models Status (NEW - 2025-09-28)
- 🎯 **Primary Model**: QWEN/qwen3-4b-fp8 via Novita AI ✅ WORKING
- 📊 **Training Dataset**: 1000 samples (800 train / 200 eval) ✅ READY
- 🎯 **Training Success**: 100% success rate, 5947 tokens
- 📈 **Evaluation Metrics**: MAE: 0.450, RMSE: 0.450 ✅ VALIDATED
- 🚀 **ML API Service**: FastAPI с **QWEN Primary**, T5, Quality Predictor, Trend Analysis ✅ WORKING
- 📁 **Results**: `results/qwen_training/` - все результаты сохранены

### Состояние системы
- ✅ **Phase 1: Kubernetes Migration ЗАВЕРШЕНА** (2025-01-19)
- ✅ **Phase 2: Multi-Region Deployment ЗАВЕРШЕНА** (2025-01-19)
- ✅ **Phase 2: GitOps Integration ЗАВЕРШЕНА** (2025-01-19)
- ✅ **Phase 4: Custom ML Models System ЗАВЕРШЕНА** (2025-09-28)
- ✅ **QWEN Primary Model НАСТРОЕНА** (2025-09-28)
- ✅ **PostgreSQL миграция завершена** (100% целостность данных)
- ✅ **Concurrent processing готов** (20 подключений в пуле)
- ✅ **Полный анализ завершен** (269,646 анализов, 100% coverage)
- ☸️ **Production Infrastructure**: Helm chart, monitoring, auto-scaling
- 🌍 **Multi-Region Architecture**: Global deployment (US-East-1, US-West-2, EU-West-1)
- 🚀 **GitOps Workflow**: ArgoCD, automated deployments, self-healing
- 🎯 **Current**: Phase 5 - Advanced AI Integration с QWEN как основной модельюcontainer orchestration, и comprehensive monitoring stack

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

### Multi-Region + GitOps Architecture
```
┌─────────────────── MULTI-REGION DEPLOYMENT ─────────────────────┐
│                                                                  │
│  ┌─── US-EAST-1 (PRIMARY) ───┐  ┌─── US-WEST-2 (REPLICA) ───┐   │
│  │  • PostgreSQL Primary     │  │  • PostgreSQL Replica     │   │
│  │  • Read/Write Operations  │──┤  • Read-Only Operations    │   │
│  │  • Streaming Replication  │  │  • Hot Standby            │   │
│  │  • ArgoCD ApplicationSet  │  │  • Regional API           │   │
│  └────────────────────────────┘  └────────────────────────────┘   │
│                   │                              │               │
│                   └─────── Replication ─────────┤               │
│                                                  │               │
│  ┌─── EU-WEST-1 (REPLICA + GDPR) ───────────────┤               │
│  │  • PostgreSQL Replica (GDPR Compliant)       │               │
│  │  • Read-Only Operations                       │               │
│  │  • Data Sovereignty Compliance                │               │
│  │  • Regional API + Monitoring                  │               │
│  └────────────────────────────────────────────────               │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────── GITOPS WORKFLOW ───────────────────────┐
│                                                           │
│  ┌─── GIT REPOSITORY ────┐    ┌─── ARGOCD CONTROLLER ───┐ │
│  │  • Helm Charts        │───▶│  • Monitors Git Repo    │ │
│  │  • K8s Manifests      │    │  • Automated Sync       │ │
│  │  • Multi-Region Config│    │  • Self-Healing         │ │
│  └───────────────────────┘    │  • Cross-Region Deploy  │ │
│                               └─────────────┬────────────┘ │
│                                            │              │
└────────────────────────────────────────────┼──────────────┘
                                             │
┌─────────────────── KUBERNETES CLUSTER ────┼──────────────┐
│                                            ▼              │
│  ┌─── INGRESS CONTROLLER ───┐                            │
│  │  • Global Load Balancing  │                            │
│  │  • SSL Termination        │                            │
│  │  • Multi-Region Routing   │                            │
│  └───────────┬───────────────┘                            │
│              │                                            │
│  ┌─────── FASTAPI SERVICE ────────┐                      │
│  │  • Regional Auto-scaling       │                      │
│  │  • HPA (CPU/Memory based)      │                      │  
│  │  • Health Probes              │                      │
│  │  • Cross-Region Load Balancing │                      │
│  └───────────┬───────────────────┘                      │
│              │                                            │
│  ┌────── POSTGRESQL + pgvector ──────┐                   │
│  │  • Primary/Replica StatefulSets   │                   │
│  │  • Cross-Region Replication       │                   │
│  │  • Vector Similarity Search       │                   │
│  │  • Regional Connection Pools      │                   │
│  └───────────┬────────────────────────┘                   │
│              │                                            │
│  ┌─────── MONITORING STACK ─────────┐                    │
│  │  ┌─── Prometheus (Multi-Region) ─┐│                    │
│  │  │  • Cross-Region Metrics       ││                    │
│  │  │  • Replication Lag Alerts     ││                    │
│  │  └────────────────────────────────┘│                    │
│  │  ┌─── Grafana (Global) ──────────┐ │                    │
│  │  │  • Multi-Region Dashboards    │ │                    │
│  │  │  • Global Performance Views   │ │                    │
│  │  └────────────────────────────────┘ │                    │
│  └─────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘

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
1. **`models/test_qwen.py`** - 🤖 **QWEN Primary ML Model** (НОВЫЙ 2025-09-28)
2. `src/database/postgres_adapter.py` - PostgreSQL connection management
3. `scripts/mass_qwen_analysis.py` - основной анализ скрипт  
4. `scripts/tools/database_diagnostics.py` - главный diagnostic tool
5. **`src/models/ml_api_service.py`** - 🚀 **ML API Service** (Production ML API)
6. **`test_ml_api.py`** - 🧪 **ML API Testing** (Test suite для ML endpoints)
7. `.env` - PostgreSQL credentials и API keys
8. `config.yaml` - система конфигурации

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
NOVITA_API_KEY=your-novita-api-key-here          # 🤖 QWEN ML Model (ОСНОВНОЙ)
GENIUS_ACCESS_TOKEN=your-genius-token
SPOTIFY_CLIENT_ID=your-spotify-client-id
SPOTIFY_CLIENT_SECRET=your-spotify-client-secret
```

---

## 🤖 QWEN ML MODEL - ОСНОВНАЯ МОДЕЛЬ ДЛЯ ОБУЧЕНИЯ

### 📊 Статус QWEN модели (2025-09-28)
- **🎯 Model**: `qwen/qwen3-4b-fp8` via Novita AI
- **✅ Status**: WORKING (100% success rate)
- **🔌 API**: https://api.novita.ai/openai (OpenAI-compatible)
- **📊 Training Dataset**: 1000 samples (800 train / 200 eval)
- **📈 Performance**: MAE: 0.450, RMSE: 0.450
- **🔢 Token Usage**: ~242 tokens per request, 5947 tokens total training
- **💾 Results**: `results/qwen_training/` - все результаты сохранены

### 🚀 QWEN команды (основные для AI агента)
```bash
# 🧪 ТЕСТИРОВАНИЕ API
python models/test_qwen.py --test-api          # Проверка подключения к Novita AI

# 📊 ПОДГОТОВКА ДАННЫХ
python models/test_qwen.py --prepare-dataset   # Загрузка 1000 samples из PostgreSQL

# 🎯 ОБУЧЕНИЕ МОДЕЛИ  
python models/test_qwen.py --train             # Симуляция обучения (5 samples)

# 📈 ОЦЕНКА КАЧЕСТВА
python models/test_qwen.py --evaluate          # Evaluation на 10 samples

# 🚀 ПОЛНЫЙ ЦИКЛ
python models/test_qwen.py --all               # API test + dataset + training + evaluation
```

### 🔧 QWEN конфигурация
```python
# Основные параметры QWEN модели
primary_model = "qwen/qwen3-4b-fp8"            # Единственная рабочая модель
base_url = "https://api.novita.ai/openai"      # Исправленный URL
temperature = 0.7                              # Оптимизированная для rap анализа
max_tokens = 20000                             # Увеличено для детального анализа
```

### 📁 QWEN файловая структура
```
models/
├── test_qwen.py                 # 🤖 QWEN Primary ML Model (ОСНОВНОЙ)
├── [УДАЛЕНО] conditional_generation.py    # GPT-2 больше не используется
├── style_transfer.py           # T5 model
├── quality_prediction.py       # Quality predictor
└── trend_analysis.py          # Trend analysis

results/qwen_training/
├── training_dataset.json       # 📊 Dataset (1000 samples)
├── training_results_*.json     # 🎯 Training results 
└── evaluation_results_*.json   # 📈 Evaluation metrics
```

### 💡 QWEN для AI агента
- **✅ QWEN** - основная модель для всех ML задач
- **❌ GPT-2** - удален, заменен на QWEN как основную модель  
- **📊 Dataset** - автоматически из PostgreSQL (57,718 треков доступно)
- **🎯 Training** - симуляция через prompt engineering (fine-tuning пока недоступен)
- **📈 Evaluation** - автоматическая оценка качества модели
- **🚀 Production** - готов к интеграции в ML API Service

---

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

### Quick Start Commands (GitOps Approach)
```bash
# OPTION 1: GitOps Deployment (Recommended)
./gitops/install-argocd.ps1                              # Install ArgoCD
kubectl port-forward svc/argocd-server -n argocd 8080:443  # Access ArgoCD UI
# https://localhost:8080 (admin/admin123)

# OPTION 2: Direct Helm Deployment
helm install rap-analyzer ./helm/rap-analyzer --create-namespace --namespace rap-analyzer

# Check deployment status
kubectl get pods -n rap-analyzer
kubectl get applications -n argocd                       # ArgoCD applications

# Access applications
kubectl port-forward svc/rap-analyzer-service 8000:8000 -n rap-analyzer
kubectl port-forward svc/grafana-service 3000:3000 -n rap-analyzer
```

### Monitoring URLs (после port-forward)
- **ArgoCD**: https://localhost:8080 (admin/admin123) - GitOps management
- **API**: http://localhost:8000/docs - FastAPI documentation
- **Grafana**: http://localhost:3000 (admin/admin123) - Monitoring dashboards
- **Prometheus**: http://localhost:9090 - Metrics collection

### GitOps Configuration
- **ArgoCD Setup**: `gitops/argocd/` - Complete ArgoCD installation
- **Applications**: `gitops/applications/rap-analyzer-app.yaml` - App configuration
- **Installation**: `gitops/install-argocd.ps1` - Automated ArgoCD deployment
- **Documentation**: `gitops/README.md` - Comprehensive GitOps guide

### Helm Configuration
- **Chart Location**: `helm/rap-analyzer/`
- **Values**: `helm/rap-analyzer/values.yaml` (80+ parameters)
- **Templates**: Kubernetes manifests в `helm/rap-analyzer/templates/`

---

**REMEMBER**: Этот проект использует Kubernetes-native архитектуру с PostgreSQL + pgvector для production deployment. Для development - используй Docker Compose. ВСЕГДА используй готовую схему БД из этого документа вместо запросов к базе! Все актуальные метрики уже указаны выше.