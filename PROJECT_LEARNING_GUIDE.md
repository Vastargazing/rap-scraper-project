# 🎓 Rap Scraper Project - Путеводитель по изучению

**Структурированный подход к изучению архитектуры проекта**

Этот документ поможет последовательно изучить все компоненты проекта от простого к сложному.

---

## 📋 Общий план изучения

### 🎯 Цель обучения
Понять полную архитектуру ML-pipeline для сбора и анализа текстовых данных, освоить паттерны производственного Python кода.

### ⏱️ Примерное время: 2-3 недели
- **Неделя 1**: Основы (scrapers, models, database)
- **Неделя 2**: Интеграции (Spotify API, AI analyzers)
- **Неделя 3**: Production (CLI, monitoring, deployment)

---

## 🗺️ Карта компонентов

```
1. FOUNDATION (базовые знания)
   ├── 📊 Data Models (Pydantic)
   ├── 🗄️ Database Design (SQLite)
   └── 🕷️ Web Scraping (Genius API)

2. INTEGRATIONS (интеграции)
   ├── 🎵 Spotify API Integration
   ├── 🤖 AI Analysis Pipeline
   └── 🔄 Data Enhancement

3. PRODUCTION (production-ready)
   ├── 🎯 CLI Interface Design
   ├── 📊 Monitoring & Logging
   └── 🧪 Testing & Quality
```

---

## 📚 Поэтапное изучение

### 🔥 ЭТАП 1: Фундамент (Сейчас изучаете)
*Базовые концепции программирования и архитектуры*

#### 1.1 📊 Модели данных
**Файл**: `src/models/models.py`
**Время**: 2-3 часа

**Что изучать**:
```python
# Основные концепции:
- Pydantic модели для валидации данных
- Type hints и строгая типизация
- Enum для категорий (Genre, Mood)
- Связи между моделями (Song ↔ AIAnalysis)
```

**Ключевые вопросы**:
- Зачем нужна валидация данных?
- Как Pydantic защищает от ошибок?
- Почему используются Enum вместо строк?

#### 1.2 🗄️ База данных
**Файлы**: `data/rap_lyrics.db`, `scripts/check_db.py`
**Время**: 2-3 часа

**Что изучать**:
```sql
-- Схема таблиц:
songs: artist, song, lyrics, url, scraped_at, album, year
ai_analysis: song_id, complexity, mood, genre, quality_score
spotify_artists: genius_name, spotify_id, name, followers
```

**Практика**:
```bash
# Изучение структуры БД
python scripts/check_db.py
sqlite3 data/rap_lyrics.db ".schema"
```

#### 1.3 🕷️ Web Scraping (ТЕКУЩИЙ ФОКУС)
**Файл**: `src/scrapers/rap_scraper_optimized.py`
**Время**: 4-6 часов

**Что изучать**:
```python
# Ключевые концепции:
- HTTP requests с retry логикой
- Парсинг HTML с BeautifulSoup
- Rate limiting и уважение к серверу
- Обработка ошибок и edge cases
- Proxy handling для стабильности
```

**Архитектурные паттерны**:
- `OptimizedGeniusScraper` класс
- Методы: `scrape_artist()`, `scrape_song()`
- Error handling и logging
- Database integration

---

### 🚀 ЭТАП 2: Интеграции
*После освоения основ переходите сюда*

#### 2.1 🎵 Spotify API Integration
**Файл**: `src/enhancers/spotify_enhancer.py`
**Время**: 3-4 часа

**Что изучать**:
```python
# Продвинутые концепции:
- OAuth2 authentication
- Rate limiting для внешних API
- Batch processing для эффективности
- Кеширование результатов
- Fuzzy matching для поиска артистов
```

**Практика**:
```bash
# Тестирование Spotify интеграции
python scripts/run_spotify_enhancement.py
python scripts/tools/check_spotify_coverage.py
```

#### 2.2 🤖 AI Analysis Pipeline
**Файл**: `src/analyzers/multi_model_analyzer.py` (1610+ строк!)
**Время**: 6-8 часов

**Что изучать**:
```python
# ML Engineering концепции:
- Multi-provider strategy (Ollama → Gemma → Mock)
- Prompt engineering для анализа текста
- Safety validation и content filtering
- Structured output с Pydantic
- Error recovery и fallback механизмы
```

**Архитектура AI Pipeline**:
- `MultiModelAnalyzer` - главный класс
- `SafetyValidator` - проверка контента
- `InterpretableAnalyzer` - понятные результаты

#### 2.3 🔄 Batch Processing
**Файл**: `scripts/tools/batch_ai_analysis.py`
**Время**: 2-3 часа

**Что изучать**:
```python
# Production ML concepts:
- Batch processing для больших данных
- Progress tracking и ETA calculations
- Memory management
- Resume/continue functionality
- Performance metrics
```

---

### 🏗️ ЭТАП 3: Production Architecture
*Финальный этап - production-ready код*

#### 3.1 🎯 CLI Interface Design
**Файл**: `scripts/rap_scraper_cli.py`
**Время**: 3-4 часа

**Что изучать**:
```python
# Advanced Python patterns:
- Command pattern для CLI
- Argparse для сложных интерфейсов
- Модульная архитектура команд
- Error handling на уровне UI
- Consistent user experience
```

#### 3.2 📊 Monitoring & Diagnostics
**Файлы**: `monitoring/`, `scripts/check_db.py`
**Время**: 2-3 часа

**Что изучать**:
```python
# Production monitoring:
- Health checks для компонентов
- Performance metrics
- Progress tracking
- Error reporting
- Status dashboards
```

#### 3.3 🧪 Testing & Quality
**Файлы**: `tests/`, `Makefile`
**Время**: 2-3 часа

**Что изучать**:
```python
# Quality assurance:
- Unit testing с pytest
- Integration testing
- Test fixtures и mocking
- TDD workflow с Makefile
- Code coverage
```

---

## 🎯 Практические задания по этапам

### ЭТАП 1: Задания для фундамента

#### Задание 1.1: Исследование моделей
```bash
# 1. Изучите все Pydantic модели
python -c "from src.models.models import *; print(Song.__fields__)"

# 2. Создайте тестовую песню и проверьте валидацию
# 3. Попробуйте нарушить валидацию и посмотрите на ошибки
```

#### Задание 1.2: Исследование БД
```bash
# 1. Подключитесь к БД и изучите данные
sqlite3 data/rap_lyrics.db

# Полезные SQL запросы:
.tables                              # Список таблиц
SELECT COUNT(*) FROM songs;          # Количество песен
SELECT artist, COUNT(*) FROM songs GROUP BY artist LIMIT 10;
```

#### Задание 1.3: Разбор скрапера (ТЕКУЩЕЕ)
```python
# Вопросы для анализа кода:
1. Как работает retry механизм в scrape_song()?
2. Зачем нужен rate limiting?
3. Как обрабатываются ошибки 404?
4. Что делает normalize_artist_name()?
5. Как работает proxy rotation?
```

### ЭТАП 2: Задания для интеграций

#### Задание 2.1: Spotify API
```bash
# 1. Запустите поиск одного артиста
python -c "from src.enhancers.spotify_enhancer import SpotifyEnhancer; ..."

# 2. Изучите fuzzy matching алгоритм
# 3. Проанализируйте rate limiting стратегию
```

#### Задание 2.2: AI Analysis
```bash
# 1. Запустите анализ одной песни
python scripts/rap_scraper_cli.py analysis --analyzer multi --limit 1

# 2. Изучите prompt templates
# 3. Проследите fallback chain: Ollama → Gemma → Mock
```

### ЭТАП 3: Production задания

#### Задание 3.1: CLI Exploration
```bash
# 1. Изучите все доступные команды
python scripts/rap_scraper_cli.py --help

# 2. Запустите каждую команду в dry-run режиме
# 3. Проанализируйте error handling
```

---

## 🔍 Ключевые концепции по темам

### 🏛️ Архитектурные паттерны
- **Repository pattern**: Database access layer
- **Strategy pattern**: Multiple AI providers
- **Command pattern**: CLI interface
- **Factory pattern**: Model creation
- **Observer pattern**: Progress monitoring

### 🛡️ Error Handling стратегии
- **Graceful degradation**: AI fallbacks
- **Retry with backoff**: Network requests
- **Circuit breaker**: API failures
- **Validation**: Data integrity
- **Logging**: Debugging information

### ⚡ Performance оптимизации
- **Batch processing**: Bulk operations
- **Connection pooling**: Database efficiency
- **Caching**: API response caching
- **Lazy loading**: Memory optimization
- **Rate limiting**: API respect

---

## 🎓 Вопросы для самопроверки

### После каждого этапа задавайте себе:

#### Технические вопросы:
1. **Зачем** этот компонент нужен?
2. **Как** он взаимодействует с другими частями?
3. **Что** произойдёт при ошибке?
4. **Почему** выбран именно такой подход?

#### Архитектурные вопросы:
1. Какие **принципы SOLID** применены?
2. Как **тестировать** этот компонент?
3. Как **масштабировать** при росте данных?
4. Какие **альтернативы** возможны?

---

## 📖 Рекомендуемый порядок изучения файлов

### 1️⃣ Неделя 1: Foundation
```
День 1-2: src/models/models.py
День 3-4: scripts/check_db.py + database exploration
День 5-7: src/scrapers/rap_scraper_optimized.py (ТЕКУЩИЙ ФОКУС)
```

### 2️⃣ Неделя 2: Integrations
```
День 1-3: src/enhancers/spotify_enhancer.py
День 4-6: src/analyzers/multi_model_analyzer.py
День 7: scripts/tools/batch_ai_analysis.py
```

### 3️⃣ Неделя 3: Production
```
День 1-2: scripts/rap_scraper_cli.py
День 3-4: monitoring/ + testing
День 5-7: Практические задания и code review
```

---

## 🎯 Следующий шаг после скрапера

После полного понимания `src/scrapers/rap_scraper_optimized.py`, переходите к:

**Файл**: `src/enhancers/spotify_enhancer.py`
**Фокус**: API интеграции и обогащение данных
**Новые концепции**: OAuth2, fuzzy matching, batch processing

---

## 📚 Дополнительные ресурсы

### Документация проекта:
- `docs/claude.md` - AI context file
- `AI_ONBOARDING_CHECKLIST.md` - Quick start guide
- `README.md` - Full project overview

### Практические команды:
```bash
# Мониторинг прогресса обучения
python scripts/rap_scraper_cli.py status

# Тестирование компонентов
python scripts/check_db.py
python scripts/tools/check_spotify_coverage.py

# Запуск анализа для понимания результатов
python scripts/tools/batch_ai_analysis.py --dry-run
```

---

**🎓 Удачи в изучении! Каждый компонент - это новый уровень в понимании production ML архитектуры.**
