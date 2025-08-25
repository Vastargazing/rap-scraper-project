# Rap Scraper Project - Claude Context

## 🎯 Цель проекта
ML пайплайн для **условной генерации рэп-лирики** с использованием структурированных метаданных из Genius API и Spotify Web API.

## 🏗️ Арх## 💡 Контекст для AI Assistant

При работе с этим проектом помни:
1. **Цель ML**: условная генерация лирики (artist + genre + mood → lyrics)
2. **Data quality**: важность structured metadata для обучения
3. **API limits**: всегда respectful usage с rate limiting
4. **Documentation**: все изменения документируются в PROJECT_DIARY
5. **Incremental progress**: проект развивается итеративно, каждый кейс добавляет value
6. **Agentic exploration**: исследуй код динамически, углубляйся по необходимости
7. **Planning-first approach**: сначала план, потом выполнение

### Workflow для новых задач:
1. **Исследуй** через agentic search
2. **Планируй** детальные шаги  
3. **Валидируй** план с пользователем
4. **Выполняй** пошагово
5. **Документируй** в PROJECT_DIARY

Этот проект - **showcase modern ML engineering practices** с акцентом на data quality, API integration, и production readiness.оекта

### Основные компоненты
- `rap_scraper.py` - базовый скрапер с Genius API (legacy)
- `rap_scraper_optimized.py` - оптимизированная версия с батчингом
- `spotify_enhancer.py` - интеграция с Spotify Web API для метаданных
- `bulk_spotify_enhancement.py` - массовая обработка треков через Spotify
- `models.py` - Pydantic модели для типизации данных
- `rap_lyrics.db` - SQLite база данных (47,971 треков, 259 артистов)

### ML анализаторы
- `langchain_analyzer.py` - анализ через LangChain + OpenAI
- `ollama_analyzer.py` - локальный анализ через Ollama
- `multi_model_analyzer.py` - сравнение разных моделей
- `optimized_analyzer.py` - оптимизированный пайплайн

### Структура данных
```sql
-- Основные таблицы
songs: id, title, artist, lyrics, url, genre, year
artists: name, total_songs, genres

-- Spotify расширения (Case 12)
spotify_artists: name, spotify_id, genres, popularity, followers_count
spotify_tracks: song_id, spotify_id, album_name, release_date, popularity
spotify_audio_features: track_id, danceability, energy, valence, tempo
```

## 🎨 Стиль кода

### Python специфика
- **Python 3.13+** с обязательными type hints
- **Pydantic модели** для валидации данных API
- **SQLite** для персистентности с контекстными менеджерами
- **Rate limiting** для всех API вызовов (respectful scraping)
- **Structured logging** для отладки и мониторинга

### Архитектурные принципы
- Разделение concerns: scraping → enhancement → analysis
- Инкрементальная обработка (resume functionality)
- Graceful degradation при API ошибках
- Batch processing для производительности

### Избегать
- Inline комментарии с очевидным содержанием
- Hardcoded credentials (используем .env)
- Блокирующие API вызовы без таймаутов
- Необработанные исключения в production коде

## 🔧 Как запускать

### Основной скрапинг
```bash
python rap_scraper_optimized.py  # Новые треки с Genius
python bulk_spotify_enhancement.py  # Обогащение метаданными
```

### ML анализ
```bash
python langchain_analyzer.py     # OpenAI анализ
python ollama_analyzer.py       # Локальный анализ
python multi_model_analyzer.py  # Сравнение моделей
```

### Утилиты
```bash
python check_db.py             # Проверка состояния БД
python setup_spotify.py        # Настройка Spotify API
python migrate_database.py     # Миграции схемы
```

## 🚀 Особенности проекта

### API интеграции
- **Genius API**: lyrics + метаданные треков (~48K треков)
- **Spotify Web API**: artist metadata + audio features (OAuth 2.0)
- **Rate limiting**: 1 req/sec для Genius, native для Spotify
- **Error handling**: 403/429 коды, automatic retry с backoff

### ML пайплайн цели
1. **Сбор данных**: lyrics + structured metadata  
2. **Feature engineering**: sentiment, complexity, audio features
3. **Conditional generation**: artist style + genre + mood → new lyrics
4. **Training data**: 47K треков с rich context для fine-tuning

### Документирование
- `PROJECT_DIARY.md` - хронология всех 12+ кейсов развития
- `TECH_SUMMARY.md` - сводка технических решений
- `PROJECT_EVOLUTION.md` - трекинг изменений архитектуры

## 📁 Ключевые файлы для понимания

### Конфигурация и модели
- @models.py - все Pydantic модели (SpotifyArtist, SpotifyTrack, etc.)
- @requirements.txt - зависимости проекта
- @.env - API credentials (не в git)

### Документация проекта  
- @PROJECT_DIARY.md - полная история кейсов с STAR format
- @TECH_SUMMARY.md - техническое резюме для интервью
- @PROJECT_EVOLUTION.md - эволюция архитектуры

### Данные и результаты
- @rap_lyrics.db - основная SQLite база
- @analysis_results/ - CSV результаты ML анализа
- @enhanced_data/ - обогащенные JSONL файлы

## 🎵 Текущее состояние (Case 12)

### Достижения
- ✅ 47,971 треков собрано из Genius API
- ✅ 258/259 артистов обогащены через Spotify (99.6% coverage)  
- ✅ Полная типизация данных через Pydantic
- ✅ Production-ready rate limiting и error handling
- ✅ Comprehensive документирование в PROJECT_DIARY

### В процессе
- 🔄 Массовое обогащение треков через Spotify API
- 🔄 Audio features extraction (работаем вокруг 403 ошибок)
- 🔄 ML feature engineering для conditional generation

### Следующие шаги
- 🎯 Завершить Spotify track enrichment
- 🎯 Implement sentiment analysis пайплайн  
- 🎯 Подготовить training dataset для fine-tuning
- 🎯 Выбрать архитектуру для conditional generation model

## � Agentic Search Philosophy

### Исследование кода как real developer
Вместо RAG с pre-indexed embeddings, используем **динамическое исследование**:

#### Начальное понимание
```bash
read_file("claude.md")                    # Общий контекст проекта
semantic_search("main pipeline")          # Ключевые компоненты  
grep_search("def main|if __name__")      # Entry points
```

#### Углубленное изучение  
```bash
list_code_usages("SpotifyEnhancer")      # Как используется класс
file_search("**/test_*.py")              # Найти тесты
get_changed_files()                      # Последние изменения
```

#### Production debugging
```bash
grep_search("error|exception|failed")    # Error handling patterns
read_file() # с контекстом               # Изучить проблемную область  
replace_string_in_file()                 # Исправить и тестировать
```

### Преимущества над RAG:
- ✅ **Live exploration** - актуальное состояние кода
- ✅ **Adaptive depth** - от overview до deep dive
- ✅ **Tool composition** - grep + semantic + read_file
- ✅ **Natural workflow** - как исследует человек-разработчик

## 📋 Planning-First Workflow

### Подход "думай, потом делай"
❌ **Плохо**: "Исправь этот баг"  
✅ **Хорошо**: "У меня баг X. Исследуй код, найди причину и предложи план исправления"

#### Фазы работы:
1. **🔍 Исследование**: Agentic search для понимания проблемы
2. **📋 Планирование**: Детальный план с шагами и рисками  
3. **✅ Валидация**: Обсуждение и корректировка плана
4. **🚀 Выполнение**: Следование утвержденному плану
5. **📝 Документирование**: Обновление PROJECT_DIARY

#### Пример workflow (Case 12):
```markdown
## Problem: Spotify API 403 errors

### 🔍 Investigation:
- grep_search("403|Forbidden") → найден error pattern
- read_file("spotify_enhancer.py") → изучен API client
- semantic_search("audio features") → понял специфику endpoint

### 📋 Plan:
1. Проверить permissions в Spotify Console
2. Модифицировать код для graceful degradation  
3. Обновить error handling
4. Тестировать на sample данных

### ✅ Validation: Plan approved
### 🚀 Execution: Implemented get_audio_features=False fallback
### 📝 Documentation: Updated PROJECT_DIARY Case 12
```

#### To-do отслеживание:
```python
# TODO: Implement audio features fallback (Case 12)  
# DONE: Rate limiting implementation (Case 12)
# PENDING: Full track enrichment testing (Case 13)
```

## 🧪 Test-Driven Development Workflow

### Принципы безопасной разработки
**Маленькие итерации → Тестирование → Коммиты → Следующая итерация**

#### Workflow цикл:
1. **🔧 Изменение кода** - небольшая фича или fix
2. **🧪 Запуск тестов** - `pytest tests/`  
3. **📝 Проверка качества** - linting, type checking
4. **💾 Коммит изменений** - в git с описанием
5. **🔄 Следующая итерация** - повторить цикл

#### Структура тестов:
```bash
tests/
├── test_spotify_enhancer.py    # Unit тесты API интеграции
├── test_models.py              # Pydantic модели валидация  
├── test_database.py            # SQLite операции
├── test_scraper.py             # Genius API тесты
└── conftest.py                 # Pytest fixtures
```

#### Команды разработки:
```bash
# Полный цикл разработки
pytest tests/ --verbose          # Запуск всех тестов
python -m mypy spotify_enhancer.py  # Type checking
python -m flake8 *.py            # Linting
git add . && git commit -m "feat: description"  # Коммит

# Быстрая проверка
pytest tests/test_spotify_enhancer.py -v  # Конкретный модуль
```

#### Преимущества:
- ✅ **Безопасные рефакторинги** - тесты ловят регрессии
- ✅ **Легкие откаты** - каждый коммит стабилен
- ✅ **Уверенность в коде** - comprehensive test coverage
- ✅ **Production readiness** - все компоненты протестированы

## �💡 Контекст для AI Assistant

При работе с этим проектом помни:
1. **Цель ML**: условная генерация лирики (artist + genre + mood → lyrics)
2. **Data quality**: важность structured metadata для обучения
3. **API limits**: всегда respectful usage с rate limiting
4. **Documentation**: все изменения документируются в PROJECT_DIARY
5. **Incremental progress**: проект развивается итеративно, каждый кейс добавляет value
6. **Agentic exploration**: исследуй код динамически, углубляйся по необходимости

Этот проект - **showcase modern ML engineering practices** с акцентом на data quality, API integration, и production readiness.
