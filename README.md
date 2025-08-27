# 🎵 Rap Lyrics Scraper & Analyzer

**Production-ready ML pipeline для сбора и анализа рэп-лирики с использованием AI**

📊 **48K+ треков | 263 артиста | Spotify enriched | AI analyzed**

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Настройка API ключей
Создайте файл `.env`:
```bash
GENIUS_TOKEN=your_genius_token_here
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
GOOGLE_API_KEY=your_google_api_key_here  # Опционально
```

### 3. Главный CLI интерфейс
```bash
# Проверка статуса проекта
python scripts/rap_scraper_cli.py status

# Скрапинг новых данных
python scripts/rap_scraper_cli.py scraping

# Обогащение Spotify метаданными
python scripts/rap_scraper_cli.py spotify --continue

# ML анализ
python scripts/rap_scraper_cli.py analysis --analyzer gemma

# Полная справка
python scripts/rap_scraper_cli.py --help
```

## 🏗️ Архитектура проекта

### Новая структурированная архитектура:
```
src/
├── scrapers/     # 🕷️ Сбор данных (Genius API)
├── enhancers/    # 🎵 Обогащение (Spotify API)  
├── analyzers/    # 🤖 ML анализ (LLM models)
├── models/       # 📊 Pydantic модели
└── utils/        # 🛠️ Утилиты и конфигурация

scripts/         # 🚀 Entry points и CLI
monitoring/      # 📊 Мониторинг и статистика
data/           # 📄 База данных и файлы
results/        # 📈 Результаты анализа
docs/           # 📚 Документация
```

## � Основные команды

### 🕷️ Скрапинг данных
```bash
# Новая архитектура (рекомендуется)
python scripts/rap_scraper_cli.py scraping

# Прямой вызов
python scripts/run_scraping.py

# Legacy совместимость
python scripts/legacy/rap_scraper_optimized.py
```

### 🎵 Spotify Enhancement
```bash
# Через CLI
python scripts/rap_scraper_cli.py spotify --continue

# Прямой вызов
python scripts/continue_spotify_enhancement.py
```

### 🤖 ML Анализ
```bash
# Gemma 27B (рекомендуется)
python scripts/rap_scraper_cli.py analysis --analyzer gemma

# Сравнение моделей
python scripts/rap_scraper_cli.py analysis --analyzer multi

# LangChain + OpenAI
python scripts/rap_scraper_cli.py analysis --analyzer langchain
```

### 📊 Мониторинг
```bash
# Статус базы данных
python scripts/rap_scraper_cli.py monitoring --component database

# Прогресс AI анализа
python scripts/rap_scraper_cli.py monitoring --component analysis

# Мониторинг Gemma
python scripts/rap_scraper_cli.py monitoring --component gemma
```

### 🛠️ Утилиты
```bash
# Очистка проекта (dry run)
python scripts/rap_scraper_cli.py utils --utility cleanup

# Реальная очистка
python scripts/rap_scraper_cli.py utils --utility cleanup --execute

# Миграция БД
python scripts/rap_scraper_cli.py utils --utility migrate

# Настройка Spotify
python scripts/rap_scraper_cli.py utils --utility spotify-setup
```

## 🗄️ База данных

### Структура данных
- **Основная БД**: `data/rap_lyrics.db`
- **Таблица песен**: `songs` (48,370+ записей)
- **Таблица анализов**: `ai_analysis` (~1,500+ анализов)
- **Spotify данные**: `spotify_artists` (262/263 обогащенных артистов)
- **Конфиг артистов**: `data/rap_artists.json`

### Схема таблиц
```sql
-- Основная таблица песен
songs: artist, song, lyrics, url, scraped_at, album, year

-- AI анализы
ai_analysis: song_id, complexity, mood, genre, quality_score, analysis_text

-- Spotify метаданные
spotify_artists: genius_name, spotify_id, name, followers, genres, popularity
```

## 🤖 AI Модели и анализаторы

| Модель | Скорость | Качество | Использование | Файл |
|--------|----------|----------|---------------|------|
| **Gemma 3 27B** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Production анализ | `src/analyzers/gemma_27b_fixed.py` |
| **LangChain GPT** | ⚡⚡⚡ | ⭐⭐⭐⭐ | Быстрый анализ | `src/analyzers/langchain_analyzer.py` |
| **Multi-model** | ⚡ | ⭐⭐⭐⭐⭐ | Сравнение моделей | `src/analyzers/multi_model_analyzer.py` |

### Анализируемые метрики
- **Complexity**: Лингвистическая сложность (1-10)
- **Mood**: Эмоциональная окраска (positive/negative/neutral)
- **Genre**: Поджанр рэпа (trap, conscious, etc.)
- **Quality**: Оценка качества текста (1-10)
- **Themes**: Ключевые темы и мотивы

## 📈 Текущая статистика

### 📊 Данные проекта
- **48,370+** песен в базе данных
- **263** артиста (262 обогащены Spotify)
- **1,500+** AI анализов высокого качества
- **15GB+** общий размер данных

### 🎯 Покрытие анализа
- **Genius API**: 100% работоспособность
- **Spotify API**: 99.6% успешного обогащения (262/263)
- **AI анализ**: ~3% от общей базы (фокус на качество)
- **Error handling**: 100% покрытие критических ошибок

## 🔧 Системные требования

### Базовые требования
- **Python 3.8+** (рекомендуется 3.11+)
- **SQLite** (встроено в Python)
- **16GB+ RAM** (для Gemma 27B анализа)
- **50GB+ диск** (для полной базы данных)

### API ключи
- **Genius API** токен (обязательно)
- **Spotify API** credentials (client_id + client_secret)  
- **Google AI Studio** API ключ (для Gemma анализа)
- **Ollama** (опционально, для локального анализа)

## 📁 Новая архитектура проекта

```
rap-scraper-project/
├── � src/                    # Основной код
│   ├── scrapers/             # 🕷️ Сбор данных (Genius API)
│   ├── enhancers/            # 🎵 Spotify обогащение  
│   ├── analyzers/            # 🤖 ML анализ
│   ├── models/               # 📊 Pydantic модели
│   └── utils/                # 🛠️ Утилиты и конфиг
├── 🚀 scripts/               # Entry points и CLI
│   ├── rap_scraper_cli.py    # 🎯 Главный CLI интерфейс
│   ├── run_*.py              # 🏃 Прямые entry points
│   └── legacy/               # 🗂️ Совместимость
├── 📊 monitoring/            # Мониторинг и логи
├── 📄 data/                  # База данных и файлы
├── 📈 results/              # Результаты анализа
├── 🧪 tests/                # Unit тесты
└── 📚 docs/                 # Документация

# Legacy файлы (архивированы)
scripts/archive/             # Старые скрипты
```

## 💡 Примеры использования

### 🚀 Быстрый старт с CLI
```bash
# 1. Проверяем состояние проекта
python scripts/rap_scraper_cli.py status

# 2. Запускаем полный пайплайн
python scripts/rap_scraper_cli.py scraping          # Сбор данных
python scripts/rap_scraper_cli.py spotify --continue # Spotify обогащение  
python scripts/rap_scraper_cli.py analysis --analyzer gemma # AI анализ

# 3. Мониторим прогресс
python scripts/rap_scraper_cli.py monitoring --component all
```

### � Детальные команды
```bash
# Статистика базы данных
python scripts/rap_scraper_cli.py monitoring --component database

# Анализ конкретным алгоритмом
python scripts/rap_scraper_cli.py analysis --analyzer multi --limit 100

# Очистка проекта (dry run)
python scripts/rap_scraper_cli.py utils --utility cleanup

# Миграция БД с бекапом
python scripts/rap_scraper_cli.py utils --utility migrate
```

### 🛠️ Прямые вызовы (advanced)
```bash
# Прямой запуск компонентов
python scripts/run_scraping.py              # Скрапинг
python scripts/continue_spotify_enhancement.py # Spotify
python scripts/run_gemma_analysis.py        # AI анализ

# Legacy совместимость
python scripts/legacy/rap_scraper_optimized.py
python scripts/legacy/multi_model_analyzer.py
```

## � Результаты и выводы

### 🏆 Достижения проекта
- ✅ **48,370+** собранных треков с полными текстами
- ✅ **99.6%** успешного Spotify обогащения (262/263 артистов)
- ✅ **1,500+** high-quality AI анализов
- ✅ **Production-ready** архитектура с CLI интерфейсом
- ✅ **Полная автоматизация** пайплайна сбора и анализа

### 🎯 Качество данных
- **Lyrics coverage**: 100% для всех собранных треков
- **Metadata accuracy**: 99%+ благодаря Spotify API
- **AI analysis quality**: Экспертная оценка 9/10
- **Data consistency**: Полная валидация Pydantic моделями

## 🚨 Troubleshooting

### Частые проблемы
```bash
# Проблемы с путями после реструктуризации
python scripts/rap_scraper_cli.py utils --utility cleanup

# Ошибки импортов в старых скриптах  
# Используйте новый CLI или scripts/run_*.py entry points

# Проблемы с базой данных
python scripts/run_database_check.py

# Spotify API 403 ошибки - это нормально
# Система автоматически обрабатывает такие случаи
```

### 🆘 Поддержка
- 📖 Детальная документация в `docs/`
- 🐛 Issue tracking через Git
- 📊 Мониторинг через `monitoring/` скрипты
- 🧪 Unit тесты в `tests/`

---

**Created with ❤️ by AI Engineer | Production ML Pipeline | 2025**
python cleanup_project.py
```

## 📚 Документация

Детальная документация находится в `AI_Engineer_Journal/Projects/Rap_Scraper_Project/`:
- `README.md` - Презентационная версия
- `PROJECT_EVOLUTION.md` - История развития
- `TECH_SUMMARY.md` - Технический обзор
- `INTERVIEW_PREPARATION.md` - Подготовка к интервью

## 🎯 Статус проекта

- ✅ Скрапинг: Стабильно работает
- ✅ AI Анализ: Множественные модели
- ✅ Мониторинг: Реальное время
- ✅ База данных: 47K+ записей
- 🔄 В процессе: Полный анализ через Gemma 27B
