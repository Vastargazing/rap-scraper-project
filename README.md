# 🎵 Rap Lyrics Scraper & Analyzer

Система для сбора и анализа текстов рэп-песен с использованием AI.

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Настройка API ключей
Создайте файл `.env`:
```bash
GENIUS_TOKEN=your_genius_token_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Скрапинг песен
```bash
# Основной скрапер
python rap_scraper_optimized.py

# Продолжение прерванного скрапинга
python resume_scraping.py
```

### 4. Анализ песен
```bash
# Основной анализатор (Ollama)
python multi_model_analyzer.py

# Премиум анализ (Gemma 3 27B)
python gemma_27b_fixed.py

# Мониторинг прогресса
python monitor_gemma_progress.py
```

## 📊 Основные скрипты

### Скрапинг
- `rap_scraper_optimized.py` - Основной скрапер с Genius.com
- `enhanced_scraper.py` - Скрапер с интегрированным AI анализом  
- `resume_scraping.py` - Продолжение прерванного скрапинга

### Анализ
- `multi_model_analyzer.py` - Главный анализатор (Ollama + DeepSeek)
- `gemma_27b_fixed.py` - Премиум анализ через Gemma 3 27B
- `run_full_analysis.py` - Batch анализ всех песен

### Мониторинг
- `check_analysis_status.py` - Статус анализов
- `monitor_gemma_progress.py` - Мониторинг Gemma анализа в реальном времени

### Утилиты
- `migrate_database.py` - Миграция схемы БД
- `merge_databases.py` - Объединение баз данных
- `models.py` - Модели данных (Pydantic)

## 🗄️ База данных

- **Основная БД**: `rap_lyrics.db`
- **Таблица песен**: `songs` (~48K записей)
- **Таблица анализов**: `ai_analysis` (~1K анализов)
- **Артисты**: `rap_artists.json`

## 🤖 AI Модели

| Модель | Скорость | Качество | Стоимость | Файл |
|--------|----------|----------|-----------|------|
| Ollama llama3.2:3b | ⚡⚡⚡ | ⭐⭐⭐ | 🆓 | `multi_model_analyzer.py` |
| Gemma 3 27B | ⚡ | ⭐⭐⭐⭐⭐ | 🆓 | `gemma_27b_fixed.py` |

## 📈 Статистика проекта

- **47,971** песен в базе данных
- **237+** уникальных артистов  
- **~1,000** AI анализов высокого качества
- **100%** покрытие error handling
- **Git LFS** для больших файлов

## 🔧 Требования

- Python 3.8+
- SQLite
- Genius API токен
- Google AI Studio API ключ (для Gemma)
- Ollama (опционально, для локального анализа)

## 📁 Структура проекта

```
rap-scraper-project/
├── 🕷️ Скрапинг
│   ├── rap_scraper_optimized.py
│   ├── enhanced_scraper.py
│   └── resume_scraping.py
├── 🤖 Анализ  
│   ├── multi_model_analyzer.py
│   ├── gemma_27b_fixed.py
│   ├── models.py
│   └── run_full_analysis.py
├── 📊 Мониторинг
│   ├── check_analysis_status.py
│   └── monitor_gemma_progress.py
├── 🛠️ Утилиты
│   ├── migrate_database.py
│   └── merge_databases.py
├── 📄 Данные
│   ├── rap_lyrics.db
│   ├── rap_artists.json
│   └── remaining_artists.json
└── ⚙️ Конфигурация
    ├── .env
    ├── requirements.txt
    └── GEMMA_SETUP.md
```

## 💡 Полезные команды

```bash
# Проверить статус анализа
python check_analysis_status.py

# Быстрая статистика Gemma
python monitor_gemma_progress.py quick

# Очистка проекта (уже выполнено)
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
