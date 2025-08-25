# AI Assistant Onboarding Checklist

## 🚀 Быстрый старт для новой сессии

### 1. Базовый контекст (ОБЯЗАТЕЛЬНО)
```
"Это проект rap-scraper для ML генерации рэп-лирики. 
Прочитай claude.md для полного контекста проекта."
```

### 2. Команды для AI assistant (копировать в чат)

#### Шаг 1: Основной контекст
```
read_file("c:\Users\VA\rap-scraper-project\claude.md")
```

#### Шаг 2: Текущее состояние проекта  
```
read_file("c:\Users\VA\rap-scraper-project\AI_Engineer_Journal\Projects\Rap_Scraper_Project\PROJECT_DIARY.md", offset=1980, limit=30)
```

#### Шаг 3: Архитектура данных
```
read_file("c:\Users\VA\rap-scraper-project\models.py")
```

#### Шаг 4: Проверка статистики БД
```
run_in_terminal("cd c:\Users\VA\rap-scraper-project; python check_db.py", "Проверка текущего состояния базы данных", false)
```

### 3. Контекстные фразы для быстрого понимания

#### Цель проекта:
```
"ML пайплайн для условной генерации рэп-лирики с использованием 
structured metadata из Genius API + Spotify Web API. 
47,971 треков, 259 артистов, готовим training data."
```

#### Текущая фаза:
```
"Case 13 завершен (claude.md + Agentic Search). 
Работаем над массовым обогащением треков через Spotify API."
```

#### Архитектурные принципы:
```
"Python 3.13+, Pydantic модели, SQLite persistence, 
rate limiting для APIs, incremental processing, 
comprehensive documentation в PROJECT_DIARY."
```

### 4. Ключевые файлы (по приоритету)

#### Tier 1 - Критически важные:
- `claude.md` - центральный контекст проекта
- `PROJECT_DIARY.md` - полная история 13 кейсов  
- `models.py` - Pydantic модели для всех данных

#### Tier 2 - Архитектура:
- `spotify_enhancer.py` - Spotify API интеграция
- `bulk_spotify_enhancement.py` - массовая обработка
- `rap_scraper_optimized.py` - основной скрапер

#### Tier 3 - Конфигурация:
- `requirements.txt` - зависимости
- `.env` - API credentials (не в git)
- `check_db.py` - утилита статистики

### 5. Быстрые команды для исследования

#### Понимание структуры:
```
file_search("*.py")
grep_search("def main|if __name__", isRegexp=true)
semantic_search("main processing pipeline")
```

#### Анализ данных:
```
grep_search("spotify_tracks|spotify_artists", isRegexp=true)
list_code_usages("SpotifyEnhancer")
get_changed_files()
```

#### Debugging:
```
grep_search("error|exception|failed", isRegexp=true)
get_terminal_output("terminal_id")
```

### 6. Шаблонные фразы для разных задач

#### Для разработки новых фич:
```
"Изучи архитектуру через claude.md, затем посмотри похожие 
реализации в проекте через semantic_search. Следуй паттернам 
Pydantic + rate limiting + error handling."
```

#### Для debugging:
```
"Проект использует structured logging и comprehensive error handling. 
Проверь логи, статус БД через check_db.py, и найди error patterns 
через grep_search."
```

#### Для ML анализа:
```
"У нас 47K треков с rich metadata для conditional generation. 
Посмотри *_analyzer.py файлы для понимания ML pipeline. 
Цель: artist + genre + mood → generated lyrics."
```

### 7. Критические моменты (НЕ ЗАБЫТЬ!)

#### API Limits:
```
"Всегда используй rate limiting! Genius: 1 req/sec, 
Spotify: native limits. См. spotify_enhancer.py для паттернов."
```

#### Data Quality:
```
"Все данные типизированы через Pydantic. Обязательно валидируй 
входящие данные перед сохранением в БД."
```

#### Documentation:
```
"Каждое значимое изменение документируется в PROJECT_DIARY 
в STAR формате для interview preparation."
```

### 8. One-liner для экстренного контекста:

```
"ML rap-lyrics project: 47K треков Genius+Spotify, Python+Pydantic+SQLite, 
13 documented cases в PROJECT_DIARY, цель - conditional generation training data. 
См. claude.md для деталей."
```

---

## 📋 Quick Copy-Paste Commands

### Minimal Setup (30 seconds):
```
read_file("c:\Users\VA\rap-scraper-project\claude.md")
```

### Full Context (2 minutes):
```
read_file("c:\Users\VA\rap-scraper-project\claude.md")
read_file("c:\Users\VA\rap-scraper-project\models.py")
run_in_terminal("cd c:\Users\VA\rap-scraper-project; python check_db.py", "Check DB status", false)
```

### Development Ready (5 minutes):
```
read_file("c:\Users\VA\rap-scraper-project\claude.md")
read_file("c:\Users\VA\rap-scraper-project\models.py")
semantic_search("main processing pipeline")
file_search("**/spotify_*.py")
get_changed_files()
```

---

*Создано: 2025-08-25. Обновляй при изменении архитектуры проекта.*
