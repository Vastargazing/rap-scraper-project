# 🎵 AI-Enhanced Rap Lyrics Dataset

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)](https://langchain.com/)
[![Gemini API](https://img.shields.io/badge/Gemini-API-orange.svg)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 🎯 **ML Pipeline для создания высококачественного датасета рэп-музыки с AI-анализом аутентичности**

## 🚀 Quick Start

```bash
# Клонирование репозитория
git clone https://github.com/Vastargazing/rap-scraper-project.git
cd rap-scraper-project

# Установка зависимостей
pip install -r requirements.txt

# Настройка API ключей в .env
GENIUS_TOKEN=your_genius_token
GOOGLE_API_KEY=your_gemini_key

# Тестирование AI анализа
python test_langchain.py
```

## 📊 Результаты

- **16,254** собранных песен с Genius.com
- **10** песен с полным AI-анализом (MVP)
- **0.735/1.0** средняя аутентичность ("живые" треки)
- **0.17/1.0** AI-likelihood (очень низкая вероятность ИИ-генерации)
- **20+** структурированных признаков для ML

## 🏗️ Архитектура

```
Data Collection → AI Enhancement → ML Dataset
     ↓               ↓              ↓
  Genius API    LangChain+Gemini   Structured
   (16k songs)   (20+ features)    JSON/CSV
```

## 🔧 Технологии

- **Data Collection**: `lyricsgenius` API, SQLite, Git LFS
- **AI Analysis**: LangChain, Google Gemini API, Pydantic
- **Data Processing**: Pandas, batch processing, caching
- **Quality**: 100% validation, error handling, logging

## 📈 ML Applications

1. **Genre Classification** (hip-hop, pop, rock)
2. **Hit Prediction** (commercial appeal scoring)
3. **AI Detection** (human vs AI-generated lyrics)
4. **Music Generation** (conditioning features)

## 📁 Ключевые файлы

- `rap_scraper.py` - Основной скрапер с Genius.com
- `langchain_analyzer.py` - AI анализ через Gemini
- `models.py` - Pydantic схемы данных
- `enhanced_scraper.py` - Интеграция AI в pipeline
- `analyze_results.py` - Статистика и визуализация

## 🎯 Для рекрутеров

Этот проект демонстрирует:
- **End-to-end ML pipeline** от сбора данных до анализа
- **LLM Engineering** с prompt optimization
- **Production-ready код** с error handling и testing
- **Scalable architecture** для больших датасетов

📄 **[Подробная документация](PROJECT_DOCUMENTATION.md)**  
🎤 **[Подготовка к интервью](INTERVIEW_PREPARATION.md)**

## 📞 Контакты

**Автор**: [Ваше имя]  
**LinkedIn**: [Ваш LinkedIn]  
**Email**: [Ваш email]

---

⭐ **Star this repo если проект показался интересным!**
