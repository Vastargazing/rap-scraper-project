"""
#!/usr/bin/env python3
⚙️ Централизованное управление конфигурацией проекта Rap Scraper

НАЗНАЧЕНИЕ:
- Загрузка и обработка конфигурации из .env, config.yaml
- Управление путями, API-ключами, лимитами

ИСПОЛЬЗОВАНИЕ:
from src.utils.config import ...

ЗАВИСИМОСТИ:
- Python 3.8+
- python-dotenv
- config.yaml

РЕЗУЛЬТАТ:
- Доступ к переменным конфигурации и путям

АВТОР: AI Assistant
ДАТА: Сентябрь 2025
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load .env from project root
env_path = PROJECT_ROOT / ".env"
load_dotenv(env_path)

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# DB_PATH removed - project migrated to PostgreSQL
# Use src.config.get_config().database for database configuration
# See src/database/postgres_adapter.py for database access

# API Configuration
GENIUS_TOKEN = os.getenv(
    "GENIUS_ACCESS_TOKEN"
)  # Исправлено: используем правильное имя переменной
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Rate limiting
GENIUS_RATE_LIMIT = 1.0  # seconds between requests
SPOTIFY_RATE_LIMIT = 0.1


# Database Configuration
def get_db_config():
    """Получение конфигурации PostgreSQL базы данных"""
    return {
        "host": os.getenv("POSTGRES_HOST", os.getenv("DB_HOST", "localhost")),
        "port": int(os.getenv("POSTGRES_PORT", os.getenv("DB_PORT", "5432"))),
        "database": os.getenv("POSTGRES_DATABASE", os.getenv("DB_NAME", "rap_lyrics")),
        "user": os.getenv("POSTGRES_USERNAME", os.getenv("DB_USER", "rap_user")),
        "password": os.getenv(
            "POSTGRES_PASSWORD", os.getenv("DB_PASSWORD", "securepassword123")
        ),
    }


# Logging configuration
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILE = PROJECT_ROOT / "logs" / "scraping.log"
