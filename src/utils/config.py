"""Centralized configuration management."""
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
DB_PATH = DATA_DIR / "rap_lyrics.db"

# API Configuration
GENIUS_TOKEN = os.getenv("GENIUS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Rate limiting
GENIUS_RATE_LIMIT = 1.0  # seconds between requests
SPOTIFY_RATE_LIMIT = 0.1

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = PROJECT_ROOT / "scraping.log"
