# 🏗️ Rap Scraper Project Restructure - Migration Guide

> **Комплексная реструктуризация проекта** для улучшения архитектуры, maintainability и production readiness ML пайплайна.

## 📑 Quick Navigation
- [🎯 Migration Overview](#-migration-overview) - понимание процесса
- [📁 New Structure](#-new-project-structure) - целевая архитектура
- [🔄 Migration Steps](#-migration-steps) - пошаговый план
- [🔧 Import Updates](#-import-updates-reference) - обновление импортов
- [✅ Validation Checklist](#-validation-checklist) - проверка работоспособности
- [🚨 Rollback Plan](#-rollback-plan) - план отката

---

## 🎯 Migration Overview

### Current State (Flat Structure)
```
rap-scraper-project/
├── rap_scraper_optimized.py
├── enhanced_scraper.py
├── multi_model_analyzer.py
├── gemma_27b_fixed.py
├── models.py
├── check_db.py
├── [30+ other files]
└── rap_lyrics.db
```

### Target State (Organized Structure)
```
rap-scraper-project/
├── src/
│   ├── scrapers/
│   ├── enhancers/
│   ├── analyzers/
│   ├── models/
│   └── utils/
├── monitoring/
├── data/
├── results/
├── tests/
├── docs/
└── scripts/
```

### Migration Philosophy
- **Zero Downtime**: Все скрипты должны работать после миграции
- **Backward Compatibility**: Старые entry points сохраняются через scripts/
- **Import Safety**: Все relative imports обновляются корректно
- **Data Preservation**: База данных и результаты остаются нетронутыми

---

## 📁 New Project Structure

### Source Code Organization
```
src/
├── scrapers/              # 🕷️ Data Collection Layer
│   ├── __init__.py
│   ├── rap_scraper_optimized.py      # Основной Genius scraper
│   ├── enhanced_scraper.py           # Scraper с AI анализом
│   ├── resume_scraping.py            # Recovery functionality
│   └── base_scraper.py               # Общий базовый класс (NEW)
│
├── enhancers/             # 🔗 Data Enhancement Layer
│   ├── __init__.py
│   ├── spotify_enhancer.py           # Spotify API integration
│   ├── bulk_spotify_enhancement.py   # Batch processing
│   └── base_enhancer.py              # Базовый класс (NEW)
│
├── analyzers/             # 🤖 ML Analysis Layer
│   ├── __init__.py
│   ├── multi_model_analyzer.py       # Ollama + DeepSeek
│   ├── langchain_analyzer.py         # LangChain + OpenAI
│   ├── ollama_analyzer.py            # Local LLM
│   ├── gemma_27b_fixed.py           # Gemma 3 27B
│   ├── optimized_analyzer.py         # Production pipeline
│   └── base_analyzer.py              # Общий интерфейс (NEW)
│
├── models/                # 📊 Data Models & Storage
│   ├── __init__.py
│   ├── models.py                     # Pydantic models
│   ├── database.py                   # SQLite operations (NEW)
│   ├── schemas.py                    # DB schemas (NEW)
│   └── validators.py                 # Data validation (NEW)
│
└── utils/                 # 🛠️ Utilities & Helpers
    ├── __init__.py
    ├── check_db.py                   # Database status
    ├── migrate_database.py           # Schema migrations
    ├── merge_databases.py            # DB utilities
    ├── setup_spotify.py              # Spotify OAuth
    ├── config.py                     # Configuration management (NEW)
    └── logging_utils.py              # Centralized logging (NEW)
```

### External Directories
```
monitoring/                # 📊 Monitoring & Status
├── __init__.py
├── check_analysis_status.py         # Analysis progress
├── monitor_gemma_progress.py        # Real-time Gemma tracking
├── project_status.py               # Live dashboard (NEW)
└── health_check.py                 # System health (NEW)

data/                      # 📄 Data Storage (Git LFS)
├── rap_lyrics.db                    # Main SQLite DB
├── rap_artists.json                # Artist metadata
├── remaining_artists.json          # Scraping queue
├── backups/                        # DB backups (NEW)
└── .gitkeep

results/                   # 📈 Output Files
├── analysis_results/               # CSV analysis outputs
├── enhanced_data/                  # JSONL enriched data
├── exports/                        # Data exports (NEW)
└── .gitkeep

tests/                     # ✅ Test Suite
├── __init__.py
├── conftest.py                     # Pytest fixtures
├── test_scrapers.py                # Scraping tests
├── test_analyzers.py               # Analysis tests
├── test_models.py                  # Data model tests
├── test_enhancers.py               # Enhancement tests
├── test_utils.py                   # Utility tests
└── integration/                    # Integration tests (NEW)
    ├── test_full_pipeline.py
    └── test_api_integration.py

docs/                      # 📚 Documentation
├── README.md                       # Main documentation
├── claude_md.md                    # AI agent context
├── PROJECT_DIARY.md                # Development history
├── TECH_SUMMARY.md                 # Technical overview
├── GEMMA_SETUP.md                  # Gemma configuration
├── API_REFERENCE.md                # Code documentation (NEW)
└── DEPLOYMENT.md                   # Production guide (NEW)

scripts/                   # 🚀 Entry Points & Main Commands
├── __init__.py
├── run_scraping.py                 # Main scraping entry point
├── run_analysis.py                 # Main analysis entry point
├── run_full_pipeline.py            # Complete workflow
├── setup_project.py                # Initial setup
└── legacy/                         # Backward compatibility
    ├── rap_scraper_optimized.py    # Wrapper script
    ├── multi_model_analyzer.py     # Wrapper script
    └── gemma_27b_fixed.py          # Wrapper script
```

---

## 🔄 Migration Steps

### Step 1: Pre-Migration Backup
```bash
# Создать backup текущего состояния
git add . && git commit -m "Pre-restructure snapshot"
git tag v1.0-flat-structure

# Backup базы данных
cp rap_lyrics.db data_backup_$(date +%Y%m%d_%H%M%S).db
```

### Step 2: Create Directory Structure
```bash
# Создать все необходимые папки
mkdir -p src/{scrapers,enhancers,analyzers,models,utils}
mkdir -p {monitoring,data,results,tests,docs,scripts}
mkdir -p results/{analysis_results,enhanced_data,exports}
mkdir -p tests/integration
mkdir -p scripts/legacy
mkdir -p data/backups

# Создать __init__.py файлы
touch src/__init__.py
touch src/{scrapers,enhancers,analyzers,models,utils}/__init__.py
touch {monitoring,tests,scripts}/__init__.py
```

### Step 3: Move Files (AI Agent Execution)
```python
# File movement mapping для AI агента
MIGRATION_MAP = {
    # Scrapers
    "rap_scraper_optimized.py": "src/scrapers/",
    "enhanced_scraper.py": "src/scrapers/", 
    "resume_scraping.py": "src/scrapers/",
    
    # Enhancers  
    "spotify_enhancer.py": "src/enhancers/",
    "bulk_spotify_enhancement.py": "src/enhancers/",
    
    # Analyzers
    "multi_model_analyzer.py": "src/analyzers/",
    "langchain_analyzer.py": "src/analyzers/",
    "ollama_analyzer.py": "src/analyzers/",
    "gemma_27b_fixed.py": "src/analyzers/",
    "optimized_analyzer.py": "src/analyzers/",
    
    # Models & Data
    "models.py": "src/models/",
    
    # Utils
    "check_db.py": "src/utils/",
    "migrate_database.py": "src/utils/", 
    "merge_databases.py": "src/utils/",
    "setup_spotify.py": "src/utils/",
    
    # Monitoring
    "check_analysis_status.py": "monitoring/",
    "monitor_gemma_progress.py": "monitoring/",
    
    # Data files
    "rap_lyrics.db": "data/",
    "rap_artists.json": "data/",
    "remaining_artists.json": "data/",
    
    # Documentation
    "claude_md.md": "docs/",
    "PROJECT_DIARY.md": "docs/",
    "TECH_SUMMARY.md": "docs/",
    "GEMMA_SETUP.md": "docs/",
}
```

### Step 4: Update Import Statements
```python
# Примеры обновления импортов

# BEFORE (в файлах)
from models import SpotifyTrack, SpotifyArtist
import check_db

# AFTER  
from src.models.models import SpotifyTrack, SpotifyArtist
from src.utils import check_db

# В src/scrapers/rap_scraper_optimized.py
# BEFORE
from models import Song, Artist

# AFTER
from ..models.models import Song, Artist
```

### Step 5: Create Entry Point Scripts
```python
# scripts/run_scraping.py
"""Main scraping entry point with backward compatibility."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.scrapers.rap_scraper_optimized import main

if __name__ == "__main__":
    main()

# scripts/legacy/rap_scraper_optimized.py  
"""Backward compatibility wrapper."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.scrapers.rap_scraper_optimized import main

if __name__ == "__main__":
    main()
```

### Step 6: Update Configuration
```python
# src/utils/config.py (NEW FILE)
"""Centralized configuration management."""
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DB_PATH = DATA_DIR / "rap_lyrics.db"

# API Configuration
GENIUS_TOKEN = os.getenv("GENIUS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Rate limiting
GENIUS_RATE_LIMIT = 1.0  # seconds between requests
SPOTIFY_RATE_LIMIT = 0.1
```

---

## 🔧 Import Updates Reference

### Основные паттерны изменений

#### 1. Cross-package imports
```python
# BEFORE
from models import SpotifyTrack
import check_db

# AFTER  
from src.models.models import SpotifyTrack
from src.utils.check_db import check_database_status
```

#### 2. Relative imports внутри src/
```python
# В src/scrapers/rap_scraper_optimized.py
# BEFORE
from models import Song

# AFTER
from ..models.models import Song
from ..utils.config import DB_PATH
```

#### 3. Path updates для файлов
```python
# BEFORE
DB_PATH = "rap_lyrics.db"

# AFTER  
from src.utils.config import DB_PATH
# или
DB_PATH = "data/rap_lyrics.db"
```

#### 4. Entry points
```python
# В scripts/run_scraping.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.scrapers.rap_scraper_optimized import main as scraper_main
```

### Critical Files Import Updates

#### src/scrapers/rap_scraper_optimized.py
```python
# Critical imports to update
from ..models.models import Song, Artist  # было: from models import
from ..utils.config import DB_PATH, GENIUS_TOKEN  # было: hardcoded paths
```

#### src/analyzers/multi_model_analyzer.py  
```python
from ..models.models import AnalysisResult
from ..utils.database import get_database_connection
```

#### monitoring/check_analysis_status.py
```python
from src.models.models import Song
from src.utils.config import DB_PATH
```

---

## ✅ Validation Checklist

### Core Functionality Tests
```bash
# 1. Database operations
python -c "from src.utils.check_db import main; main()"

# 2. Main scraping pipeline  
python scripts/run_scraping.py --limit 1 --dry-run

# 3. Analysis pipeline
python scripts/run_analysis.py --sample 1

# 4. Monitoring
python monitoring/check_analysis_status.py

# 5. Legacy compatibility
python scripts/legacy/rap_scraper_optimized.py --help
```

### Import Validation
```python
# Test all critical imports
python -c "
from src.scrapers.rap_scraper_optimized import main
from src.analyzers.multi_model_analyzer import analyze_song  
from src.models.models import Song, SpotifyTrack
from src.utils.config import DB_PATH
print('✅ All imports successful')
"
```

### Database Path Verification
```python
# Verify database accessibility
python -c "
from src.utils.config import DB_PATH
import sqlite3
conn = sqlite3.connect(DB_PATH)
cursor = conn.execute('SELECT COUNT(*) FROM songs')
count = cursor.fetchone()[0]
print(f'✅ Database accessible: {count} songs')
conn.close()
"
```

### API Functionality Check
```bash
# Test основных API endpoints
python -c "
from src.scrapers.rap_scraper_optimized import test_genius_connection
from src.enhancers.spotify_enhancer import test_spotify_connection  
test_genius_connection()
test_spotify_connection()
print('✅ API connections working')
"
```

---

## 🚨 Rollback Plan

### Quick Rollback (если что-то пошло не так)
```bash
# 1. Вернуться к предыдущему коммиту
git reset --hard v1.0-flat-structure

# 2. Восстановить базу данных из backup
cp data_backup_YYYYMMDD_HHMMSS.db rap_lyrics.db

# 3. Проверить работоспособность
python rap_scraper_optimized.py --help
```

### Частичный откат (если только некоторые компоненты сломались)
```bash
# Откатить конкретный файл
git checkout HEAD~1 -- src/scrapers/rap_scraper_optimized.py

# Временно использовать legacy wrappers
cp scripts/legacy/rap_scraper_optimized.py ./
```

---

## 🎯 Post-Migration Improvements

### New Features to Implement After Migration

#### 1. Centralized Configuration
```python
# src/utils/config.py enhancements
- Environment-based configs (dev/prod)
- Validation для all API keys
- Dynamic path resolution
- Logging configuration
```

#### 2. Improved Error Handling
```python
# src/utils/exceptions.py (NEW)
class ScraperError(Exception): pass
class AnalysisError(Exception): pass
class DatabaseError(Exception): pass
```

#### 3. Better Testing Infrastructure  
```python
# tests/conftest.py enhancements
- Test database fixtures
- Mock API responses  
- Integration test helpers
```

#### 4. CLI Interface Improvements
```python
# scripts/cli.py (NEW)
import click

@click.group()
def cli():
    """Rap Scraper Project CLI."""
    pass

@cli.command()
@click.option('--limit', default=100)
def scrape(limit):
    """Run scraping pipeline."""
    # Implementation
```

---

## 📊 Migration Success Metrics

### Before vs After Comparison

| Metric | Before (Flat) | After (Structured) | Improvement |
|--------|---------------|-------------------|-------------|
| **Navigation** | ❌ Hard to find files | ✅ Logical grouping | 🚀 5x faster |
| **Import Clarity** | ❌ Unclear dependencies | ✅ Clear module paths | 📈 Better maintainability |
| **Testing** | ❌ No test structure | ✅ Organized test suite | ✅ Test coverage ready |
| **Scaling** | ❌ Naming conflicts risk | ✅ Namespace isolation | 🔧 Production ready |
| **New Developer** | ❌ 30min+ to understand | ✅ 5min to navigate | 👥 Better DX |

---

## 💡 AI Agent Collaboration Tips

### Working Together During Migration
1. **Step-by-Step Execution**: Выполняем по одному step за раз с validation
2. **Import Tracking**: Ведем список всех updated imports для review  
3. **Test After Each Step**: Не переходим к следующему шагу пока current не работает
4. **Communication**: Сообщаем о любых неожиданных issues немедленно

### AI Agent Responsibilities
- ✅ File movement execution
- ✅ Import statement updates
- ✅ Path resolution fixes
- ✅ Entry point script creation
- ✅ Basic validation testing

### Human Responsibilities  
- ✅ Final validation и approval
- ✅ Business logic verification
- ✅ Performance testing
- ✅ Git commits и documentation

---

**Этот migration guide обеспечивает безопасный переход к production-ready структуре проекта с zero downtime и backward compatibility.** 🚀