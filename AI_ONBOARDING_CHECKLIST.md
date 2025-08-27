# AI Assistant Onboarding Checklist

> **Universal guide для любого AI агента**, начинающего работу с rap-scraper проектом

## 🎯 Quick Start (выбери свой сценарий)

### ⚡ Express Mode (30 секунд)
```bash
# Только базовый контекст
read_file("docs/claude.md")
```
**Когда использовать**: Простые вопросы, quick fixes, понимание общей архитектуры

### 🚀 Standard Mode (2 минуты)  
```bash
# Полный контекст для работы
read_file("docs/claude.md")
read_file("src/models/models.py") 
run_terminal("python src/utils/check_db.py")
# ИЛИ используй новый CLI
run_terminal("python scripts/rap_scraper_cli.py status")
```
**Когда использовать**: Разработка новых фич, debugging, analysis tasks

### 🔬 Deep Dive Mode (5 минут)
```bash
# Comprehensive understanding
read_file("docs/claude.md")
read_file("docs/PROJECT_DIARY.md")
read_file("src/models/models.py")
read_file("README.md")  # Новая архитектура и CLI
semantic_search("main processing pipeline")
file_search("src/**/*spotify*.py")
get_changed_files()
run_terminal("python scripts/rap_scraper_cli.py status")
```
**Когда использовать**: Архитектурные изменения, complex debugging, ML pipeline work

---

## 📖 Context Layers (читать по потребности)

### Layer 1: Project Foundation (MANDATORY)
```yaml
File: docs/claude.md
Purpose: Central project context, architecture, AI workflow
Read When: Always (first thing to read)
Key Info: ML goals, 48K+ tracks, Spotify+Genius APIs, restructured architecture
```

### Layer 2: Data Models (HIGH PRIORITY)
```yaml  
File: src/models/models.py
Purpose: Pydantic schemas, API contracts, data validation
Read When: Working with data structures, API integration
Key Info: SpotifyTrack, SpotifyArtist, AudioFeatures models
```

### Layer 3: Project Architecture (NEW - HIGH PRIORITY)
```yaml
File: README.md
Purpose: New CLI interface, structured architecture, production setup
Read When: Using project tools, understanding new organization
Key Info: scripts/rap_scraper_cli.py, src/ structure, entry points
```

### Layer 4: Project History (CONTEXT DEPENDENT)
```yaml
File: docs/PROJECT_DIARY.md  
Purpose: Complete evolution history, 13+ documented cases
Read When: Understanding decisions, avoiding past mistakes
Key Info: STAR methodology, technical debt, lessons learned
```

### Layer 5: Implementation Details (AS NEEDED)
```yaml
Files: src/enhancers/spotify_enhancer.py, src/scrapers/rap_scraper_optimized.py, etc.
Purpose: Specific implementation patterns
Read When: Modifying existing code, understanding algorithms
Key Info: Rate limiting patterns, error handling, batch processing
```

---

## 🤖 AI Agent Personas & Workflows

### 👨‍💻 Developer Agent
```python
# Primary tasks: Code changes, new features, refactoring
def developer_onboarding():
    read_file("docs/claude.md")                    # Architecture understanding
    read_file("src/models/models.py")              # Data contracts
    read_file("README.md")                         # CLI and new structure
    semantic_search("error handling patterns")     # Code quality standards
    grep_search("TODO|FIXME", isRegexp=True)      # Known issues
    run_terminal("python -m pytest tests/")       # Test status
```

### 🔍 Debugging Agent  
```python
# Primary tasks: Issue investigation, error resolution
def debugging_onboarding():
    read_file("docs/claude.md")                              # Context
    read_file("README.md")                                   # Current tools
    grep_search("error|exception|failed", isRegexp=True)    # Error patterns
    get_terminal_output("last_run")                         # Recent failures
    run_terminal("python scripts/rap_scraper_cli.py status") # System health
    list_code_usages("problematic_function")               # Usage analysis
```

### 📊 ML/Analysis Agent
```python  
# Primary tasks: Data analysis, feature engineering, ML pipeline
def ml_onboarding():
    read_file("docs/claude.md")                         # ML objectives
    read_file("src/models/models.py")                   # Feature schemas
    file_search("src/analyzers/**/*.py")               # Existing ML code
    run_terminal("python scripts/rap_scraper_cli.py status") # Dataset overview
    semantic_search("conditional generation")          # ML architecture
```

### 📝 Documentation Agent
```python
# Primary tasks: Documentation updates, technical writing  
def docs_onboarding():
    read_file("docs/claude.md")           # Current docs state
    read_file("docs/PROJECT_DIARY.md")    # History to document
    read_file("README.md")                # Current documentation
    get_changed_files()                   # Recent changes
    grep_search("TODO.*doc", isRegexp=True) # Doc gaps
    file_search("docs/**/*.md")           # All documentation
```

---

## 🎯 Task-Specific Quick Commands

### New Feature Development
```bash
# 1. Understand existing patterns
semantic_search("similar feature functionality")
list_code_usages("BaseClass")
read_file("README.md")  # Check CLI structure

# 2. Check dependencies  
grep_search("import.*spotify", isRegexp=True)
read_file("requirements.txt")

# 3. Plan implementation
grep_search("def.*process", isRegexp=True)  # Find processing patterns
file_search("src/**/*.py")  # Browse new structure
```

### Bug Investigation
```bash
# 1. Reproduce issue using CLI
run_terminal("python scripts/rap_scraper_cli.py status")
run_terminal("python scripts/legacy/problematic_script.py --debug")

# 2. Find error patterns
grep_search("Exception|Error", isRegexp=True)
get_terminal_output("error_session")

# 3. Locate relevant code in new structure
semantic_search("error_symptom description")
list_code_usages("failing_function")
file_search("src/**/*relevant*.py")
```

### Data Analysis  
```bash
# 1. Current dataset state using new CLI
run_terminal("python scripts/rap_scraper_cli.py status")
run_terminal("python scripts/rap_scraper_cli.py monitoring --component database")

# 2. Analysis tools available
file_search("src/analyzers/**/*.py")
semantic_search("sentiment analysis OR complexity metrics")

# 3. ML pipeline status
grep_search("training.*ready", isRegexp=True)
```

### API Integration Work
```bash
# 1. Existing API patterns in new structure
semantic_search("rate limiting implementation")  
list_code_usages("requests.get")
file_search("src/enhancers/**/*.py")

# 2. Authentication setup
read_file(".env.example")  # API keys structure
grep_search("auth.*token", isRegexp=True)

# 3. Error handling patterns
semantic_search("API error handling")
file_search("src/utils/**/*.py")
```

---

## 💡 Intelligence Boosters

### Context Shortcuts (copy-paste готовые фразы)

#### Project Summary (elevator pitch):
```
"Production-ready ML pipeline для conditional rap lyrics generation. 48K+ треков из Genius+Spotify APIs, 
новая структурированная архитектура src/, единый CLI интерфейс, Python+Pydantic+SQLite stack. 
Цель: artist_style + genre + mood → authentic generated lyrics."
```

#### Current Status (что сейчас происходит):
```
"Проект полностью реструктурирован: новая архитектура src/, CLI интерфейс, обновленная документация.
Active: Gemma 27B анализ (11,860+ треков проанализировано), Spotify обогащение завершено (262/263).
База: 48,508 песен, 265 артистов. Next: Завершение AI анализа + ML feature engineering."
```

#### Technical Stack (для понимания технологий):
```
"Python 3.13, новая структура src/{scrapers,enhancers,analyzers,models,utils}/, 
Pydantic validation, SQLite persistence, CLI интерфейс scripts/rap_scraper_cli.py,
async/await patterns, comprehensive rate limiting, production-ready architecture."
```

### Common Pitfalls (чего избегать)

#### ❌ Don't Do This:
```python
# Использовать старые пути
from models import SpotifyTrack  # Устарело!
python check_db.py              # Используй CLI!

# Игнорировать rate limits
requests.get(url)  # Без delays

# Не валидировать данные  
data = {"field": raw_api_response}  # Без Pydantic

# Забывать новую архитектуру
# Прямые вызовы вместо CLI интерфейса
```

#### ✅ Do This Instead:
```python  
# Новая структурированная архитектура
from src.models.models import SpotifyTrack
python scripts/rap_scraper_cli.py status  # Используй CLI

# Respectful API usage
await self.rate_limiter.wait()
response = await session.get(url, timeout=30)

# Type-safe data handling
track_data = SpotifyTrack(**api_response)

# Используй новые entry points
python scripts/rap_scraper_cli.py analysis --analyzer gemma
```

---

## 🔧 Environment Setup Validation

### Prerequisites Check
```bash
# System requirements
python --version  # Should be 3.13+
pip list | grep pydantic  # Core dependencies

# Project setup (новые пути!)
ls docs/claude.md docs/PROJECT_DIARY.md src/models/models.py  # Key files present
sqlite3 data/rap_lyrics.db ".tables"  # Database accessible
ls scripts/rap_scraper_cli.py  # CLI interface present
```

### API Credentials Validation  
```bash
# Check .env setup
cat .env.example  # Template structure
test -f .env && echo "✅ .env exists" || echo "❌ Need to create .env"

# Test API access (новые пути)
python -c "from src.enhancers.spotify_enhancer import SpotifyEnhancer; print('✅ Spotify OK')"
python -c "import os; print('✅ Genius key:', 'GENIUS_ACCESS_TOKEN' in os.environ)"
```

### Database Health Check
```bash
# Quick DB validation using new CLI
python scripts/rap_scraper_cli.py status              # Comprehensive overview
python scripts/rap_scraper_cli.py monitoring --component database  # Health check

# Legacy direct access (if needed)
python src/utils/check_db.py                    # Basic stats
```

---

## 🚨 Emergency Protocols

### If Agent Seems Lost
```markdown
**RESET PROTOCOL**: 
1. read_file("docs/claude.md") - восстанови основной контекст
2. read_file("README.md") - изучи новую CLI архитектуру
3. Объясни твою конкретную задачу в 1-2 предложениях  
4. Выбери подходящий persona workflow выше
5. Следуй step-by-step инструкциям для твоего типа задачи
```

### If Database Issues
```markdown
**DB RECOVERY**:
1. run_terminal("python scripts/rap_scraper_cli.py status")
2. run_terminal("python src/utils/check_db.py --diagnose")  # Direct access
3. Если corruption: run_terminal("python src/utils/migrate_database.py --repair") 
4. Если missing tables: auto-created on next script run
5. Backup есть в data/backups/ директории
```

### If API Errors
```markdown  
**API TROUBLESHOOTING**:
- 403 Forbidden: Check API keys в .env, verify rate limits
- 429 Too Many Requests: Automatic backoff implemented
- Timeout errors: Network issue, retry механизм активен
- Authentication: run_terminal("python src/utils/setup_spotify.py --refresh")
```

### If CLI Issues
```markdown
**CLI TROUBLESHOOTING**:
- Command not found: Убедись что используешь python scripts/rap_scraper_cli.py
- Import errors: Проверь что все файлы в src/ структуре
- Path issues: Используй CLI вместо прямых вызовов
- Legacy scripts: Доступны в scripts/legacy/ если нужно
```

### If Performance Issues
```markdown
**PERFORMANCE DEBUG**:
1. Check available memory: run_terminal("python -c 'import psutil; print(psutil.virtual_memory())'")
2. Database size: run_terminal("du -h rap_lyrics.db")  
3. Enable batch processing: Use --batch-size parameter
4. Monitor progress: All scripts show progress bars
```

---

## 📊 Success Metrics & Validation

### Knowledge Acquisition Checklist
- [ ] **Project Goal Clear**: Понимаю цель conditional lyrics generation
- [ ] **Architecture Mapped**: Знаю основные компоненты pipeline  
- [ ] **Data Flow Understood**: Genius → Enhancement → Analysis → ML
- [ ] **Current State Known**: Case 13 status, что работает/что нет
- [ ] **My Role Defined**: Понимаю конкретную задачу в контексте проекта

### Technical Readiness Checklist  
- [ ] **Environment Setup**: Python 3.13+, dependencies installed
- [ ] **Database Access**: Can run check_db.py successfully
- [ ] **API Understanding**: Know rate limits, error patterns  
- [ ] **Code Patterns**: Familiar с Pydantic, async/await, error handling
- [ ] **Testing Approach**: Know how to run tests, validate changes

### Communication Readiness Checklist
- [ ] **Context Layering**: Умею выбирать правильный уровень детализации  
- [ ] **Problem Scoping**: Могу четко сформулировать техническую задачу
- [ ] **Solution Planning**: Планирую before executing, валидирую с пользователем
- [ ] **Documentation Style**: Следую STAR methodology для updates
- [ ] **Error Reporting**: Умею diagnostic information и reproduction steps

---

## 🔄 Maintenance & Updates

### When to Update This Checklist
- [ ] New major features added (new Case in PROJECT_DIARY)
- [ ] Architecture changes (new APIs, different data models)  
- [ ] Tool changes (new testing framework, different deployment)
- [ ] Common issues pattern changes (new error types, solutions)

### Update Procedure
```bash
# 1. Document changes
echo "Updated: $(date)" >> AI_ONBOARDING_CHECKLIST.md

# 2. Test with fresh agent session
# Follow your own checklist completely

# 3. Validate all copy-paste commands work
# Every code block should execute successfully

# 4. Cross-reference with claude.md
# Ensure consistency between documents
```

---

## 📚 Quick Reference Card

```yaml
Project: Production-ready ML rap lyrics generation (conditional)
Stack: Python 3.13 + Pydantic + SQLite + APIs + CLI
Architecture: src/{scrapers,enhancers,analyzers,models,utils}/ + scripts/
Data: 48.5K+ tracks, 265 artists, rich metadata, 11.8K+ AI analyses
Status: Реструктуризация завершена, Gemma анализ активен (прогресс: 24.4%)
Goal: artist + genre + mood → generated lyrics

Key Files:
  - docs/claude.md: Central context
  - src/models/models.py: Data schemas  
  - README.md: CLI interface & new architecture
  - docs/PROJECT_DIARY.md: Complete history

APIs: 
  - Genius: lyrics + metadata (1 req/sec)
  - Spotify: audio features + artist data (OAuth)

CLI Commands:
  - Status: python scripts/rap_scraper_cli.py status
  - Scraping: python scripts/rap_scraper_cli.py scraping
  - Analysis: python scripts/rap_scraper_cli.py analysis --analyzer gemma
  - Monitoring: python scripts/rap_scraper_cli.py monitoring --component all

Entry Modes:
  - Express: read_file("docs/claude.md")
  - Standard: + src/models/models.py + CLI status  
  - Deep: + README.md + docs/PROJECT_DIARY.md

Emergency: Reset with docs/claude.md + README.md, define task clearly
```

---

*Created: 2025-08-26 | Version: 3.0 - Post-Restructure | Updated: 2025-08-26 | Next Review: После завершения Gemma анализа*