# AI Assistant Onboarding Checklist

> **Universal guide для любого AI агента**, начинающего работу с rap-scraper проектом

## 🎯 Quick Start (выбери свой сценарий)

### ⚡ Express Mode (30 секунд)
```bash
# Только базовый контекст
read_file("docs/claude.md")  # Обновлен после cleanup!
```
**Когда использовать**: Простые вопросы, quick fixes, понимание новой организованной архитектуры

### 🚀 Standard Mode (2 минуты)  
```bash
# Полный контекст для работы с обновленной структурой
read_file("docs/claude.md")                                 # Обновленная архитектура
read_file("src/models/models.py")                           # Data schemas
run_terminal("python scripts/rap_scraper_cli.py status")    # Unified CLI interface
list_dir("scripts/")                                        # Новая организация
```
**Когда использовать**: Разработка, debugging, работа с новой структурой проекта

### 🔬 Deep Dive Mode (5 минут)
```bash
# Comprehensive understanding после cleanup
read_file("docs/claude.md")                                 # Обновленный контекст
read_file("docs/PROJECT_DIARY.md")                          # История включая cleanup
read_file("src/models/models.py")                           # Data models
list_dir("scripts/tools/")                                  # Production utilities
list_dir("scripts/development/")                            # Development tools  
semantic_search("batch processing OR multi-model analysis") # Core functionality
run_terminal("python scripts/rap_scraper_cli.py status")    # Current state
```
**Когда использовать**: Архитектурные изменения, complex debugging, работа с новыми инструментами

---

## 📖 Context Layers (читать по потребности)

### Layer 1: Project Foundation (MANDATORY - UPDATED!)
```yaml
File: docs/claude.md
Purpose: Central project context, architecture, post-cleanup workflow
Read When: Always (first thing to read - recently updated!)
Key Info: 53K+ tracks, organized structure, CLI-first approach, tools reorganization
```

### Layer 2: Data Models (HIGH PRIORITY)
```yaml  
File: src/models/models.py
Purpose: Pydantic schemas, API contracts, data validation
Read When: Working with data structures, API integration
Key Info: SpotifyTrack, SpotifyArtist, AudioFeatures models
```

### Layer 3: New Project Organization (NEW - HIGH PRIORITY)
```yaml
Structure: scripts/{tools,development,legacy}/, src/{scrapers,enhancers,analyzers}/
Purpose: Clean organized architecture, production tools, development workflow
Read When: Using any project tools, understanding new organization
Key Info: Unified CLI, batch processing tools, development utilities
```

### Layer 4: Project History (CONTEXT DEPENDENT)
```yaml
File: docs/PROJECT_DIARY.md  
Purpose: Complete evolution history, includes recent cleanup case
Read When: Understanding decisions, avoiding past mistakes, learning from cleanup
Key Info: STAR methodology, cleanup decisions, why scripts were moved/removed
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
    read_file("docs/claude.md")                         # Updated architecture
    read_file("src/models/models.py")                   # Data contracts
    list_dir("scripts/tools/")                          # Production utilities
    list_dir("scripts/development/")                    # Development tools
    semantic_search("error handling patterns")         # Code quality standards
    grep_search("TODO|FIXME", isRegexp=True)          # Known issues
    run_terminal("python -m pytest tests/")           # Test status
    run_terminal("python scripts/rap_scraper_cli.py status")  # System health
```

### 🔍 Debugging Agent  
```python
# Primary tasks: Issue investigation, error resolution
def debugging_onboarding():
    read_file("docs/claude.md")                                    # Updated context
    list_dir("scripts/tools/")                                     # Available tools
    grep_search("error|exception|failed", isRegexp=True)          # Error patterns
    get_terminal_output("last_run")                               # Recent failures
    run_terminal("python scripts/rap_scraper_cli.py status")      # System health
    run_terminal("python scripts/tools/check_spotify_coverage.py") # Diagnostics
    list_code_usages("problematic_function")                     # Usage analysis
```

### 📊 ML/Analysis Agent
```python  
# Primary tasks: Data analysis, feature engineering, ML pipeline
def ml_onboarding():
    read_file("docs/claude.md")                                    # Updated ML objectives
    read_file("src/models/models.py")                              # Feature schemas
    file_search("src/analyzers/**/*.py")                          # Existing ML code
    run_terminal("python scripts/rap_scraper_cli.py status")      # Dataset overview
    run_terminal("python scripts/tools/batch_ai_analysis.py --dry-run")  # Batch tools
    semantic_search("multi-model analysis OR gemma")              # Current ML stack
```

### 📝 Documentation Agent
```python
# Primary tasks: Documentation updates, technical writing  
def docs_onboarding():
    read_file("docs/claude.md")                        # Updated docs state
    read_file("docs/PROJECT_DIARY.md")                 # History including cleanup
    list_dir("scripts/tools/")                         # New tools to document
    list_dir("scripts/development/")                   # Development workflow
    get_changed_files()                                # Recent changes
    grep_search("TODO.*doc", isRegexp=True)           # Doc gaps
    file_search("docs/**/*.md")                       # All documentation
    read_file("scripts/tools/README.md")              # New tools documentation
```

---

## 🎯 Task-Specific Quick Commands

### New Feature Development
```bash
# 1. Understand organized structure
list_dir("scripts/tools/")                              # Production utilities
list_dir("scripts/development/")                        # Development tools
read_file("docs/claude.md")                            # Updated architecture
semantic_search("similar feature functionality")

# 2. Check dependencies and patterns
grep_search("import.*spotify", isRegexp=True)
list_code_usages("BaseClass") 
file_search("src/**/*.py")                             # Browse organized structure

# 3. Plan implementation using existing tools
run_terminal("python scripts/rap_scraper_cli.py --help")  # Available commands
grep_search("def.*process", isRegexp=True)              # Find processing patterns
```

### Bug Investigation
```bash
# 1. Use unified CLI for diagnosis
run_terminal("python scripts/rap_scraper_cli.py status")
run_terminal("python scripts/tools/check_spotify_coverage.py")  # Coverage diagnostics
run_terminal("python scripts/rap_scraper_cli.py monitoring --component analysis")

# 2. Find error patterns in organized code
grep_search("Exception|Error", isRegexp=True)
get_terminal_output("error_session")
semantic_search("error_symptom description")

# 3. Locate relevant code using new structure
list_code_usages("failing_function")
file_search("src/**/*relevant*.py")                     # Organized src structure
```

### Data Analysis  
```bash
# 1. Current dataset state using unified CLI
run_terminal("python scripts/rap_scraper_cli.py status")
run_terminal("python scripts/rap_scraper_cli.py monitoring --component database")
run_terminal("python scripts/tools/batch_ai_analysis.py --dry-run")  # Batch processing

# 2. Analysis tools in organized structure
file_search("src/analyzers/**/*.py")
run_terminal("python src/analyzers/create_visual_analysis.py")  # Portfolio dashboard
semantic_search("multi-model analysis OR gemma")

# 3. ML pipeline status  
grep_search("training.*ready", isRegexp=True)
run_terminal("python scripts/rap_scraper_cli.py analysis --analyzer multi")
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
"Production-ready ML pipeline для conditional rap lyrics generation. 53K+ треков из Genius+Spotify APIs, 
новая организованная архитектура scripts/{tools,development,legacy}/ + src/, единый CLI интерфейс, 
Python+Pydantic+SQLite stack. Цель: artist_style + genre + mood → authentic generated lyrics.
Недавно: проект полностью реорганизован, удалены избыточные скрипты, добавлены production инструменты."
```

#### Current Status (что сейчас происходит):
```
"Проект полностью реорганизован (авг 2025): новая структура scripts/{tools,development,legacy}/, 
унифицированный CLI интерфейс, удалено 5+ избыточных скриптов, реабилитированы batch processing инструменты.
Active: 38K+ треков ожидают AI анализа, Spotify обогащение завершено (~99.6%).
База: 53,300+ песен, 328 артистов. Next: Завершение AI анализа через batch_ai_analysis.py"
```

#### Technical Stack (для понимания технологий):
```
"Python 3.13, организованная структура: scripts/{tools,development,legacy}/ + src/{scrapers,enhancers,analyzers,models,utils}/, 
Pydantic validation, SQLite persistence, единый CLI scripts/rap_scraper_cli.py,
async/await patterns, comprehensive rate limiting, production batch processing tools,
rehabilitated batch_ai_analysis.py for large-scale operations."
```

### Common Pitfalls (чего избегать)

#### ❌ Don't Do This:
```python
# Использовать старые/удаленные пути
from models import SpotifyTrack              # Устарело!
python project_status.py                    # Удален! Используй CLI!
python enhanced_scraper.py                  # Удален как избыточный!

# Игнорировать новую организацию
python scripts/some_random_script.py        # Проверь scripts/{tools,development,legacy}/

# Забывать про batch processing
# Для большой обработки используй scripts/tools/batch_ai_analysis.py

# Пропускать unified CLI
python src/scrapers/rap_scraper_optimized.py  # Используй CLI интерфейс!
```

#### ✅ Do This Instead:
```python  
# Новая организованная архитектура
from src.models.models import SpotifyTrack
python scripts/rap_scraper_cli.py status                    # Unified CLI

# Используй правильные инструменты из организованной структуры
python scripts/tools/batch_ai_analysis.py --batch-size 25   # Production batching
python scripts/tools/check_spotify_coverage.py             # Diagnostics
python scripts/development/test_fixed_scraper.py           # Development testing

# Respectful API usage (unchanged)
await self.rate_limiter.wait()
response = await session.get(url, timeout=30)

# Type-safe data handling (unchanged)
track_data = SpotifyTrack(**api_response)

# Используй organized entry points
python scripts/rap_scraper_cli.py analysis --analyzer gemma
python scripts/rap_scraper_cli.py monitoring --component analysis
```

---

## 🔧 Environment Setup Validation

### Prerequisites Check
```bash
# System requirements
python --version  # Should be 3.13+
pip list | grep pydantic  # Core dependencies

### Project setup (обновленная структура!)
```bash
# System requirements
python --version  # Should be 3.13+
pip list | grep pydantic  # Core dependencies

# Organized project structure validation
ls docs/claude.md docs/PROJECT_DIARY.md src/models/models.py     # Key files present
ls scripts/rap_scraper_cli.py                                   # Unified CLI
ls scripts/tools/batch_ai_analysis.py scripts/tools/check_spotify_coverage.py  # Production tools
ls scripts/development/test_fixed_scraper.py                    # Development tools
sqlite3 data/rap_lyrics.db ".tables"                           # Database accessible
```
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
# Unified CLI health check (preferred)
python scripts/rap_scraper_cli.py status                           # Comprehensive overview
python scripts/rap_scraper_cli.py monitoring --component database  # Health check
python scripts/tools/check_spotify_coverage.py                     # Coverage analysis

# Legacy direct access (if needed)
python src/utils/check_db.py                                       # Basic stats  
```

---

## 🚨 Emergency Protocols

### If Agent Seems Lost
```markdown
**RESET PROTOCOL**: 
1. read_file("docs/claude.md") - восстанови обновленный контекст
2. list_dir("scripts/") - изучи новую организованную структуру
3. Объясни твою конкретную задачу в 1-2 предложениях  
4. Выбери подходящий persona workflow выше
5. Используй scripts/tools/ для production, scripts/development/ для testing
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
- Import errors: Проверь что все файлы в организованной src/ структуре
- Path issues: Используй CLI вместо прямых вызовов удаленных скриптов
- Legacy scripts: Доступны в scripts/legacy/ если нужно
- Production tools: Используй scripts/tools/ для batch operations
```

### If Performance Issues
```markdown
**PERFORMANCE DEBUG**:
1. Check system resources: run_terminal("python -c 'import psutil; print(psutil.virtual_memory())'")
2. Database size: run_terminal("python scripts/rap_scraper_cli.py status")  
3. Use batch processing: run_terminal("python scripts/tools/batch_ai_analysis.py --batch-size 25 --dry-run")
4. Monitor progress: run_terminal("python scripts/rap_scraper_cli.py monitoring --component analysis")
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
Stack: Python 3.13 + Pydantic + SQLite + APIs + Unified CLI
Architecture: scripts/{tools,development,legacy}/ + src/{scrapers,enhancers,analyzers,models,utils}/
Data: 53K+ tracks, 328 artists, rich metadata, 38K+ pending AI analysis
Status: Полная реорганизация завершена (авг 2025), batch processing tools готовы
Goal: artist + genre + mood → generated lyrics

Key Files:
  - docs/claude.md: Central context (UPDATED!)
  - src/models/models.py: Data schemas  
  - scripts/rap_scraper_cli.py: Unified CLI interface
  - docs/PROJECT_DIARY.md: Complete history + cleanup case

Organized Structure:
  - scripts/tools/: batch_ai_analysis.py, check_spotify_coverage.py
  - scripts/development/: test_fixed_scraper.py, scrape_artist_one.py
  - scripts/legacy/: backward compatibility
  - src/: core modules (scrapers, enhancers, analyzers, models)

APIs: 
  - Genius: lyrics + metadata (1 req/sec)
  - Spotify: audio features + artist data (OAuth)

CLI Commands:
  - Status: python scripts/rap_scraper_cli.py status
  - Batch AI: python scripts/tools/batch_ai_analysis.py --batch-size 25
  - Coverage: python scripts/tools/check_spotify_coverage.py
  - Analysis: python scripts/rap_scraper_cli.py analysis --analyzer gemma
  - Monitoring: python scripts/rap_scraper_cli.py monitoring --component all

Entry Modes:
  - Express: read_file("docs/claude.md")
  - Standard: + list_dir("scripts/") + CLI status  
  - Deep: + PROJECT_DIARY.md + tools exploration

Emergency: Reset with docs/claude.md + list organized structure
```

---

*Created: 2025-08-26 | Version: 4.0 - Post-Cleanup | Updated: 2025-08-27 | Next Review: После завершения batch AI анализа*