# AI Assistant Onboarding Checklist

> **Universal guide для любого AI агента**, начинающего работу с rap-scraper проектом

## 🎯 Quick Start (выбери свой сценарий)

### ⚡ Express Mode (30 секунд)
```bash
# Только базовый контекст
read_file("claude.md")
```
**Когда использовать**: Простые вопросы, quick fixes, понимание общей архитектуры

### 🚀 Standard Mode (2 минуты)  
```bash
# Полный контекст для работы
read_file("claude.md")
read_file("models.py") 
run_terminal("python check_db.py")
```
**Когда использовать**: Разработка новых фич, debugging, analysis tasks

### 🔬 Deep Dive Mode (5 минут)
```bash
# Comprehensive understanding
read_file("claude.md")
read_file("PROJECT_DIARY.md")
read_file("models.py")
semantic_search("main processing pipeline")
file_search("**/*spotify*.py")
get_changed_files()
run_terminal("python check_db.py --stats")
```
**Когда использовать**: Архитектурные изменения, complex debugging, ML pipeline work

---

## 📖 Context Layers (читать по потребности)

### Layer 1: Project Foundation (MANDATORY)
```yaml
File: claude.md
Purpose: Central project context, architecture, AI workflow
Read When: Always (first thing to read)
Key Info: ML goals, 47K tracks, Spotify+Genius APIs, Case 13 status
```

### Layer 2: Data Models (HIGH PRIORITY)
```yaml  
File: models.py
Purpose: Pydantic schemas, API contracts, data validation
Read When: Working with data structures, API integration
Key Info: SpotifyTrack, SpotifyArtist, AudioFeatures models
```

### Layer 3: Project History (CONTEXT DEPENDENT)
```yaml
File: PROJECT_DIARY.md  
Purpose: Complete evolution history, 13 documented cases
Read When: Understanding decisions, avoiding past mistakes
Key Info: STAR methodology, technical debt, lessons learned
```

### Layer 4: Implementation Details (AS NEEDED)
```yaml
Files: spotify_enhancer.py, rap_scraper_optimized.py, etc.
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
    read_file("claude.md")                    # Architecture understanding
    read_file("models.py")                    # Data contracts
    semantic_search("error handling patterns") # Code quality standards
    grep_search("TODO|FIXME", isRegexp=True)  # Known issues
    run_terminal("python -m pytest tests/")   # Test status
```

### 🔍 Debugging Agent  
```python
# Primary tasks: Issue investigation, error resolution
def debugging_onboarding():
    read_file("claude.md")                              # Context
    grep_search("error|exception|failed", isRegexp=True) # Error patterns
    get_terminal_output("last_run")                     # Recent failures
    run_terminal("python check_db.py --diagnose")      # System health
    list_code_usages("problematic_function")           # Usage analysis
```

### 📊 ML/Analysis Agent
```python  
# Primary tasks: Data analysis, feature engineering, ML pipeline
def ml_onboarding():
    read_file("claude.md")                    # ML objectives
    read_file("models.py")                    # Feature schemas
    file_search("**/*analyzer*.py")          # Existing ML code
    run_terminal("python check_db.py --ml-stats") # Dataset overview
    semantic_search("conditional generation") # ML architecture
```

### 📝 Documentation Agent
```python
# Primary tasks: Documentation updates, technical writing  
def docs_onboarding():
    read_file("claude.md")           # Current docs state
    read_file("PROJECT_DIARY.md")    # History to document
    get_changed_files()              # Recent changes
    grep_search("TODO.*doc", isRegexp=True) # Doc gaps
    file_search("**/*.md")           # All documentation
```

---

## 🎯 Task-Specific Quick Commands

### New Feature Development
```bash
# 1. Understand existing patterns
semantic_search("similar feature functionality")
list_code_usages("BaseClass")

# 2. Check dependencies  
grep_search("import.*spotify", isRegexp=True)
read_file("requirements.txt")

# 3. Plan implementation
grep_search("def.*process", isRegexp=True)  # Find processing patterns
```

### Bug Investigation
```bash
# 1. Reproduce issue
run_terminal("python problematic_script.py --debug")

# 2. Find error patterns
grep_search("Exception|Error", isRegexp=True)
get_terminal_output("error_session")

# 3. Locate relevant code
semantic_search("error_symptom description")
list_code_usages("failing_function")
```

### Data Analysis  
```bash
# 1. Current dataset state
run_terminal("python check_db.py --detailed-stats")

# 2. Analysis tools available
file_search("**/*analyzer*.py")
semantic_search("sentiment analysis OR complexity metrics")

# 3. ML pipeline status
grep_search("training.*ready", isRegexp=True)
```

### API Integration Work
```bash
# 1. Existing API patterns
semantic_search("rate limiting implementation")  
list_code_usages("requests.get")

# 2. Authentication setup
read_file(".env.example")  # API keys structure
grep_search("auth.*token", isRegexp=True)

# 3. Error handling patterns
semantic_search("API error handling")
```

---

## 💡 Intelligence Boosters

### Context Shortcuts (copy-paste готовые фразы)

#### Project Summary (elevator pitch):
```
"ML pipeline для conditional rap lyrics generation. 47K треков из Genius+Spotify APIs, 
Python+Pydantic+SQLite architecture, 13 documented evolution cases. 
Цель: artist_style + genre + mood → authentic generated lyrics."
```

#### Current Status (что сейчас происходит):
```
"Case 13 completed: claude.md overhaul + agentic search methodology. 
Active: Bulk Spotify track enrichment (массовое обогащение metadata). 
Next: Audio features extraction + ML feature engineering phase."
```

#### Technical Stack (для понимания технологий):
```
"Python 3.13, Pydantic validation, SQLite persistence, 
async/await patterns, comprehensive rate limiting, 
STAR documentation methodology, test-driven development."
```

### Common Pitfalls (чего избегать)

#### ❌ Don't Do This:
```python
# Игнорировать rate limits
requests.get(url)  # Без delays

# Не валидировать данные  
data = {"field": raw_api_response}  # Без Pydantic

# Забывать документировать
# Крупные изменения без PROJECT_DIARY update
```

#### ✅ Do This Instead:
```python  
# Respectful API usage
await self.rate_limiter.wait()
response = await session.get(url, timeout=30)

# Type-safe data handling
track_data = SpotifyTrack(**api_response)

# Comprehensive documentation  
# Update PROJECT_DIARY.md in STAR format
```

---

## 🔧 Environment Setup Validation

### Prerequisites Check
```bash
# System requirements
python --version  # Should be 3.13+
pip list | grep pydantic  # Core dependencies

# Project setup
ls claude.md PROJECT_DIARY.md models.py  # Key files present
sqlite3 rap_lyrics.db ".tables"  # Database accessible
```

### API Credentials Validation  
```bash
# Check .env setup
cat .env.example  # Template structure
test -f .env && echo "✅ .env exists" || echo "❌ Need to create .env"

# Test API access
python -c "from spotify_enhancer import SpotifyEnhancer; print('✅ Spotify OK')"
python -c "import os; print('✅ Genius key:', 'GENIUS_ACCESS_TOKEN' in os.environ)"
```

### Database Health Check
```bash
# Quick DB validation
python check_db.py                    # Basic stats
python check_db.py --detailed-stats   # Comprehensive overview
python check_db.py --integrity-check  # Data consistency
```

---

## 🚨 Emergency Protocols

### If Agent Seems Lost
```markdown
**RESET PROTOCOL**: 
1. read_file("claude.md") - восстанови основной контекст
2. Объясни твою конкретную задачу в 1-2 предложениях  
3. Выбери подходящий persona workflow выше
4. Следуй step-by-step инструкциям для твоего типа задачи
```

### If Database Issues
```markdown
**DB RECOVERY**:
1. run_terminal("python check_db.py --diagnose")
2. Если corruption: run_terminal("python migrate_database.py --repair") 
3. Если missing tables: auto-created on next script run
4. Backup есть в enhanced_data/ директории
```

### If API Errors
```markdown  
**API TROUBLESHOOTING**:
- 403 Forbidden: Check API keys в .env, verify rate limits
- 429 Too Many Requests: Automatic backoff implemented
- Timeout errors: Network issue, retry механизм активен
- Authentication: run_terminal("python setup_spotify.py --refresh")
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
Project: ML rap lyrics generation (conditional)
Stack: Python 3.13 + Pydantic + SQLite + APIs  
Data: 47K tracks, 259 artists, rich metadata
Status: Case 13 complete, bulk enrichment active
Goal: artist + genre + mood → generated lyrics

Key Files:
  - claude.md: Central context
  - models.py: Data schemas  
  - PROJECT_DIARY.md: Complete history
  - check_db.py: System status

APIs: 
  - Genius: lyrics + metadata (1 req/sec)
  - Spotify: audio features + artist data (OAuth)

Commands:
  - Express: read_file("claude.md")
  - Standard: + models.py + check_db.py  
  - Deep: + PROJECT_DIARY.md + semantic_search

Emergency: Reset with claude.md, define task clearly
```

---

*Created: 2025-08-26 | Version: 2.0 | Next Review: After Case 14*