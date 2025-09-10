# AI Assistant Onboarding Checklist (PostgreSQL Migration Complete)

> **Universal guide для любого AI агента**, начинающего работу с PostgreSQL-powered rap scraper проектом

## 🎯 Quick Start (выбери свой сценарий)

### ⚡ Express Mode (30 секунд)
```bash
# Только базовый контекст после PostgreSQL миграции
read_file("docs/claude.md")  # Полностью обновлен для PostgreSQL архитектуры!
run_terminal("python scripts/tools/database_diagnostics.py --quick")  # MAIN diagnostic tool
```
**Когда использовать**: Простые вопросы, quick fixes, понимание PostgreSQL архитектуры

### 🚀 Standard Mode (2 минуты)  
```bash
# Полный контекст PostgreSQL ecosystem
read_file("docs/claude.md")                    # PostgreSQL-focused architecture  
read_file("src/utils/config.py", limit=30)     # get_db_config function (NEW!)
read_file(".env")                              # PostgreSQL configuration
run_terminal("python scripts/tools/database_diagnostics.py")  # MAIN diagnostic tool
run_terminal("python scripts/mass_qwen_analysis.py --test")  # Analysis test
list_dir("scripts/")                          # PostgreSQL-compatible scripts
```
**Когда использовать**: Разработка, debugging, работа с PostgreSQL системой

### 🔬 Deep Dive Mode (5 минут)
```bash
# Comprehensive understanding PostgreSQL ecosystem
read_file("docs/claude.md")                           # Полный PostgreSQL контекст
read_file("src/utils/config.py")                      # get_db_config function (UPDATED!)
read_file("scripts/tools/database_diagnostics.py", limit=50)  # MAIN diagnostic tool
read_file("scripts/mass_qwen_analysis.py", limit=50)  # Main analysis script
run_terminal("python scripts/tools/database_diagnostics.py")  # Full diagnostics
run_terminal("python scripts/tools/database_diagnostics.py --analysis")  # AI analysis status
semantic_search("postgresql OR concurrent OR diagnostics") # PostgreSQL functionality
```
**Когда использовать**: Архитектурные изменения, complex debugging, concurrent processing

---

## 📖 Context Layers (читать по потребности)

### Layer 1: PostgreSQL Architecture Foundation (MANDATORY - UPDATED!)
```yaml
File: docs/claude.md
Purpose: Central PostgreSQL context, concurrent processing, migration achievements
Read When: Always (first thing to read - completely rewritten for PostgreSQL!)
Key Info: 57,718 tracks, PostgreSQL 15, concurrent script execution, migration complete
```

### Layer 2: Database Configuration (HIGH PRIORITY - UPDATED!)
```yaml  
File: src/utils/config.py
Purpose: get_db_config() function, PostgreSQL connection management, environment variables
Read When: Connection issues, credentials problems, database setup
Key Info: POSTGRES_* and DB_* environment variables, fallback authentication, connection parameters
```

### Layer 3: PostgreSQL Configuration (HIGH PRIORITY - NEW!)
```yaml
File: .env
Purpose: PostgreSQL connection credentials, API keys, system settings
Read When: Connection issues, credentials problems, configuration changes
Key Info: PostgreSQL credentials, Novita/Qwen API, Genius/Spotify keys, pool settings
```

### Layer 4: Analysis Scripts (HIGH PRIORITY - UPDATED!)
```yaml
Files: scripts/mass_qwen_analysis.py, scripts/db_browser.py
Purpose: PostgreSQL-compatible analysis and database tools
Read When: Running analysis, database exploration, concurrent processing
Key Info: No confirmation prompts, PostgreSQL queries, concurrent-safe operations
```

### Layer 5: Database Diagnostics (HIGH PRIORITY - UPDATED!)
```yaml
File: scripts/tools/database_diagnostics.py
Purpose: MAIN PostgreSQL diagnostic tool - health, statistics, schema, analysis status
Read When: Database issues, analysis problems, unanalyzed tracks, schema verification
Key Info: --quick, --analysis, --unanalyzed, --schema flags, comprehensive PostgreSQL diagnostics
```

### Layer 6: AI Project Tools (NEW!)
```yaml
Files: scripts/tools/ai_context_manager.py, scripts/tools/ai_project_analyzer.py
Purpose: Интеллектуальный анализ архитектуры, управление контекстом для AI
Read When: Аудит архитектуры, автоматизация AI-навигации, поиск дубликатов и legacy
Key Info: AST-парсинг, приоритезация файлов, workspace генерация, метрики для микросервисов
```

### Layer 7: Legacy Compatibility (AS NEEDED - ARCHIVED)
```yaml
Files: scripts/archive/, data/data_backup_*.db
Purpose: SQLite legacy system, backup data
Read When: Emergency fallback, reference for old functionality
Key Info: SQLite backup scripts, historical data, migration rollback
```

---

## 🤖 AI Agent Personas & Workflows

### 👨‍💻 Database Developer Agent (Updated for PostgreSQL)
```python
# Primary tasks: PostgreSQL development, concurrent processing, database optimization
def database_developer_onboarding():
    read_file("docs/claude.md")                         # PostgreSQL architecture
    read_file("src/database/postgres_adapter.py")       # Database layer
    read_file(".env")                                   # Connection configuration
    run_terminal("python check_stats.py")              # Database health
    run_terminal("psql -h localhost -U rap_user -d rap_lyrics -c 'SELECT version();'")  # Direct test
    semantic_search("postgresql OR connection OR pool")  # Database patterns
    run_terminal("python scripts/mass_qwen_analysis.py --test")  # PostgreSQL compatibility
```

### 🔍 Debugging Agent (Enhanced for PostgreSQL)
```python
# Primary tasks: PostgreSQL debugging, connection issues, concurrent problems
def postgresql_debugging_onboarding():
    read_file("docs/claude.md")                         # PostgreSQL context
    run_terminal("python check_stats.py")              # Database statistics
    read_file("src/database/postgres_adapter.py", limit=50)  # Connection layer
    grep_search("PostgreSQLManager|asyncpg", isRegexp=True)  # PostgreSQL usage
    get_terminal_output("last_run")                     # Recent failures
    semantic_search("error OR connection OR timeout")   # PostgreSQL errors
    run_terminal("python scripts/db_browser.py")       # Interactive testing
```

### 📊 Analysis Agent (Modernized for PostgreSQL + Qwen)
```python  
# Primary tasks: Mass analysis, Qwen integration, concurrent processing
def analysis_onboarding():
    read_file("docs/claude.md")                                # PostgreSQL + analysis context
    run_terminal("python check_stats.py")                     # Current statistics
    read_file("scripts/mass_qwen_analysis.py", limit=50)      # Main analysis script
    run_terminal("python scripts/mass_qwen_analysis.py --test")  # Quick validation
    semantic_search("qwen OR analysis OR concurrent")         # Analysis patterns
    run_terminal("python check_overlap.py")                   # Overlap analysis
```

### 🚀 Migration Agent (New for PostgreSQL)
```python
# Primary tasks: Database migration, data integrity, performance optimization
def migration_onboarding():
    read_file("docs/claude.md")                        # Migration achievements
    read_file("scripts/migrate_to_postgresql.py", limit=50)  # Migration pipeline
    run_terminal("python check_stats.py")             # Migration verification
    run_terminal("python check_overlap.py")           # Data consistency
    semantic_search("migration OR integrity")         # Migration patterns
    list_dir("scripts/archive/")                      # Legacy system reference
```

### 📝 Documentation Agent (Updated for PostgreSQL)
```python
# Primary tasks: PostgreSQL documentation, migration guides, concurrent processing
def docs_onboarding():
    read_file("docs/claude.md")                        # Current PostgreSQL state
    read_file("README.md", limit=100)                  # Project overview (needs update)
    read_file("docs/PROGRESS.md", limit=50)            # Progress tracking (needs update)
    list_dir("docs/")                                 # All documentation
    get_changed_files()                               # Recent PostgreSQL changes
    grep_search("TODO.*postgresql", isRegexp=True)    # PostgreSQL documentation gaps
```

---

## 🤖 AI NAVIGATION MAP (FOR VS CODE)
**Core Principle:** Always prioritize `PostgreSQL` and `Microservices` architecture. Legacy (SQLite) is for reference only.

**Critical Files for AI Analysis:**
- `src/database/postgres_adapter.py` - **#1 Priority**. Check for async patterns, connection handling.
- `main.py` - Central orchestration. Check for component imports and CLI logic.
- `scripts/mass_qwen_analysis.py` - Main analysis script. Check for PostgreSQL queries and batch logic.
- `config.yaml` & `.env` - Configuration sources. Check for hardcoded values.
- `tests/test_integration_comprehensive.py` - Gold standard for expected behavior.
- `scripts/tools/ai_context_manager.py` — AI Context Manager: динамическое управление контекстом, приоритезация файлов, workspace генерация
- `scripts/tools/ai_project_analyzer.py` — AI Project Analyzer: семантический анализ архитектуры, AST-парсинг, метрики для микросервисов

**Deprecated/Legacy Areas (Low Priority for New Features):**
- `scripts/archive/`
- `data/data_backup_*.db`
- Any script with `_sqlite.py` in name.

**AI Analysis Commands (Use these in VS Code Terminal or via AI):**
- `grep -r "sqlite3" src/ scripts/` - Find any lingering SQLite usage.
- `grep -r "PostgreSQLManager" --include="*.py" .` - Find all correct PostgreSQL usages.
- `find . -name "*.py" -type f -exec grep -l "hardcoded_password\|SECRET_KEY" {} \;` - Find security issues.
- `git ls-files | grep -E '\.py$' | xargs wc -l | sort -nr` - Find largest (potentially complex) files.

---

## 🎯 Task-Specific Quick Commands (PostgreSQL-Updated)

### PostgreSQL Development
```bash
# 1. Database layer understanding
read_file("src/database/postgres_adapter.py")          # Connection management
read_file(".env")                                      # Configuration
run_terminal("python check_stats.py")                 # Database health

# 2. Connection testing
run_terminal("psql -h localhost -U rap_user -d rap_lyrics -c '\dt'")  # Table structure
run_terminal("python scripts/db_browser.py")          # Interactive exploration
run_terminal("python -c 'from src.database.postgres_adapter import PostgreSQLManager; print(\"✅ Import OK\")'")

# 3. Performance analysis
run_terminal("python check_overlap.py")               # Analysis efficiency
semantic_search("connection pool OR async")           # Performance patterns
```

### Analysis Development (PostgreSQL-Compatible)
```bash
# 1. Current analysis status
run_terminal("python check_stats.py")                 # Statistics overview
run_terminal("python scripts/mass_qwen_analysis.py --test")  # Quick test
grep_search("PostgreSQLManager", includePattern="scripts/**")  # PostgreSQL usage

# 2. Concurrent analysis testing
# Terminal 1: python scripts/mass_qwen_analysis.py --batch 10
# Terminal 2: python scripts/db_browser.py
# Verify both run simultaneously without conflicts

# 3. Analysis optimization
semantic_search("qwen OR analysis OR batch")          # Analysis patterns
run_terminal("python check_overlap.py")               # Overlap efficiency
list_code_usages("analyze_song")                      # Analysis methods
```

### Migration & Data Work
```bash
# 1. Migration verification
read_file("scripts/migrate_to_postgresql.py", limit=50)  # Migration reference
run_terminal("python check_stats.py")                   # Data verification
run_terminal("python check_overlap.py")                 # Consistency check

# 2. Data integrity analysis
run_terminal("psql -h localhost -U rap_user -d rap_lyrics -c 'SELECT COUNT(*) FROM tracks WHERE lyrics IS NOT NULL;'")
semantic_search("migration OR integrity OR verification")  # Data patterns

# 3. Performance monitoring
run_terminal("python scripts/db_browser.py")            # Interactive analysis
grep_search("async.*conn", isRegexp=True)               # Async patterns
```

### Concurrent Processing Development
```bash
# 1. Concurrent capability verification
# Test multiple scripts simultaneously:
# python scripts/mass_qwen_analysis.py --test &
# python scripts/db_browser.py
# Both should work without database locks

# 2. Connection pool analysis
semantic_search("connection pool OR concurrent")        # Pool patterns
read_file("src/database/postgres_adapter.py", limit=100)  # Pool configuration

# 3. Performance under load
run_terminal("python -c \"
import asyncio
from src.database.postgres_adapter import PostgreSQLManager
async def test_pool():
    db = PostgreSQLManager()
    await db.initialize()
    print(f'Pool size: {db.pool.get_size()}')
    await db.close()
asyncio.run(test_pool())
\"")
```

---

## 💡 Intelligence Boosters

### Context Shortcuts (copy-paste готовые фразы)

#### Project Summary (PostgreSQL-focused):
```
"Enterprise-ready PostgreSQL ML pipeline для rap lyrics analysis. 57,718 треков migrated от SQLite, 
PostgreSQL 15 с connection pooling (20 connections), concurrent script execution capability, 
Qwen API integration для mass analysis. Python+asyncpg+PostgreSQL stack. 
Цель: scalable concurrent text analysis с database integrity.
Недавно: complete SQLite → PostgreSQL migration с 100% data preservation."
```

#### Current Status (post-migration):
```
"Проект успешно завершил PostgreSQL migration (сент 2025): SQLite → PostgreSQL 15, 
57,717 треков + 54,170 анализов migrated, concurrent script execution enabled, 
Qwen analysis pipeline (19,852 completed, 37,866 remaining).
Status: PostgreSQL healthy, 93.9% analysis coverage, concurrent processing verified.
Performance: 34.4% Qwen coverage, connection pool efficient, multi-script capability working."
```

#### Technical Stack (PostgreSQL-enhanced):
```
"Python 3.13+ + PostgreSQL 15, concurrent processing architecture: 
src/database/postgres_adapter.py (connection pooling), asyncpg/psycopg2 drivers,
scripts/mass_qwen_analysis.py (no confirmation prompts), 
Qwen API integration via Novita, Genius/Spotify data enrichment,
.env configuration management, concurrent script execution capability,
migration tools, statistics/diagnostics системы."
```

### Common Pitfalls (чего избегать)

#### ❌ Don't Do This:
```python
# Использовать устаревшие SQLite patterns
import sqlite3                                      # Устарело! Используй PostgreSQL
db_connection.execute(query)                        # Используй async PostgreSQL!

# Игнорировать connection pooling
# Прямые psycopg2 connections вместо adapter
import psycopg2; conn = psycopg2.connect(...)      # Используй PostgreSQLManager!

# Забывать про concurrent processing capabilities
# Запуск одного скрипта когда можно несколько параллельно

# Использовать старые SQLite scripts
python scripts/archive/mass_qwen_analysis_sqlite.py  # Используй PostgreSQL версии!

# Игнорировать .env configuration
hardcoded_postgres_credentials                      # Используй .env!
```

#### ✅ Do This Instead:
```python  
# Modern PostgreSQL approach
from src.database.postgres_adapter import PostgreSQLManager
async with db_manager.get_connection() as conn:    # Proper connection pooling

# Concurrent processing utilization
# Terminal 1: python scripts/mass_qwen_analysis.py
# Terminal 2: python scripts/db_browser.py
# Оба работают одновременно без конфликтов

# Proper configuration management
from dotenv import load_dotenv; load_dotenv()
POSTGRES_HOST = os.getenv("POSTGRES_HOST")

# PostgreSQL-compatible scripts
python scripts/mass_qwen_analysis.py --test        # Updated for PostgreSQL
python check_stats.py                              # PostgreSQL statistics
python scripts/db_browser.py                       # PostgreSQL browser

# Migration awareness
python scripts/migrate_to_postgresql.py            # Migration tools
python check_overlap.py                            # Data verification

# Statistics and monitoring
python check_stats.py                              # Database health
python check_overlap.py                            # Analysis efficiency
```

---

## 🔧 Environment Setup Validation (PostgreSQL-Updated)

### Prerequisites Check (PostgreSQL-Enhanced)
```bash
# System requirements
python --version  # Should be 3.8+ (3.13+ recommended)
psql --version    # PostgreSQL 15+ required
pip list | grep asyncpg  # Async PostgreSQL driver
pip list | grep psycopg2  # Sync PostgreSQL driver

# PostgreSQL service validation
sudo systemctl status postgresql  # Linux service status
# Windows: Check services.msc for PostgreSQL service
psql -h localhost -U rap_user -d rap_lyrics -c "SELECT version();"  # Direct connection test
```

### Database Configuration Validation
```bash
# Check PostgreSQL configuration
cat .env | grep POSTGRES                           # PostgreSQL credentials
psql -h localhost -U rap_user -d rap_lyrics -c "\dt"  # Table structure
python -c "from src.database.postgres_adapter import PostgreSQLManager; print('✅ Adapter OK')"

# Database health check
python check_stats.py                              # Statistics validation
python scripts/db_browser.py                       # Interactive exploration
```

### System Health Check (PostgreSQL Interface)
```bash
# Primary health validation using PostgreSQL
python check_stats.py                              # Complete database statistics
python scripts/mass_qwen_analysis.py --test        # Analysis pipeline test
python check_overlap.py                            # Data consistency check

# Expected outputs:
# - 📈 Всего треков с текстами: 57,718
# - 🤖 Проанализировано Qwen: 19,852
# - ⚡ Остается для Qwen: 37,866
# - 🎯 Покрытие Qwen: 34.4%
```

### Concurrent Processing Check
```bash
# Test concurrent capability (main migration goal)
# Terminal 1:
python scripts/mass_qwen_analysis.py --test &

# Terminal 2 (simultaneously):
python scripts/db_browser.py

# Expected: Both scripts run without database locks or conflicts
# PostgreSQL should handle multiple connections via connection pool
```

### Legacy System Check (Archived)
```bash
# SQLite backup validation (emergency fallback)
ls scripts/archive/                                # Archived SQLite scripts
ls data/data_backup_*.db                          # SQLite backup files
python scripts/archive/mass_qwen_analysis_sqlite.py --dry-run  # Legacy test

# Migration verification
python scripts/migrate_to_postgresql.py --verify   # Migration validation
```

---

## 🚨 Emergency Protocols

### If Agent Seems Lost
```markdown
**RESET PROTOCOL (PostgreSQL-UPDATED)**: 
1. read_file("docs/claude.md") - восстанови PostgreSQL контекст
2. run_terminal("python check_stats.py") - проверь database health
3. read_file("src/database/postgres_adapter.py", limit=50) - изучи database layer
4. Объясни твою конкретную задачу в 1-2 предложениях  
5. Выбери подходящий persona workflow выше
6. Используй PostgreSQL для всех database операций
```

### If PostgreSQL Issues
```markdown
**POSTGRESQL RECOVERY**:
1. run_terminal("python check_stats.py")           # Database diagnostics
2. run_terminal("psql -h localhost -U rap_user -d rap_lyrics -c 'SELECT 1;'")  # Direct test
3. cat .env | grep POSTGRES                        # Check credentials
4. sudo systemctl restart postgresql               # Service restart (Linux)
5. python scripts/db_browser.py                    # Interactive testing
```

### If Migration Issues
```markdown
**MIGRATION RECOVERY**:
1. run_terminal("python check_stats.py")                    # Verify current state
2. run_terminal("python check_overlap.py")                  # Data consistency
3. ls data/data_backup_*.db                                # Check SQLite backup
4. python scripts/migrate_to_postgresql.py --verify        # Migration validation
5. Emergency: use scripts/archive/ for SQLite fallback
```

### If Concurrent Processing Problems
```markdown
**CONCURRENT TROUBLESHOOTING**:
- Connection pool exhaustion: check PostgreSQL max_connections
- Script conflicts: verify scripts use PostgreSQLManager properly
- Performance issues: monitor connection pool usage
- Database locks: check for long-running transactions
- Recovery: restart PostgreSQL service, reduce concurrent scripts
```

### If Analysis Pipeline Issues
```markdown
**ANALYSIS DEBUG (POSTGRESQL)**:
1. Check Qwen API: run_terminal("python scripts/mass_qwen_analysis.py --test")
2. Database connectivity: run_terminal("python check_stats.py")
3. API credentials: cat .env | grep NOVITA_API_KEY
4. Script compatibility: grep -r "PostgreSQLManager" scripts/
5. Analysis efficiency: run_terminal("python check_overlap.py")
```

---

## 📊 Success Metrics & Validation (PostgreSQL-Enhanced)

### PostgreSQL Knowledge Checklist (New)
- [ ] **PostgreSQL Setup Clear**: Понимаю connection configuration и credentials
- [ ] **Database Layer Mapped**: PostgreSQLManager, connection pooling, async operations  
- [ ] **Migration Understanding**: Знаю SQLite → PostgreSQL migration process
- [ ] **Concurrent Processing**: Понимаю multi-script execution capabilities
- [ ] **Statistics Tools**: Могу использовать check_stats.py, check_overlap.py
- [ ] **Analysis Integration**: Знаю Qwen + PostgreSQL integration
- [ ] **Performance Monitoring**: Понимаю connection pool efficiency

### Technical Readiness Checklist (PostgreSQL-Enhanced)
- [ ] **PostgreSQL Access**: psql connection works with rap_user credentials
- [ ] **Database Health**: `python check_stats.py` shows 57,718 tracks
- [ ] **Analysis Pipeline**: `python scripts/mass_qwen_analysis.py --test` works
- [ ] **Concurrent Capability**: Can run multiple scripts simultaneously
- [ ] **Connection Pool**: PostgreSQLManager initializes without errors
- [ ] **API Integration**: Qwen analysis completes successfully
- [ ] **Data Consistency**: check_overlap.py shows proper statistics

### Migration Success Checklist (New)
- [ ] **Data Integrity**: 57,717 tracks + 54,170 analyses migrated полностью
- [ ] **Schema Compatibility**: PostgreSQL tables match SQLite structure
- [ ] **Performance Improvement**: Concurrent access works без конфликтов
- [ ] **Script Compatibility**: All major scripts work with PostgreSQL
- [ ] **Backup Availability**: SQLite backups preserved in archive
- [ ] **Configuration Updated**: .env contains PostgreSQL credentials
- [ ] **Legacy Preservation**: SQLite scripts archived for reference

### Production Readiness Checklist (PostgreSQL-Enhanced)
- [ ] **Database Performance**: Connection pool efficient, queries fast
- [ ] **Concurrent Stability**: Multiple scripts run без database locks
- [ ] **Analysis Throughput**: Qwen processing ~2 tracks/min sustained
- [ ] **Error Handling**: Graceful degradation under connection limits
- [ ] **Monitoring Tools**: Statistics and diagnostics working properly
- [ ] **Recovery Procedures**: Emergency fallback to SQLite available
- [ ] **Documentation Complete**: PostgreSQL setup and usage documented

### Communication Readiness Checklist (PostgreSQL-Enhanced)
- [ ] **Migration Communication**: Умею объяснить SQLite → PostgreSQL benefits
- [ ] **Concurrent Processing**: Понимаю multi-script advantages
- [ ] **Performance Context**: Знаю connection pooling и scalability benefits
- [ ] **Problem Scoping**: Могу isolate PostgreSQL vs application issues
- [ ] **Solution Planning**: Планирую considering database constraints
- [ ] **Migration Expertise**: Могу помочь с similar database migrations

---

## 🔄 Maintenance & Updates

### When to Update This Checklist
- [ ] PostgreSQL schema changes (new tables, indexes, constraints)
- [ ] New concurrent processing features (batch improvements, parallel analysis)
- [ ] Analysis pipeline updates (new APIs, analyzer improvements)  
- [ ] Performance optimizations (connection pool tuning, query optimization)

### Update Procedure
```bash
# 1. Document PostgreSQL changes
echo "PostgreSQL Updated: $(date)" >> AI_ONBOARDING_CHECKLIST.md

# 2. Test with fresh PostgreSQL validation
python check_stats.py  # Database health
python check_overlap.py  # Data consistency
python scripts/mass_qwen_analysis.py --test  # Analysis pipeline

# 3. Validate all PostgreSQL commands work
# Every database command should execute successfully

# 4. Cross-reference with claude.md
# Ensure PostgreSQL consistency between documents
```

---

## 📚 Quick Reference Card (PostgreSQL-Updated)

```yaml
Project: Enterprise PostgreSQL ML rap lyrics analysis with concurrent processing
Stack: Python 3.13+ + PostgreSQL 15 + asyncpg/psycopg2 + Qwen API + concurrent scripts
Architecture: PostgreSQL database + connection pooling + async operations + multi-script support
Data: 57,718 tracks, 54,170 analyses, 19,852 Qwen completed, 37,866 remaining
Status: SQLite → PostgreSQL migration complete, concurrent processing verified

Key PostgreSQL Files:
  - src/database/postgres_adapter.py: Database layer (~200 lines)
  - .env: PostgreSQL credentials and configuration
  - scripts/mass_qwen_analysis.py: Main analysis script (PostgreSQL-compatible)
  - check_stats.py: Database statistics and health monitoring
  - check_overlap.py: Analysis overlap and efficiency verification
  - scripts/migrate_to_postgresql.py: Migration tools and verification

PostgreSQL Configuration:
  - Database: rap_lyrics
  - User: rap_user  
  - Host: localhost:5432
  - Pool: 20 max connections
  - Drivers: asyncpg (async) + psycopg2 (sync)

Migration Achievements:
  - ✅ 57,717 tracks migrated (100% preservation)
  - ✅ 54,170 analysis results migrated (100% preservation)
  - ✅ Concurrent script execution enabled
  - ✅ Connection pooling implemented (20 connections)
  - ✅ All major scripts PostgreSQL-compatible
  - ✅ SQLite backup preserved in scripts/archive/

PostgreSQL Commands:
  - Database health: python check_stats.py
  - Analysis test: python scripts/mass_qwen_analysis.py --test
  - Overlap check: python check_overlap.py
  - Database browser: python scripts/db_browser.py
  - Direct access: psql -h localhost -U rap_user -d rap_lyrics
  
Concurrent Processing Test:
  - Terminal 1: python scripts/mass_qwen_analysis.py --batch 50
  - Terminal 2: python scripts/db_browser.py
  - Both should run simultaneously without conflicts

Analysis Status:
  - Total tracks: 57,718
  - Qwen analyzed: 19,852 (34.4%)
  - Gemma analyzed: 34,320 (59.5%)
  - Overall coverage: 93.9%
  - Remaining for Qwen: 37,866

API Integration:
  - Qwen: Novita API (NOVITA_API_KEY in .env)
  - Genius: Lyrics scraping (GENIUS_ACCESS_TOKEN in .env)
  - Spotify: Audio features (SPOTIFY_CLIENT_ID/SECRET in .env)

Emergency Fallback:
  - SQLite scripts: scripts/archive/
  - Data backup: data/data_backup_*.db
  - Legacy diagnostics: scripts/tools/database_diagnostics.py

Entry Modes:
  - Express: read_file("docs/claude.md") + python check_stats.py
  - Standard: + PostgreSQL adapter + configuration + analysis test
  - Deep: + migration context + concurrent testing + performance validation

Emergency: Reset with docs/claude.md + PostgreSQL health check + database statistics
```

---

*Created: 2025-08-26 | Version: 6.0 - Updated for PostgreSQL Migration | Updated: 2025-09-08 | Next Review: After analysis pipeline optimization*