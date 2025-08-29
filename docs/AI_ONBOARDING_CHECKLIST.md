# AI Assistant Onboarding Checklist (Post-Refactoring)

> **Universal guide для любого AI агента**, начинающего работу с modernized rap-scraper проектом

## 🎯 Quick Start (выбери свой сценарий)

### ⚡ Express Mode (30 секунд)
```bash
# Только базовый контекст после рефакторинга
read_file("docs/claude.md")  # Полностью обновлен для микросервисной архитектуры!
run_terminal("python main.py --info")  # Новый unified entry point
```
**Когда использовать**: Простые вопросы, quick fixes, понимание новой архитектуры

### 🚀 Standard Mode (2 минуты)  
```bash
# Полный контекст microservices архитектуры
read_file("docs/claude.md")                    # Обновленная архитектура  
read_file("main.py", limit=50)                 # Unified entry point (653 lines)
read_file("config.yaml")                       # Centralized configuration
run_terminal("python main.py --info")          # System status
list_dir("src/analyzers/")                     # 4 specialized analyzers
list_dir("src/cli/")                          # CLI component system
```
**Когда использовать**: Разработка, debugging, работа с новой микросервисной структурой

### 🔬 Deep Dive Mode (5 минут)
```bash
# Comprehensive understanding microservices ecosystem
read_file("docs/claude.md")                           # Полный контекст
read_file("main.py", limit=100)                       # Central integration point
read_file("tests/test_integration_comprehensive.py", limit=50)  # Test architecture
read_file("docker-compose.yml")                       # Container orchestration
list_dir("src/")                                      # Component structure
semantic_search("microservice OR analyzer OR main.py") # Core functionality
run_terminal("python main.py --test")                 # Validation
```
**Когда использовать**: Архитектурные изменения, complex debugging, системная интеграция

---

## 📖 Context Layers (читать по потребности)

### Layer 1: Modern Architecture Foundation (MANDATORY - UPDATED!)
```yaml
File: docs/claude.md
Purpose: Central microservices context, main.py architecture, post-refactoring workflow
Read When: Always (first thing to read - completely rewritten for new architecture!)
Key Info: 54K+ tracks, microservices design, main.py unified interface, Docker ready
```

### Layer 2: Unified Entry Point (HIGH PRIORITY - NEW!)
```yaml  
File: main.py
Purpose: Central application integration, CLI interface, component orchestration
Read When: Understanding system integration, debugging workflows
Key Info: 653 lines, 7-option interactive menu, command-line flags, error handling
```

### Layer 3: Microservices Components (HIGH PRIORITY - NEW!)
```yaml
Structure: src/analyzers/, src/cli/, src/models/
Purpose: Specialized analyzers, CLI components, data models
Read When: Working with specific components, adding features
Key Info: 4 analyzers (algorithmic_basic, gemma, ollama, hybrid), modular CLI system
```

### Layer 4: Configuration & Testing (HIGH PRIORITY - NEW!)
```yaml
Files: config.yaml, tests/test_integration_comprehensive.py
Purpose: Centralized settings, comprehensive test suite
Read When: Configuration changes, validation, quality assurance
Key Info: YAML-based config, 400+ lines test suite, 15 test methods
```

### Layer 5: Docker & Production (CONTEXT DEPENDENT - NEW!)
```yaml
Files: Dockerfile, docker-compose.yml
Purpose: Containerization, production deployment, monitoring stack
Read When: Deployment, scaling, production troubleshooting
Key Info: Multi-service setup, Prometheus + Grafana, health checks
```

### Layer 6: Legacy Compatibility (AS NEEDED)
```yaml
Files: scripts/rap_scraper_cli.py, scripts/tools/, scripts/development/
Purpose: Backwards compatibility, specialized tools
Read When: Using legacy functionality, batch processing
Key Info: Original CLI preserved, production tools available
```

---

## 🤖 AI Agent Personas & Workflows

### 👨‍💻 Developer Agent (Updated for Microservices)
```python
# Primary tasks: Component development, microservice integration, feature enhancement
def developer_onboarding():
    read_file("docs/claude.md")                         # Microservices architecture
    read_file("main.py", limit=100)                     # Central integration point
    read_file("config.yaml")                           # Configuration system
    list_dir("src/analyzers/")                         # 4 specialized analyzers
    list_dir("src/cli/")                              # CLI component system
    list_dir("src/models/")                           # Data models
    semantic_search("BaseAnalyzer OR CLI")             # Component interfaces
    run_terminal("python main.py --test")             # Test infrastructure
    run_terminal("python main.py --info")             # System health
```

### 🔍 Debugging Agent (Enhanced for New Architecture)
```python
# Primary tasks: Component debugging, integration issues, performance analysis
def debugging_onboarding():
    read_file("docs/claude.md")                                # Updated context
    run_terminal("python main.py --info")                     # System status
    run_terminal("python main.py --test")                     # Test validation
    list_dir("tests/")                                        # Test infrastructure
    grep_search("error|exception|failed", isRegexp=True)      # Error patterns
    get_terminal_output("last_run")                           # Recent failures
    semantic_search("error handling OR validation")           # Error management
    list_code_usages("problematic_function")                 # Usage analysis
```

### 📊 ML/Analysis Agent (Modernized for 4 Analyzers)
```python  
# Primary tasks: Analyzer development, performance optimization, ML pipeline
def ml_onboarding():
    read_file("docs/claude.md")                                # ML objectives updated
    read_file("main.py", limit=50)                             # Analyzer integration
    list_dir("src/analyzers/")                                # 4 specialized analyzers
    read_file("src/analyzers/algorithmic_basic.py", limit=50)  # Fast baseline
    read_file("src/analyzers/hybrid_analyzer.py", limit=50)    # Combined approach
    run_terminal("python main.py --analyze 'test text'")      # Quick test
    run_terminal("python main.py --benchmark")                # Performance metrics
    semantic_search("sentiment OR confidence OR analysis")     # Analysis patterns
```

### � DevOps Agent (New for Docker Architecture)
```python
# Primary tasks: Container deployment, monitoring setup, production readiness
def devops_onboarding():
    read_file("docs/claude.md")                        # Production context
    read_file("Dockerfile")                           # Container specification
    read_file("docker-compose.yml")                   # Service orchestration
    run_terminal("docker-compose ps")                 # Service status
    run_terminal("docker-compose logs rap-scraper")   # Application logs
    semantic_search("monitoring OR health check")     # Observability
    list_dir("monitoring/")                          # Legacy monitoring tools
```

### 📝 Documentation Agent (Updated for New Structure)
```python
# Primary tasks: Documentation maintenance, architecture explanation, guides
def docs_onboarding():
    read_file("docs/claude.md")                        # Current state
    read_file("README.md", limit=100)                  # Project overview
    read_file("FINAL_COMPLETION_REPORT.md", limit=50) # Refactoring summary
    list_dir("docs/")                                 # All documentation
    get_changed_files()                               # Recent changes
    grep_search("TODO.*doc", isRegexp=True)           # Documentation gaps
    file_search("**/*.md")                           # All markdown files
    semantic_search("documentation OR guide")         # Documentation patterns
```

---

## 🎯 Task-Specific Quick Commands (Updated)

### New Feature Development (Microservices Approach)
```bash
# 1. Understand component architecture
read_file("main.py", limit=50)                         # Integration patterns
list_dir("src/analyzers/")                             # Analyzer components
list_dir("src/cli/")                                   # CLI components
read_file("src/models/analysis_models.py")             # Data models
semantic_search("BaseAnalyzer OR Component")           # Interface patterns

# 2. Check existing implementations
grep_search("class.*Analyzer", isRegexp=True)          # Analyzer patterns
list_code_usages("BaseClass")                          # Interface usage
run_terminal("python main.py --info")                  # Component status

# 3. Plan implementation using microservice patterns
read_file("config.yaml")                              # Configuration reference
run_terminal("python main.py --test")                 # Validation framework
grep_search("def.*process", isRegexp=True)            # Processing patterns
```

### Bug Investigation (Enhanced Diagnostics)
```bash
# 1. Use unified diagnostics
run_terminal("python main.py --info")                  # Complete system status
run_terminal("python main.py --test")                  # Test suite validation
run_terminal("python main.py --analyze 'test'")       # Quick functionality test

# 2. Component-level debugging
grep_search("Exception|Error", isRegexp=True)          # Error patterns
get_terminal_output("error_session")                   # Recent failures
semantic_search("error_symptom description")           # Similar issues

# 3. Microservice isolation
list_code_usages("failing_function")                   # Component usage
read_file("tests/test_integration_comprehensive.py", limit=50)  # Test patterns
run_terminal("pytest tests/ -v")                       # Detailed test results
```

### Data Analysis (4-Analyzer Ecosystem)
```bash
# 1. Analyzer ecosystem overview
run_terminal("python main.py --info")                  # Available analyzers
run_terminal("python main.py --benchmark")             # Performance comparison
list_dir("src/analyzers/")                            # All analyzers

# 2. Interactive analysis testing
run_terminal("python main.py")                        # Interactive menu (option 2)
run_terminal("python main.py --analyze 'sample text' --analyzer hybrid")
run_terminal("python main.py --batch test_file.txt")  # Batch processing

# 3. Legacy batch processing (if needed)
run_terminal("python scripts/rap_scraper_cli.py status")
run_terminal("python scripts/tools/batch_ai_analysis.py --dry-run")
semantic_search("sentiment OR confidence OR analysis") # Analysis patterns
```

### Docker & Production Work (New)
```bash
# 1. Container architecture
read_file("docker-compose.yml")                       # Service definitions
read_file("Dockerfile")                              # Container specification
run_terminal("docker-compose ps")                    # Service status

# 2. Production deployment
run_terminal("docker-compose up -d")                 # Deploy stack
run_terminal("docker-compose logs -f rap-scraper")   # Application logs
run_terminal("docker-compose exec rap-scraper python main.py --info")

# 3. Monitoring and health
# Access Grafana: http://localhost:3000
# Access Prometheus: http://localhost:9090
semantic_search("monitoring OR health check")         # Observability patterns
```

### Component Integration Work (New)
```bash
# 1. Integration patterns
read_file("main.py", limit=100)                      # Central orchestration
grep_search("import.*src", isRegexp=True)            # Component imports
list_code_usages("AppConfig")                        # Configuration usage

# 2. Component interfaces
read_file("src/models/analysis_models.py")           # Data contracts
semantic_search("BaseAnalyzer OR interface")         # Interface patterns
grep_search("async def", isRegexp=True)              # Async patterns

# 3. Testing integration
run_terminal("python main.py --test")                # Integration tests
pytest tests/test_integration_comprehensive.py -v    # Detailed validation
```

---

## 💡 Intelligence Boosters

### Context Shortcuts (copy-paste готовые фразы)

### Context Shortcuts (copy-paste готовые фразы)

#### Project Summary (elevator pitch):
```
"Enterprise-ready microservices ML pipeline для rap lyrics analysis. 54K+ треков из Genius+Spotify APIs, 
modern Docker-first architecture с unified main.py entry point, 4 specialized analyzers 
(algorithmic_basic, gemma, ollama, hybrid), comprehensive pytest test suite, production monitoring stack.
Python+Pydantic+SQLite+Docker stack. Цель: scalable text analysis с real-time response.
Недавно: complete 4-phase refactoring от monolithic → microservices architecture."
```

#### Current Status (что сейчас происходит):
```
"Проект успешно завершил 4-фазный рефакторинг (авг 2025): microservices architecture, 
unified main.py interface (653 lines), 4 specialized analyzers, Docker containerization, 
comprehensive testing (15 test methods), centralized config.yaml. Production ready.
Status: 54,568 songs в БД, 4/4 analyzers operational, 100% test pass rate.
Validated performance: analysis_time 0.0s, confidence 0.86, batch success 100%."
```

#### Technical Stack (для понимания технологий):
```
"Python 3.8+, microservices architecture: main.py (653 lines) + src/{analyzers,cli,models}/, 
4 specialized analyzers (algorithmic_basic, gemma, ollama, hybrid), 
unified CLI с interactive menu, Pydantic validation, centralized config.yaml,
Docker multi-service deployment (5 containers), Prometheus+Grafana monitoring,
comprehensive pytest suite (400+ lines), async/await patterns, production-ready."
```

### Common Pitfalls (чего избегать)

### Common Pitfalls (чего избегать)

#### ❌ Don't Do This:
```python
# Использовать устаревшие monolithic patterns
from models import SpotifyTrack                      # Устарело! Используй src.models
python some_old_script.py                           # Используй main.py interface!

# Игнорировать microservices architecture
# Прямые вызовы компонентов вместо main.py
python src/analyzers/algorithmic_basic.py           # Используй main.py!
python src/cli/batch_processor.py                   # Используй unified interface!

# Забывать про centralized configuration
hardcoded_values = "secret_key"                     # Используй config.yaml!
os.environ.get("SOME_VAR")                          # Используй AppConfig!

# Пропускать Docker deployment benefits
pip install requirements.txt                        # Рассмотри docker-compose!

# Игнорировать comprehensive test suite
# Изменения без тестирования
# Всегда: python main.py --test
```

#### ✅ Do This Instead:
```python  
# Modern microservices approach
from src.models.analysis_models import AnalysisResult
python main.py --analyze "text"                     # Unified interface
python main.py --info                               # System status

# Proper component interaction через main.py
python main.py                                      # Interactive menu
python main.py --batch input.txt --analyzer hybrid # Batch processing
python main.py --benchmark                          # Performance testing

# Centralized configuration management
from src.models.config_models import AppConfig
config = AppConfig.from_yaml("config.yaml")

# Production deployment
docker-compose up -d                                # Multi-service deployment
docker-compose exec rap-scraper python main.py --info

# Quality assurance
python main.py --test                              # Comprehensive testing
pytest tests/test_integration_comprehensive.py     # Detailed validation

# Legacy system compatibility (when needed)
python scripts/rap_scraper_cli.py status           # Backwards compatibility
python scripts/tools/batch_ai_analysis.py          # Specialized tools
```

---

## 🔧 Environment Setup Validation (Updated)

### Prerequisites Check (Enhanced)
```bash
# System requirements
python --version  # Should be 3.8+ (3.11+ recommended)
pip list | grep pydantic  # Core dependencies
docker --version  # Optional, for containerized deployment

# Microservices architecture validation
ls main.py config.yaml Dockerfile docker-compose.yml     # Core files present
ls docs/claude.md docs/AI_ONBOARDING_CHECKLIST.md       # Documentation updated
ls src/analyzers/ src/cli/ src/models/                  # Component structure
ls tests/test_integration_comprehensive.py              # Test infrastructure
sqlite3 data/rap_lyrics.db ".tables"                    # Database accessible
```

### Configuration Validation (Centralized)
```bash
# Check centralized config
cat config.yaml                                         # YAML configuration
python -c "from src.models.config_models import AppConfig; AppConfig.from_yaml('config.yaml')"

# Legacy .env check (if still used)
test -f .env && echo "✅ .env exists" || echo "❌ Need to create .env"
cat .env.example                                        # Template structure
```

### System Health Check (Main.py Interface)
```bash
# Primary health validation using unified interface
python main.py --info                                   # Complete system status
python main.py --test                                   # Test suite validation
python main.py --analyze "test text"                    # Quick functionality test

# Expected outputs:
# - Analysis time: 0.0s
# - Sentiment: "neutral" or "positive"  
# - Confidence: >0.7
# - 4/4 analyzers ready
# - Database: 54,568 records
```

### Docker Environment Check (Production)
```bash
# Container validation
docker-compose up -d --build                           # Fresh deployment
docker-compose ps                                      # Service status
docker-compose logs rap-scraper                       # Application logs

# Container health validation
docker-compose exec rap-scraper python main.py --info
docker-compose exec rap-scraper python main.py --test

# Monitoring endpoints
curl http://localhost:9090/metrics                     # Prometheus metrics
curl http://localhost:3000                            # Grafana dashboard
```

### Legacy System Check (Compatibility)
```bash
# Backwards compatibility validation
python scripts/rap_scraper_cli.py status              # Original CLI
python scripts/tools/batch_ai_analysis.py --dry-run   # Production tools
python scripts/check_db.py                            # Database diagnostics

# API credentials validation  
python -c "from src.enhancers.spotify_enhancer import SpotifyEnhancer; print('✅ Spotify OK')"
python -c "import os; print('✅ Genius key:', 'GENIUS_ACCESS_TOKEN' in os.environ)"
```

---

## 🚨 Emergency Protocols

## 🚨 Emergency Protocols (Updated)

### If Agent Seems Lost
```markdown
**RESET PROTOCOL (UPDATED)**: 
1. read_file("docs/claude.md") - восстанови microservices контекст
2. run_terminal("python main.py --info") - проверь unified system status
3. read_file("main.py", limit=50) - изучи central integration point
4. Объясни твою конкретную задачу в 1-2 предложениях  
5. Выбери подходящий persona workflow выше
6. Используй main.py для всех операций, Docker для production
```

### If System Issues
```markdown
**SYSTEM RECOVERY**:
1. run_terminal("python main.py --info")  # Complete diagnostics
2. run_terminal("python main.py --test")  # Validate all components
3. run_terminal("docker-compose ps")      # Container status (if using Docker)
4. Если failures: check logs via docker-compose logs rap-scraper
5. Если container issues: docker-compose down && docker-compose up -d
```

### If Database Issues
```markdown
**DB RECOVERY (UPDATED)**:
1. run_terminal("python main.py --info")                    # Database status
2. run_terminal("python scripts/check_db.py")               # Legacy diagnostics  
3. Если corruption: run_terminal("python src/utils/migrate_database.py --repair") 
4. Если missing tables: auto-created on next main.py run
5. Backup available в data/backups/ directory
```

### If Configuration Problems
```markdown
**CONFIG TROUBLESHOOTING (NEW)**:
- Missing config.yaml: copy from config.yaml.example
- Invalid YAML syntax: validate with online YAML checker
- API keys: check both config.yaml and .env (legacy)
- Component configuration: run_terminal("python main.py --info")
- Docker environment: check docker-compose.yml settings
```

### If Performance Issues
```markdown
**PERFORMANCE DEBUG (ENHANCED)**:
1. Check system resources: run_terminal("python main.py --benchmark")
2. Analyzer performance: run_terminal("python main.py --analyze 'test' --analyzer algorithmic_basic")
3. Container resources: run_terminal("docker stats") (if using Docker)
4. Database optimization: run_terminal("python scripts/rap_scraper_cli.py monitoring --component database")
5. Memory usage: prefer algorithmic_basic for speed, hybrid for quality
```

### If Docker Issues (New)
```markdown
**DOCKER TROUBLESHOOTING**:
- Port conflicts: check if 8080, 9090, 3000 are available
- Memory issues: ensure 16GB+ RAM for AI models
- Permission errors: check volume mount permissions
- Service failures: docker-compose logs service_name
- Fresh start: docker-compose down && docker-compose up -d --build
```

---

## 📊 Success Metrics & Validation (Enhanced)

### Knowledge Acquisition Checklist (Updated)
- [ ] **Microservices Architecture Clear**: Понимаю main.py + component structure
- [ ] **4 Analyzers Mapped**: algorithmic_basic, gemma, ollama, hybrid capabilities  
- [ ] **Docker Deployment**: Знаю containerization и monitoring stack
- [ ] **Configuration System**: Понимаю config.yaml + legacy .env
- [ ] **Testing Framework**: Знаю pytest suite и validation methods
- [ ] **My Role Defined**: Понимаю задачу в context microservices ecosystem

### Technical Readiness Checklist (Enhanced)
- [ ] **Environment Setup**: Python 3.8+, Docker optional, dependencies installed
- [ ] **System Validation**: `python main.py --info` shows 4/4 analyzers ready
- [ ] **Performance Check**: `python main.py --benchmark` runs successfully
- [ ] **Configuration Access**: Can read config.yaml and understand structure
- [ ] **Component Understanding**: Know src/{analyzers,cli,models}/ boundaries
- [ ] **Testing Capability**: Can run `python main.py --test` with 100% pass rate

### Production Readiness Checklist (New)
- [ ] **Docker Competency**: Can deploy with `docker-compose up -d`
- [ ] **Monitoring Access**: Can access Grafana (3000) and Prometheus (9090)
- [ ] **Container Health**: Know how to check `docker-compose ps` and logs
- [ ] **Performance Monitoring**: Understand resource usage and optimization
- [ ] **Security Awareness**: Know container hardening and resource limits
- [ ] **Scalability Understanding**: Know multi-service architecture benefits

### Communication Readiness Checklist (Enhanced)
- [ ] **Architecture Communication**: Умею объяснить microservices benefits
- [ ] **Component Interaction**: Понимаю main.py orchestration patterns
- [ ] **Docker Context**: Могу обсуждать containerization и production deployment
- [ ] **Performance Context**: Знаю benchmarking и optimization approaches
- [ ] **Problem Scoping**: Могу isolate issues в specific microservice components
- [ ] **Solution Planning**: Планирую considering component boundaries и integration

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

## 📚 Quick Reference Card (Updated)

```yaml
Project: Enterprise microservices ML rap lyrics analysis
Stack: Python 3.8+ + Pydantic + SQLite + Docker + main.py unified interface
Architecture: main.py (653 lines) + src/{analyzers,cli,models}/ + Docker stack
Data: 54,568 tracks, 345 artists, 4 specialized analyzers, production monitoring
Status: 4-phase refactoring complete, production ready, 100% test pass rate
Performance: analysis_time 0.0s, confidence 0.86, batch success 100%

Key Files:
  - main.py: Unified entry point (653 lines, 7-option menu)
  - docs/claude.md: Microservices context (COMPLETELY UPDATED!)
  - config.yaml: Centralized configuration system
  - tests/test_integration_comprehensive.py: Test suite (400+ lines)
  - docker-compose.yml: Multi-service deployment

Microservices Architecture:
  - src/analyzers/: algorithmic_basic, gemma, ollama, hybrid
  - src/cli/: text_analyzer, batch_processor, performance_monitor
  - src/models/: analysis_models, config_models, database_models
  - tests/: comprehensive pytest suite with 15 test methods

Docker Stack:
  - rap-scraper: Main application container
  - ollama: AI model server  
  - nginx: Reverse proxy
  - prometheus: Metrics collection
  - grafana: Data visualization

Main.py Commands:
  - Interactive: python main.py
  - Quick analysis: python main.py --analyze "text"
  - System status: python main.py --info
  - Performance test: python main.py --benchmark
  - Test suite: python main.py --test
  - Batch processing: python main.py --batch file.txt

Docker Commands:
  - Deploy: docker-compose up -d
  - Status: docker-compose ps
  - Logs: docker-compose logs rap-scraper
  - Execute: docker-compose exec rap-scraper python main.py --info
  - Monitoring: http://localhost:3000 (Grafana), http://localhost:9090 (Prometheus)

Legacy Compatibility:
  - Original CLI: python scripts/rap_scraper_cli.py status
  - Production tools: python scripts/tools/batch_ai_analysis.py
  - Database check: python scripts/check_db.py

Entry Modes:
  - Express: read_file("docs/claude.md") + python main.py --info
  - Standard: + component exploration + config.yaml
  - Deep: + Docker context + test suite + performance validation

Emergency: Reset with docs/claude.md + main.py --info + component structure
```

---

*Created: 2025-08-26 | Version: 5.0 - Post-Microservices Refactoring | Updated: 2025-08-29 | Next Review: After production deployment optimization*