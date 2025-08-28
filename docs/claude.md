# Rap Scraper Project — AI Agent Context

> **ML pipeline for conditional rap lyrics generation**, using structured metadata from the Genius API and the Spotify Web API.

## 📑 Quick navigation
- [🚀 Quick Start](#-quick-start) — get running in 5 minutes
- [📊 Project Status](#-project-status-live-dashboard) — current metrics
- [🏗️ Architecture](#-architecture) — system overview
- [🤖 AI Agent Workflow](#-ai-agent-workflow) — how to work with the project
- [🔧 Commands Reference](#-commands-reference) — main commands
- [🚨 Troubleshooting](#-troubleshooting) — common fixes

---

## 🚀 Quick Start

### Prerequisites
```bash
# Requirements
Python 3.13+
SQLite 3.x
API keys: Genius (required), Spotify (optional for enrichment)
```

### 2-minute setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configuration
# Copy the template and fill your API keys
# PowerShell example:
# Copy-Item .env.example .env; notepad .env

# 3. Check status via CLI
python scripts/rap_scraper_cli.py status
```

### First run (recommended via unified CLI)
```bash
# End-to-end smoke test (small sample)
python scripts/rap_scraper_cli.py scraping --limit 10
python scripts/rap_scraper_cli.py spotify --limit 10 --continue
python scripts/rap_scraper_cli.py analysis --analyzer multi --limit 5

# Database stats
python scripts/rap_scraper_cli.py monitoring --component database
```

---

## 📊 Project Status (Live Dashboard)

### Current metrics
- 📁 **Dataset**: 53,300 tracks, 328 artists (full dataset)
- 🎯 **Spotify Coverage**: ~99.6% artists enriched, 38K+ tracks pending AI analysis
- 🔄 **Pipeline Status**: Cleaned and optimized - removed 5+ redundant scripts
- 🤖 **AI Analysis**: Multi-model pipeline ready (Ollama, Gemma 27B, fallback modes)
- ✨ **NEW: ML Features**: 17-feature extraction pipeline (rhyme density, TTR, metaphor detection, flow patterns)
- 🚀 **CLI-First**: Unified interface via `rap_scraper_cli.py` - single entry point

### Active components (quick checks)
```bash
# Unified CLI - single entry point for all operations
python scripts/rap_scraper_cli.py status
python scripts/rap_scraper_cli.py monitoring --component database

# Production tools (moved from clutter)
python scripts/tools/batch_ai_analysis.py --dry-run
python scripts/tools/check_spotify_coverage.py

# Key data locations
- data/rap_lyrics.db (main SQLite DB)
- analysis_results/ (ML outputs)
- enhanced_data/ (enriched JSONL)
```

---

## 🏗️ Architecture

```mermaid
graph TD
    A[Genius API] -->|Lyrics + Metadata| B[src/scrapers/rap_scraper_optimized.py]
    B -->|SQLite| C[(Database)]
    C -->|Enhancement| D[src/enhancers/spotify_enhancer.py]
    E[Spotify API] -->|Audio Features| D
    D -->|Enriched Data| F[Analysis Pipeline]
    F -->|Features| G[ML Training Dataset]

    H[src/analyzers/langchain_analyzer.py] -->|OpenAI| F
    I[src/analyzers/ollama_analyzer.py] -->|Local LLM| F
    J[src/analyzers/multi_model_analyzer.py] -->|Comparison| F
```

### Core components

#### Data collection layer
- `src/scrapers/rap_scraper_optimized.py` — optimized Genius scraper with proxy handling
- `src/enhancers/spotify_enhancer.py` — Spotify metadata enrichment
- `src/enhancers/bulk_spotify_enhancement.py` — bulk enhancement utilities

#### Data models & storage
- `src/models/models.py` — Pydantic models (SpotifyArtist, SpotifyTrack, etc.)
- `data/rap_lyrics.db` — SQLite DB with enhanced schema

#### ML analysis layer
- `src/analyzers/multi_model_analyzer.py` — multi-provider AI analysis (Ollama → Gemma → Mock)
- `src/analyzers/gemma_27b_fixed.py` — Gemma 27B integration
- `src/analyzers/simplified_feature_analyzer.py` — **NEW**: ML feature engineering (17 features)
- `src/analyzers/advanced_feature_analyzer.py` — **NEW**: NLTK-powered advanced features
- `src/analyzers/create_visual_analysis.py` — portfolio-ready dashboard generation

#### Tools & utilities
- `scripts/tools/batch_ai_analysis.py` — production batch processor (rehabilitated)
- `scripts/tools/check_spotify_coverage.py` — coverage diagnostics
- `monitoring/` — real-time progress tracking

---

## 🤖 AI Agent Workflow

### Context priority (read in this order)
1. `docs/claude.md` — this document (project context)
2. `docs/PROJECT_DIARY.md` — full development history and cases
3. `src/models/models.py` — data structures and API contracts
4. The working file related to the current task

### Investigation protocol (standard)
```python
def investigate_issue(problem_description):
    # 1. Understand scope: semantic_search on the error
    semantic_search(f"error {problem_description}")

    # 2. Find relevant code
    grep_search(f"def.*{main_component}")
    list_code_usages("ProblemClass")

    # 3. Deep dive
    read_file("problematic_module.py")

    # 4. Pattern search
    grep_search("similar_error_pattern")

    # 5. Return a detailed plan
    return detailed_plan_with_validation_steps
```

### Response format (required)
When reporting investigation results, follow this structure:

```markdown
## 🔍 Investigation Summary
- **Current Understanding**: short summary
- **Knowledge Gaps**: what needs more research

## 📋 Findings
- **Root Cause**: technical cause
- **Impact**: affected components
- **Code Locations**: files/lines

## 🚀 Solution Plan
1. **Step 1**: action + expected outcome
2. **Step 2**: action + validation
3. **Step N**: ...

## ✅ Validation Needed
- **User Approval**: yes/no
- **Testing**: tests or checks to run
```

---

## 🔧 Development phases

### Phase 1: Research & understanding
```bash
semantic_search("main pipeline functionality")
grep_search("def main|if __name__")
file_search("**/test_*.py")
get_changed_files()
```

### Phase 2: Planning
- Draft a step-by-step plan with risks and rollback criteria
- Identify dependencies and side effects
- Define success criteria and validation steps

### Phase 3: Validation
- Discuss plan with the user
- Adjust based on feedback
- Approve before execution

### Phase 4: Execution
- Execute in small, tested steps
- Run tests after each change
- Handle errors gracefully

### Phase 5: Documentation
- Update `PROJECT_DIARY` with STAR-format notes
- Add code comments for complex logic
- Update README/docs if behavior or CLI changes

---

## 🔧 Commands reference

### Canonical pipeline (CLI-first architecture)
```bash
# Core scraping workflow
python scripts/rap_scraper_cli.py scraping                    # Production mode
python scripts/rap_scraper_cli.py scraping --test            # Test mode (3 artists)
python scripts/rap_scraper_cli.py scraping --artist "Drake"  # Single artist

# Spotify enrichment workflow  
python scripts/rap_scraper_cli.py spotify                     # New enhancement
python scripts/rap_scraper_cli.py spotify --continue         # Resume existing

# AI analysis workflow
python scripts/rap_scraper_cli.py analysis --analyzer gemma  # Gemma 27B
python scripts/rap_scraper_cli.py analysis --analyzer multi  # Multi-model comparison

# NEW: ML Feature Engineering (17 features)
python scripts/rap_scraper_cli.py mlfeatures                 # All tracks with simplified features
python scripts/rap_scraper_cli.py mlfeatures --limit 50      # Process 50 tracks only
python scripts/rap_scraper_cli.py mlfeatures --export results.json  # Export to JSON

# Monitoring & utilities
python scripts/rap_scraper_cli.py monitoring --component database
python scripts/rap_scraper_cli.py monitoring --component analysis
python scripts/rap_scraper_cli.py monitoring --component gemma
python scripts/rap_scraper_cli.py utilities --utility spotify-setup
python scripts/rap_scraper_cli.py status
```

### Production tools (new organized structure)
```bash
# Batch processing (rehabilitated from archive)
python scripts/tools/batch_ai_analysis.py --batch-size 25 --dry-run
python scripts/tools/batch_ai_analysis.py --batch-size 50 --max-batches 100

# Diagnostics & coverage analysis
python scripts/tools/check_spotify_coverage.py

# Portfolio & visualization
python src/analyzers/create_visual_analysis.py
```

### Development & testing
```bash
# Development scripts (organized)
python scripts/development/test_fixed_scraper.py          # Test scraper fixes
python scripts/development/scrape_artist_one.py "Artist" # Single artist testing
python scripts/development/run_scraping_debug.py         # Debug mode

# Legacy compatibility (maintained)
python scripts/legacy/run_analysis.py                     # Legacy analysis wrapper
```

---

## 🎨 Code style & architecture guidelines

### Python standards
- Python 3.13+ with type hints
- Pydantic models for API validation
- SQLite usage with context managers
- Rate limiting for all external API calls
- Structured logging with correlation IDs where useful

### Architectural principles
- Separation of concerns: scraping → enhancement → analysis
- Incremental processing and resume capability
- Graceful degradation for partial failures
- Batch processing for performance

### Anti-patterns to avoid
- Hardcoded credentials (use `.env`)
- Blocking API calls without timeout
- Unhandled exceptions in production
- Magic numbers without constants

---

## 🧪 Testing & quality

### Recommended cycle
```bash
# Full development cycle
pytest tests/ --verbose
python -m mypy src/
python -m flake8 src/
```

### Quick module check
```bash
pytest tests/test_spotify_enhancer.py -v
```

### Test structure (expected)
```
tests/
├── test_spotify_enhancer.py
├── test_models.py
├── test_database.py
├── test_scraper.py
└── conftest.py
```

---

## 🚨 Troubleshooting

### Common API issues
- Genius API: 403 Forbidden → check `GENIUS_TOKEN` in `.env` and rate limits (1 req/sec)
- Spotify API: 403 on audio features → fallback mode disables audio features
- Token expired → run `python src/utils/setup_spotify.py --refresh`

### Database problems
```bash
# Corrupted DB
python src/utils/migrate_database.py --repair

# Missing tables (auto-fix)
python src/utils/check_db.py --create-missing
```

### Performance issues
- Slow scraping: increase batch_size or enable streaming
- Memory issues: lower batch sizes or use streaming mode
- API rate limits: automatic backoff is implemented — check logs

### Development issues
```bash
# Dependency/import errors
pip install -r requirements.txt --upgrade

# Type checking errors
python -m mypy --install-types

# Test failures
pytest tests/ --tb=short
```

---

## 🎯 ML pipeline goals

### Project structure (post-cleanup)
```
scripts/
├── rap_scraper_cli.py          # 🎯 UNIFIED CLI - single entry point
├── continue_spotify_enhancement.py  # Resume Spotify enhancement
├── run_spotify_enhancement.py      # New Spotify enhancement  
├── check_db.py                     # Database diagnostics
├── development/                    # 🧪 Development & testing
│   ├── test_fixed_scraper.py      # Test scraper fixes
│   ├── scrape_artist_one.py       # Single artist testing
│   └── run_scraping_debug.py      # Debug mode
├── legacy/                         # 🗂️ Legacy compatibility
│   └── run_analysis.py            # Legacy analysis wrapper
├── tools/                          # 🛠️ Production utilities
│   ├── batch_ai_analysis.py       # Batch processor (rehabilitated)
│   └── check_spotify_coverage.py  # Coverage diagnostics
└── utils/                          # 🔧 Project utilities
    └── cleanup_project.py          # Project organization tool

src/
├── scrapers/
│   └── rap_scraper_optimized.py   # Core scraper with proxy handling
├── enhancers/
│   ├── spotify_enhancer.py        # Spotify API integration
│   └── bulk_spotify_enhancement.py # Bulk processing
├── analyzers/
│   ├── multi_model_analyzer.py    # Multi-provider AI analysis
│   ├── gemma_27b_fixed.py         # Gemma 27B integration
│   └── create_visual_analysis.py  # Portfolio dashboard
└── models/
    └── models.py                   # Pydantic schemas

monitoring/
├── monitor_gemma_progress.py      # Real-time Gemma monitoring
└── check_analysis_status.py       # Analysis status overview
```

## 🎯 ML pipeline goals

### Project goal
**Conditional Rap Lyrics Generation** — artist style + genre + mood → authentic generated lyrics

### Training pipeline
1. Data collection: 53K+ tracks with rich metadata ✅
2. Feature engineering: sentiment, complexity, audio features 🔄
3. AI analysis: Multi-model pipeline (Ollama, Gemma 27B) 🔄  
4. Model training: fine-tuning on structured dataset 📋
5. Generation: controlled sampling for style/mood 📋

### Available features for training
- Lyrics text (full corpus)
- Artist metadata
- Spotify audio features (energy, valence, danceability)
- Genre labels
- Sentiment and complexity metrics

---

## 📁 Key files reference

### Config & models
- `src/models/models.py` — Pydantic schemas
- `requirements.txt` — Python dependencies
- `.env` — API credentials (DO NOT commit; create from `.env.example`)

### Documentation & history
- `docs/PROJECT_DIARY.md` — detailed project case history
- `docs/TECH_SUMMARY.md` — technical summary
- `docs/PROJECT_EVOLUTION.md` — architecture change log
- `FEATURE_ENGINEERING_GUIDE.md` — **NEW**: ML features documentation (17 features)

### Data & results
- `data/rap_lyrics.db` — primary SQLite DB
- `analysis_results/` — CSV/JSON outputs from analysis
- `enhanced_data/` — enriched JSONL exports

---

## 📈 Project evolution (post-cleanup status)

### Recently completed ✅
- **Project organization**: Removed 5+ redundant scripts, organized into logical folders
- **CLI consolidation**: Single entry point via `rap_scraper_cli.py` 
- **Tool rehabilitation**: Moved `batch_ai_analysis.py` from archive to tools/
- **Code quality**: Enhanced error handling, documentation, and user experience
- **53,300 tracks** collected with ~99.6% Spotify enrichment coverage

### In progress 🔄
- **AI Analysis pipeline**: 38K+ tracks pending analysis via multi-model system
- **Batch processing**: Production-ready batch analyzer for large-scale analysis
- **Feature engineering**: Audio features + sentiment analysis for ML training

### Next milestones 📋
- Complete AI analysis coverage using batch tools
- Prepare final ML training dataset with rich features
- Architect conditional generation model
- Production deployment pipeline

---

## 💡 AI assistant guidelines

### Core philosophy
1. ML-first: every change should improve training data quality
2. Data quality over quantity
3. Respect external APIs: apply rate limits and retries
4. Incremental progress with measurable value
5. Production readiness and observability

### Working principles for agents
- **CLI-first approach**: Use unified `rap_scraper_cli.py` as primary interface
- **Organized exploration**: Check `scripts/tools/`, `scripts/development/` for utilities
- **Systematic investigation**: Use semantic_search → grep_search → read_file workflow
- **Clean architecture**: Respect separation between development/, legacy/, tools/
- **Production readiness**: Test changes thoroughly, document in PROJECT_DIARY
- **Batch-friendly**: Use `scripts/tools/batch_ai_analysis.py` for large operations

### Success metrics
- Code quality: types, tests, documentation
- Pipeline reliability: resume capability, error handling
- ML readiness: feature completeness and consistency
- Developer UX: clear CLI and debugging tools

---

**This project is a showcase of modern ML engineering practices**, emphasizing data quality, API integration, and production readiness for conditional text generation.