# 🎵 Rap Lyrics Scraper & Analyzer

**Production-ready ML pipeline for collecting and analyzing rap lyrics using AI**

📊 **54.5K+ tracks | 345 artists | Spotify enriched | AI analyzed**

## 🚀 Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
Create a `.env` file:
```bash
GENIUS_TOKEN=your_genius_token_here
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
GOOGLE_API_KEY=your_google_api_key_here  # Optional
```

### 3. Main CLI interface
```bash
# Check project status
python scripts/rap_scraper_cli.py status

# Scrape new data
python scripts/rap_scraper_cli.py scraping

# Continue Spotify metadata enrichment
python scripts/rap_scraper_cli.py spotify --continue

# Run ML analysis
python scripts/rap_scraper_cli.py analysis --analyzer gemma

# Full help
python scripts/rap_scraper_cli.py --help
```

## 🏗️ Project architecture

### Structured layout (post-cleanup):
```
src/
├── scrapers/     # 🕷️ Data collection (Genius API)
├── enhancers/    # 🎵 Enrichment (Spotify API)  
├── analyzers/    # 🤖 ML analysis (LLM models)
├── models/       # 📊 Pydantic models
└── utils/        # 🛠️ Utilities and config

scripts/              # 🚀 Organized entry points  
├── rap_scraper_cli.py      # 🎯 Unified CLI interface
├── continue_spotify_enhancement.py  # Resume Spotify enhancement
├── run_spotify_enhancement.py      # New Spotify enhancement
├── check_db.py             # Database diagnostics
├── tools/                  # 🛠️ Production utilities
│   ├── batch_ai_analysis.py       # Batch AI processor (rehabilitated)
│   └── check_spotify_coverage.py  # Coverage diagnostics
├── development/            # 🧪 Development & testing
│   ├── test_fixed_scraper.py      # Test scraper fixes
│   ├── scrape_artist_one.py       # Single artist testing
│   └── run_scraping_debug.py      # Debug mode
├── legacy/                 # 🗂️ Backwards compatibility
│   └── run_analysis.py            # Legacy analysis wrapper
└── utils/                  # 🔧 Project utilities
    └── cleanup_project.py         # Project organization tool

monitoring/       # 📊 Monitoring and stats
data/             # 📄 Database and files
results/          # 📈 Analysis outputs
docs/             # 📚 Documentation
```

## � Contextual documentation for AI-assisted development

This repository includes agent-friendly, contextual documentation designed to help AI assistants onboard and act quickly. Key files:

- `docs/claude.md` — a prioritized project context file with architecture, workflows, CLI examples, and investigation protocols for AI agents.
- `AI_ONBOARDING_CHECKLIST.md` — an onboarding checklist and step-by-step command templates created specifically for autonomous or assisted agents.

These documents provide a layered reading order, command snippets, and troubleshooting protocols so an AI assistant can immediately understand the codebase and run actionable tasks.


## 🔧 Core commands (updated)

### 🕷️ Scraping
```bash
# Unified CLI (recommended)
python scripts/rap_scraper_cli.py scraping                    # Production mode
python scripts/rap_scraper_cli.py scraping --test            # Test mode (3 artists)
python scripts/rap_scraper_cli.py scraping --artist "Drake"  # Single artist

# Development & testing
python scripts/development/test_fixed_scraper.py             # Test scraper fixes
python scripts/development/scrape_artist_one.py "Artist"    # Single artist testing
```

### 🎵 Spotify enrichment
```bash
# Via unified CLI
python scripts/rap_scraper_cli.py spotify                    # New enhancement
python scripts/rap_scraper_cli.py spotify --continue        # Resume existing

# Direct invocation
python scripts/continue_spotify_enhancement.py              # Resume enhancement
python scripts/run_spotify_enhancement.py                   # New enhancement
```

### 🤖 ML analysis & batch processing
```bash
# Unified CLI analysis
python scripts/rap_scraper_cli.py analysis --analyzer gemma  # Gemma 27B
python scripts/rap_scraper_cli.py analysis --analyzer multi  # Multi-model comparison

# NEW: ML Feature Engineering (17 features)
python scripts/rap_scraper_cli.py mlfeatures                 # All tracks with simplified features
python scripts/rap_scraper_cli.py mlfeatures --limit 50      # Process 50 tracks only
python scripts/rap_scraper_cli.py mlfeatures --export results.json  # Export to JSON

# Production batch processing (rehabilitated tool)
python scripts/tools/batch_ai_analysis.py --batch-size 25    # Batch AI analysis
python scripts/tools/batch_ai_analysis.py --dry-run          # Test run

# Legacy compatibility
python scripts/legacy/run_analysis.py                        # Legacy analysis wrapper
```

### �️ Utilities & diagnostics
```bash
# Database monitoring
python scripts/rap_scraper_cli.py monitoring --component database
python scripts/rap_scraper_cli.py monitoring --component analysis
python scripts/rap_scraper_cli.py monitoring --component gemma

# Coverage & diagnostics (new organized tools)
python scripts/tools/check_spotify_coverage.py              # Spotify coverage analysis

# Utilities via CLI
python scripts/rap_scraper_cli.py utilities --utility spotify-setup

# Direct database check
python scripts/check_db.py                                  # Database diagnostics
```

## 🗄️ Database

### Data layout (updated)
- **Primary DB**: `data/rap_lyrics.db`
- **Songs table**: `songs` (54,568 records)
- **AI analyses pending**: ~36,929 tracks awaiting analysis
- **Spotify data**: `spotify_artists` (~97.3% coverage)
- **Artists config**: `data/rap_artists.json`

### Table schemas
```sql
-- Main songs table
songs: artist, song, lyrics, url, scraped_at, album, year

-- AI analyses
ai_analysis: song_id, complexity, mood, genre, quality_score, analysis_text

-- Spotify metadata
spotify_artists: genius_name, spotify_id, name, followers, genres, popularity
```

## 🤖 AI models & analyzers

| Model | Speed | Quality | Usage | File |
|-------|-------|---------|-------|------|
| **Gemma 3 27B** | ⚡⚡ | ⭐⭐⭐⭐⭐ | Production analysis | `src/analyzers/gemma_27b_fixed.py` |
| **Multi-model** | ⚡ | ⭐⭐⭐⭐⭐ | Model comparison | `src/analyzers/multi_model_analyzer.py` |
| **ML Feature Extractor** | ⚡⚡⚡ | ⭐⭐⭐⭐ | Feature engineering | `src/analyzers/simplified_feature_analyzer.py` |
| **Advanced ML Features** | ⚡⚡ | ⭐⭐⭐⭐⭐ | NLTK-powered analysis | `src/analyzers/advanced_feature_analyzer.py` |
| **Batch Processor** | ⚡ | ⭐⭐⭐⭐ | Large-scale analysis | `scripts/tools/batch_ai_analysis.py` |

### Metrics analyzed
**AI Analysis:**
- **Complexity**: Linguistic complexity (1-10)
- **Mood**: Sentiment / tonal classification (positive/negative/neutral)
- **Genre**: Rap subgenre (trap, conscious, etc.)
- **Quality**: Text quality score (1-10)
- **Themes**: Key themes and motifs

**ML Feature Engineering (17 features):**
- **Rhyme Analysis**: Rhyme density, perfect/near rhymes, rhyme scheme patterns
- **Vocabulary Diversity**: Type-Token Ratio (TTR), unique word percentages
- **Metaphor Detection**: Metaphorical language patterns and wordplay
- **Flow Patterns**: Syllable analysis, stress patterns, rhythm metrics
- **Structural Features**: Verse/chorus detection, repetition analysis
- **Linguistic Metrics**: Sentence complexity, word lengths, readability scores

## 📈 Current stats

### Project data
- **54,568** songs in the database
- **345** artists (334 enriched with Spotify)
- **17,639** high-quality AI analyses
- **213MB** total database size

### Coverage
- **Genius API**: 100% availability
- **Spotify API**: 97.1% enrichment success (334/345 artists)
- **AI analysis**: ~32.3% of the full dataset (17,639/54,568)
- **Error handling**: critical errors covered

## 🔧 System requirements

### Minimum
- **Python 3.8+** (3.11+ recommended)
- **SQLite** (bundled with Python)
- **16GB+ RAM** (for Gemma 27B analysis)
- **5GB+ disk** (for the full dataset and models)

### API keys
- **Genius API** token (required)
- **Spotify API** credentials (client_id + client_secret)
- **Google AI Studio** API key (for Gemma analysis)
- **Ollama** (optional, for local experimentation)

## 📁 Project layout (summary)

```
rap-scraper-project/
├── src/                    # Core code
│   ├── scrapers/           # 🕷️ Data collection (Genius API)
│   ├── enhancers/          # 🎵 Spotify enrichment
│   ├── analyzers/          # 🤖 ML analysis
│   ├── models/             # 📊 Pydantic models
│   └── utils/              # 🛠️ Utilities & config
├── scripts/                # 🚀 Entry points and CLI
│   ├── rap_scraper_cli.py  # 🎯 Main CLI
│   ├── run_*.py            # 🏃 Direct entry points
│   └── legacy/             # 🗂️ Backwards compatibility
├── monitoring/             # Monitoring & logs
├── data/                   # Database & files
├── results/                # Analysis outputs
├── tests/                  # Unit tests
└── docs/                   # Documentation

# Legacy files (archived)
scripts/archive/           # Old scripts
```

## 💡 Examples

### 🚀 Quick CLI workflow
```bash
# 1. Check project status
python scripts/rap_scraper_cli.py status

# 2. Run the full pipeline
python scripts/rap_scraper_cli.py scraping          # Scrape data
python scripts/rap_scraper_cli.py spotify --continue # Spotify enrichment
python scripts/rap_scraper_cli.py analysis --analyzer gemma # AI analysis

# 3. # 3. Monitor progress
python scripts/rap_scraper_cli.py monitoring --component all
```

### Detailed commands
```bash
# Database stats
python scripts/rap_scraper_cli.py monitoring --component database

# Monitor all components at once
python scripts/rap_scraper_cli.py monitoring --component all

# Run analysis with a specific algorithm
python scripts/rap_scraper_cli.py analysis --analyzer multi --limit 100

# Project cleanup (dry run)
python scripts/rap_scraper_cli.py utils --utility cleanup

# Migrate DB with backup
python scripts/rap_scraper_cli.py utils --utility migrate
```

### 🛠️ Direct invocations (advanced)
```bash
# Direct component runs
python scripts/run_scraping.py                # Scraping
python scripts/continue_spotify_enhancement.py # Spotify enrichment
python scripts/run_gemma_analysis.py          # AI analysis

# Legacy compatibility
python scripts/legacy/rap_scraper_optimized.py
python scripts/legacy/multi_model_analyzer.py
```

## ✅ Results & takeaways

### Achievements
- ✅ **53,300+** collected tracks with full lyrics
- ✅ **99.6%** Spotify enrichment success (327/328 artists)
- ✅ **15,000+** high-quality AI analyses
- ✅ **Production-ready** architecture with organized CLI
- ✅ **Fully automated** collection & analysis pipeline
- ✅ **Cleaned codebase** with organized scripts structure

### Data quality
- **Lyrics coverage**: 100% for collected tracks
- **Metadata accuracy**: 99%+ (Spotify)
- **AI analysis quality**: expert-rated 9/10
- **Data consistency**: fully validated with Pydantic models

## 🚨 Troubleshooting

### Common issues
```bash
# Database diagnostics after cleanup
python scripts/check_db.py

# Import errors from legacy scripts
# Use the new CLI: python scripts/rap_scraper_cli.py --help

# Path problems after restructuring  
# All scripts moved to organized folders: scripts/{tools,development,legacy}/

# Spotify API rate limits are handled automatically
# Gemma analysis uses batch processing for efficiency
```

### Support
- 📖 Detailed docs in `docs/`
- 🐛 Report issues via Git
- 📊 Monitoring via `monitoring/` scripts
- 🧪 Unit tests in `tests/`

---

**Created with ❤️ by AI Engineer | Production ML Pipeline | 2025**

## 📚 Documentation

Detailed project documentation is stored in `AI_Engineer_Journal/Projects/Rap_Scraper_Project/`:
- `README.md` - Presentation version
- `PROJECT_EVOLUTION.md` - Development history
- `TECH_SUMMARY.md` - Technical overview
- `INTERVIEW_PREPARATION.md` - Interview prep

## 🎯 Project status

- ✅ Scraping: stable with proxy handling
- ✅ AI analysis: multiple models available (Gemma 27B, Multi-model)
- ✅ Monitoring: real-time via CLI
- ✅ Database: 54K+ records
- ✅ Codebase: organized and cleaned up
- 🔄 In progress: batch AI analysis (~37K tracks pending)
