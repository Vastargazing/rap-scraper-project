# 🎵 Rap Lyrics Scraper & Analyzer

**Production-ready ML pipeline for collecting and analyzing rap lyrics using AI**

📊 **48K+ tracks | 263 artists | Spotify enriched | AI analyzed**

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

### Structured layout:
```
src/
├── scrapers/     # 🕷️ Data collection (Genius API)
├── enhancers/    # 🎵 Enrichment (Spotify API)
├── analyzers/    # 🤖 ML analysis (LLM models)
├── models/       # 📊 Pydantic models
└── utils/        # 🛠️ Utilities and config

scripts/         # 🚀 Entry points and CLI
monitoring/      # 📊 Monitoring and stats
data/             # 📄 Database and files
results/         # 📈 Analysis outputs
docs/            # 📚 Documentation
```

## 🔧 Core commands

### 🕷️ Scraping
```bash
# Recommended (new architecture)
python scripts/rap_scraper_cli.py scraping

# Direct invocation
python scripts/run_scraping.py

# Legacy compatibility
python scripts/legacy/rap_scraper_optimized.py
```

### 🎵 Spotify enrichment
```bash
# Via CLI
python scripts/rap_scraper_cli.py spotify --continue

# Direct invocation
python scripts/continue_spotify_enhancement.py
```

### 🤖 ML analysis
```bash
# Gemma 27B (recommended)
python scripts/rap_scraper_cli.py analysis --analyzer gemma

# Model comparison
python scripts/rap_scraper_cli.py analysis --analyzer multi

# LangChain + OpenAI
python scripts/rap_scraper_cli.py analysis --analyzer langchain
```

### 📊 Monitoring
```bash
# Database status
python scripts/rap_scraper_cli.py monitoring --component database

# AI analysis progress
python scripts/rap_scraper_cli.py monitoring --component analysis

# Gemma monitoring
python scripts/rap_scraper_cli.py monitoring --component gemma
```

### 🛠️ Utilities
```bash
# Project cleanup (dry run)
python scripts/rap_scraper_cli.py utils --utility cleanup

# Perform cleanup
python scripts/rap_scraper_cli.py utils --utility cleanup --execute

# Migrate the database
python scripts/rap_scraper_cli.py utils --utility migrate

# Spotify setup helper
python scripts/rap_scraper_cli.py utils --utility spotify-setup
```

## 🗄️ Database

### Data layout
- **Primary DB**: `data/rap_lyrics.db`
- **Songs table**: `songs` (48,370+ records)
- **Analyses table**: `ai_analysis` (~1,500+ analyses)
- **Spotify data**: `spotify_artists` (262/263 artists enriched)
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
| **LangChain GPT** | ⚡⚡⚡ | ⭐⭐⭐⭐ | Fast analysis | `src/analyzers/langchain_analyzer.py` |
| **Multi-model** | ⚡ | ⭐⭐⭐⭐⭐ | Model comparison | `src/analyzers/multi_model_analyzer.py` |

### Metrics analyzed
- **Complexity**: Linguistic complexity (1-10)
- **Mood**: Sentiment / tonal classification (positive/negative/neutral)
- **Genre**: Rap subgenre (trap, conscious, etc.)
- **Quality**: Text quality score (1-10)
- **Themes**: Key themes and motifs

## 📈 Current stats

### Project data
- **48,370+** songs in the database
- **263** artists (262 enriched with Spotify)
- **1,500+** high-quality AI analyses
- **15GB+** total data size

### Coverage
- **Genius API**: 100% availability
- **Spotify API**: 99.6% enrichment success (262/263)
- **AI analysis**: ~3% of the full dataset (quality-focused)
- **Error handling**: critical errors covered

## 🔧 System requirements

### Minimum
- **Python 3.8+** (3.11+ recommended)
- **SQLite** (bundled with Python)
- **16GB+ RAM** (for Gemma 27B analysis)
- **50GB+ disk** (for the full dataset)

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

# 3. Monitor progress
python scripts/rap_scraper_cli.py monitoring --component all
```

### Detailed commands
```bash
# Database stats
python scripts/rap_scraper_cli.py monitoring --component database

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
- ✅ **48,370+** collected tracks with full lyrics
- ✅ **99.6%** Spotify enrichment success (262/263 artists)
- ✅ **1,500+** high-quality AI analyses
- ✅ **Production-ready** architecture with a CLI
- ✅ **Fully automated** collection & analysis pipeline

### Data quality
- **Lyrics coverage**: 100% for collected tracks
- **Metadata accuracy**: 99%+ (Spotify)
- **AI analysis quality**: expert-rated 9/10
- **Data consistency**: fully validated with Pydantic models

## 🚨 Troubleshooting

### Common issues
```bash
# Path problems after restructuring
python scripts/rap_scraper_cli.py utils --utility cleanup

# Import errors from legacy scripts
# Use the new CLI or the scripts/run_*.py entry points

# Database issues
python scripts/run_database_check.py

# Spotify API 403 responses may occur; they are handled automatically
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

- ✅ Scraping: stable
- ✅ AI analysis: multiple models available
- ✅ Monitoring: real-time
- ✅ Database: 47K+ records
- 🔄 In progress: full Gemma 27B analysis
