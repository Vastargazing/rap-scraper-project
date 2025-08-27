# 🛠️ Tools Directory

Collection of standalone utilities and diagnostic tools for the rap scraper project.

## Available Tools

### 🤖 batch_ai_analysis.py
**Batch AI Analysis Tool** - Production-grade dataset processor for large-scale AI analysis.

```bash
# Basic usage
python scripts/tools/batch_ai_analysis.py --batch-size 25

# Custom configuration  
python scripts/tools/batch_ai_analysis.py --batch-size 50 --sleep 2 --max-batches 100

# Test run (dry-run mode)
python scripts/tools/batch_ai_analysis.py --dry-run
```

**Features:**
- 📊 Configurable batch processing
- 🔄 Automatic resume support  
- ⏸️ Graceful interruption handling
- 🧪 Dry-run mode for testing
- 📈 Progress tracking and metrics

### 📊 check_spotify_coverage.py
**Spotify Coverage Analyzer** - Quick diagnostic tool for Spotify data completeness.

```bash
python scripts/tools/check_spotify_coverage.py
```

### 🎨 create_cli_showcase.py  
**CLI Showcase Generator** - Creates demonstration materials for project presentation.

```bash
python scripts/tools/create_cli_showcase.py
```

## Usage Guidelines

- These tools are **standalone utilities** that don't require CLI integration
- Use for **debugging**, **diagnostics**, and **specialized processing**
- All tools support `--help` for detailed usage information
- Tools are designed to be **safe** and **non-destructive**

## Development

When adding new tools:
1. Include comprehensive help text and examples
2. Add error handling and logging
3. Support dry-run mode where applicable
4. Update this README
