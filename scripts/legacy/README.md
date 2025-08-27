# 📦 Legacy Scripts

This folder contains legacy scripts that have been replaced by the unified CLI interface.

## 📋 Contents

### 🕷️ Scraping Scripts
- **`run_scraping.py`** - Simple wrapper for main scraper
  - **Replaced by**: `python scripts/rap_scraper_cli.py scraping`

### 🤖 Analysis Scripts  
- **`run_analysis.py`** - Multi-model analysis wrapper
  - **Replaced by**: `python scripts/rap_scraper_cli.py analysis --analyzer multi`
  
- **`run_gemma_analysis.py`** - Gemma analysis wrapper
  - **Replaced by**: `python scripts/rap_scraper_cli.py analysis --analyzer gemma`

### 🔧 Utilities
- **`check_db.py`** - Database check utility (legacy version)
- **`multi_model_analyzer.py`** - Legacy analyzer (moved from scripts root)
- **`rap_scraper_optimized.py`** - Legacy scraper (moved from scripts root)

## 🎯 Why Moved?

These scripts were **simple wrappers** that just called main functions without adding value:

```python
# Typical legacy script pattern
from src.module import main
if __name__ == "__main__":
    main()
```

## 🚀 Migration Guide

### Old way:
```bash
python scripts/run_scraping.py
python scripts/run_analysis.py  
python scripts/run_gemma_analysis.py
```

### New way (CLI):
```bash
python scripts/rap_scraper_cli.py scraping
python scripts/rap_scraper_cli.py analysis --analyzer multi
python scripts/rap_scraper_cli.py analysis --analyzer gemma
```

## ✅ Benefits of CLI Approach

1. **Single Entry Point** - One command for all operations
2. **Consistent Interface** - Same argument patterns everywhere
3. **More Options** - Debug, test, single artist modes
4. **Better Help** - Built-in documentation and examples
5. **Less Code** - No duplicate wrapper scripts

---

*These scripts are kept for reference and emergency fallback, but the CLI is now the recommended approach.*
