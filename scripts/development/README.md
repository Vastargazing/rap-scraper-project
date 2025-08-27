# 🔧 Development Scripts

This folder contains development and testing scripts that were created during the scraper optimization process.

## 📋 Scripts Overview

### 🎤 Single Artist Testing
- **`scrape_artist_one.py`** - Test scraping for a single artist
  ```bash
  python development/scrape_artist_one.py "Drake"
  ```

### 🧪 Testing & Debugging
- **`test_fixed_scraper.py`** - Test the fixed scraper on multiple artists
- **`run_scraping_debug.py`** - Debug mode with detailed logging and file analysis

### 🚀 Enhanced Modes
- **`run_scraping_improved.py`** - Improved scraper with additional checks
- **`run_remaining_artists.py`** - Continue scraping remaining artists from list

## 🎯 Usage via CLI

**Recommended approach**: Use the main CLI instead of running these scripts directly:

```bash
# Single artist
python scripts/rap_scraper_cli.py scraping --artist "Drake"

# Test mode
python scripts/rap_scraper_cli.py scraping --test

# Debug mode
python scripts/rap_scraper_cli.py scraping --debug

# Continue remaining
python scripts/rap_scraper_cli.py scraping --continue
```

## 📝 Notes

- These scripts were created to solve specific proxy and connection issues
- They implement enhanced error handling and retry logic
- All functionality is now integrated into the main CLI
- Keep for reference and development purposes

## 🔄 Migration

If you were using these scripts directly, update your workflow:

```bash
# Old way
python scripts/scrape_artist_one.py "Artist Name"

# New way (CLI)
python scripts/rap_scraper_cli.py scraping --artist "Artist Name"
```
