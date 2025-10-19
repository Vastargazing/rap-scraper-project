#!/usr/bin/env python3
"""
Legacy scraping entry point with backward compatibility.

⚠️  DEPRECATED: Consider using the main CLI instead:
    python scripts/rap_scraper_cli.py scraping

This script is kept for backward compatibility and simple automation.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.scrapers.rap_scraper_optimized import main

if __name__ == "__main__":
    print("🔄 Legacy scraper starting...")
    print("💡 Tip: Use 'python scripts/rap_scraper_cli.py scraping' for more options")
    print("=" * 60)
    main()
