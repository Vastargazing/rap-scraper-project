#!/usr/bin/env python3
"""Main scraping entry point with backward compatibility."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.scrapers.rap_scraper_optimized import main

if __name__ == "__main__":
    main()
