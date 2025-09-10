#!/usr/bin/env python3
"""Backward compatibility wrapper for rap_scraper_optimized.py"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.scrapers.rap_scraper_optimized import main

if __name__ == "__main__":
    main()
