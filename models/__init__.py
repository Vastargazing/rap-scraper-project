# Models package for rap-scraper-project
"""
ML Models Package
================

This package contains machine learning models for rap lyrics analysis:
- test_qwen.py: QWEN primary ML model (ОСНОВНОЙ) 
- style_transfer.py: T5-based style transfer  
- quality_predictor.py: Quality assessment model
- trend_analysis.py: Trend analysis model
"""

__version__ = "1.0.0"
__author__ = "Rap Scraper Project"

# Import main models for easy access
try:
    from .style_transfer import *
    from .quality_predictor import *
    from .trend_analysis import *
    # QWEN model is imported separately in test_qwen.py when needed
except ImportError as e:
    print(f"⚠️ Some models not available: {e}")