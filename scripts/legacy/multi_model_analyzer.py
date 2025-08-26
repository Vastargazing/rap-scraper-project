#!/usr/bin/env python3
"""Backward compatibility wrapper for multi_model_analyzer.py"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analyzers.multi_model_analyzer import main

if __name__ == "__main__":
    main()
