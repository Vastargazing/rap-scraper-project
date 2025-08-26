#!/usr/bin/env python3
"""Backward compatibility wrapper for check_db.py"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.check_db import main

if __name__ == "__main__":
    main()
