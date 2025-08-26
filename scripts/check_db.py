#!/usr/bin/env python3
"""Database status check entry point."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.check_db import main

if __name__ == "__main__":
    main()
