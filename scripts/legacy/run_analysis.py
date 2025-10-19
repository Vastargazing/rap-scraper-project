#!/usr/bin/env python3
"""Main analysis entry point."""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.analyzers.multi_model_analyzer import main

if __name__ == "__main__":
    main()
