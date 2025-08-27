#!/usr/bin/env python3
"""Spotify enhancement entry point."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.enhancers.spotify_enhancer import main

if __name__ == "__main__":
    main()
