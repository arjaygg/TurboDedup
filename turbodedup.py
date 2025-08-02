#!/usr/bin/env python3
"""
TurboDedup Entry Point

This script provides a convenient entry point for running TurboDedup
without requiring package installation.

Usage:
    python3 turbodedup.py [options]
    
This is equivalent to:
    python3 -m turbodedup.cli.main [options]
"""

import sys
import os

# Add the current directory to Python path so we can import turbodedup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from turbodedup.cli.main import main
    main()