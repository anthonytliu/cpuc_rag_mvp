#!/usr/bin/env python3
"""
CPUC RAG System - Main Application Entry Point

This is the main entry point for the CPUC RAG Streamlit application.
It handles the import path setup and launches the actual app from the src structure.

Usage:
    streamlit run app.py

Author: Claude Code
"""

import sys
from pathlib import Path

# Add src to Python path to enable imports
current_dir = Path.cwd()
src_path = current_dir / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import and run the main app
if __name__ == "__main__":
    try:
        from ui.app import main
        main()
    except ImportError as e:
        print(f"❌ Failed to import app: {e}")
        print("Please ensure you're running from the CPUC_REG_RAG root directory")
        sys.exit(1)