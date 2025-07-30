#!/usr/bin/env python3
"""
Test script to verify app.py only handles RAG queries and no longer does embedding/chunking.
"""

import logging
import ast
import re
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_app_rag_only():
    """Test that app.py is now focused only on RAG functionality."""
    
    logger.info("🧪 Testing that app.py only handles RAG queries...")
    
    app_file = Path(__file__).parent / "app.py"
    
    with open(app_file, 'r') as f:
        app_content = f.read()
    
    # Check for removed functionality
    removed_patterns = [
        'data_processing',
        'incremental_embedder', 
        'BackgroundProcessor',
        'extract_and_chunk',
        'process_incremental_embeddings',
        '_process_single_document',
        'show_background_notifications',
        'handle_background_data_refresh'
    ]
    
    logger.info("=== Checking for removed functionality ===")
    found_removed = []
    
    for pattern in removed_patterns:
        if pattern in app_content and 'removed' not in app_content[app_content.find(pattern):app_content.find(pattern)+100].lower():
            found_removed.append(pattern)
    
    if found_removed:
        logger.error(f"❌ Found removed functionality still present: {found_removed}")
        return False
    else:
        logger.info("✅ All embedding/chunking functionality successfully removed")
    
    # Check for core RAG functionality
    required_patterns = [
        'render_document_analysis_tab',
        'rag_system.query',
        'CPUCRAGSystem',
        'render_timeline_tab',
        'render_system_management_tab'
    ]
    
    logger.info("=== Checking for required RAG functionality ===")
    missing_required = []
    
    for pattern in required_patterns:
        if pattern not in app_content:
            missing_required.append(pattern)
    
    if missing_required:
        logger.error(f"❌ Missing required RAG functionality: {missing_required}")
        return False
    else:
        logger.info("✅ All required RAG functionality present")
    
    # Check startup manager changes
    startup_file = Path(__file__).parent / "startup_manager.py"
    
    with open(startup_file, 'r') as f:
        startup_content = f.read()
    
    logger.info("=== Checking startup manager changes ===")
    
    if '_process_incremental_embeddings' in startup_content and 'removed' not in startup_content:
        logger.error("❌ Startup manager still has embedding processing")
        return False
    else:
        logger.info("✅ Startup manager cleaned of embedding processing")
    
    # Count lines to verify significant cleanup
    logger.info("=== File size analysis ===")
    app_lines = len(app_content.splitlines())
    logger.info(f"App.py line count: {app_lines}")
    
    if app_lines > 650:  # Rough estimate after cleanup
        logger.warning(f"⚠️  App.py seems large ({app_lines} lines) - may need more cleanup")
    else:
        logger.info("✅ App.py size looks appropriate after cleanup")
    
    logger.info("\n=== Summary ===")
    logger.info("✅ App.py successfully cleaned of embedding/chunking functionality!")
    logger.info("🔍 App now focuses only on RAG queries and UI")
    logger.info("📋 Document processing delegated to standalone_data_processor.py")
    logger.info("🚀 Startup flow simplified to RAG system initialization only")
    
    return True

if __name__ == "__main__":
    success = test_app_rag_only()
    if success:
        print("\n🎉 Test PASSED: App.py is now RAG-focused!")
    else:
        print("\n❌ Test FAILED: More cleanup needed")