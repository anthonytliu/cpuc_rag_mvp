#!/usr/bin/env python3
"""
Test script for the PDF Scheduler functionality
"""

import sys
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.resolve()))

from pdf_scheduler import create_pdf_scheduler
from pdf_scraper_core import check_for_new_pdfs
from rag_core import CPUCRAGSystem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_scraper_core():
    """Test the core scraper functionality"""
    logger.info("Testing PDF scraper core...")
    
    try:
        # Test checking for new PDFs
        new_urls, metadata = check_for_new_pdfs("R2207005", headless=True)
        
        logger.info(f"Found {len(new_urls)} new PDFs")
        if new_urls:
            logger.info("Sample new URLs:")
            for i, url in enumerate(new_urls[:3]):
                logger.info(f"  {i+1}. {url}")
        
        logger.info(f"Metadata: {metadata}")
        
        return True
        
    except Exception as e:
        logger.error(f"Scraper core test failed: {e}")
        return False

def test_scheduler():
    """Test the scheduler functionality"""
    logger.info("Testing PDF scheduler...")
    
    try:
        # Create a scheduler (without RAG system for testing)
        scheduler = create_pdf_scheduler(rag_system=None, check_interval_hours=24)  # 24 hours for testing
        
        # Test status methods
        status = scheduler.get_status()
        logger.info(f"Initial scheduler status: {status}")
        
        # Test force check
        logger.info("Testing force check...")
        scheduler.start()
        
        # Wait a moment
        time.sleep(2)
        
        # Get status again
        status = scheduler.get_status()
        logger.info(f"Scheduler status after start: {status}")
        
        # Stop the scheduler
        scheduler.stop()
        
        logger.info("Scheduler test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Scheduler test failed: {e}")
        return False

def test_integration():
    """Test integration with RAG system"""
    logger.info("Testing scheduler integration with RAG system...")
    
    try:
        # Initialize RAG system
        rag_system = CPUCRAGSystem()
        
        # Create scheduler with RAG system
        scheduler = create_pdf_scheduler(rag_system=rag_system, check_interval_hours=24)
        
        # Test callbacks
        callback_called = False
        
        def on_new_pdfs(count):
            nonlocal callback_called
            callback_called = True
            logger.info(f"Callback: {count} new PDFs downloaded")
        
        scheduler.on_new_pdfs_downloaded = on_new_pdfs
        
        # Start scheduler
        scheduler.start()
        
        # Wait a moment
        time.sleep(2)
        
        # Get status
        status = scheduler.get_status()
        logger.info(f"Integration test status: {status}")
        
        # Stop scheduler
        scheduler.stop()
        
        logger.info("Integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Starting PDF Scheduler Tests")
    logger.info("=" * 50)
    
    # Test 1: Scraper core
    test1_passed = test_scraper_core()
    
    # Test 2: Scheduler
    test2_passed = test_scheduler()
    
    # Test 3: Integration
    test3_passed = test_integration()
    
    # Summary
    logger.info("=" * 50)
    logger.info("📋 Test Results:")
    logger.info(f"  - Scraper core: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    logger.info(f"  - Scheduler: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    logger.info(f"  - Integration: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        logger.info("🎉 All tests passed! Scheduler system is ready.")
    else:
        logger.warning("⚠️  Some tests failed. Check logs for details.")

if __name__ == "__main__":
    main()