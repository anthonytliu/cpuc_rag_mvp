#!/usr/bin/env python3
"""
Test Scraped Data Loading Fix

This script tests the fix for loading scraped PDF data when the file uses
proceeding-specific naming (e.g., R1311005_scraped_pdf_history.json instead
of scraped_pdf_history.json).
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from standalone_data_processor import load_scraped_pdf_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_scraped_data_loading():
    """Test loading scraped data for different naming patterns."""
    logger.info("🧪 Testing scraped PDF data loading fix")
    
    test_cases = [
        {
            'proceeding': 'R2207005',
            'expected_files': ['scraped_pdf_history.json', 'R2207005_scraped_pdf_history.json'],
            'should_work': True
        },
        {
            'proceeding': 'R1311005', 
            'expected_files': ['R1311005_scraped_pdf_history.json'],
            'should_work': True
        }
    ]
    
    for test_case in test_cases:
        proceeding = test_case['proceeding']
        logger.info(f"\n--- Testing {proceeding} ---")
        
        # Check which files exist
        proceeding_dir = Path(f"cpuc_proceedings/{proceeding}")
        if not proceeding_dir.exists():
            logger.warning(f"Proceeding directory doesn't exist: {proceeding_dir}")
            continue
            
        logger.info(f"Proceeding directory: {proceeding_dir}")
        
        # List expected files and check which exist
        for expected_file in test_case['expected_files']:
            file_path = proceeding_dir / expected_file
            exists = "✅" if file_path.exists() else "❌"
            logger.info(f"  {exists} {expected_file}")
        
        # Test the loader
        try:
            scraped_data = load_scraped_pdf_data(proceeding)
            
            if scraped_data:
                logger.info(f"✅ Successfully loaded {len(scraped_data)} documents")
                
                # Show sample data
                if scraped_data:
                    first_key = next(iter(scraped_data.keys()))
                    sample_doc = scraped_data[first_key]
                    logger.info(f"  Sample document: {sample_doc.get('title', 'Unknown')}")
                    logger.info(f"  Sample URL: {sample_doc.get('url', 'Unknown')}")
                
                if test_case['should_work']:
                    logger.info(f"✅ Test passed for {proceeding}")
                else:
                    logger.error(f"❌ Test failed: Expected failure but got success for {proceeding}")
            else:
                if test_case['should_work']:
                    logger.error(f"❌ Test failed: Expected success but got failure for {proceeding}")
                else:
                    logger.info(f"✅ Test passed: Expected failure for {proceeding}")
                    
        except Exception as e:
            logger.error(f"❌ Test failed with exception for {proceeding}: {e}")
    
    logger.info("\n🏁 Test completed")


def test_specific_proceedings():
    """Test specific proceedings that were having issues."""
    logger.info("🔍 Testing specific problematic proceedings")
    
    # Test proceedings that might have naming pattern issues
    test_proceedings = []
    
    # Find all proceeding directories
    cpuc_dir = Path("cpuc_proceedings")
    if cpuc_dir.exists():
        for proc_dir in cpuc_dir.iterdir():
            if proc_dir.is_dir() and proc_dir.name.startswith('R'):
                test_proceedings.append(proc_dir.name)
    
    logger.info(f"Found {len(test_proceedings)} proceeding directories")
    
    successful_loads = 0
    failed_loads = 0
    
    for proceeding in sorted(test_proceedings)[:10]:  # Test first 10
        logger.info(f"\nTesting {proceeding}...")
        
        try:
            scraped_data = load_scraped_pdf_data(proceeding)
            if scraped_data:
                logger.info(f"  ✅ Loaded {len(scraped_data)} documents")
                successful_loads += 1
            else:
                logger.info(f"  ❌ No data loaded")
                failed_loads += 1
        except Exception as e:
            logger.error(f"  💥 Exception: {e}")
            failed_loads += 1
    
    logger.info(f"\n📊 Results: {successful_loads} successful, {failed_loads} failed")


def main():
    logger.info("=" * 60)
    logger.info("SCRAPED DATA LOADING TEST")
    logger.info("=" * 60)
    
    # Test 1: Known test cases
    test_scraped_data_loading()
    
    # Test 2: Survey multiple proceedings
    logger.info("\n" + "-" * 60)
    test_specific_proceedings()
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()