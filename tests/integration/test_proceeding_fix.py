#!/usr/bin/env python3
"""
Test script to verify proceeding context propagation fix.

This script tests that the proceeding parameter is properly passed through
the function call chain to resolve the mismatch issue.
"""

import json
import logging
from pathlib import Path
import data_processing

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_proceeding_context():
    """Test that proceeding context is properly propagated"""
    
    test_proceeding = 'R1202009'
    logger.info(f"Testing proceeding context with: {test_proceeding}")
    
    # Load scraped data to get a test URL
    from core.config import get_proceeding_file_paths
    proceeding_paths = get_proceeding_file_paths(test_proceeding)
    
    history_file = proceeding_paths['scraped_pdf_history']
    if not history_file.exists():
        history_file = proceeding_paths['scraped_pdf_history_alt']
    
    if not history_file.exists():
        logger.error(f"No scraped PDF history found for {test_proceeding}")
        return False
    
    with open(history_file, 'r') as f:
        scraped_data = json.load(f)
    
    if not scraped_data:
        logger.error(f"No scraped data found")
        return False
    
    # Get first URL for testing
    first_record = next(iter(scraped_data.values()))
    test_url = first_record.get('url')
    test_title = first_record.get('title', first_record.get('filename', 'test'))
    
    if not test_url:
        logger.error(f"No URL found in first record")
        return False
    
    logger.info(f"Testing with URL: {test_url}")
    logger.info(f"Testing with title: {test_title}")
    
    # Test the function with proceeding context
    try:
        chunks = data_processing.extract_and_chunk_with_docling_url(
            test_url, test_title, test_proceeding
        )
        
        logger.info(f"Successfully processed {len(chunks)} chunks")
        
        if chunks:
            # Check first chunk metadata
            first_chunk = chunks[0]
            metadata = first_chunk.metadata
            
            logger.info(f"First chunk metadata keys: {list(metadata.keys())}")
            logger.info(f"Proceeding number in metadata: {metadata.get('proceeding_number', 'NOT_FOUND')}")
            logger.info(f"Source URL: {metadata.get('source_url', 'NOT_FOUND')}")
            logger.info(f"Document type: {metadata.get('document_type', 'NOT_FOUND')}")
            
            return True
        else:
            logger.warning("No chunks extracted")
            return False
            
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        return False

if __name__ == '__main__':
    print("🧪 Testing Proceeding Context Fix")
    print("=" * 50)
    
    success = test_proceeding_context()
    
    if success:
        print("✅ Proceeding context test PASSED")
    else:
        print("❌ Proceeding context test FAILED")
    
    exit(0 if success else 1)