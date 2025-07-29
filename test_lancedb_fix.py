#!/usr/bin/env python3
"""
Test script to verify LanceDB schema fix
"""

import logging
from rag_core import CPUCRAGSystem
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_lancedb_addition():
    """Test adding documents to existing LanceDB"""
    
    test_proceeding = 'R1202009'
    logger.info(f"Testing LanceDB addition for: {test_proceeding}")
    
    # Initialize RAG system
    rag_system = CPUCRAGSystem(current_proceeding=test_proceeding)
    
    # Get a test URL
    from config import get_proceeding_file_paths
    proceeding_paths = get_proceeding_file_paths(test_proceeding)
    
    history_file = proceeding_paths['scraped_pdf_history']
    if not history_file.exists():
        history_file = proceeding_paths['scraped_pdf_history_alt']
    
    with open(history_file, 'r') as f:
        scraped_data = json.load(f)
    
    # Get first URL that hasn't been processed yet
    first_record = next(iter(scraped_data.values()))
    test_url_data = {
        'url': first_record.get('url'),
        'title': first_record.get('title', first_record.get('filename', 'test')),
        'id': 'test_doc'
    }
    
    logger.info(f"Testing with URL: {test_url_data['url']}")
    
    # Process single URL
    try:
        result = rag_system._process_single_url(test_url_data)
        
        if result['success'] and result['chunks']:
            logger.info(f"✅ Processing successful: {result['chunk_count']} chunks")
            
            # Test adding to vector store
            url_hash = result.get('url_hash', 'test_hash')
            success = rag_system.add_document_incrementally(
                result['chunks'], 
                url_hash, 
                test_url_data
            )
            
            if success:
                logger.info("✅ Successfully added to LanceDB!")
                return True
            else:
                logger.error("❌ Failed to add to LanceDB")
                return False
        else:
            logger.error(f"❌ Processing failed: {result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == '__main__':
    print("🧪 Testing LanceDB Schema Fix")
    print("=" * 50)
    
    success = test_lancedb_addition()
    
    if success:
        print("✅ LanceDB test PASSED")
    else:
        print("❌ LanceDB test FAILED")
    
    exit(0 if success else 1)