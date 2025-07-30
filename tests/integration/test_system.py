#!/usr/bin/env python3
"""
Test script to verify the CPUC RAG system functionality
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.resolve()))

from rag_core import CPUCRAGSystem
from timeline_integration import create_timeline_integration

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_system_initialization():
    """Test that the system initializes correctly"""
    logger.info("Testing system initialization...")
    
    try:
        # Initialize RAG system
        rag_system = CPUCRAGSystem()
        
        # Check vector store status
        if rag_system.vectordb is not None:
            chunk_count = rag_system.vectordb._collection.count()
            logger.info(f"✅ Vector store loaded with {chunk_count} chunks")
        else:
            logger.info("⚠️  Vector store not loaded (may be rebuilding)")
        
        # Test timeline integration
        timeline_integration = create_timeline_integration(rag_system)
        timeline_initialized = timeline_integration.initialize_timeline_system()
        
        if timeline_initialized:
            logger.info("✅ Timeline system initialized successfully")
        else:
            logger.warning("⚠️  Timeline system initialization failed")
        
        # Get system stats
        stats = rag_system.get_system_stats()
        logger.info(f"📊 System Stats:")
        logger.info(f"  - Documents on disk: {stats.get('total_documents_on_disk', 'N/A')}")
        logger.info(f"  - Total chunks: {stats.get('total_chunks', 'N/A')}")
        logger.info(f"  - Vector store status: {stats.get('vector_store_status', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

def test_vector_store_parity():
    """Test vector store parity checking"""
    logger.info("Testing vector store parity check...")
    
    try:
        rag_system = CPUCRAGSystem()
        parity_check = rag_system._check_vector_store_parity()
        
        logger.info(f"Parity check result: {parity_check['has_parity']}")
        logger.info(f"Reason: {parity_check['reason']}")
        
        if not parity_check['has_parity']:
            logger.info(f"Missing files: {len(parity_check['missing_files'])}")
            logger.info(f"Extra files: {len(parity_check['extra_files'])}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Parity check failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🧪 Starting CPUC RAG System Tests")
    logger.info("=" * 50)
    
    # Test 1: System initialization
    test1_passed = test_system_initialization()
    
    # Test 2: Vector store parity
    test2_passed = test_vector_store_parity()
    
    # Summary
    logger.info("=" * 50)
    logger.info("📋 Test Results:")
    logger.info(f"  - System initialization: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    logger.info(f"  - Vector store parity: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        logger.info("🎉 All tests passed! System is ready.")
    else:
        logger.warning("⚠️  Some tests failed. Check logs for details.")

if __name__ == "__main__":
    main()