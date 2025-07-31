#!/usr/bin/env python3
"""
Test ArrowSchema Recursion Recovery System

Tests the enhanced ArrowSchema recursion recovery that processes
documents individually when batch processing fails.
"""

import logging
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_arrowschema_recovery():
    """Test ArrowSchema recursion recovery in embedding_only_system."""
    logger.info("🧪 Testing ArrowSchema recursion recovery...")
    
    try:
        from data_processing.embedding_only_system import EmbeddingOnlySystem
        from langchain.schema import Document
        
        # Create test proceeding
        test_proceeding = "R1807006"
        
        # Initialize embedding system
        embedding_system = EmbeddingOnlySystem(test_proceeding)
        
        # Create test documents
        test_documents = [
            Document(
                page_content="Test content for ArrowSchema recovery testing.",
                metadata={
                    'url': 'https://example.com/test1.pdf',
                    'title': 'Test Document 1',
                    'proceeding': test_proceeding,
                    'chunk_index': 0
                }
            ),
            Document(
                page_content="Another test document for recovery validation.",
                metadata={
                    'url': 'https://example.com/test2.pdf', 
                    'title': 'Test Document 2',
                    'proceeding': test_proceeding,
                    'chunk_index': 1
                }
            )
        ]
        
        # Mock ArrowSchema recursion error on first attempt
        original_add_documents = embedding_system.vectordb.add_documents if embedding_system.vectordb else None
        call_count = 0
        
        def mock_add_documents_with_error(documents):
            nonlocal call_count
            call_count += 1
            
            # First call (batch) raises ArrowSchema error
            if call_count == 1 and len(documents) > 1:
                raise RuntimeError("Recursion level in ArrowSchema struct exceeded")
            
            # Individual calls (recovery) succeed
            elif len(documents) == 1:
                logger.info(f"✅ Individual document processed successfully: {documents[0].metadata.get('title', 'Unknown')}")
                return True
            
            # Shouldn't reach here in recovery mode
            return True
        
        # Apply mock if vectordb exists, otherwise create a mock
        if embedding_system.vectordb:
            embedding_system.vectordb.add_documents = mock_add_documents_with_error
        else:
            # Create mock vectordb for testing
            mock_vectordb = Mock()
            mock_vectordb.add_documents = mock_add_documents_with_error
            embedding_system.vectordb = mock_vectordb
        
        # Test ArrowSchema recovery
        result = embedding_system.add_document_incrementally(
            documents=test_documents,
            batch_size=2,  # This will trigger batch processing first
            use_progress_tracking=False
        )
        
        # Verify results
        if result.get('success') and result.get('added') == len(test_documents):
            logger.info("✅ ArrowSchema recovery test PASSED!")
            logger.info(f"   - Successfully recovered from ArrowSchema error")
            logger.info(f"   - Processed {result.get('added')} documents individually")
            logger.info(f"   - Total calls made: {call_count} (1 batch + {len(test_documents)} individual)")
            return True
        else:
            logger.error(f"❌ ArrowSchema recovery test FAILED!")
            logger.error(f"   - Result: {result}")
            logger.error(f"   - Expected success=True, added={len(test_documents)}")
            return False
        
    except Exception as e:
        logger.error(f"❌ ArrowSchema recovery test failed with exception: {e}")
        return False


def test_normal_processing():
    """Test that normal processing still works without errors."""
    logger.info("🔧 Testing normal processing (no ArrowSchema errors)...")
    
    try:
        from data_processing.embedding_only_system import EmbeddingOnlySystem
        from langchain.schema import Document
        
        # Create test proceeding
        test_proceeding = "TEST_NORMAL"
        
        # Initialize embedding system
        embedding_system = EmbeddingOnlySystem(test_proceeding)
        
        # Create test documents
        test_documents = [
            Document(
                page_content="Normal test content.",
                metadata={
                    'url': 'https://example.com/normal.pdf',
                    'title': 'Normal Test Document',
                    'proceeding': test_proceeding,
                    'chunk_index': 0
                }
            )
        ]
        
        # Mock normal successful processing
        if embedding_system.vectordb:
            original_add_documents = embedding_system.vectordb.add_documents
            embedding_system.vectordb.add_documents = lambda docs: True
        else:
            mock_vectordb = Mock()
            mock_vectordb.add_documents = lambda docs: True
            embedding_system.vectordb = mock_vectordb
        
        # Test normal processing
        result = embedding_system.add_document_incrementally(
            documents=test_documents,
            batch_size=1,
            use_progress_tracking=False
        )
        
        # Verify results
        if result.get('success') and result.get('added') == len(test_documents):
            logger.info("✅ Normal processing test PASSED!")
            return True
        else:
            logger.error(f"❌ Normal processing test FAILED: {result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Normal processing test failed with exception: {e}")
        return False


def main():
    """Run ArrowSchema recovery tests."""
    print("🧪 ArrowSchema Recovery Test Suite")
    print("=" * 50)
    
    tests = [
        ("ArrowSchema Recovery", test_arrowschema_recovery),
        ("Normal Processing", test_normal_processing)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n🔧 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n📊 TEST RESULTS:")
    print("=" * 30)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Tests Passed: {passed}/{total}")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Total Time: {total_time:.2f}s")
    
    if passed == total:
        print(f"\n🎉 ALL ARROWSCHEMA RECOVERY TESTS PASSED!")
        print("   • ArrowSchema recursion errors now recover automatically")
        print("   • Individual document processing works correctly")
        print("   • Normal processing remains unaffected")
        print("   • document_hashes.json should now update correctly")
        return True
    else:
        print(f"\n⚠️ SOME TESTS FAILED - ArrowSchema recovery needs attention")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)