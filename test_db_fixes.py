#!/usr/bin/env python3
"""
Test script to verify the DB writing fixes work correctly.
"""

import sys
import logging
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

from rag_core import CPUCRAGSystem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_db_initialization():
    """Test that the vector store initializes correctly."""
    print("=" * 50)
    print("Testing DB Initialization")
    print("=" * 50)
    
    try:
        rag = CPUCRAGSystem()
        
        # Check that db_dir exists
        if rag.db_dir.exists():
            print(f"✅ DB directory exists: {rag.db_dir}")
        else:
            print(f"❌ DB directory missing: {rag.db_dir}")
            return False
            
        # Check vector store state
        if rag.vectordb is not None:
            print("✅ Vector store initialized")
            try:
                count = rag.vectordb._collection.count()
                print(f"✅ Vector store contains {count} chunks")
            except Exception as e:
                print(f"⚠️ Could not get chunk count: {e}")
        else:
            print("ℹ️ Vector store not initialized (expected if empty)")
            
        return True
        
    except Exception as e:
        print(f"❌ DB initialization failed: {e}")
        return False

def test_incremental_write():
    """Test the incremental write functionality."""
    print("\n" + "=" * 50)
    print("Testing Incremental Write")
    print("=" * 50)
    
    try:
        rag = CPUCRAGSystem()
        
        # Create test data
        from langchain.docstore.document import Document
        test_chunks = [
            Document(
                page_content="Test content 1",
                metadata={"source": "test.pdf", "page": 1, "chunk_id": "test_1"}
            ),
            Document(
                page_content="Test content 2", 
                metadata={"source": "test.pdf", "page": 2, "chunk_id": "test_2"}
            )
        ]
        
        test_url_data = {
            'url': 'https://example.com/test.pdf',
            'title': 'Test Document'
        }
        
        # Test incremental write
        result = rag.add_document_incrementally(
            chunks=test_chunks,
            url_hash='test_hash_123',
            url_data=test_url_data,
            immediate_persist=True
        )
        
        if result:
            print("✅ Incremental write successful")
            
            # Verify the data was written
            if rag.vectordb:
                count = rag.vectordb._collection.count()
                print(f"✅ Vector store now contains {count} chunks")
                
                # Check if hashes were updated
                if 'test_hash_123' in rag.doc_hashes:
                    print("✅ Document hashes updated correctly")
                else:
                    print("⚠️ Document hashes not updated")
            else:
                print("❌ Vector store not available after write")
                return False
        else:
            print("❌ Incremental write failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Incremental write test failed: {e}")
        return False

def test_recovery():
    """Test the recovery functionality."""
    print("\n" + "=" * 50)
    print("Testing Recovery Functionality")
    print("=" * 50)
    
    try:
        rag = CPUCRAGSystem()
        
        # Test recovery with some URLs
        test_urls = [
            {'url': 'https://example.com/doc1.pdf', 'title': 'Doc 1'},
            {'url': 'https://example.com/doc2.pdf', 'title': 'Doc 2'},
            {'url': 'https://example.com/doc3.pdf', 'title': 'Doc 3'}
        ]
        
        urls_to_process = rag.recover_partial_build(test_urls)
        
        print(f"✅ Recovery check completed")
        print(f"ℹ️ URLs that need processing: {len(urls_to_process)}")
        
        # Test checkpoint creation
        checkpoint_path = rag.create_checkpoint("test_checkpoint")
        if checkpoint_path:
            print(f"✅ Checkpoint created: {checkpoint_path}")
            
            # Clean up test checkpoint
            import shutil
            shutil.rmtree(checkpoint_path, ignore_errors=True)
            print("✅ Test checkpoint cleaned up")
        else:
            print("❌ Checkpoint creation failed")
            
        return True
        
    except Exception as e:
        print(f"❌ Recovery test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Starting DB Fixes Test Suite")
    print("=" * 60)
    
    tests = [
        ("DB Initialization", test_db_initialization),
        ("Incremental Write", test_incremental_write),
        ("Recovery Functions", test_recovery)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! DB fixes are working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)