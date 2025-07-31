#!/usr/bin/env python3
"""
Test the minimal metadata schema fix for ArrowSchema recursion issues.

This test validates that the minimal schema prevents the ArrowSchema recursion
errors that were causing every document embedding to fail.
"""

import logging
import sys
import tempfile
from pathlib import Path
from langchain.schema import Document

# Add src to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_minimal_metadata_schema():
    """Test the minimal metadata schema creation and validation."""
    logger.info("🧪 Testing minimal metadata schema...")
    
    try:
        from data_processing.minimal_metadata_schema import MinimalMetadataSchema, normalize_document_metadata_minimal
        
        # Test 1: Create minimal metadata
        minimal_metadata = MinimalMetadataSchema.create_minimal_metadata(
            pdf_url="https://test.com/document.pdf",
            document_title="Test Document",
            proceeding="TEST_PROC",
            processing_method="docling"
        )
        
        # Validate schema
        if not MinimalMetadataSchema.validate_schema(minimal_metadata):
            logger.error("❌ Minimal metadata schema validation failed")
            return False
        
        # Check field count (should be only 5 essential fields)
        if len(minimal_metadata) != len(MinimalMetadataSchema.ESSENTIAL_FIELDS):
            logger.error(f"❌ Expected {len(MinimalMetadataSchema.ESSENTIAL_FIELDS)} fields, got {len(minimal_metadata)}")
            return False
        
        # Test 2: Document normalization
        test_doc = Document(
            page_content="Test content for document processing",
            metadata={
                'source': 'https://test.com/document.pdf',
                'title': 'Test Document',
                'proceeding': 'TEST_PROC',
                # Add some extra fields that should be filtered out
                'extra_field': 'should_be_removed',
                'complex_nested': {'nested': 'data'},
                'large_list': list(range(100))
            }
        )
        
        normalized_doc = normalize_document_metadata_minimal(test_doc, 'test')
        
        # Validate normalized document has only essential fields
        if not MinimalMetadataSchema.validate_schema(normalized_doc.metadata):
            logger.error("❌ Normalized document schema validation failed")
            return False
        
        # Check that extra fields were removed
        if len(normalized_doc.metadata) != len(MinimalMetadataSchema.ESSENTIAL_FIELDS):
            logger.error(f"❌ Normalized doc has {len(normalized_doc.metadata)} fields, expected {len(MinimalMetadataSchema.ESSENTIAL_FIELDS)}")
            return False
        
        logger.info("✅ Minimal metadata schema test PASSED!")
        logger.info(f"   Schema has {len(MinimalMetadataSchema.ESSENTIAL_FIELDS)} essential fields only")
        logger.info(f"   Fields: {list(MinimalMetadataSchema.ESSENTIAL_FIELDS)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Minimal metadata schema test failed: {e}")
        return False


def test_embedding_with_minimal_schema():
    """Test that embedding works with minimal schema without ArrowSchema recursion."""
    logger.info("🧪 Testing embedding with minimal schema...")
    
    try:
        from data_processing.embedding_only_system import EmbeddingOnlySystem
        from data_processing.minimal_metadata_schema import normalize_document_metadata_minimal
        
        # Create temporary test proceeding
        test_proceeding = "TEST_MIN_SCHEMA"
        
        # Create test documents with minimal schema
        test_docs = []
        for i in range(3):
            # Create document with complex metadata (to simulate real-world scenario)
            complex_doc = Document(
                page_content=f"Test content for document {i}",
                metadata={
                    'source': f'https://test.com/doc{i}.pdf',
                    'title': f'Test Document {i}',
                    'proceeding': test_proceeding,
                    # Add complex fields that used to cause recursion
                    'complex_nested_data': {'level1': {'level2': {'level3': 'deep'}}},
                    'large_array': list(range(50)),
                    'datetime_fields': {'created': '2024-01-01', 'modified': '2024-01-02'},
                    'processing_metadata': {
                        'chunks': 10,
                        'confidence': 0.95,
                        'methods': ['docling', 'chonkie']
                    }
                }
            )
            
            # Normalize to minimal schema
            minimal_doc = normalize_document_metadata_minimal(complex_doc, 'test')
            test_docs.append(minimal_doc)
        
        # Create embedding system
        embedding_system = EmbeddingOnlySystem(test_proceeding)
        
        # Try to add documents (this used to fail with ArrowSchema recursion)
        result = embedding_system.add_document_incrementally(
            documents=test_docs,
            batch_size=5,
            use_progress_tracking=False
        )
        
        if not result.get('success'):
            logger.error(f"❌ Embedding failed: {result.get('error', 'Unknown error')}")
            return False
        
        added_count = result.get('added', 0)
        if added_count != len(test_docs):
            logger.error(f"❌ Expected {len(test_docs)} documents added, got {added_count}")
            return False
        
        # Verify vector count
        vector_count = embedding_system.get_vector_count()
        if vector_count != len(test_docs):
            logger.error(f"❌ Expected {len(test_docs)} vectors, got {vector_count}")
            return False
        
        logger.info("✅ Embedding with minimal schema test PASSED!")
        logger.info(f"   Successfully embedded {added_count} documents without ArrowSchema recursion")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding with minimal schema test failed: {e}")
        return False


def main():
    """Run minimal schema fix tests."""
    print("🧪 Minimal Schema Fix Test Suite")
    print("=" * 45)
    
    tests = [
        ("Minimal Metadata Schema", test_minimal_metadata_schema),
        ("Embedding with Minimal Schema", test_embedding_with_minimal_schema)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔧 Running {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
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
    
    if passed == total:
        print(f"\n🎉 ALL MINIMAL SCHEMA TESTS PASSED!")
        print("   • ArrowSchema recursion issue should be resolved")
        print("   • Document embedding should work without errors")
        print("   • Only 5 essential metadata fields are used")
        print("   • Complex nested metadata is filtered out")
        return True
    else:
        print(f"\n⚠️ MINIMAL SCHEMA ISSUES DETECTED")
        print("   • Review failed tests before deploying fix")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)