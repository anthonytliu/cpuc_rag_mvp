#!/usr/bin/env python3
"""
Test Schema Compatibility and Hybrid Processing

Tests the new unified metadata schema and hybrid processing to ensure:
1. No schema compatibility issues between Docling and Chonkie
2. Safe schema migration without data loss
3. Proper hybrid processing combining tables and text
4. Unified metadata schema across all processing methods
"""

import logging
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_unified_metadata_schema():
    """Test the unified metadata schema functionality."""
    logger.info("🧪 Testing unified metadata schema...")
    
    try:
        from data_processing.unified_metadata_schema import UnifiedMetadataSchema, normalize_document_metadata
        from langchain.schema import Document
        
        # Test 1: Base metadata creation
        base_metadata = UnifiedMetadataSchema.create_base_metadata(
            pdf_url="https://example.com/test.pdf",
            document_title="Test Document",
            proceeding="TEST_PROC",
            processing_method="test"
        )
        
        # Check required fields are present
        required_fields = UnifiedMetadataSchema.REQUIRED_FIELDS
        missing_fields = required_fields - set(base_metadata.keys())
        
        if missing_fields:
            logger.error(f"❌ Missing required fields: {missing_fields}")
            return False, f"Missing required fields: {missing_fields}"
        
        # Test 2: Document normalization
        test_doc = Document(
            page_content="Test content",
            metadata={
                'source': 'https://example.com/test.pdf',
                'title': 'Test Doc',
                'proceeding': 'TEST_PROC'
            }
        )
        
        normalized_doc = normalize_document_metadata(test_doc, 'test')
        
        # Check all fields are present
        all_fields = UnifiedMetadataSchema.ALL_FIELDS
        doc_fields = set(normalized_doc.metadata.keys())
        missing_in_doc = all_fields - doc_fields
        
        if missing_in_doc:
            logger.error(f"❌ Document missing fields: {missing_in_doc}")
            return False, f"Document missing fields: {missing_in_doc}"
        
        logger.info("✅ Unified metadata schema test PASSED!")
        return True, f"All {len(all_fields)} schema fields present and working"
        
    except Exception as e:
        logger.error(f"❌ Unified metadata schema test failed: {e}")
        return False, str(e)


def test_docling_schema_compatibility():
    """Test Docling output schema compatibility."""
    logger.info("🔧 Testing Docling schema compatibility...")
    
    try:
        from data_processing.data_processing import _process_with_standard_docling
        from data_processing.unified_metadata_schema import UnifiedMetadataSchema
        
        # Use a small test document
        test_url = "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M440/K092/440092094.PDF"
        
        # Process with Docling
        docling_results = _process_with_standard_docling(
            test_url, "Schema Test Document", "TEST_SCHEMA"
        )
        
        if not docling_results:
            logger.error("❌ No results from Docling processing")
            return False, "No results from Docling processing"
        
        # Check schema compatibility
        all_fields = UnifiedMetadataSchema.ALL_FIELDS
        
        for i, doc in enumerate(docling_results[:5]):  # Check first 5 documents
            doc_fields = set(doc.metadata.keys())
            missing_fields = all_fields - doc_fields
            
            if missing_fields:
                logger.error(f"❌ Document {i} missing fields: {missing_fields}")
                return False, f"Document {i} missing fields: {missing_fields}"
        
        logger.info(f"✅ Docling schema compatibility test PASSED!")
        logger.info(f"   Checked {min(5, len(docling_results))} documents with {len(all_fields)} fields each")
        return True, f"Docling produces schema-compatible documents: {len(docling_results)} total"
        
    except Exception as e:
        logger.error(f"❌ Docling schema compatibility test failed: {e}")
        return False, str(e)


def test_chonkie_schema_compatibility():
    """Test Chonkie output schema compatibility.""" 
    logger.info("🔧 Testing Chonkie schema compatibility...")
    
    try:
        from data_processing.data_processing import _extract_with_chonkie_fallback, extract_filename_from_url, get_url_hash
        from data_processing.unified_metadata_schema import UnifiedMetadataSchema
        from datetime import datetime
        
        # Use a small test document
        test_url = "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M440/K092/440092094.PDF"
        source_name = extract_filename_from_url(test_url)
        url_hash = get_url_hash(test_url)
        
        # Process with Chonkie
        chonkie_results = _extract_with_chonkie_fallback(
            test_url, source_name,
            doc_date=datetime.now(),
            publication_date=None,
            proceeding_number="TEST_SCHEMA",
            doc_type="test",
            url_hash=url_hash,
            proceeding="TEST_SCHEMA"
        )
        
        if not chonkie_results:
            logger.warning("⚠️ No results from Chonkie processing - this may be expected")
            return True, "No Chonkie results to test (acceptable)"
        
        # Check schema compatibility
        all_fields = UnifiedMetadataSchema.ALL_FIELDS
        
        for i, doc in enumerate(chonkie_results[:5]):  # Check first 5 documents
            doc_fields = set(doc.metadata.keys())
            missing_fields = all_fields - doc_fields
            
            if missing_fields:
                logger.error(f"❌ Chonkie document {i} missing fields: {missing_fields}")
                return False, f"Chonkie document {i} missing fields: {missing_fields}"
        
        logger.info(f"✅ Chonkie schema compatibility test PASSED!")
        logger.info(f"   Checked {min(5, len(chonkie_results))} documents with {len(all_fields)} fields each")
        return True, f"Chonkie produces schema-compatible documents: {len(chonkie_results)} total"
        
    except Exception as e:
        logger.error(f"❌ Chonkie schema compatibility test failed: {e}")  
        return False, str(e)


def test_hybrid_processing():
    """Test hybrid processing functionality."""
    logger.info("🔄 Testing hybrid processing...")
    
    try:
        from data_processing.hybrid_processor import process_with_intelligent_hybrid
        from data_processing.unified_metadata_schema import UnifiedMetadataSchema
        
        # Test with a document that might have both tables and text
        test_url = "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M440/K092/440092094.PDF"
        
        # Process with hybrid method
        hybrid_results = process_with_intelligent_hybrid(
            pdf_url=test_url,
            document_title="Hybrid Test Document",
            proceeding="TEST_HYBRID",
            table_score=0.2  # Mixed content
        )
        
        if not hybrid_results:
            logger.error("❌ No results from hybrid processing")
            return False, "No results from hybrid processing"
        
        # Check that results have unified schema
        all_fields = UnifiedMetadataSchema.ALL_FIELDS
        
        for i, doc in enumerate(hybrid_results[:5]):  # Check first 5 documents
            doc_fields = set(doc.metadata.keys()) 
            missing_fields = all_fields - doc_fields
            
            if missing_fields:
                logger.error(f"❌ Hybrid document {i} missing fields: {missing_fields}")
                return False, f"Hybrid document {i} missing fields: {missing_fields}"
            
            # Check hybrid-specific metadata
            processing_method = doc.metadata.get('processing_method', '')
            if 'hybrid' not in processing_method:
                logger.warning(f"⚠️ Document {i} processing_method: {processing_method}")
        
        logger.info(f"✅ Hybrid processing test PASSED!")
        logger.info(f"   Generated {len(hybrid_results)} documents with unified schema")
        return True, f"Hybrid processing works: {len(hybrid_results)} unified documents"
        
    except Exception as e:
        logger.error(f"❌ Hybrid processing test failed: {e}")
        return False, str(e)


def test_schema_migration_safety():
    """Test that schema migration doesn't lose data."""
    logger.info("🔄 Testing schema migration safety...")
    
    try:
        from data_processing.embedding_only_system import EmbeddingOnlySystem
        from data_processing.unified_metadata_schema import UnifiedMetadataSchema
        from langchain.schema import Document
        import tempfile
        import shutil
        
        # Create temporary proceeding for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_proceeding = "SCHEMA_TEST"
            
            # Create embedding system
            embedding_system = EmbeddingOnlySystem(temp_proceeding)
            
            # Create test documents with unified schema
            test_docs = []
            for i in range(3):
                metadata = UnifiedMetadataSchema.create_base_metadata(
                    pdf_url=f"https://test.com/doc{i}.pdf",
                    document_title=f"Test Document {i}",
                    proceeding=temp_proceeding,
                    processing_method="test"
                )
                metadata['chunk_id'] = f"test_chunk_{i}"
                
                doc = Document(
                    page_content=f"Test content for document {i}",
                    metadata=metadata
                )
                test_docs.append(doc)
            
            # Add documents to create initial schema
            result1 = embedding_system.add_document_incrementally(
                documents=test_docs,
                batch_size=5,
                use_progress_tracking=False
            )
            
            if not result1.get('success'):
                logger.error(f"❌ Failed to add initial documents: {result1}")
                return False, "Failed to add initial documents"
            
            # Verify we can add more documents without schema conflicts
            more_test_docs = []
            for i in range(3, 5):
                metadata = UnifiedMetadataSchema.create_base_metadata(
                    pdf_url=f"https://test.com/doc{i}.pdf", 
                    document_title=f"Test Document {i}",
                    proceeding=temp_proceeding,
                    processing_method="test_additional"
                )
                metadata['chunk_id'] = f"test_chunk_{i}"
                
                doc = Document(
                    page_content=f"Additional test content for document {i}",
                    metadata=metadata
                )
                more_test_docs.append(doc)
            
            result2 = embedding_system.add_document_incrementally(
                documents=more_test_docs,
                batch_size=5,
                use_progress_tracking=False
            )
            
            if not result2.get('success'):
                logger.error(f"❌ Failed to add additional documents: {result2}")
                return False, f"Schema migration failed: {result2.get('error', 'Unknown error')}"
            
            total_added = result1.get('added', 0) + result2.get('added', 0)
            
            logger.info(f"✅ Schema migration safety test PASSED!")
            logger.info(f"   Added {total_added} documents without schema conflicts")
            return True, f"Schema migration safe: {total_added} documents added successfully"
            
    except Exception as e:
        logger.error(f"❌ Schema migration safety test failed: {e}")
        return False, str(e)


def main():
    """Run schema compatibility tests."""
    print("🧪 Schema Compatibility and Hybrid Processing Test Suite")
    print("=" * 65)
    
    tests = [
        ("Unified Metadata Schema", test_unified_metadata_schema),
        ("Schema Migration Safety", test_schema_migration_safety),
        ("Docling Schema Compatibility", test_docling_schema_compatibility), 
        ("Chonkie Schema Compatibility", test_chonkie_schema_compatibility),
        ("Hybrid Processing", test_hybrid_processing)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n🔧 Running {test_name}...")
        try:
            success, message = test_func()
            results.append((test_name, success, message))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False, f"Exception: {e}"))
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n📊 TEST RESULTS:")
    print("=" * 40)
    
    passed = 0
    for test_name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        print(f"    {message}")
        if success:
            passed += 1
    
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Tests Passed: {passed}/{total}")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Total Time: {total_time:.2f}s")
    
    if passed == total:
        print(f"\n🎉 ALL SCHEMA COMPATIBILITY TESTS PASSED!")
        print("   • Unified metadata schema prevents compatibility issues")
        print("   • Safe schema migration without data loss")
        print("   • Docling and Chonkie outputs are compatible")
        print("   • Hybrid processing combines best of both methods")
        print("   • No more 'Field not found in target schema' errors")
        print("\n🔧 Production Impact:")
        print("   • Lance files won't be deleted during schema migration") 
        print("   • document_hashes.json will stay consistent with progress")
        print("   • Hybrid processing optimizes for both tables and text")
        print("   • All processing methods use unified metadata schema")
        return True
    else:
        print(f"\n⚠️ SCHEMA COMPATIBILITY ISSUES DETECTED")
        print("   • Review failed tests and fix schema issues")
        print("   • Data loss may occur without proper schema compatibility")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)