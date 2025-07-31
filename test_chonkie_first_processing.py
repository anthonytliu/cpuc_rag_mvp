#!/usr/bin/env python3
"""
Test Chonkie-First Processing Approach

Tests the new Chonkie-first processing pipeline with Docling fallback.
All results should use the proven Chonkie metadata schema to prevent
ArrowSchema recursion issues.
"""

import logging
import sys
import time
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_chonkie_schema_validation():
    """Test that the Chonkie schema validation works correctly."""
    logger.info("🔧 Testing Chonkie schema validation...")
    
    try:
        from data_processing.chonkie_schema import ChonkieSchema
        from langchain.schema import Document
        
        # Test 1: Valid Chonkie metadata
        valid_metadata = ChonkieSchema.create_base_metadata(
            pdf_url="https://test.com/doc.pdf",
            source_name="test_doc",
            proceeding="TEST_PROC",
            chunk_info={
                'text': 'Test content',
                'start_index': 0,
                'end_index': 12,
                'token_count': 2,
                'level': 0,
                'strategy': 'chonkie'
            }
        )
        
        if not ChonkieSchema.validate_metadata(valid_metadata):
            logger.error("❌ Valid metadata failed validation")
            return False
        
        logger.info(f"✅ Chonkie schema validation works - {len(valid_metadata)} fields validated")
        return True
        
    except Exception as e:
        logger.error(f"❌ Chonkie schema validation test failed: {e}")
        return False


def test_docling_to_chonkie_conversion():
    """Test conversion of Docling documents to Chonkie schema."""
    logger.info("🔄 Testing Docling to Chonkie conversion...")
    
    try:
        from data_processing.docling_to_chonkie_converter import convert_single_docling_document
        from data_processing.chonkie_schema import ChonkieSchema
        from langchain.schema import Document
        
        # Create mock Docling document
        docling_doc = Document(
            page_content="This is a test table with financial data.",
            metadata={
                'source_url': 'https://test.com/doc.pdf',
                'source': 'test_doc',
                'page': 1,
                'content_type': 'table',
                'chunk_id': 'test_chunk_123',
                'document_type': 'financial',
                'last_checked': '2024-01-01T00:00:00'
            }
        )
        
        # Convert to Chonkie schema
        converted_doc = convert_single_docling_document(
            doc=docling_doc,
            pdf_url="https://test.com/doc.pdf",
            source_name="test_doc",
            proceeding="TEST_CONV",
            chunk_index=0
        )
        
        # Validate converted document
        if not ChonkieSchema.validate_metadata(converted_doc.metadata):
            logger.error("❌ Converted document failed Chonkie schema validation")
            logger.error(f"Metadata: {list(converted_doc.metadata.keys())}")
            return False
        
        # Check that content type was properly converted
        content_type = converted_doc.metadata.get('content_type', '')
        if 'docling' not in content_type:
            logger.error(f"❌ Content type not properly converted: {content_type}")
            return False
            
        logger.info("✅ Docling to Chonkie conversion works correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Docling to Chonkie conversion test failed: {e}")
        return False


def test_chonkie_first_processing():
    """Test the complete Chonkie-first processing pipeline."""
    logger.info("🚀 Testing Chonkie-first processing pipeline...")
    
    try:
        from data_processing.data_processing import process_with_chonkie_first_approach
        from data_processing.chonkie_schema import ChonkieSchema
        
        # Test with a small PDF document
        test_url = "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M440/K092/440092094.PDF"
        test_title = "Chonkie First Test Document"
        test_proceeding = "TEST_CHONKIE_FIRST"
        
        logger.info(f"Processing: {test_url}")
        
        # Process with Chonkie-first approach
        start_time = time.time()
        documents = process_with_chonkie_first_approach(
            pdf_url=test_url,
            document_title=test_title,
            proceeding=test_proceeding,
            enable_ocr_fallback=False  # Disable for faster testing
        )
        processing_time = time.time() - start_time
        
        if not documents or len(documents) == 0:
            logger.error("❌ Chonkie-first processing returned no documents")
            return False
        
        logger.info(f"✅ Processing successful: {len(documents)} documents in {processing_time:.2f}s")
        
        # Validate all documents have Chonkie schema
        for i, doc in enumerate(documents[:5]):  # Check first 5
            if not ChonkieSchema.validate_metadata(doc.metadata):
                logger.error(f"❌ Document {i} failed Chonkie schema validation")
                logger.error(f"Fields: {list(doc.metadata.keys())}")
                return False
        
        # Check specific Chonkie schema fields
        first_doc = documents[0]
        required_fields = ['source_url', 'proceeding_number', 'chunk_id', 'char_start', 'char_end']
        for field in required_fields:
            if field not in first_doc.metadata:
                logger.error(f"❌ Missing required field: {field}")
                return False
        
        logger.info("✅ All documents pass Chonkie schema validation")
        return True
        
    except Exception as e:
        logger.error(f"❌ Chonkie-first processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_with_chonkie_schema():
    """Test that embedding works without ArrowSchema recursion."""
    logger.info("⚡ Testing embedding with Chonkie schema...")
    
    try:
        from data_processing.embedding_only_system import EmbeddingOnlySystem
        from data_processing.chonkie_schema import create_chonkie_document
        
        # Create test proceeding
        test_proceeding = "TEST_EMBED_CHONKIE"
        
        # Create test documents with Chonkie schema
        test_docs = []
        for i in range(3):
            chunk_info = {
                'text': f'Test content for document {i}',
                'start_index': i * 100,
                'end_index': i * 100 + 50,
                'token_count': 8,
                'level': 0,
                'strategy': 'test',
                'document_type': 'test',
                'last_checked': '',
                'document_date': '',
                'publication_date': '',
                'supersedes_priority': 0.5
            }
            
            doc = create_chonkie_document(
                text=f'Test content for document {i}',
                chunk_info=chunk_info,
                pdf_url=f'https://test.com/doc{i}.pdf',
                source_name=f'test_doc_{i}',
                proceeding=test_proceeding
            )
            test_docs.append(doc)
        
        # Create embedding system and test embedding
        embedding_system = EmbeddingOnlySystem(test_proceeding)
        
        logger.info("Attempting to embed documents (checking for ArrowSchema recursion)...")
        result = embedding_system.add_document_incrementally(
            documents=test_docs,
            batch_size=5,
            use_progress_tracking=False
        )
        
        if not result.get('success'):
            logger.error(f"❌ Embedding failed: {result.get('error', 'Unknown error')}")
            return False
        
        added_count = result.get('added', 0)
        logger.info(f"✅ Embedding successful: {added_count} documents embedded without ArrowSchema recursion")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {e}")
        return False


def test_r2110002_sample():
    """Test processing a few documents from R2110002 proceeding."""
    logger.info("📄 Testing with R2110002 sample documents...")
    
    try:
        from data_processing.data_processing import process_with_chonkie_first_approach
        from data_processing.embedding_only_system import EmbeddingOnlySystem
        
        # Test with a few URLs from R2110002
        test_urls = [
            "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M560/K138/560138226.PDF",
            "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M560/K101/560101185.PDF"
        ]
        
        proceeding = "TEST_R2110002"
        embedding_system = EmbeddingOnlySystem(proceeding)
        
        total_processed = 0
        total_embedded = 0
        
        for i, pdf_url in enumerate(test_urls):
            logger.info(f"Processing document {i+1}/{len(test_urls)}: {pdf_url}")
            
            # Process document
            documents = process_with_chonkie_first_approach(
                pdf_url=pdf_url,
                document_title=f"R2110002 Test Doc {i+1}",
                proceeding=proceeding,
                enable_ocr_fallback=False
            )
            
            if not documents:
                logger.warning(f"⚠️ No documents from {pdf_url}")
                continue
            
            total_processed += len(documents)
            logger.info(f"✅ Processed {len(documents)} chunks from document {i+1}")
            
            # Test embedding
            result = embedding_system.add_document_incrementally(
                documents=documents,
                batch_size=10,
                use_progress_tracking=False
            )
            
            if result.get('success'):
                added = result.get('added', 0)
                total_embedded += added
                logger.info(f"✅ Embedded {added} chunks successfully")
            else:
                logger.error(f"❌ Embedding failed: {result.get('error')}")
                return False
        
        logger.info(f"🎉 R2110002 sample test completed:")
        logger.info(f"   📄 Processed: {total_processed} chunks")
        logger.info(f"   ⚡ Embedded: {total_embedded} chunks") 
        logger.info(f"   🔧 No ArrowSchema recursion errors!")
        
        return total_processed > 0 and total_embedded > 0
        
    except Exception as e:
        logger.error(f"❌ R2110002 sample test failed: {e}")
        return False


def main():
    """Run all Chonkie-first processing tests."""
    print("🧪 Chonkie-First Processing Test Suite")
    print("=" * 50)
    
    tests = [
        ("Chonkie Schema Validation", test_chonkie_schema_validation),
        ("Docling to Chonkie Conversion", test_docling_to_chonkie_conversion), 
        ("Chonkie-First Processing Pipeline", test_chonkie_first_processing),
        ("Embedding with Chonkie Schema", test_embedding_with_chonkie_schema),
        ("R2110002 Sample Processing", test_r2110002_sample)
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
        print(f"\n🎉 ALL CHONKIE-FIRST TESTS PASSED!")
        print("   ✅ Chonkie-first processing approach works correctly")
        print("   ✅ Docling fallback converts to Chonkie schema properly")
        print("   ✅ No ArrowSchema recursion errors detected")
        print("   ✅ All metadata follows proven Chonkie schema structure")
        print("   ✅ Ready for R2110002 full proceeding processing")
        return True
    else:
        print(f"\n⚠️ SOME TESTS FAILED")
        print("   • Review failed tests before deploying to production")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)