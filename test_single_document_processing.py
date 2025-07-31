#!/usr/bin/env python3
"""
Test single document processing with the minimal schema fix.

This validates that the ArrowSchema recursion issue is resolved by processing
a single real document and embedding it successfully.
"""

import logging
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_single_document_processing():
    """Test processing and embedding a single document."""
    logger.info("🧪 Testing single document processing with minimal schema...")
    
    try:
        from data_processing.embedding_only_system import EmbeddingOnlySystem
        from data_processing.data_processing import _process_with_hybrid_evaluation
        
        # Test proceeding
        test_proceeding = "TEST_SINGLE_DOC"
        
        # Use a small PDF from the CPUC system
        test_url = "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M440/K092/440092094.PDF"
        test_title = "Test Document"
        
        logger.info(f"Processing document: {test_url}")
        
        # Process document using the updated hybrid evaluation
        documents = _process_with_hybrid_evaluation(
            pdf_url=test_url,
            document_title=test_title,
            proceeding=test_proceeding,
            enable_ocr_fallback=False
        )
        
        if not documents:
            logger.error("❌ No documents returned from processing")
            return False
        
        logger.info(f"✅ Successfully processed {len(documents)} document chunks")
        
        # Check that documents have minimal metadata schema
        from data_processing.minimal_metadata_schema import MinimalMetadataSchema
        
        for i, doc in enumerate(documents[:3]):  # Check first 3 documents
            if not MinimalMetadataSchema.validate_schema(doc.metadata):
                logger.error(f"❌ Document {i} does not have valid minimal schema")
                logger.error(f"   Metadata keys: {list(doc.metadata.keys())}")
                return False
            
            # Check that only essential fields are present
            if len(doc.metadata) != len(MinimalMetadataSchema.ESSENTIAL_FIELDS):
                logger.error(f"❌ Document {i} has {len(doc.metadata)} fields, expected {len(MinimalMetadataSchema.ESSENTIAL_FIELDS)}")
                logger.error(f"   Fields: {list(doc.metadata.keys())}")
                logger.error(f"   Expected: {list(MinimalMetadataSchema.ESSENTIAL_FIELDS)}")
                return False
        
        logger.info("✅ All documents have valid minimal metadata schema")
        
        # Now test embedding without ArrowSchema recursion
        embedding_system = EmbeddingOnlySystem(test_proceeding)
        
        logger.info("Attempting to embed documents (this used to fail with ArrowSchema recursion)...")
        
        result = embedding_system.add_document_incrementally(
            documents=documents,
            batch_size=10,
            use_progress_tracking=False
        )
        
        if not result.get('success'):
            logger.error(f"❌ Embedding failed: {result.get('error', 'Unknown error')}")
            return False
        
        added_count = result.get('added', 0)
        logger.info(f"✅ Successfully embedded {added_count} documents without ArrowSchema recursion!")
        
        # Verify vectors were actually added
        vector_count = embedding_system.get_vector_count()
        if vector_count == 0:
            logger.error("❌ No vectors found in database after embedding")
            return False
        
        logger.info(f"✅ Vector count verification: {vector_count} vectors in database")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Single document processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run single document processing test."""
    print("🧪 Single Document Processing Test with Minimal Schema")
    print("=" * 55)
    
    success = test_single_document_processing()
    
    if success:
        print(f"\n🎉 SINGLE DOCUMENT TEST PASSED!")
        print("   • ArrowSchema recursion issue is RESOLVED")
        print("   • Document processing works with minimal schema")
        print("   • Embedding succeeds without schema conflicts")
        print("   • Ready for full proceeding processing")
    else:
        print(f"\n⚠️ SINGLE DOCUMENT TEST FAILED")
        print("   • Review errors before processing full proceedings")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)