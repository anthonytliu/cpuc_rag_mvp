#!/usr/bin/env python3
"""
Test script to verify OCR fallback functionality for 0-chunk PDFs.
"""

import logging
import data_processing

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ocr_fallback():
    """Test OCR fallback on the problematic PDF"""
    
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/EFILE/CM/162841.PDF"
    test_proceeding = 'R1202009'
    
    print("🧪 TESTING OCR FALLBACK FOR 0-CHUNK PDF")
    print("=" * 60)
    print(f"📄 URL: {test_url}")
    print(f"📋 Proceeding: {test_proceeding}")
    
    try:
        print("\n🔄 Testing with OCR fallback enabled...")
        chunks = data_processing.extract_and_chunk_with_docling_url(
            test_url, 
            "162841", 
            test_proceeding,
            enable_ocr_fallback=True
        )
        
        print(f"✅ Processing completed: {len(chunks)} chunks extracted")
        
        if chunks:
            print(f"\n📊 First chunk analysis:")
            first_chunk = chunks[0]
            print(f"   Content length: {len(first_chunk.page_content)}")
            print(f"   Content preview: {first_chunk.page_content[:300]}...")
            print(f"   Metadata keys: {list(first_chunk.metadata.keys())}")
            print(f"   Extraction method: {first_chunk.metadata.get('extraction_method', 'standard')}")
            print(f"   Content type: {first_chunk.metadata.get('content_type', 'unknown')}")
            
            return True
        else:
            print("❌ Still extracted 0 chunks even with OCR fallback")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Detailed error:")
        return False

def compare_standard_vs_ocr():
    """Compare standard processing vs OCR fallback"""
    
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/EFILE/CM/162841.PDF"
    test_proceeding = 'R1202009'
    
    print("\n🔄 COMPARISON: Standard vs OCR Processing")
    print("=" * 60)
    
    # Test without OCR fallback
    print("📋 Testing standard processing (no OCR)...")
    try:
        standard_chunks = data_processing.extract_and_chunk_with_docling_url(
            test_url, "162841", test_proceeding, enable_ocr_fallback=False
        )
        print(f"   Standard processing: {len(standard_chunks)} chunks")
    except Exception as e:
        print(f"   Standard processing failed: {e}")
        standard_chunks = []
    
    # Test with OCR fallback
    print("📋 Testing with OCR fallback...")
    try:
        ocr_chunks = data_processing.extract_and_chunk_with_docling_url(
            test_url, "162841", test_proceeding, enable_ocr_fallback=True
        )
        print(f"   OCR processing: {len(ocr_chunks)} chunks")
    except Exception as e:
        print(f"   OCR processing failed: {e}")
        ocr_chunks = []
    
    print(f"\n📊 Comparison Results:")
    print(f"   Standard: {len(standard_chunks)} chunks")
    print(f"   OCR: {len(ocr_chunks)} chunks")
    print(f"   Improvement: +{len(ocr_chunks) - len(standard_chunks)} chunks")
    
    if len(ocr_chunks) > len(standard_chunks):
        print("✅ OCR fallback successfully extracted more content!")
        return True
    else:
        print("❌ OCR fallback did not improve extraction")
        return False

if __name__ == '__main__':
    print("🧪 OCR FALLBACK TEST")
    print("=" * 80)
    print("Testing OCR fallback functionality for PDFs that extract 0 chunks")
    print("=" * 80)
    
    # Run basic OCR test
    success1 = test_ocr_fallback()
    
    # Run comparison test
    success2 = compare_standard_vs_ocr()
    
    overall_success = success1 and success2
    
    if overall_success:
        print("\n✅ OCR FALLBACK TESTS PASSED")
    else:
        print("\n❌ OCR FALLBACK TESTS FAILED")
    
    exit(0 if overall_success else 1)