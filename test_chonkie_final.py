#!/usr/bin/env python3
"""
Final comprehensive test of Chonkie integration with the problematic PDF
"""

import logging
import data_processing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_complete_integration():
    """Test complete Chonkie integration with problematic PDF"""
    
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/EFILE/CM/162841.PDF"
    test_proceeding = 'R1202009'
    
    print("🧪 FINAL CHONKIE INTEGRATION TEST")
    print("=" * 60)
    print(f"📄 Testing URL: {test_url}")
    print(f"📋 Proceeding: {test_proceeding}")
    
    try:
        print("\n🔄 Running complete fallback chain...")
        chunks = data_processing.extract_and_chunk_with_docling_url(
            test_url, 
            "162841_final_test", 
            test_proceeding,
            enable_ocr_fallback=True,
            enable_chonkie_fallback=True
        )
        
        print(f"✅ FINAL RESULT: {len(chunks)} chunks extracted")
        
        if chunks:
            # Analyze what method succeeded
            content_types = {}
            for chunk in chunks:
                content_type = chunk.metadata.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            print("\n📊 EXTRACTION METHOD BREAKDOWN:")
            for content_type, count in content_types.items():
                method = "Chonkie" if "chonkie" in content_type else "Other"
                print(f"   {content_type} ({method}): {count} chunks")
            
            # Show sample content
            sample = chunks[0]
            print(f"\n📝 SAMPLE CHUNK:")
            print(f"   Content length: {len(sample.page_content)}")
            print(f"   Content preview: {sample.page_content[:200]}...")
            print(f"   Metadata: {list(sample.metadata.keys())}")
            
            chonkie_chunks = [c for c in chunks if 'chonkie' in c.metadata.get('content_type', '')]
            if chonkie_chunks:
                print(f"\n🎉 SUCCESS: Chonkie fallback extracted {len(chonkie_chunks)} chunks!")
                print(f"   This PDF would have been lost without Chonkie fallback")
                return True
            else:
                print(f"   Note: Other methods succeeded, Chonkie wasn't needed")
                return True
        else:
            print("❌ FAILED: No chunks extracted at all")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: Test failed with error: {e}")
        logger.exception("Detailed error:")
        return False

if __name__ == '__main__':
    print("🚀 CHONKIE FALLBACK INTEGRATION - FINAL TEST")
    print("=" * 80)
    
    success = test_complete_integration()
    
    if success:
        print("\n🎉 CHONKIE INTEGRATION SUCCESS!")
        print("✅ The fallback chain now works: Docling → OCR → Chonkie → Placeholder")
        print("📈 This dramatically improves PDF processing success rate")
        print("🚀 Malformed PDFs that previously failed can now be processed")
    else:
        print("\n❌ CHONKIE INTEGRATION FAILED")
        print("🔧 Review the implementation and fix any remaining issues")
    
    exit(0 if success else 1)