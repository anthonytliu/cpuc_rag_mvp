#!/usr/bin/env python3
"""
Test the placeholder document creation for failed PDFs
"""

import logging
import data_processing

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_placeholder_creation():
    """Test placeholder creation for failing PDF"""
    
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/EFILE/CM/162841.PDF"
    test_proceeding = 'R1202009'
    
    print("🧪 TESTING PLACEHOLDER DOCUMENT CREATION")
    print("=" * 60)
    print(f"📄 URL: {test_url}")
    print(f"📋 Proceeding: {test_proceeding}")
    
    try:
        print("\n🔄 Processing with placeholder fallback...")
        chunks = data_processing.extract_and_chunk_with_docling_url(
            test_url, 
            "162841", 
            test_proceeding,
            enable_ocr_fallback=True
        )
        
        print(f"✅ Processing completed: {len(chunks)} chunks returned")
        
        if chunks:
            print(f"\n📊 Chunk analysis:")
            for i, chunk in enumerate(chunks):
                print(f"   Chunk {i+1}:")
                print(f"      Content length: {len(chunk.page_content)}")
                print(f"      Content type: {chunk.metadata.get('content_type', 'unknown')}")
                print(f"      Processing status: {chunk.metadata.get('processing_status', 'unknown')}")
                print(f"      Extraction method: {chunk.metadata.get('extraction_method', 'unknown')}")
                
                if chunk.metadata.get('content_type') == 'extraction_failure':
                    print(f"      ✅ Placeholder document created successfully")
                    print(f"      Content preview: {chunk.page_content[:200]}...")
                    return True
                else:
                    print(f"      📝 Regular chunk with content")
            
            return True
        else:
            print("❌ No chunks returned - placeholder creation failed")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Detailed error:")
        return False

def test_with_rag_system():
    """Test if the placeholder documents work with the RAG system"""
    
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/EFILE/CM/162841.PDF"
    test_proceeding = 'R1202009'
    
    print(f"\n🤖 TESTING RAG SYSTEM INTEGRATION")
    print("=" * 60)
    
    try:
        from rag_core import CPUCRAGSystem
        
        # Initialize RAG system
        rag_system = CPUCRAGSystem(current_proceeding=test_proceeding)
        
        # Process the problematic URL
        url_data = {
            'url': test_url,
            'title': '162841',
            'id': 'test_placeholder'
        }
        
        print("⏳ Testing RAG system processing...")
        result = rag_system._process_single_url(url_data)
        
        if result['success'] and result['chunks']:
            print(f"✅ RAG processing successful: {result['chunk_count']} chunks")
            
            # Try adding to vector store
            url_hash = data_processing.get_url_hash(test_url)
            success = rag_system.add_document_incrementally(
                chunks=result['chunks'],
                url_hash=url_hash,
                url_data=url_data,
                immediate_persist=True
            )
            
            if success:
                print("✅ Successfully added placeholder to vector store")
                return True
            else:
                print("❌ Failed to add placeholder to vector store")
                return False
        else:
            print(f"❌ RAG processing failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ RAG integration test failed: {e}")
        logger.exception("Detailed error:")
        return False

if __name__ == '__main__':
    print("🧪 PLACEHOLDER DOCUMENT TEST")
    print("=" * 80)
    print("Testing placeholder creation for malformed PDFs")
    print("=" * 80)
    
    # Test placeholder creation
    success1 = test_placeholder_creation()
    
    # Test RAG system integration
    success2 = test_with_rag_system()
    
    overall_success = success1 and success2
    
    if overall_success:
        print("\n✅ PLACEHOLDER TESTS PASSED")
        print("📝 Malformed PDFs will now be tracked with placeholder documents")
        print("🚀 This prevents infinite retry loops and provides visibility into failed processing")
    else:
        print("\n❌ PLACEHOLDER TESTS FAILED")
    
    exit(0 if overall_success else 1)