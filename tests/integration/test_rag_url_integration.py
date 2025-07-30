#!/usr/bin/env python3
# Test RAG system integration with URL-based processing

import logging
from rag_core import CPUCRAGSystem

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_rag_url_integration():
    """Test the RAG system with URL-based processing"""
    
    logger.info("🧪 Testing RAG system with URL-based processing...")
    
    # Test URLs (using ArXiv for reliable testing)
    test_urls = [
        {
            'url': 'https://arxiv.org/pdf/2408.09869',
            'title': 'Docling Technical Report'
        }
    ]
    
    try:
        # Initialize RAG system
        logger.info("Step 1: Initializing RAG system...")
        rag_system = CPUCRAGSystem()
        
        # Build vector store from URLs
        logger.info("Step 2: Building vector store from URLs...")
        rag_system.build_vector_store_from_urls(test_urls)
        
        # Check system stats
        logger.info("Step 3: Checking system statistics...")
        stats = rag_system.get_system_stats()
        
        logger.info("System Statistics:")
        for key, value in stats.items():
            logger.info(f"   {key}: {value}")
        
        # Test a simple query
        logger.info("Step 4: Testing query functionality...")
        
        # Generator-based query
        query_text = "What is document parsing?"
        result_generator = rag_system.query(query_text)
        
        final_result = None
        for result in result_generator:
            if isinstance(result, str):
                logger.info(f"   Status: {result}")
            elif isinstance(result, dict):
                final_result = result
                break
        
        if final_result:
            answer = final_result.get("answer", "No answer generated")
            sources = final_result.get("sources", [])
            confidence = final_result.get("confidence_indicators", {})
            
            logger.info("Query Results:")
            logger.info(f"   Answer length: {len(answer)} characters")
            logger.info(f"   Number of sources: {len(sources)}")
            logger.info(f"   Confidence level: {confidence.get('overall_confidence', 'Unknown')}")
            
            # Show first part of answer
            if answer:
                logger.info(f"   Answer preview: {answer[:200]}...")
            
            # Show source info
            if sources:
                logger.info("   Source info:")
                for i, source in enumerate(sources[:3]):  # Show first 3 sources
                    logger.info(f"     {i+1}. {source.get('document', 'Unknown')} (Page: {source.get('page', 'Unknown')})")
            
            return True
        else:
            logger.error("❌ No query result received")
            return False
            
    except Exception as e:
        logger.error(f"❌ RAG URL integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_url_integration()
    if success:
        logger.info("🎉 RAG URL integration test completed successfully!")
    else:
        logger.error("💥 RAG URL integration test failed!")