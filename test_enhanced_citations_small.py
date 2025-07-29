#!/usr/bin/env python3
"""
Test Enhanced Citations with Small Dataset

This script tests the enhanced citation system with a small subset of R2207005 documents
to verify that character position tracking is working correctly.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from rag_core import CPUCRAGSystem
from test_citation_accuracy_r2207005 import CitationAccuracyTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_small_test_vector_store(proceeding: str = "R2207005", num_docs: int = 5):
    """Create a small test vector store with enhanced citations."""
    logger.info(f"🧪 Creating small test vector store for {proceeding} with {num_docs} documents")
    
    try:
        # Step 1: Clear existing test vector store
        test_proceeding = f"{proceeding}_test"
        vector_store_path = Path(f"local_lance_db/{test_proceeding}")
        
        if vector_store_path.exists():
            import shutil
            shutil.rmtree(vector_store_path)
            logger.info(f"✅ Removed existing test vector store")
        
        # Step 2: Load a subset of proceeding data
        proceeding_dir = Path(f"cpuc_proceedings/{proceeding}")
        scraped_data_file = proceeding_dir / f"{proceeding}_scraped_pdf_history.json"
        
        if not scraped_data_file.exists():
            scraped_data_file = proceeding_dir / "scraped_pdf_history.json"
        
        if not scraped_data_file.exists():
            logger.error(f"❌ No scraped PDF data found for {proceeding}")
            return False
        
        with open(scraped_data_file, 'r') as f:
            scraped_data_dict = json.load(f)
        
        # Take only the first few documents
        scraped_data = list(scraped_data_dict.values())[:num_docs]
        
        logger.info(f"📊 Selected {len(scraped_data)} documents for testing:")
        for i, doc in enumerate(scraped_data):
            title = doc.get('title', 'No title')[:50]
            logger.info(f"  {i+1}. {title}...")
        
        # Step 3: Initialize RAG system for test proceeding
        logger.info("⚙️ Initializing RAG system for test proceeding...")
        
        rag_system = CPUCRAGSystem(current_proceeding=test_proceeding)
        
        # Step 4: Process documents with enhanced chunking
        logger.info("🔄 Processing documents with enhanced citation system...")
        
        start_time = time.time()
        
        success = rag_system.build_vector_store_from_urls(
            pdf_urls=scraped_data,
            force_rebuild=True,
            incremental_mode=False
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"📊 Processing completed:")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        logger.info(f"   Build successful: {success}")
        
        if success:
            # Get stats from the test vector store
            stats = rag_system.get_system_stats()
            total_chunks = stats.get('total_chunks', 0)
            logger.info(f"   Total chunks created: {total_chunks}")
            
            return test_proceeding
        else:
            logger.error("❌ Test vector store build failed")
            return None
            
    except Exception as e:
        logger.error(f"💥 Test vector store creation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def test_enhanced_citations_with_small_dataset():
    """Test enhanced citations using the small dataset."""
    logger.info("🔍 Testing enhanced citations with small dataset")
    
    # Create small test vector store
    test_proceeding = create_small_test_vector_store(num_docs=3)
    
    if not test_proceeding:
        logger.error("❌ Failed to create test vector store")
        return False
    
    try:
        # Initialize citation tester with test proceeding
        logger.info(f"🧪 Initializing citation tester for {test_proceeding}")
        tester = CitationAccuracyTester(proceeding=test_proceeding)
        
        # Test with a simple query
        test_query = "What are the main objectives of this proceeding?"
        
        logger.info(f"🔍 Testing query: '{test_query}'")
        
        result = tester.test_single_query(test_query)
        
        if result.get('error'):
            logger.error(f"❌ Query test failed: {result['error']}")
            return False
        
        # Check results
        citations = result.get('citations', [])
        validation_results = result.get('validation_results', [])
        metrics = result.get('metrics', {})
        
        logger.info(f"📊 Test Results:")
        logger.info(f"   Citations found: {len(citations)}")
        logger.info(f"   Has citations: {metrics.get('has_citations', False)}")
        
        if len(citations) > 0:
            logger.info(f"   Citation accuracy: {metrics.get('accuracy_rate', 0):.1f}%")
            logger.info(f"   Content precision: {metrics.get('precision_rate', 0):.1f}%")
            
            # Show sample citations
            logger.info("📝 Sample citations:")
            for i, citation in enumerate(citations[:3]):
                logger.info(f"   {i+1}. {citation.get('filename', 'Unknown')} page {citation.get('page', 'N/A')}")
        
        # Check for enhanced metadata in the response
        sources = result.get('response', {})
        if isinstance(sources, str):
            # Check if response contains enhanced citation formats
            import re
            enhanced_citations = re.findall(r'\[CITE:[^,]+,page_\d+,chars_\d+-\d+', sources)
            
            if enhanced_citations:
                logger.info(f"✅ Found {len(enhanced_citations)} enhanced citations with character positions!")
                for citation in enhanced_citations[:2]:
                    logger.info(f"   Example: {citation}]")
            else:
                logger.warning("⚠️ No enhanced citations with character positions found")
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Enhanced citation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main function to test enhanced citations."""
    print("🚀 Enhanced Citation System Test with Small Dataset")
    print("=" * 60)
    
    print("This will create a small test vector store with 3 documents")
    print("to verify the enhanced citation system is working correctly.")
    print()
    
    success = test_enhanced_citations_with_small_dataset()
    
    if success:
        print("\n🎉 Enhanced citation test completed successfully!")
        print("The system is ready for full-scale testing.")
    else:
        print("\n❌ Enhanced citation test failed. Check the logs for details.")


if __name__ == "__main__":
    main()