#!/usr/bin/env python3
"""
Rebuild Vector Store with Enhanced Citation System

This script rebuilds the R2207005 vector store using the enhanced Chonkie
chunking with character position tracking for improved citation accuracy.
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from rag_core import CPUCRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def rebuild_vector_store_with_enhanced_citations(proceeding: str = "R2207005"):
    """Rebuild vector store with enhanced citation system."""
    logger.info(f"🔄 Starting enhanced vector store rebuild for {proceeding}")
    
    try:
        # Step 1: Clear existing vector store
        logger.info("🗑️ Clearing existing vector store...")
        
        vector_store_path = Path(f"local_lance_db/{proceeding}")
        
        if vector_store_path.exists():
            import shutil
            shutil.rmtree(vector_store_path)
            logger.info(f"✅ Removed existing vector store at {vector_store_path}")
        
        # Step 2: Load proceeding data
        logger.info(f"📂 Loading proceeding data for {proceeding}...")
        
        proceeding_dir = Path(f"cpuc_proceedings/{proceeding}")
        scraped_data_file = proceeding_dir / f"{proceeding}_scraped_pdf_history.json"
        
        if not scraped_data_file.exists():
            # Try generic name
            scraped_data_file = proceeding_dir / "scraped_pdf_history.json"
        
        if not scraped_data_file.exists():
            logger.error(f"❌ No scraped PDF data found for {proceeding}")
            return False
        
        with open(scraped_data_file, 'r') as f:
            scraped_data_dict = json.load(f)
        
        # Convert dictionary to list format expected by RAG system
        scraped_data = list(scraped_data_dict.values())
        
        logger.info(f"📊 Found {len(scraped_data)} documents to process")
        
        # Step 3: Initialize RAG system with enhanced processing
        logger.info("⚙️ Initializing RAG system for enhanced processing...")
        
        rag_system = CPUCRAGSystem(current_proceeding=proceeding)
        
        # Step 4: Process documents with enhanced chunking
        logger.info("🔄 Processing documents with enhanced citation system...")
        
        start_time = time.time()
        
        # Use the RAG system's build method for complete processing
        success = rag_system.build_vector_store_from_urls(
            pdf_urls=scraped_data,
            force_rebuild=True,
            incremental_mode=False  # Do a complete rebuild
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"📊 Processing completed:")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        logger.info(f"   Build successful: {success}")
        
        if success:
            # Verify the new vector store
            verify_enhanced_vector_store(proceeding)
            return True
        else:
            logger.error("❌ Vector store build failed")
            return False
            
    except Exception as e:
        logger.error(f"💥 Vector store rebuild failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def verify_enhanced_vector_store(proceeding: str):
    """Verify the enhanced vector store has character position data."""
    logger.info(f"🔍 Verifying enhanced vector store for {proceeding}")
    
    try:
        # Initialize RAG system with the new vector store
        rag_system = CPUCRAGSystem(current_proceeding=proceeding)
        
        # Get system stats
        stats = rag_system.get_system_stats()
        total_chunks = stats.get('total_chunks', 0)
        
        logger.info(f"📊 Vector store verification:")
        logger.info(f"   Total chunks: {total_chunks}")
        
        if total_chunks > 0:
            # Test a sample query to check for enhanced metadata
            logger.info("🧪 Testing sample query for enhanced citation data...")
            
            query = "What are the main objectives of this proceeding?"
            response_generator = rag_system.query(query)
            
            # Get the response
            final_result = None
            for result in response_generator:
                if isinstance(result, dict):
                    final_result = result
                    break
            
            if final_result:
                sources = final_result.get('sources', [])
                enhanced_sources = 0
                
                for source in sources:
                    if any(key in source for key in ['char_start', 'char_end', 'char_length']):
                        enhanced_sources += 1
                
                logger.info(f"   Sources with enhanced data: {enhanced_sources}/{len(sources)}")
                
                if enhanced_sources > 0:
                    logger.info("✅ Enhanced citation system is active!")
                    
                    # Show sample enhanced metadata
                    for source in sources[:3]:  # Show first 3 sources
                        if 'char_start' in source:
                            logger.info(f"   Sample: {source.get('source', 'Unknown')} chars {source.get('char_start', 'N/A')}-{source.get('char_end', 'N/A')}")
                else:
                    logger.warning("⚠️ No enhanced citation data found - may need troubleshooting")
            else:
                logger.warning("⚠️ No response received from test query")
        else:
            logger.error("❌ Vector store appears to be empty")
            
    except Exception as e:
        logger.error(f"💥 Vector store verification failed: {e}")


def main():
    """Main function to rebuild enhanced vector store."""
    print("🚀 Enhanced Citation System Vector Store Rebuild")
    print("=" * 60)
    
    proceeding = "R2207005"
    
    print(f"Proceeding: {proceeding}")
    print(f"Mode: Enhanced Chonkie chunking with character positions")
    print()
    
    # Confirm rebuild
    confirm = input("This will delete and rebuild the vector store. Continue? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Rebuild cancelled.")
        return
    
    print("\n🔄 Starting rebuild...")
    
    success = rebuild_vector_store_with_enhanced_citations(proceeding)
    
    if success:
        print("\n🎉 Enhanced vector store rebuild completed successfully!")
        print("The citation system now has character-level position tracking.")
    else:
        print("\n❌ Rebuild failed. Check the logs for details.")


if __name__ == "__main__":
    main()