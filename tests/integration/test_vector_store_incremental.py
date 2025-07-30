#!/usr/bin/env python3
"""
Test Vector Store Incremental Addition

This script tests whether the vector store is properly accumulating chunks
or overwriting them during incremental addition.
"""

import logging
import sys
from pathlib import Path
import time
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config
from rag_core import CPUCRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_incremental_addition():
    """Test if incremental addition is working correctly."""
    logger.info("🧪 Testing vector store incremental addition behavior")
    
    proceeding = "R2207005"
    
    # Step 1: Get current vector store state
    logger.info("📊 Step 1: Check initial vector store state")
    
    try:
        rag_system = CPUCRAGSystem(current_proceeding=proceeding)
        initial_stats = rag_system.get_system_stats()
        initial_chunks = initial_stats.get('total_chunks', 0)
        logger.info(f"Initial chunks in vector store: {initial_chunks}")
        
        # Get sample documents from embedding status
        embedding_status_path = Path(f"cpuc_proceedings/{proceeding}/embeddings/embedding_status.json")
        if not embedding_status_path.exists():
            logger.error("No embedding status file found")
            return False
        
        with open(embedding_status_path, 'r') as f:
            embedding_status = json.load(f)
        
        embedded_docs = embedding_status.get('embedded_documents', {})
        if not embedded_docs:
            logger.error("No embedded documents found")
            return False
        
        # Get 3 test documents
        test_docs = list(embedded_docs.items())[:3]
        logger.info(f"Selected {len(test_docs)} test documents")
        
        # Step 2: Add documents one by one and check accumulation
        logger.info("🔄 Step 2: Test incremental addition")
        
        for i, (doc_hash, doc_info) in enumerate(test_docs, 1):
            logger.info(f"\n--- Adding Document {i}/3: {doc_info.get('title', 'Unknown')} ---")
            
            # Process the document
            url_data = {
                'url': doc_info['url'],
                'title': doc_info.get('title', f'Test Document {i}')
            }
            
            # Get chunks for this document
            processing_result = rag_system._process_single_url(url_data)
            
            if not processing_result['success']:
                logger.error(f"Failed to process document {i}: {processing_result.get('error')}")
                continue
            
            chunks = processing_result.get('chunks', [])
            if not chunks:
                logger.error(f"No chunks extracted for document {i}")
                continue
            
            logger.info(f"Extracted {len(chunks)} chunks from document {i}")
            
            # Get current vector store state before addition
            pre_add_stats = rag_system.get_system_stats()
            pre_add_chunks = pre_add_stats.get('total_chunks', 0)
            logger.info(f"Chunks before addition: {pre_add_chunks}")
            
            # Add to vector store
            success = rag_system.add_document_incrementally(
                chunks=chunks,
                url_hash=doc_hash,
                url_data=url_data,
                immediate_persist=True
            )
            
            if success:
                logger.info(f"✅ Successfully added document {i}")
            else:
                logger.error(f"❌ Failed to add document {i}")
                continue
            
            # Check vector store state after addition
            post_add_stats = rag_system.get_system_stats()
            post_add_chunks = post_add_stats.get('total_chunks', 0)
            logger.info(f"Chunks after addition: {post_add_chunks}")
            
            # Calculate expected vs actual
            expected_chunks = pre_add_chunks + len(chunks)
            actual_increase = post_add_chunks - pre_add_chunks
            
            logger.info(f"Expected total chunks: {expected_chunks}")
            logger.info(f"Actual total chunks: {post_add_chunks}")
            logger.info(f"Chunks added: {actual_increase} (expected {len(chunks)})")
            
            if post_add_chunks < expected_chunks:
                logger.error(f"🚨 ISSUE DETECTED: Vector store not accumulating properly!")
                logger.error(f"  Expected {expected_chunks} chunks, but only have {post_add_chunks}")
                logger.error(f"  Missing {expected_chunks - post_add_chunks} chunks")
                
                # Check if this is an overwriting issue
                if post_add_chunks == len(chunks):
                    logger.error("🔥 CRITICAL: Vector store is being OVERWRITTEN instead of APPENDED TO!")
                    logger.error("  This explains why only the last document's chunks remain")
                
                return False
            elif post_add_chunks == expected_chunks:
                logger.info(f"✅ Correct accumulation: {actual_increase} chunks added")
            else:
                logger.warning(f"⚠️ Unexpected: Got more chunks than expected ({post_add_chunks} vs {expected_chunks})")
            
            # Brief pause between additions
            time.sleep(1)
        
        # Step 3: Final verification
        logger.info("\n📋 Step 3: Final verification")
        final_stats = rag_system.get_system_stats()
        final_chunks = final_stats.get('total_chunks', 0)
        
        logger.info(f"Final vector store chunks: {final_chunks}")
        logger.info(f"Initial chunks: {initial_chunks}")
        logger.info(f"Net increase: {final_chunks - initial_chunks}")
        
        # Test retrieval to make sure chunks are actually accessible
        logger.info("\n🔍 Step 4: Test chunk retrieval")
        try:
            # Use similarity search to get all documents
            all_docs = rag_system.vectordb.similarity_search("", k=10000)
            retrievable_chunks = len(all_docs)
            logger.info(f"Retrievable chunks via similarity_search: {retrievable_chunks}")
            
            if retrievable_chunks != final_chunks:
                logger.warning(f"⚠️ Discrepancy: Stats show {final_chunks} but can only retrieve {retrievable_chunks}")
            else:
                logger.info("✅ Chunk retrieval matches stats")
        
        except Exception as e:
            logger.error(f"Failed to test retrieval: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        return False


def test_lancedb_table_behavior():
    """Test LanceDB table creation and addition behavior directly."""
    logger.info("🔬 Testing LanceDB table behavior directly")
    
    try:
        import lancedb
        from langchain_community.vectorstores import LanceDB
        from langchain.schema import Document
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        # Create test data
        test_docs_1 = [
            Document(page_content="Test document 1 content", metadata={"source": "test1", "id": "1"}),
            Document(page_content="Test document 2 content", metadata={"source": "test2", "id": "2"})
        ]
        
        test_docs_2 = [
            Document(page_content="Test document 3 content", metadata={"source": "test3", "id": "3"}),
            Document(page_content="Test document 4 content", metadata={"source": "test4", "id": "4"})
        ]
        
        # Create test directory
        test_db_dir = Path("test_lance_db")
        if test_db_dir.exists():
            import shutil
            shutil.rmtree(test_db_dir)
        
        test_db_dir.mkdir()
        
        # Initialize embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'}
        )
        
        # Test 1: Create vector store with first batch
        logger.info("Creating vector store with first batch (2 documents)")
        db = lancedb.connect(str(test_db_dir))
        
        vectordb = LanceDB.from_documents(
            documents=test_docs_1,
            embedding=embedding_model,
            connection=db,
            table_name="test_documents",
            mode="append"  # Fix: Use append mode
        )
        
        # Check count after first batch
        count_1 = len(vectordb.similarity_search("", k=100))
        logger.info(f"After first batch: {count_1} documents")
        
        # Test 2: Add second batch
        logger.info("Adding second batch (2 more documents)")
        vectordb.add_documents(test_docs_2)
        
        # Check count after second batch
        count_2 = len(vectordb.similarity_search("", k=100))
        logger.info(f"After second batch: {count_2} documents")
        
        if count_2 == 4:
            logger.info("✅ LanceDB correctly accumulates documents")
        elif count_2 == 2:
            logger.error("🚨 LanceDB is overwriting instead of accumulating!")
        else:
            logger.warning(f"⚠️ Unexpected count: {count_2}")
        
        # Clean up
        import shutil
        shutil.rmtree(test_db_dir)
        
        return count_2 == 4
        
    except Exception as e:
        logger.error(f"LanceDB test failed: {e}")
        return False


def main():
    logger.info("=" * 60)
    logger.info("VECTOR STORE INCREMENTAL ADDITION TEST")
    logger.info("=" * 60)
    
    # Test 1: Basic LanceDB behavior
    logger.info("\n🧪 Test 1: Basic LanceDB accumulation behavior")
    lancedb_ok = test_lancedb_table_behavior()
    
    if not lancedb_ok:
        logger.error("❌ Basic LanceDB test failed - investigation needed")
        return
    
    logger.info("✅ Basic LanceDB test passed")
    
    # Test 2: RAG system incremental addition
    logger.info("\n🧪 Test 2: RAG system incremental addition")
    rag_ok = test_incremental_addition()
    
    if rag_ok:
        logger.info("✅ All tests passed!")
    else:
        logger.error("❌ Vector store incremental addition issue detected")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()