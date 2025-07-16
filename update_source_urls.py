#!/usr/bin/env python3
"""
Update existing vector store chunks with source_url metadata
This script adds the missing source_url field to existing chunks without rebuilding the entire vector store.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.resolve()))

from rag_core import CPUCRAGSystem
from data_processing import get_source_url_from_filename
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_chunks_with_source_urls():
    """Update existing chunks with source_url metadata"""
    
    # Initialize RAG system
    rag_system = CPUCRAGSystem()
    
    # Get all chunks
    logger.info("Loading existing chunks...")
    all_chunks = rag_system.vectordb.get()
    
    total_chunks = len(all_chunks['ids'])
    logger.info(f"Found {total_chunks} chunks to update")
    
    # Track updates
    updated_count = 0
    failed_count = 0
    
    # Process chunks in batches
    batch_size = 1000
    
    for i in range(0, total_chunks, batch_size):
        batch_end = min(i + batch_size, total_chunks)
        logger.info(f"Processing batch {i//batch_size + 1}: chunks {i+1}-{batch_end}")
        
        # Get batch data
        batch_ids = all_chunks['ids'][i:batch_end]
        batch_metadatas = all_chunks['metadatas'][i:batch_end]
        
        # Update metadata for this batch
        updated_metadatas = []
        
        for chunk_id, metadata in zip(batch_ids, batch_metadatas):
            try:
                # Check if source_url already exists
                if 'source_url' in metadata and metadata['source_url']:
                    updated_metadatas.append(metadata)
                    continue
                
                # Get filename from metadata
                filename = metadata.get('source', '')
                
                if filename:
                    # Map filename to source URL
                    source_url = get_source_url_from_filename(filename)
                    
                    if source_url:
                        # Add source_url to metadata
                        metadata['source_url'] = source_url
                        updated_count += 1
                        
                        if updated_count % 100 == 0:
                            logger.info(f"Updated {updated_count} chunks so far...")
                    else:
                        logger.warning(f"Could not find source URL for file: {filename}")
                        failed_count += 1
                else:
                    logger.warning(f"No filename found in metadata for chunk: {chunk_id}")
                    failed_count += 1
                
                updated_metadatas.append(metadata)
                
            except Exception as e:
                logger.error(f"Error updating chunk {chunk_id}: {e}")
                failed_count += 1
                updated_metadatas.append(metadata)
        
        # Update the batch in vector store
        try:
            rag_system.vectordb.update(
                ids=batch_ids,
                metadatas=updated_metadatas
            )
            logger.info(f"Successfully updated batch {i//batch_size + 1}")
            
        except Exception as e:
            logger.error(f"Failed to update batch {i//batch_size + 1}: {e}")
    
    logger.info(f"Update complete!")
    logger.info(f"Successfully updated: {updated_count} chunks")
    logger.info(f"Failed to update: {failed_count} chunks")
    
    # Verify a few samples
    logger.info("Verifying updates...")
    sample_chunks = rag_system.vectordb.get(limit=5)
    
    for i, (chunk_id, metadata) in enumerate(zip(sample_chunks['ids'], sample_chunks['metadatas'])):
        source_url = metadata.get('source_url', 'NOT FOUND')
        filename = metadata.get('source', 'NO FILENAME')
        logger.info(f"Sample {i+1}: {filename} -> {source_url}")

if __name__ == "__main__":
    print("🔄 Updating existing vector store chunks with source URLs...")
    print("This will add the missing source_url metadata to existing chunks.")
    print("The vector store will NOT be rebuilt - only metadata will be updated.")
    print()
    
    update_chunks_with_source_urls()
    print("✅ Update complete! Citations should now work properly.")