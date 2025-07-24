#!/usr/bin/env python3
"""
Incremental Embedding System

Manages incremental chunking and embedding of PDFs with progress tracking
and robust error handling to prevent single points of mass failure.

Features:
- Track ingested PDFs and their status
- Incremental processing of new documents
- Progress notifications and error recovery
- Metadata synchronization with embedding status

Author: Claude Code
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
import hashlib
from tqdm import tqdm

import config
from rag_core import CPUCRAGSystem

logger = logging.getLogger(__name__)


class IncrementalEmbedder:
    """Manages incremental embedding with progress tracking and error recovery."""
    
    def __init__(self, proceeding: str, progress_callback: Optional[Callable] = None):
        """
        Initialize incremental embedder for a specific proceeding.
        
        Args:
            proceeding: Proceeding number (e.g., "R2207005")
            progress_callback: Optional callback for progress updates
        """
        self.proceeding = proceeding
        self.progress_callback = progress_callback
        
        # Initialize RAG system
        self.rag_system = CPUCRAGSystem(current_proceeding=proceeding)
        
        # Set up paths
        self.proceeding_paths = config.get_proceeding_file_paths(proceeding)
        self.embedding_status_file = self.proceeding_paths['embedding_status']
        
        # Initialize embedding status tracking
        self.embedding_status = self._load_embedding_status()
        
        logger.info(f"Incremental embedder initialized for {proceeding}")
    
    def process_incremental_embeddings(self) -> Dict:
        """
        Process incremental embeddings for new/updated documents.
        
        Returns:
            Dictionary with processing results
        """
        try:
            self._update_progress("Starting incremental embedding process...", 0)
            
            # Step 1: Load scraped PDF metadata
            self._update_progress("Loading document metadata...", 10)
            scraped_metadata = self._load_scraped_metadata()
            
            if not scraped_metadata:
                logger.info("No scraped metadata found")
                return {'status': 'no_metadata', 'documents_processed': 0}
            
            # Step 2: Identify documents needing embedding
            self._update_progress("Identifying documents for embedding...", 20)
            documents_to_process = self._identify_documents_for_embedding(scraped_metadata)
            
            if not documents_to_process:
                logger.info("All documents are already embedded")
                self._update_progress("All documents up to date", 100)
                return {'status': 'up_to_date', 'documents_processed': 0}
            
            # Step 3: Process documents incrementally
            self._update_progress(f"Processing {len(documents_to_process)} documents...", 30)
            processing_results = self._process_documents_incrementally(documents_to_process)
            
            # Step 4: Update metadata and status
            self._update_progress("Updating metadata and status...", 90)
            self._update_embedding_status(processing_results)
            self._sync_metadata_with_embeddings(scraped_metadata)
            
            self._update_progress("Incremental embedding completed!", 100)
            
            return {
                'status': 'completed',
                'documents_processed': len(processing_results['successful']),
                'successful': len(processing_results['successful']),
                'failed': len(processing_results['failed']),
                'processing_results': processing_results
            }
            
        except Exception as e:
            logger.error(f"Incremental embedding failed: {e}")
            self._update_progress(f"Embedding failed: {str(e)}", -1)
            return {
                'status': 'error',
                'error': str(e),
                'documents_processed': 0
            }
    
    def _load_scraped_metadata(self) -> Dict:
        """Load scraped PDF metadata from JSON file."""
        try:
            history_file = self.proceeding_paths['scraped_pdf_history']
            
            if not history_file.exists():
                logger.warning(f"No scraped metadata file found: {history_file}")
                return {}
            
            with open(history_file, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded metadata for {len(metadata)} documents")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load scraped metadata: {e}")
            return {}
    
    def _load_embedding_status(self) -> Dict:
        """Load embedding status tracking."""
        try:
            if self.embedding_status_file.exists():
                with open(self.embedding_status_file, 'r') as f:
                    status = json.load(f)
                logger.info(f"Loaded embedding status for {len(status.get('embedded_documents', {}))} documents")
                return status
            else:
                # Initialize new status file
                return {
                    'last_updated': datetime.now().isoformat(),
                    'embedded_documents': {},
                    'failed_documents': {},
                    'total_embedded': 0,
                    'version': '1.0'
                }
                
        except Exception as e:
            logger.error(f"Failed to load embedding status: {e}")
            return {
                'last_updated': datetime.now().isoformat(),
                'embedded_documents': {},
                'failed_documents': {},
                'total_embedded': 0,
                'version': '1.0'
            }
    
    def _identify_documents_for_embedding(self, scraped_metadata: Dict) -> List[Dict]:
        """Identify documents that need embedding."""
        documents_to_process = []
        embedded_docs = self.embedding_status.get('embedded_documents', {})
        
        for doc_hash, metadata in scraped_metadata.items():
            # Add hash to metadata for processing
            enhanced_metadata = metadata.copy()
            enhanced_metadata['hash'] = doc_hash
            
            # Check if document is already embedded
            if doc_hash in embedded_docs:
                # Check if document was updated
                embedded_date = embedded_docs[doc_hash].get('embedding_date')
                updated_date = metadata.get('last_updated')
                
                if updated_date and embedded_date and updated_date > embedded_date:
                    logger.info(f"Document {doc_hash} was updated, re-embedding")
                    documents_to_process.append(enhanced_metadata)
            else:
                # New document
                documents_to_process.append(enhanced_metadata)
        
        logger.info(f"Identified {len(documents_to_process)} documents for embedding")
        return documents_to_process
    
    def _process_documents_incrementally(self, documents: List[Dict]) -> Dict:
        """Process documents incrementally with error recovery."""
        successful = []
        failed = []
        
        total_docs = len(documents)
        
        # Create progress bar for current proceeding
        progress_bar = tqdm(
            documents, 
            desc=f"📄 Processing {self.proceeding}",
            unit="doc",
            disable=config.DEBUG,  # Hide progress bar in debug mode
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        for i, doc_metadata in enumerate(documents):
            try:
                doc_title = doc_metadata.get('title', 'Unknown')
                
                # Update progress bar
                progress_bar.set_postfix_str(f"{doc_title[:30]}...")
                
                if config.VERBOSE_LOGGING:
                    logger.debug(f"Processing document {i+1}/{total_docs}: {doc_title}")
                
                # Process single document
                result = self._process_single_document(doc_metadata)
                
                if result['success']:
                    successful.append({
                        'hash': doc_metadata['hash'],
                        'url': doc_metadata['url'],
                        'title': doc_title,
                        'processing_time': result.get('processing_time', 0)
                    })
                    if config.VERBOSE_LOGGING:
                        logger.debug(f"✅ Successfully processed: {doc_title}")
                else:
                    failed.append({
                        'hash': doc_metadata['hash'],
                        'url': doc_metadata['url'],
                        'error': result.get('error', 'Unknown error')
                    })
                    if config.VERBOSE_LOGGING:
                        logger.warning(f"❌ Failed to process: {doc_title} - {result.get('error', 'Unknown error')}")
                
                progress_bar.update(1)
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
            except Exception as e:
                error_msg = f"Failed to process document {doc_metadata.get('url', 'Unknown')}: {e}"
                if config.VERBOSE_LOGGING:
                    logger.error(error_msg)
                failed.append({
                    'hash': doc_metadata['hash'],
                    'url': doc_metadata.get('url', 'Unknown'),
                    'error': str(e)
                })
                progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Print clear completion message with line break
        print(f"\n✅ {self.proceeding} completed: {len(successful)} successful, {len(failed)} failed\n" + "="*60)
        
        return {
            'successful': successful,
            'failed': failed,
            'total_processed': len(documents)
        }
    
    def _process_single_document(self, doc_metadata: Dict) -> Dict:
        """Process a single document for embedding."""
        start_time = time.time()
        
        try:
            url = doc_metadata['url']
            title = doc_metadata.get('title', 'Unknown')
            
            logger.debug(f"Processing document: {title} ({url})")
            
            # Use RAG system to build vector store for this specific URL
            # This is a simplified approach - in reality, you might want to download the PDF first
            pdf_urls = [{
                'url': url,
                'filename': f"{doc_metadata['hash']}.pdf"
            }]
            
            success = self.rag_system.build_vector_store_from_urls(pdf_urls, force_rebuild=False, incremental_mode=True)
            
            processing_time = time.time() - start_time
            
            if success:
                logger.debug(f"Successfully processed {title} in {processing_time:.2f}s")
                return {
                    'success': True,
                    'processing_time': processing_time
                }
            else:
                return {
                    'success': False,
                    'error': 'Vector store building failed',
                    'processing_time': processing_time
                }
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to process document: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }
    
    def _update_embedding_status(self, processing_results: Dict):
        """Update embedding status with processing results."""
        try:
            # Update embedded documents
            for doc in processing_results['successful']:
                self.embedding_status['embedded_documents'][doc['hash']] = {
                    'url': doc['url'],
                    'title': doc['title'],
                    'embedding_date': datetime.now().isoformat(),
                    'processing_time': doc['processing_time']
                }
            
            # Update failed documents
            for doc in processing_results['failed']:
                self.embedding_status['failed_documents'][doc['hash']] = {
                    'url': doc['url'],
                    'error': doc['error'],
                    'last_attempt': datetime.now().isoformat()
                }
            
            # Update totals
            self.embedding_status['total_embedded'] = len(self.embedding_status['embedded_documents'])
            self.embedding_status['last_updated'] = datetime.now().isoformat()
            
            # Save status
            self._save_embedding_status()
            
        except Exception as e:
            logger.error(f"Failed to update embedding status: {e}")
    
    def _save_embedding_status(self):
        """Save embedding status to file."""
        try:
            # Ensure directory exists
            self.embedding_status_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.embedding_status_file, 'w') as f:
                json.dump(self.embedding_status, f, indent=2)
                
            logger.debug(f"Saved embedding status to {self.embedding_status_file}")
            
        except Exception as e:
            logger.error(f"Failed to save embedding status: {e}")
    
    def _sync_metadata_with_embeddings(self, scraped_metadata: Dict):
        """Sync scraped metadata with embedding status."""
        try:
            embedded_docs = self.embedding_status.get('embedded_documents', {})
            
            # Update scraped metadata with embedding status
            for doc_hash, metadata in scraped_metadata.items():
                if doc_hash in embedded_docs:
                    metadata['embedding_status'] = 'embedded'
                    metadata['embedding_date'] = embedded_docs[doc_hash]['embedding_date']
                else:
                    metadata['embedding_status'] = 'pending'
            
            # Save updated metadata
            history_file = self.proceeding_paths['scraped_pdf_history']
            with open(history_file, 'w') as f:
                json.dump(scraped_metadata, f, indent=2)
            
            logger.info("Synchronized metadata with embedding status")
            
        except Exception as e:
            logger.error(f"Failed to sync metadata with embeddings: {e}")
    
    def get_embedding_status(self) -> Dict:
        """Get current embedding status summary."""
        try:
            embedded_count = len(self.embedding_status.get('embedded_documents', {}))
            failed_count = len(self.embedding_status.get('failed_documents', {}))
            
            return {
                'total_embedded': embedded_count,
                'total_failed': failed_count,
                'last_updated': self.embedding_status.get('last_updated'),
                'status': 'ready' if embedded_count > 0 else 'empty'
            }
            
        except Exception as e:
            logger.error(f"Failed to get embedding status: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _update_progress(self, message: str, progress: int):
        """Update progress with message and percentage."""
        logger.info(f"[{progress}%] {message}")
        if self.progress_callback:
            self.progress_callback(message, progress)


# Convenience functions

def create_incremental_embedder(proceeding: str, progress_callback=None) -> IncrementalEmbedder:
    """Factory function to create incremental embedder."""
    return IncrementalEmbedder(proceeding, progress_callback=progress_callback)


def process_incremental_embeddings(proceeding: str, progress_callback=None) -> Dict:
    """Process incremental embeddings for a proceeding."""
    embedder = create_incremental_embedder(proceeding, progress_callback)
    return embedder.process_incremental_embeddings()


if __name__ == "__main__":
    # Test the incremental embedder
    import sys
    
    proceeding = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_PROCEEDING
    
    logging.basicConfig(level=logging.INFO)
    
    def progress_logger(message: str, progress: int):
        print(f"Progress: {progress}% - {message}")
    
    results = process_incremental_embeddings(proceeding, progress_callback=progress_logger)
    
    if results['status'] == 'completed':
        print(f"\n✅ Incremental embedding completed for {proceeding}")
        print(f"Documents processed: {results['documents_processed']}")
        print(f"Successful: {results['successful']}")
        print(f"Failed: {results['failed']}")
    else:
        print(f"\n⚠️ Incremental embedding status: {results['status']}")
        if 'error' in results:
            print(f"Error: {results['error']}")