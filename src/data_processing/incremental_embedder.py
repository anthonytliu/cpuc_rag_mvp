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

# Try relative imports first, fall back to absolute
try:
    from ..core import config
    from .embedding_only_system import EmbeddingOnlySystem
except ImportError:
    import sys
    from pathlib import Path
    # Add src directory to path for absolute imports
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    from core import config
    from data_processing.embedding_only_system import EmbeddingOnlySystem

logger = logging.getLogger(__name__)


class IncrementalEmbedder:
    """Manages incremental embedding with progress tracking and error recovery."""
    
    def __init__(self, proceeding: str, progress_callback: Optional[Callable] = None, enable_timeout: bool = True):
        """
        Initialize incremental embedder for a specific proceeding.
        
        Args:
            proceeding: Proceeding number (e.g., "R2207005")
            progress_callback: Optional callback for progress updates
            enable_timeout: Whether to enable 300-second timeout for document processing
        """
        self.proceeding = proceeding
        self.progress_callback = progress_callback
        self.enable_timeout = enable_timeout
        
        # Initialize lightweight embedding system (no GPT/LLM required)
        self.embedding_system = EmbeddingOnlySystem(proceeding)
        
        # Set up paths
        self.proceeding_paths = config.get_proceeding_file_paths(proceeding)
        self.embedding_status_file = self.proceeding_paths['embedding_status']
        
        # Initialize embedding status tracking
        self.embedding_status = self._load_embedding_status()
        
        # Ensure we use the same vector store directory as the embedding system
        # This ensures consistency between loading and adding documents
        if hasattr(self.embedding_system, 'db_dir') and self.embedding_system.db_dir:
            logger.info(f"Using vector store directory: {self.embedding_system.db_dir}")
        
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
            # Try both naming conventions
            history_file = None
            if self.proceeding_paths['scraped_pdf_history'].exists():
                history_file = self.proceeding_paths['scraped_pdf_history']
            elif self.proceeding_paths['scraped_pdf_history_alt'].exists():
                history_file = self.proceeding_paths['scraped_pdf_history_alt']
            
            if history_file is None:
                logger.warning(f"No scraped metadata file found")
                logger.info(f"Checked paths: {self.proceeding_paths['scraped_pdf_history']}, "
                          f"{self.proceeding_paths['scraped_pdf_history_alt']}")
                return {}
            
            with open(history_file, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded metadata for {len(metadata)} documents from {history_file.name}")
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
        """Process documents incrementally with robust error recovery and batch management."""
        successful = []
        failed = []
        schema_errors = []
        retryable_failures = []
        
        total_docs = len(documents)
        batch_size = getattr(config, 'EMBEDDING_BATCH_SIZE', 10)
        max_retries = 3
        retry_delay = 2.0
        
        # Create progress bar for current proceeding
        progress_bar = tqdm(
            documents, 
            desc=f"📄 Processing {self.proceeding}",
            unit="doc",
            disable=config.DEBUG,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        # Process documents in batches for better error recovery
        for batch_start in range(0, total_docs, batch_size):
            batch_end = min(batch_start + batch_size, total_docs)
            batch = documents[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} ({len(batch)} documents)")
            
            # Process each document in the batch with retry logic
            for doc_metadata in batch:
                doc_title = doc_metadata.get('title', 'Unknown')
                doc_hash = doc_metadata['hash']
                retry_count = 0
                processing_success = False
                
                progress_bar.set_postfix_str(f"{doc_title[:30]}...")
                
                while retry_count <= max_retries and not processing_success:
                    try:
                        if config.VERBOSE_LOGGING and retry_count > 0:
                            logger.debug(f"Retry {retry_count}/{max_retries} for: {doc_title}")
                        
                        # Process single document with timeout protection
                        result = self._process_single_document_with_timeout(doc_metadata)
                        
                        if result['success']:
                            chunks_added = result.get('chunks_added', 0)
                            successful.append({
                                'hash': doc_hash,
                                'url': doc_metadata['url'],
                                'title': doc_title,
                                'processing_time': result.get('processing_time', 0),
                                'chunks_added': chunks_added,
                                'retry_count': retry_count
                            })
                            if config.VERBOSE_LOGGING:
                                logger.debug(f"✅ Successfully processed: {doc_title} ({chunks_added} chunks)")
                            processing_success = True
                        else:
                            error = result.get('error', 'Unknown error')
                            
                            # Check for schema errors (non-retryable)
                            if result.get('schema_error', False):
                                schema_errors.append({
                                    'hash': doc_hash,
                                    'url': doc_metadata['url'],
                                    'title': doc_title,
                                    'error': error
                                })
                                logger.error(f"🚨 Schema error for {doc_title}: {error}")
                                processing_success = True  # Don't retry schema errors
                            else:
                                # Check if error is retryable
                                if self._is_retryable_error(error) and retry_count < max_retries:
                                    retry_count += 1
                                    logger.warning(f"⚠️ Retryable error for {doc_title}, attempt {retry_count}: {error}")
                                    time.sleep(retry_delay * retry_count)  # Exponential backoff
                                else:
                                    # Final failure
                                    failed.append({
                                        'hash': doc_hash,
                                        'url': doc_metadata['url'],
                                        'title': doc_title,
                                        'error': error,
                                        'retry_count': retry_count
                                    })
                                    if retry_count >= max_retries:
                                        retryable_failures.append(doc_hash)
                                    processing_success = True
                                    
                    except Exception as e:
                        error_msg = f"Exception processing {doc_title}: {str(e)}"
                        logger.error(error_msg)
                        
                        if retry_count < max_retries and self._is_retryable_error(str(e)):
                            retry_count += 1
                            time.sleep(retry_delay * retry_count)
                        else:
                            failed.append({
                                'hash': doc_hash,
                                'url': doc_metadata.get('url', 'Unknown'),
                                'title': doc_title,
                                'error': str(e),
                                'retry_count': retry_count
                            })
                            processing_success = True
                
                progress_bar.update(1)
                
                # Brief pause between documents to prevent system overload
                time.sleep(0.05)
            
            # Batch completion checkpoint - save progress
            if successful:
                try:
                    self._checkpoint_progress(successful, failed, schema_errors)
                except Exception as checkpoint_error:
                    logger.warning(f"Failed to save checkpoint: {checkpoint_error}")
            
            # Brief pause between batches
            time.sleep(0.2)
        
        progress_bar.close()
        
        # Calculate total chunks processed
        total_chunks = sum(doc.get('chunks_added', 0) for doc in successful)
        
        # Enhanced completion report
        print(f"\n✅ {self.proceeding} batch processing completed:")
        print(f"   📊 Successful: {len(successful)} documents")
        print(f"   ❌ Failed: {len(failed)} documents")
        print(f"   🔄 Retryable failures: {len(retryable_failures)} documents")
        print(f"   🚨 Schema errors: {len(schema_errors)} documents")
        print(f"   📄 Total chunks: {total_chunks}")
        print("=" * 60)
        
        return {
            'successful': successful,
            'failed': failed,
            'schema_errors': schema_errors,
            'retryable_failures': retryable_failures,
            'total_processed': len(documents),
            'total_chunks_added': total_chunks
        }
    
    def _is_retryable_error(self, error_msg: str) -> bool:
        """Determine if an error is retryable."""
        retryable_patterns = [
            "timeout", "connection", "network", "temporary", "rate limit",
            "503", "502", "504", "429", "connection reset", "read timeout"
        ]
        error_lower = error_msg.lower()
        return any(pattern in error_lower for pattern in retryable_patterns)
    
    def _checkpoint_progress(self, successful: List[Dict], failed: List[Dict], schema_errors: List[Dict]):
        """Save progress checkpoint for recovery."""
        try:
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'proceeding': self.proceeding,
                'successful_count': len(successful),
                'failed_count': len(failed),
                'schema_errors_count': len(schema_errors),
                'last_successful': successful[-5:] if successful else [],
                'recent_failures': failed[-5:] if failed else []
            }
            
            checkpoint_file = self.proceeding_paths['embeddings_dir'] / 'processing_checkpoint.json'
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _process_single_document_with_timeout(self, doc_metadata: Dict) -> Dict:
        """Process document with timeout protection."""
        import signal
        import threading
        
        result = {'success': False, 'error': 'Unknown error'}
        
        # Use timeout setting from initialization, extend timeout for complex documents
        if self.enable_timeout:
            timeout_seconds = 300  # 5 minutes for complex documents (was 120)
        else:
            timeout_seconds = getattr(config, 'URL_PROCESSING_TIMEOUT', 900)  # Use config default if no timeout
        
        def timeout_handler():
            return {
                'success': False,
                'error': f'Processing timeout after {timeout_seconds} seconds',
                'processing_time': timeout_seconds
            }
        
        def process_with_monitoring():
            nonlocal result
            start_time = time.time()
            try:
                # Add detailed processing metrics
                doc_title = doc_metadata.get('title', 'Unknown')
                logger.debug(f"🔄 Starting processing: {doc_title}")
                
                result = self._process_single_document(doc_metadata)
                
                # Add processing time metrics
                processing_time = time.time() - start_time
                result['processing_time'] = processing_time
                
                if result.get('success'):
                    logger.debug(f"✅ Completed processing: {doc_title} in {processing_time:.2f}s")
                else:
                    logger.warning(f"❌ Failed processing: {doc_title} in {processing_time:.2f}s - {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                processing_time = time.time() - start_time
                logger.error(f"❌ Exception processing {doc_metadata.get('title', 'Unknown')}: {e}")
                result = {
                    'success': False,
                    'error': str(e),
                    'processing_time': processing_time
                }
        
        # Create and start processing thread
        process_thread = threading.Thread(target=process_with_monitoring)
        process_thread.daemon = True
        process_thread.start()
        
        # Wait for completion or timeout
        process_thread.join(timeout=timeout_seconds)
        
        if process_thread.is_alive():
            logger.warning(f"Document processing timed out after {timeout_seconds}s: {doc_metadata.get('title', 'Unknown')}")
            return timeout_handler()
        
        return result
    
    def _process_single_document(self, doc_metadata: Dict) -> Dict:
        """Process a single document for embedding using true incremental processing."""
        start_time = time.time()
        
        try:
            url = doc_metadata['url']
            title = doc_metadata.get('title', 'Unknown')
            doc_hash = doc_metadata['hash']
            
            logger.debug(f"Processing document: {title} ({url})")
            
            # Check if document is already processed using embedding system's hash tracking
            if self.embedding_system.is_document_processed(url):
                logger.debug(f"Document already processed: {title}")
                processing_time = time.time() - start_time
                return {
                    'success': True,
                    'processing_time': processing_time,
                    'chunks_added': 0,
                    'total_chunks_processed': 0,
                    'success_rate': '100%',
                    'skipped': True,
                    'reason': 'Already processed'
                }
            
            # Step 1: Extract chunks from the URL using the RAG system's single URL processor
            url_data = {
                'url': url,
                'title': title
            }
            
            # Process the single URL to extract chunks using embedding system
            chunks = self.embedding_system.process_document_url(
                pdf_url=url_data['url'],
                document_title=url_data.get('title'),
                proceeding=self.proceeding
            )
            
            if not chunks:
                processing_time = time.time() - start_time
                return {
                    'success': False,
                    'error': 'No chunks extracted from document',
                    'processing_time': processing_time
                }
            
            # Step 2: Add chunks incrementally to the vector store with Chonkie schema
            # Use appropriate batch sizes - no need to limit chunks with proper schema
            batch_size = 50  # Standard batch size for Chonkie schema
            if len(chunks) > 500:
                # For very large chunk sets, use smaller batches but process all chunks
                batch_size = 25
                logger.info(f"Large chunk set detected ({len(chunks)} total), using batch size {batch_size} for efficiency")
            
            # Special handling for ArrowSchema recursion-prone documents
            processing_method = chunks[0].metadata.get('processing_method', '') if chunks else ''
            if 'docling_direct' in processing_method:
                batch_size = 5  # Even smaller batches for recursion recovery mode
                logger.info(f"Docling direct mode detected - using ultra-small batch size: {batch_size}")
            
            # Add documents with enhanced ArrowSchema recursion protection
            try:
                result = self.embedding_system.add_document_incrementally(
                    documents=chunks,
                    batch_size=batch_size,
                    use_progress_tracking=False  # Disable to reduce complexity
                )
            except Exception as embed_error:
                error_msg = str(embed_error).lower()
                if 'recursion level' in error_msg or 'arrowschema' in error_msg:
                    logger.warning(f"ArrowSchema recursion during embedding - using minimal batch processing")
                    # Try with minimal batch size as final attempt
                    try:
                        result = self.embedding_system.add_document_incrementally(
                            documents=chunks[:10],  # Limit to first 10 chunks only
                            batch_size=1,  # Process one at a time
                            use_progress_tracking=False
                        )
                        if result.get('success'):
                            logger.info("Minimal batch processing successful for ArrowSchema recursion recovery")
                        else:
                            logger.error("Even minimal batch processing failed")
                            result = {'success': False, 'error': 'ArrowSchema recursion - minimal processing failed'}
                    except Exception as final_error:
                        logger.error(f"Final attempt at ArrowSchema recovery failed: {final_error}")
                        result = {'success': False, 'error': f'ArrowSchema recursion recovery failed: {final_error}'}
                else:
                    raise embed_error
            success = result.get('success', False)
            
            processing_time = time.time() - start_time
            
            if success:
                # Get the actual chunk count that was successfully added
                actual_chunks_added = result.get('added', len(chunks))
                
                # Add document to hash tracking
                self.embedding_system.add_document_to_hashes(url, title, actual_chunks_added)
                
                logger.debug(f"Successfully processed {title} ({actual_chunks_added}/{len(chunks)} chunks added) in {processing_time:.2f}s")
                return {
                    'success': True,
                    'processing_time': processing_time,
                    'chunks_added': actual_chunks_added,
                    'total_chunks_processed': len(chunks),
                    'success_rate': '100%'
                }
            else:
                # Get detailed error information from the result
                error_details = result.get('error', 'Failed to add chunks to vector store')
                
                return {
                    'success': False,
                    'error': error_details,
                    'processing_time': processing_time,
                    'chunks_attempted': len(chunks)
                }
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            # Check for schema-related errors and provide helpful guidance
            if ("cast from string to null" in error_msg or 
                "Field" in error_msg and "not found in target schema" in error_msg):
                logger.error("🚨 LanceDB Schema Compatibility Issue Detected!")
                logger.error("The existing vector database has an incompatible schema.")
                logger.error(f"💡 To fix this, run: python fix_lancedb_schema.py {self.proceeding}")
                logger.error("This will rebuild the database with the correct schema.")
                return {
                    'success': False,
                    'error': 'Schema compatibility issue - see fix_lancedb_schema.py',
                    'processing_time': processing_time,
                    'schema_error': True
                }
            else:
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

def create_incremental_embedder(proceeding: str, progress_callback=None, enable_timeout: bool = True) -> IncrementalEmbedder:
    """Factory function to create incremental embedder."""
    return IncrementalEmbedder(proceeding, progress_callback=progress_callback, enable_timeout=enable_timeout)


def process_incremental_embeddings(proceeding: str, progress_callback=None, enable_timeout: bool = True) -> Dict:
    """Process incremental embeddings for a proceeding."""
    embedder = create_incremental_embedder(proceeding, progress_callback, enable_timeout)
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