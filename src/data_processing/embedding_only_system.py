#!/usr/bin/env python3
"""
Embedding-Only System for Data Processing

This module provides lightweight embedding capabilities without requiring
GPT/LLM access. It's specifically designed for the data processing pipeline
to create and store embeddings efficiently.

Features:
- Embedding model initialization only (no LLM)
- Direct LanceDB access for vector storage
- Document processing and chunking
- Schema migration support
- Minimal resource overhead

Author: Claude Code
"""

import logging
import os
import time
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import lancedb
from langchain.schema import Document

# Try relative imports first, fall back to absolute
try:
    from ..core import config
    from ..core import models
    from .data_processing import _process_with_hybrid_evaluation
    from ..monitoring.enhanced_progress_tracker import EnhancedProgressTracker, ProcessingStage
except ImportError:
    import sys
    from pathlib import Path
    # Add src directory to path for absolute imports
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    from core import config
    from core import models
    from data_processing.data_processing import _process_with_hybrid_evaluation
    from monitoring.enhanced_progress_tracker import EnhancedProgressTracker, ProcessingStage

logger = logging.getLogger(__name__)


class EmbeddingOnlySystem:
    """Lightweight system for embedding creation without LLM dependency."""
    
    def __init__(self, proceeding: str):
        """
        Initialize embedding-only system for a specific proceeding.
        
        Args:
            proceeding: Proceeding number (e.g., "R2207005")
        """
        self.proceeding = proceeding
        self.embedding_model = None
        self.vectordb = None
        self.lance_db = None
        self.db_dir = None
        
        # Initialize document hash tracking (like RAG core)
        proceeding_paths = config.get_proceeding_file_paths(proceeding)
        self.doc_hashes_file = proceeding_paths['document_hashes']
        self.doc_hashes = self._load_doc_hashes()
        
        # Initialize embedding model only (no LLM)
        self._initialize_embedding_model()
        
        # Initialize vector store
        self._initialize_vector_store()
        
        logger.info(f"EmbeddingOnlySystem initialized for {proceeding}")
    
    def _initialize_embedding_model(self):
        """Initialize only the embedding model."""
        try:
            logger.info("🔄 Initializing embedding model (no LLM required)...")
            self.embedding_model = models.get_embedding_model()
            logger.info("✅ Embedding model initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding model: {e}")
            raise
    
    def _initialize_vector_store(self):
        """Initialize LanceDB vector store for the proceeding."""
        try:
            # Use the same path structure as the main RAG system - use config paths
            proceeding_paths = config.get_proceeding_file_paths(self.proceeding)
            self.db_dir = proceeding_paths['vector_db']
            
            logger.info(f"Using LanceDB location: {self.db_dir}")
            
            # Ensure directory exists
            self.db_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"LanceDB directory ensured: {self.db_dir}")
            
            # Connect to LanceDB
            self.lance_db = lancedb.connect(str(self.db_dir))
            logger.info(f"Connected to LanceDB at {self.db_dir}")
            
            # Try to load existing table or note that we'll create one when needed
            table_name = f"{self.proceeding}_documents"
            try:
                table = self.lance_db.open_table(table_name)
                logger.info(f"Loaded existing LanceDB table: {table_name}")
                
                # Initialize vectordb wrapper for compatibility
                from langchain_community.vectorstores import LanceDB
                self.vectordb = LanceDB(
                    connection=self.lance_db,
                    table_name=table_name,
                    embedding=self.embedding_model
                )
                
            except Exception:
                logger.info(f"No existing LanceDB table found. Will create {table_name} when needed.")
                self.vectordb = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize vector store: {e}")
            raise
    
    def add_document_incrementally(self, documents: List[Document], batch_size: int = 50, 
                                 use_progress_tracking: bool = True) -> Dict[str, Any]:
        """
        Add documents to the vector store incrementally.
        
        Args:
            documents: List of LangChain Document objects to add
            batch_size: Number of documents to process in each batch
            use_progress_tracking: Whether to show progress for large batches
            
        Returns:
            Dictionary with operation results
        """
        if not documents:
            return {"success": True, "added": 0, "message": "No documents to add"}
        
        # Initialize progress tracking for large batches
        tracker = None
        if use_progress_tracking and len(documents) > 100:
            tracker = EnhancedProgressTracker(
                document_title=f"{len(documents)} document chunks",
                estimated_size_mb=0  # Size unknown at this point
            )
            tracker.start_stage(ProcessingStage.EMBEDDING, 
                              total_items=len(documents),
                              message=f"Generating embeddings for {len(documents)} chunks...")
        
        try:
            # Initialize vectordb if not already done
            if self.vectordb is None:
                from langchain_community.vectorstores import LanceDB
                table_name = f"{self.proceeding}_documents"
                
                # Create the vector store with first batch
                logger.info(f"Creating new LanceDB table: {table_name}")
                if tracker:
                    tracker.update_progress(0, "Creating new vector database...")
                
                self.vectordb = LanceDB.from_documents(
                    documents[:batch_size],
                    self.embedding_model,
                    connection=self.lance_db,
                    table_name=table_name
                )
                
                remaining_docs = documents[batch_size:]
                added_count = batch_size
                
                if tracker:
                    tracker.update_progress(batch_size, f"Created database with {batch_size} initial chunks")
            else:
                remaining_docs = documents
                added_count = 0
                
                if tracker:
                    tracker.update_progress(0, "Using existing vector database...")
            
            # Add remaining documents in batches
            if remaining_docs:
                total_batches = (len(remaining_docs) + batch_size - 1) // batch_size
                
                for batch_idx, i in enumerate(range(0, len(remaining_docs), batch_size)):
                    batch = remaining_docs[i:i + batch_size]
                    
                    try:
                        if tracker:
                            batch_num = batch_idx + 1
                            tracker.update_progress(
                                added_count, 
                                f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)"
                            )
                        
                        # Add batch to existing vector store
                        self.vectordb.add_documents(batch)
                        added_count += len(batch)
                        
                        logger.debug(f"Added batch {batch_idx + 1}: {len(batch)} documents")
                        
                        if tracker:
                            tracker.update_progress(
                                added_count,
                                f"Completed batch {batch_num}/{total_batches} - {added_count}/{len(documents)} total"
                            )
                        
                    except Exception as e:
                        error_msg = str(e).lower()
                        
                        # Check if this is an ArrowSchema recursion issue
                        if ('recursion level' in error_msg and ('arrowschema' in error_msg or 'struct' in error_msg)) or \
                           'recursion limit' in error_msg or \
                           ('arrow' in error_msg and 'recursion' in error_msg):
                            logger.warning(f"ArrowSchema recursion error detected during embedding: {e}")
                            logger.info("Attempting to recover with minimal batch processing...")
                            
                            if tracker:
                                tracker.update_progress(added_count, "ArrowSchema recursion detected - using recovery mode...")
                            
                            # Try to add documents one at a time (minimal batch processing)
                            try:
                                for single_doc in batch:
                                    self.vectordb.add_documents([single_doc])
                                    added_count += 1
                                    
                                logger.info(f"✅ ArrowSchema recursion recovery successful: processed {len(batch)} documents individually")
                                
                                if tracker:
                                    tracker.update_progress(added_count, f"Recovery successful - {len(batch)} docs processed individually")
                                
                                # Recovery succeeded - continue with normal flow
                                continue
                            
                            except Exception as recovery_error:
                                logger.error(f"❌ ArrowSchema recursion recovery failed: {recovery_error}")
                                if tracker:
                                    tracker.fail_stage(ProcessingStage.EMBEDDING, f"ArrowSchema recursion recovery failed: {recovery_error}")
                                    tracker.finish(success=False, message="ArrowSchema recursion could not be resolved")
                                
                                # Re-raise with specific error message for upstream handling
                                raise RuntimeError(f"ArrowSchema recursion level exceeded - recovery failed: {recovery_error}")
                        
                        # Check if this is a schema compatibility issue
                        elif "not found in target schema" in str(e):
                            logger.warning("Schema compatibility issue detected. Attempting migration...")
                            
                            if tracker:
                                tracker.update_progress(added_count, "Schema migration in progress...")
                            
                            # Attempt schema migration
                            if self._attempt_schema_migration():
                                logger.info("Schema migration successful. Retrying batch...")
                                if tracker:
                                    tracker.update_progress(added_count, "Migration successful, retrying batch...")
                                
                                # Retry the batch after migration
                                if self.vectordb is not None:
                                    self.vectordb.add_documents(batch)
                                    added_count += len(batch)
                                else:
                                    # If vectordb is still None after migration, create it with this batch
                                    from langchain_community.vectorstores import LanceDB
                                    table_name = f"{self.proceeding}_documents"
                                    
                                    logger.info(f"Creating new LanceDB table after migration: {table_name}")
                                    self.vectordb = LanceDB.from_documents(
                                        batch,
                                        self.embedding_model,
                                        connection=self.lance_db,
                                        table_name=table_name
                                    )
                                    added_count += len(batch)
                            else:
                                logger.error("Schema migration failed. Cannot add documents.")
                                if tracker:
                                    tracker.fail_stage(ProcessingStage.EMBEDDING, "Schema migration failed")
                                    tracker.finish(success=False)
                                raise e
                        else:
                            if tracker:
                                tracker.fail_stage(ProcessingStage.EMBEDDING, str(e))
                                tracker.finish(success=False)
                            raise e
            
            if tracker:
                tracker.complete_stage(ProcessingStage.EMBEDDING, 
                                     f"All {added_count} chunks embedded successfully")
                tracker.start_stage(ProcessingStage.STORING, 
                                  message="Finalizing database storage...")
                tracker.complete_stage(ProcessingStage.STORING, "Storage completed")
                tracker.finish(success=True, message=f"Successfully processed {added_count} document chunks")
            
            logger.info(f"✅ Successfully added {added_count} documents to vector store")
            
            return {
                "success": True,
                "added": added_count,
                "message": f"Successfully added {added_count} documents"
            }
            
        except Exception as e:
            if tracker:
                tracker.fail_stage(tracker.current_stage, str(e))
                tracker.finish(success=False, message=f"Failed to add documents: {e}")
            
            logger.error(f"❌ Failed to add documents: {e}")
            return {
                "success": False,
                "added": 0,
                "error": str(e),
                "message": f"Failed to add documents: {e}"
            }
    
    def _attempt_schema_migration(self) -> bool:
        """
        Attempt to migrate the LanceDB schema to support enhanced citation metadata.
        Preserves existing data by reading it and re-inserting with the new schema.
        
        Returns:
            True if migration successful, False otherwise
        """
        try:
            logger.info("🔄 Starting schema migration for enhanced citation support...")
            
            table_name = f"{self.proceeding}_documents"
            table_path = self.db_dir / f"{table_name}.lance"
            
            # Check if table exists and has data
            existing_data = []
            if table_path.exists():
                try:
                    # Read existing data before migration
                    table = self.lance_db.open_table(table_name)
                    row_count = table.count_rows()
                    
                    if row_count > 0:
                        logger.info(f"📊 Found {row_count} existing vectors to preserve during migration")
                        
                        # No backup folder creation - preserve data in memory only
                        logger.info("📦 Preserving existing data in memory for schema migration (no backup folders created)")
                        
                        # Read all data to preserve it
                        logger.info("📖 Reading existing data for preservation...")
                        try:
                            # Convert existing vectors back to documents
                            df = table.to_pandas()
                            
                            import pandas as pd
                            from langchain.schema import Document
                            for _, row in df.iterrows():
                                # Reconstruct Document objects from the stored data
                                page_content = row.get('text', row.get('page_content', ''))
                                
                                # Reconstruct metadata, handling potential schema differences
                                metadata = {}
                                
                                # Use Chonkie schema to prevent ArrowSchema recursion
                                from .chonkie_schema import ChonkieSchema
                                
                                # Only preserve fields that exist in Chonkie schema
                                chonkie_fields = ChonkieSchema.SCHEMA_FIELDS.keys()
                                
                                # Preserve Chonkie-compatible fields from the existing schema
                                for col in df.columns:
                                    if col in chonkie_fields and not pd.isna(row[col]):
                                        # Convert numpy types to Python native types for JSON compatibility
                                        value = row[col]
                                        if hasattr(value, 'item'):  # numpy scalar
                                            value = value.item()
                                        elif hasattr(value, 'tolist'):  # numpy array
                                            value = value.tolist()
                                        metadata[col] = value
                                
                                # Ensure basic fields are present for chunk_info creation
                                metadata.setdefault('source_url', f'migration://{self.proceeding}/doc_{len(existing_data)}')
                                metadata.setdefault('source', f'{self.proceeding} Document {len(existing_data)}')
                                metadata.setdefault('proceeding_number', self.proceeding)
                                
                                # Create chunk_info for Chonkie schema
                                chunk_info = {
                                    'text': page_content,
                                    'start_index': metadata.get('char_start', len(existing_data) * 1000),
                                    'end_index': metadata.get('char_end', len(existing_data) * 1000 + len(page_content)),
                                    'token_count': metadata.get('token_count', len(page_content.split())),
                                    'level': metadata.get('chunk_level', 0),
                                    'strategy': 'legacy_migration',
                                    'page': metadata.get('page', 1),
                                    'line_number': metadata.get('line_number', 1),
                                    'document_type': 'migrated',
                                    'last_checked': '',
                                    'document_date': '',
                                    'publication_date': '',
                                    'supersedes_priority': 0.5
                                }
                                
                                # Create document with Chonkie schema
                                from .chonkie_schema import create_chonkie_document
                                normalized_doc = create_chonkie_document(
                                    text=page_content,
                                    chunk_info=chunk_info,
                                    pdf_url=metadata.get('source_url', f'migration://{self.proceeding}'),
                                    source_name=metadata.get('source', f'{self.proceeding} Document'),
                                    proceeding=self.proceeding
                                )
                                
                                # The normalized_doc already has proper Chonkie schema metadata
                                existing_data.append(normalized_doc)
                            
                            logger.info(f"📊 Preserved {len(existing_data)} documents for re-insertion")
                            
                        except Exception as read_error:
                            logger.warning(f"⚠️ Could not read existing data for preservation: {read_error}")
                            existing_data = []
                    else:
                        logger.info("📊 No existing data found, proceeding with clean migration")
                    
                    # Remove the incompatible table
                    logger.info(f"🗑️ Removing incompatible table: {table_path}")
                    import shutil
                    shutil.rmtree(table_path)
                    
                except Exception as table_error:
                    logger.warning(f"⚠️ Error accessing existing table: {table_error}")
                    logger.info("🗑️ Removing potentially corrupted table")
                    import shutil
                    if table_path.exists():
                        shutil.rmtree(table_path)
            else:
                logger.info("🗑️ No existing table found, creating fresh schema")
            
            # Reinitialize vector store with enhanced schema
            logger.info("🔄 Reinitializing vector store with enhanced schema...")
            self.vectordb = None
            self._initialize_vector_store()
            
            # Re-insert preserved data if any exists
            if existing_data:
                logger.info(f"🔄 Re-inserting {len(existing_data)} preserved documents...")
                
                # Create the vector store with the first batch of existing data
                from langchain_community.vectorstores import LanceDB
                batch_size = 50
                first_batch = existing_data[:batch_size]
                remaining_data = existing_data[batch_size:]
                
                self.vectordb = LanceDB.from_documents(
                    first_batch,
                    self.embedding_model,
                    connection=self.lance_db,
                    table_name=table_name
                )
                
                # Add remaining data in batches
                if remaining_data:
                    for i in range(0, len(remaining_data), batch_size):
                        batch = remaining_data[i:i + batch_size]
                        self.vectordb.add_documents(batch)
                
                logger.info(f"✅ Successfully re-inserted {len(existing_data)} documents with new schema")
            
            logger.info("✅ Schema migration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema migration failed: {e}")
            logger.error("To fix manually, delete the table directory and rebuild:")
            logger.error(f"rm -rf {self.db_dir}")
            return False
    
    def get_vector_count(self) -> int:
        """Get the number of vectors in the store."""
        try:
            if self.vectordb is None:
                return 0
            
            table_name = f"{self.proceeding}_documents"
            table = self.lance_db.open_table(table_name)
            return table.count_rows()
            
        except Exception as e:
            logger.error(f"Error getting vector count: {e}")
            return 0
    
    def process_document_url(self, pdf_url: str, document_title: str = None, 
                           proceeding: str = None, enable_ocr_fallback: bool = True,
                           use_progress_tracking: bool = True) -> List[Document]:
        """
        Process a PDF URL into Document objects ready for embedding.
        
        Args:
            pdf_url: URL to the PDF document
            document_title: Optional title for the document
            proceeding: Proceeding number (defaults to self.proceeding)
            enable_ocr_fallback: Whether to enable OCR fallback
            use_progress_tracking: Whether to use enhanced progress tracking for large files
            
        Returns:
            List of Document objects with enhanced metadata
        """
        if proceeding is None:
            proceeding = self.proceeding
        
        if document_title is None:
            # Extract document name from URL
            document_title = pdf_url.split('/')[-1].replace('.PDF', '').replace('.pdf', '')
        
        # Initialize progress tracker for large files
        tracker = None
        if use_progress_tracking:
            # Quick size check to determine if we need progress tracking
            try:
                import requests
                response = requests.head(pdf_url, timeout=10)
                if response.status_code == 200:
                    content_length = response.headers.get('content-length')
                    if content_length:
                        size_mb = int(content_length) / (1024 * 1024)
                        if size_mb > 5:  # Use progress tracking for files > 5MB
                            tracker = EnhancedProgressTracker(
                                document_title=document_title,
                                estimated_size_mb=size_mb
                            )
                            tracker.start_stage(ProcessingStage.INITIALIZING, 
                                              message=f"Processing {size_mb:.1f}MB PDF: {document_title}")
            except Exception as e:
                logger.debug(f"Could not determine file size for progress tracking: {e}")
        
        try:
            if tracker:
                tracker.start_stage(ProcessingStage.DOWNLOADING, 
                                  message="Downloading and extracting PDF content...")
            
            # Use the existing intelligent hybrid processing
            result = _process_with_hybrid_evaluation(
                pdf_url=pdf_url,
                document_title=document_title,
                proceeding=proceeding,
                enable_ocr_fallback=enable_ocr_fallback
            )
            
            if tracker:
                if result and isinstance(result, list):
                    tracker.complete_stage(ProcessingStage.DOWNLOADING, 
                                         f"Successfully processed into {len(result)} chunks")
                    tracker.finish(success=True, 
                                 message=f"Generated {len(result)} document chunks ready for embedding")
                else:
                    tracker.fail_stage(ProcessingStage.DOWNLOADING, "No documents generated from PDF")
            
            if result and isinstance(result, list):
                logger.info(f"✅ Processed PDF into {len(result)} documents")
                return result
            else:
                logger.warning(f"⚠️ No documents returned from PDF processing")
                return []
                
        except Exception as e:
            if tracker:
                tracker.fail_stage(tracker.current_stage, str(e))
                tracker.finish(success=False, message=f"Processing failed: {e}")
            
            logger.error(f"❌ Failed to process document URL {pdf_url}: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the embedding system.
        
        Returns:
            Dictionary with health status information
        """
        status = {
            "proceeding": self.proceeding,
            "embedding_model_ready": self.embedding_model is not None,
            "vector_store_ready": self.vectordb is not None,
            "lance_db_connected": self.lance_db is not None,
            "vector_count": self.get_vector_count(),
            "db_path": str(self.db_dir) if self.db_dir else None,
            "timestamp": datetime.now().isoformat()
        }
        
        # Overall health
        status["healthy"] = all([
            status["embedding_model_ready"],
            status["lance_db_connected"]
        ])
        
        return status
    
    def _load_doc_hashes(self) -> Dict[str, Dict]:
        """
        Load document hashes from the JSON file.
        
        Returns:
            Dictionary mapping file hashes to document metadata
        """
        if self.doc_hashes_file.exists():
            try:
                with open(self.doc_hashes_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to load document hashes: {e}")
                return {}
        return {}
    
    def _save_doc_hashes(self):
        """Save document hashes to JSON file."""
        # Ensure the directory exists
        self.doc_hashes_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.doc_hashes_file, 'w') as f:
            json.dump(self.doc_hashes, f, indent=2)
    
    def _calculate_url_hash(self, url: str) -> str:
        """Calculate hash for a document URL."""
        return hashlib.md5(url.encode()).hexdigest()
    
    def is_document_processed(self, url: str) -> bool:
        """Check if a document has already been processed."""
        url_hash = self._calculate_url_hash(url)
        return url_hash in self.doc_hashes
    
    def add_document_to_hashes(self, url: str, title: str, chunk_count: int):
        """Add a processed document to the hash tracking."""
        url_hash = self._calculate_url_hash(url)
        self.doc_hashes[url_hash] = {
            "url": url,
            "title": title,
            "last_processed": datetime.now().isoformat(),
            "chunk_count": chunk_count,
            "total_chunks_processed": chunk_count,
            "success_rate": "100.0%"
        }
        self._save_doc_hashes()
        logger.info(f"Added document to hashes: {title} ({chunk_count} chunks)")
    
    def find_chunk_limited_documents(self, chunk_limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find documents that were processed with the old chunk limit and need reprocessing.
        
        Args:
            chunk_limit: The chunk limit to check for (default 100)
            
        Returns:
            List of document metadata dictionaries that need reprocessing
        """
        limited_docs = []
        
        for url_hash, doc_info in self.doc_hashes.items():
            chunk_count = doc_info.get('chunk_count', 0)
            if chunk_count == chunk_limit:
                # This document was likely limited by the old chunk restriction
                limited_docs.append({
                    'url_hash': url_hash,
                    'url': doc_info.get('url', ''),
                    'title': doc_info.get('title', ''),
                    'chunk_count': chunk_count,
                    'last_processed': doc_info.get('last_processed', ''),
                    'needs_reprocessing': True,
                    'reason': f'Document has exactly {chunk_limit} chunks (likely old limit applied)'
                })
        
        if limited_docs:
            logger.info(f"Found {len(limited_docs)} documents with {chunk_limit}-chunk limit that need reprocessing")
        else:
            logger.info(f"No documents found with {chunk_limit}-chunk limit")
            
        return limited_docs
    
    def reprocess_chunk_limited_documents(self, chunk_limit: int = 100, 
                                        max_reprocess: int = None,
                                        enable_ocr_fallback: bool = True) -> Dict[str, Any]:
        """
        Automatically reprocess documents that were limited by the old chunk restriction.
        
        Args:
            chunk_limit: The chunk limit to check for (default 100)
            max_reprocess: Maximum number of documents to reprocess (None for all)
            enable_ocr_fallback: Whether to enable OCR fallback during reprocessing
            
        Returns:
            Dictionary with reprocessing results
        """
        logger.info(f"🔄 Checking for documents with {chunk_limit}-chunk limit to reprocess...")
        
        # Find documents that need reprocessing
        limited_docs = self.find_chunk_limited_documents(chunk_limit)
        
        if not limited_docs:
            return {
                'success': True,
                'reprocessed': 0,
                'message': f'No documents with {chunk_limit}-chunk limit found'
            }
        
        # Limit reprocessing if requested
        if max_reprocess and len(limited_docs) > max_reprocess:
            limited_docs = limited_docs[:max_reprocess]
            logger.info(f"Limiting reprocessing to {max_reprocess} documents")
        
        reprocessed_count = 0
        failed_count = 0
        total_new_chunks = 0
        
        logger.info(f"🚀 Starting reprocessing of {len(limited_docs)} documents...")
        
        for i, doc_info in enumerate(limited_docs):
            url = doc_info['url']
            title = doc_info['title']
            old_chunk_count = doc_info['chunk_count']
            
            logger.info(f"📄 Reprocessing {i+1}/{len(limited_docs)}: {title}")
            logger.info(f"   URL: {url}")
            logger.info(f"   Old chunk count: {old_chunk_count}")
            
            try:
                # Process document with new unlimited approach
                documents = self.process_document_url(
                    pdf_url=url,
                    document_title=title,
                    proceeding=self.proceeding,
                    enable_ocr_fallback=enable_ocr_fallback,
                    use_progress_tracking=False  # Disable for batch reprocessing
                )
                
                if documents and len(documents) > 0:
                    new_chunk_count = len(documents)
                    logger.info(f"   ✅ Reprocessed: {new_chunk_count} chunks (was {old_chunk_count})")
                    
                    # Add to vector store (this will replace old embeddings)
                    result = self.add_document_incrementally(
                        documents=documents,
                        batch_size=50,
                        use_progress_tracking=False
                    )
                    
                    if result.get('success'):
                        # Update hash tracking
                        self.add_document_to_hashes(url, title, new_chunk_count)
                        reprocessed_count += 1
                        total_new_chunks += new_chunk_count
                        
                        gain = new_chunk_count - old_chunk_count
                        if gain > 0:
                            logger.info(f"   📈 Gained {gain} additional chunks from full processing!")
                        elif gain == 0:
                            logger.info(f"   ➡️ Same chunk count - document was not limited")
                        else:
                            logger.info(f"   📉 Fewer chunks than before ({gain}) - processing may have improved")
                    else:
                        logger.error(f"   ❌ Failed to embed reprocessed chunks: {result.get('error')}")
                        failed_count += 1
                else:
                    logger.warning(f"   ⚠️ Reprocessing returned no documents")
                    failed_count += 1
                    
            except Exception as e:
                logger.error(f"   ❌ Failed to reprocess {title}: {e}")
                failed_count += 1
        
        logger.info(f"🎯 Reprocessing complete:")
        logger.info(f"   ✅ Successfully reprocessed: {reprocessed_count}/{len(limited_docs)}")
        logger.info(f"   ❌ Failed: {failed_count}/{len(limited_docs)}")
        logger.info(f"   📊 Total new chunks added: {total_new_chunks}")
        
        return {
            'success': reprocessed_count > 0,
            'reprocessed': reprocessed_count,
            'failed': failed_count,
            'total_new_chunks': total_new_chunks,
            'message': f'Reprocessed {reprocessed_count}/{len(limited_docs)} documents with chunk limit issues'
        }


def create_embedding_system(proceeding: str) -> EmbeddingOnlySystem:
    """
    Factory function to create an EmbeddingOnlySystem instance.
    
    Args:
        proceeding: Proceeding number
        
    Returns:
        Initialized EmbeddingOnlySystem instance
    """
    return EmbeddingOnlySystem(proceeding)


if __name__ == "__main__":
    # Test the embedding system
    import sys
    
    if len(sys.argv) > 1:
        test_proceeding = sys.argv[1]
    else:
        test_proceeding = "R2207005"
    
    print(f"🧪 Testing EmbeddingOnlySystem with proceeding: {test_proceeding}")
    
    try:
        system = create_embedding_system(test_proceeding)
        health = system.health_check()
        
        print("📊 Health Check Results:")
        for key, value in health.items():
            print(f"  {key}: {value}")
        
        if health["healthy"]:
            print("✅ EmbeddingOnlySystem is healthy and ready!")
        else:
            print("❌ EmbeddingOnlySystem has issues")
            
    except Exception as e:
        print(f"❌ Failed to initialize EmbeddingOnlySystem: {e}")
        sys.exit(1)