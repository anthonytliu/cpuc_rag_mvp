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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import lancedb
from langchain.schema import Document

# Try relative imports first, fall back to absolute
try:
    from ..core import config
except ImportError:
    from core import config
# Try relative imports first, fall back to absolute
try:
    from ..core import models
    from .data_processing import _process_with_hybrid_evaluation
    from ..monitoring.enhanced_progress_tracker import EnhancedProgressTracker, ProcessingStage
except ImportError:
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
                        # Check if this is a schema compatibility issue
                        if "not found in target schema" in str(e):
                            logger.warning("Schema compatibility issue detected. Attempting migration...")
                            
                            if tracker:
                                tracker.update_progress(added_count, "Schema migration in progress...")
                            
                            # Attempt schema migration
                            if self._attempt_schema_migration():
                                logger.info("Schema migration successful. Retrying batch...")
                                if tracker:
                                    tracker.update_progress(added_count, "Migration successful, retrying batch...")
                                
                                # Retry the batch after migration
                                self.vectordb.add_documents(batch)
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
        
        Returns:
            True if migration successful, False otherwise
        """
        try:
            logger.info("🔄 Starting schema migration for enhanced citation support...")
            
            # Backup existing vector store
            backup_path = self.db_dir.parent / f"{self.db_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"📦 Creating backup at: {backup_path}")
            
            if self.db_dir.exists():
                import shutil
                shutil.copytree(self.db_dir, backup_path)
                logger.info("✅ Backup created successfully")
            
            # Remove the incompatible table
            table_name = f"{self.proceeding}_documents"
            table_path = self.db_dir / f"{table_name}.lance"
            
            if table_path.exists():
                logger.info(f"🗑️ Removing incompatible table: {table_path}")
                import shutil
                shutil.rmtree(table_path)
            
            # Reinitialize vector store with enhanced schema
            logger.info("🔄 Reinitializing vector store with enhanced schema...")
            self.vectordb = None
            self._initialize_vector_store()
            
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