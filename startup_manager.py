#!/usr/bin/env python3
"""
Startup Manager for CPUC RAG System

Implements the complete startup sequence:
1. Select first proceeding from config
2. Initialize DB and folders
3. Run scraper workflow
4. Implement incremental embedding system

Author: Claude Code
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import hashlib
import time

import config
from rag_core import CPUCRAGSystem

logger = logging.getLogger(__name__)


class StartupManager:
    """Manages the complete startup sequence for the CPUC RAG system."""
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        """
        Initialize the startup manager.
        
        Args:
            progress_callback: Optional callback function for progress updates
        """
        self.progress_callback = progress_callback
        self.current_proceeding = None
        self.rag_system = None
        self.base_dir = Path(__file__).parent
        
        logger.info("Startup Manager initialized")
    
    def execute_startup_sequence(self) -> Dict:
        """
        Execute the complete startup sequence with robust error handling.
        
        Returns:
            Dictionary with startup results and status
        """
        startup_errors = []
        startup_warnings = []
        
        try:
            self._update_progress("Starting CPUC RAG system...", 0)
            logger.info("="*60)
            logger.info("STARTING CPUC RAG SYSTEM INITIALIZATION")
            logger.info("="*60)
            
            # Step 1: Select first proceeding
            try:
                self._update_progress("Selecting default proceeding...", 10)
                self.current_proceeding = self._select_first_proceeding()
                logger.info(f"✅ Selected proceeding: {self.current_proceeding}")
            except Exception as e:
                error_msg = f"Failed to select proceeding: {e}"
                logger.error(f"❌ {error_msg}")
                startup_errors.append(error_msg)
                raise
            
            # Step 2: Initialize the database and folders
            try:
                self._update_progress("Initializing database and folders...", 20)
                db_exists = self._initialize_database_and_folders()
                logger.info(f"✅ Database initialization complete (existed: {db_exists})")
            except Exception as e:
                error_msg = f"Failed to initialize database: {e}"
                logger.error(f"❌ {error_msg}")
                startup_errors.append(error_msg)
                raise
            
            # Step 3: Run scraper workflow (standard scraper)
            try:
                self._update_progress("Running scraper workflow...", 30)
                scraper_results = self._run_standard_scraper_workflow()
                
                if scraper_results.get('success', True):
                    logger.info(f"✅ Scraper workflow completed")
                else:
                    warning_msg = f"Scraper failed: {scraper_results.get('error', 'Unknown error')}"
                    logger.warning(f"⚠️ {warning_msg}")
                    startup_warnings.append(warning_msg)
                    
            except Exception as e:
                error_msg = f"Scraper workflow failed: {e}"
                logger.error(f"❌ {error_msg}")
                startup_errors.append(error_msg)
                # Continue with startup even if scraper fails
                scraper_results = {'success': False, 'error': str(e)}
            
            # Step 4: Process embeddings (with fallbacks)
            try:
                self._update_progress("Processing embeddings...", 70)
                embedding_results = self._process_incremental_embeddings()
                
                if embedding_results.get('status') in ['completed', 'up_to_date', 'no_data']:
                    logger.info(f"✅ Embedding processing completed ({embedding_results.get('status')})")
                    if embedding_results.get('standard_embedding_used'):
                        startup_warnings.append("Used standard embedding (incremental not available)")
                else:
                    warning_msg = f"Embedding processing issue: {embedding_results.get('status')}"
                    logger.warning(f"⚠️ {warning_msg}")
                    startup_warnings.append(warning_msg)
                    
            except Exception as e:
                error_msg = f"Embedding processing failed: {e}"
                logger.error(f"❌ {error_msg}")
                startup_errors.append(error_msg)
                # Continue with startup even if embedding fails
                embedding_results = {'status': 'error', 'error': str(e)}
            
            self._update_progress("Startup sequence completed!", 100)
            logger.info("="*60)
            logger.info("STARTUP SEQUENCE COMPLETED")
            logger.info(f"Errors: {len(startup_errors)}, Warnings: {len(startup_warnings)}")
            logger.info("="*60)
            
            # Determine overall success
            critical_failure = len(startup_errors) > 2 or not self.rag_system
            
            return {
                'success': not critical_failure,
                'proceeding': self.current_proceeding,
                'db_existed': db_exists,
                'scraper_results': scraper_results,
                'embedding_results': embedding_results,
                'rag_system': self.rag_system,
                'startup_errors': startup_errors,
                'startup_warnings': startup_warnings,
                'fallbacks_used': len(startup_warnings) > 0
            }
            
        except Exception as e:
            logger.error(f"❌ CRITICAL STARTUP FAILURE: {e}")
            logger.error("="*60)
            self._update_progress(f"Critical startup failure: {str(e)}", -1)
            
            return {
                'success': False,
                'error': str(e),
                'proceeding': self.current_proceeding,
                'startup_errors': startup_errors + [str(e)],
                'startup_warnings': startup_warnings,
                'critical_failure': True
            }
    
    def _select_first_proceeding(self) -> str:
        """Step 1: Select the first proceeding from AVAILABLE_PROCEEDINGS."""
        first_proceeding = config.get_first_proceeding()
        self.current_proceeding = first_proceeding
        logger.info(f"Selected first proceeding: {first_proceeding}")
        return first_proceeding
    
    def _initialize_database_and_folders(self) -> bool:
        """
        Step 2: Initialize database and create necessary folders.
        
        Returns:
            True if database already existed, False if created new
        """
        try:
            # Check if DB folder exists
            proceeding_db_path = self.base_dir / "local_chroma_db" / self.current_proceeding
            db_existed = proceeding_db_path.exists()
            
            if db_existed:
                logger.info(f"Database folder exists for {self.current_proceeding}")
                # Initialize RAG system with existing DB
                self.rag_system = CPUCRAGSystem(current_proceeding=self.current_proceeding)
            else:
                logger.info(f"Creating new database folder for {self.current_proceeding}")
                proceeding_db_path.mkdir(parents=True, exist_ok=True)
                
                # Initialize RAG system with new DB
                self.rag_system = CPUCRAGSystem(current_proceeding=self.current_proceeding)
            
            # Create additional required folders
            folders_to_create = [
                self.base_dir / "cpuc_pdfs" / self.current_proceeding,
                self.base_dir / "cpuc_csvs"
            ]
            
            for folder in folders_to_create:
                folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created/verified folder: {folder}")
            
            return db_existed
            
        except Exception as e:
            logger.error(f"Failed to initialize database and folders: {e}")
            raise
    
    def _run_standard_scraper_workflow(self) -> Dict:
        """Run standard scraper workflow."""
        try:
            from cpuc_scraper import CPUCUnifiedScraper
            
            self._update_progress("Initializing scraper...", 35)
            scraper = CPUCUnifiedScraper(headless=True)
            
            self._update_progress("Running scraper...", 40)
            results = scraper.scrape_proceeding_pdfs(self.current_proceeding)
            
            self._update_progress("Scraper completed", 65)
            
            # Convert to expected format
            return {
                'success': True,
                'csv_results': results.get('csv_urls', []),
                'google_results': results.get('google_urls', []),
                'metadata_count': len(results.get('csv_urls', [])),
                'total_scraped': results.get('total_scraped', 0)
            }
            
        except Exception as e:
            logger.error(f"Scraper failed: {e}")
            raise
    
    def _analyze_document_differences(self) -> Dict:
        """
        Analyze differences between old and new scraped data.
        
        Returns:
            Dictionary with difference analysis
        """
        try:
            proceeding_paths = config.get_proceeding_file_paths(self.current_proceeding)
            history_file = proceeding_paths['scraped_pdf_history']
            
            if not history_file.exists():
                logger.info("No existing history file, all documents are new")
                return {'new_documents': [], 'updated_documents': [], 'status': 'first_run'}
            
            # Load existing history
            with open(history_file, 'r') as f:
                existing_history = json.load(f)
            
            # For now, return basic analysis
            # This will be enhanced when we implement the full metadata extraction
            return {
                'existing_count': len(existing_history),
                'status': 'analyzed'
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze document differences: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _process_incremental_embeddings(self) -> Dict:
        """
        Step 4: Process incremental embeddings for new/updated documents.
        
        Returns:
            Dictionary with embedding results
        """
        try:
            # Try incremental embedder first, fallback to standard RAG build
            try:
                from incremental_embedder import create_incremental_embedder
                logger.info("Using incremental embedder")
                
                def embedder_progress(message, progress):
                    overall_progress = 70 + int((progress / 100) * 30)
                    self._update_progress(message, overall_progress)
                
                embedder = create_incremental_embedder(self.current_proceeding, embedder_progress)
                embedding_results = embedder.process_incremental_embeddings()
                
            except ImportError as ie:
                logger.warning(f"Incremental embedder not available ({ie}), using standard RAG build")
                embedding_results = self._run_standard_embedding()
            
            self._update_progress("Embedding processing completed", 100)
            return embedding_results
            
        except Exception as e:
            logger.error(f"Embedding processing failed: {e}")
            # Return a default result to allow startup to continue
            return {
                'status': 'error',
                'error': str(e),
                'documents_processed': 0,
                'fallback_used': True
            }
    
    def _run_standard_embedding(self) -> Dict:
        """Run standard RAG system embedding as fallback."""
        try:
            if not self.rag_system:
                logger.warning("No RAG system available for embedding")
                return {
                    'status': 'no_rag_system',
                    'documents_processed': 0
                }
            
            self._update_progress("Checking vector store status...", 75)
            
            # Check if we have any data to process
            proceeding_paths = config.get_proceeding_file_paths(self.current_proceeding)
            history_file = proceeding_paths['scraped_pdf_history']
            
            if not history_file.exists():
                logger.info("No scraped PDF history found, skipping embedding")
                return {
                    'status': 'no_data',
                    'documents_processed': 0
                }
            
            self._update_progress("Building vector store...", 85)
            
            # Try to build vector store
            success = self.rag_system.build_vector_store()
            
            if success:
                self._update_progress("Vector store built successfully", 95)
                return {
                    'status': 'completed',
                    'documents_processed': 1,  # We don't have exact count in standard mode
                    'standard_embedding_used': True
                }
            else:
                logger.warning("Vector store building failed")
                return {
                    'status': 'build_failed',
                    'documents_processed': 0
                }
                
        except Exception as e:
            logger.error(f"Standard embedding failed: {e}")
            raise
    
    def _update_progress(self, message: str, progress: int):
        """Update progress with message and percentage."""
        logger.info(f"[{progress}%] {message}")
        if self.progress_callback:
            self.progress_callback(message, progress)
    
    def get_system_status(self) -> Dict:
        """Get current system status."""
        try:
            if not self.current_proceeding:
                return {'status': 'not_initialized'}
            
            proceeding_paths = config.get_proceeding_file_paths(self.current_proceeding)
            
            status = {
                'proceeding': self.current_proceeding,
                'db_folder_exists': proceeding_paths['vector_db'].exists(),
                'csv_file_exists': proceeding_paths['result_csv'].exists(),
                'history_file_exists': proceeding_paths['scraped_pdf_history'].exists(),
                'rag_system_ready': self.rag_system is not None
            }
            
            # Add vector store status if RAG system exists
            if self.rag_system:
                parity_check = self.rag_system._check_vector_store_parity()
                status['vector_store_parity'] = parity_check['has_parity']
                status['missing_documents'] = len(parity_check.get('missing_files', []))
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'status': 'error', 'error': str(e)}


# Utility functions for startup management

def create_startup_manager(progress_callback: Optional[Callable] = None) -> StartupManager:
    """Factory function to create startup manager."""
    return StartupManager(progress_callback=progress_callback)


def run_startup_sequence(progress_callback: Optional[Callable] = None) -> Dict:
    """
    Convenience function to run the complete startup sequence.
    
    Args:
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with startup results
    """
    manager = create_startup_manager(progress_callback)
    return manager.execute_startup_sequence()


if __name__ == "__main__":
    # Test the startup sequence
    logging.basicConfig(level=logging.INFO)
    
    def progress_logger(message: str, progress: int):
        print(f"Progress: {progress}% - {message}")
    
    results = run_startup_sequence(progress_callback=progress_logger)
    
    if results['success']:
        print("✅ Startup sequence completed successfully!")
        print(f"Selected proceeding: {results['proceeding']}")
    else:
        print(f"❌ Startup sequence failed: {results.get('error', 'Unknown error')}")