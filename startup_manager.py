#!/usr/bin/env python3
"""
Startup Manager for CPUC RAG System

Implements the complete startup sequence:
1. Select first proceeding from config
2. Initialize DB and folders
3. Process embeddings from existing data
4. Initialize RAG system

Note: Document discovery/scraping has been moved to standalone_scraper.py

Author: Claude Code
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable
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
            
            # Step 3: Skip scraper workflow (moved to standalone process)
            logger.info("⏭️ Skipping scraper workflow - use standalone_scraper.py for document discovery")
            scraper_results = {'success': True, 'skipped': True, 'message': 'Scraper moved to standalone process'}
            
            # Step 4: Initialize RAG system (without processing embeddings)
            try:
                self._update_progress("Initializing RAG system...", 70)
                
                # Just initialize the RAG system - it will load existing data if available
                if not self.rag_system:
                    self.rag_system = CPUCRAGSystem(current_proceeding=self.current_proceeding)
                
                # Check RAG system status
                stats = self.rag_system.get_system_stats()
                chunk_count = stats.get('total_chunks', 0)
                
                if chunk_count > 0:
                    logger.info(f"✅ RAG system initialized with {chunk_count} existing chunks")
                    embedding_results = {'status': 'loaded_existing', 'chunks_loaded': chunk_count}
                else:
                    logger.info("✅ RAG system initialized (no existing chunks found)")
                    logger.info("💡 Use standalone_data_processor.py to process documents")
                    embedding_results = {'status': 'ready_for_data', 'chunks_loaded': 0}
                    
            except Exception as e:
                error_msg = f"RAG system initialization failed: {e}"
                logger.error(f"❌ {error_msg}")
                startup_errors.append(error_msg)
                # Continue with startup even if RAG init fails
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
                self.base_dir / "cpuc_csvs"
            ]
            
            for folder in folders_to_create:
                folder.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created/verified folder: {folder}")
            
            return db_existed
            
        except Exception as e:
            logger.error(f"Failed to initialize database and folders: {e}")
            raise
    
    # _run_standard_scraper_workflow removed - use standalone_scraper.py for document discovery
    
    # Orphaned function _analyze_document_differences removed - never called in current implementation
    
    # Embedding processing functions removed - handled by standalone_data_processor.py
    
    def _update_progress(self, message: str, progress: int):
        """Update progress with message and percentage."""
        logger.info(f"[{progress}%] {message}")
        if self.progress_callback:
            self.progress_callback(message, progress)
    
    # Orphaned function get_system_status removed - never called in current implementation


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