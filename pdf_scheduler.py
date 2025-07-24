#!/usr/bin/env python3
"""
PDF Scheduler for CPUC RAG System

This module provides background job scheduling for automated PDF checking,
downloading, and model updates. It runs every hour to check for new
documents and automatically updates the RAG system.

Author: Claude Code
"""

import json
import logging
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Callable
import schedule

import config
# Import existing modules
from cpuc_scraper import CPUCSimplifiedScraper, scrape_proceeding_pdfs

logger = logging.getLogger(__name__)


class PDFScheduler:
    """Background scheduler for automated PDF checking and downloading"""
    
    def __init__(self, rag_system=None, check_interval_hours: int = 1):
        """
        Initialize the PDF scheduler
        
        Args:
            rag_system: The RAG system instance to update
            check_interval_hours: How often to check for new PDFs (default: 1 hour)
        """
        self.rag_system = rag_system
        self.check_interval_hours = check_interval_hours
        self.is_running = False
        self.scheduler_thread = None
        self.last_check_time = None
        self.last_download_count = 0
        self.status_file = Path("pdf_scheduler_status.json")
        
        # Initialize status
        self.status = {
            'last_check': None,
            'last_download_count': 0,
            'is_running': False,
            'next_check': None,
            'total_downloads': 0,
            'errors': []
        }
        
        # Load existing status
        self._load_status()
        
        # Setup callback functions
        self.on_new_pdfs_downloaded: Optional[Callable] = None
        self.on_check_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None
        self.on_rag_updated: Optional[Callable] = None
        
        # Initialize unified document scraper for enhanced discovery
        try:
            # Import config here to avoid circular imports
            import config
            
            self.document_scraper = CPUCSimplifiedScraper(
                headless=True
            )
            logger.info("Unified document scraper initialized for scheduler")
        except Exception as e:
            logger.warning(f"Could not initialize unified document scraper: {e}")
            self.document_scraper = None
        
        logger.info(f"PDF Scheduler initialized with {check_interval_hours}-hour interval")
    
    def _load_status(self):
        """Load scheduler status from file"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    saved_status = json.load(f)
                    self.status.update(saved_status)
                    
                    # Parse datetime strings
                    if self.status['last_check']:
                        self.last_check_time = datetime.fromisoformat(self.status['last_check'])
                    
                    logger.info(f"Loaded scheduler status. Last check: {self.status['last_check']}")
        except Exception as e:
            logger.warning(f"Could not load scheduler status: {e}")
    
    def _save_status(self):
        """Save scheduler status to file"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save scheduler status: {e}")
    
    def start(self):
        """Start the background scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self.is_running = True
        self.status['is_running'] = True
        
        # Schedule the job to run every N hours
        schedule.every(self.check_interval_hours).hours.do(self._check_for_new_pdfs)
        
        # Also run immediately on startup
        self._schedule_immediate_check()
        
        # Start the scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info(f"PDF Scheduler started. Next check in {self.check_interval_hours} hours")
        self._save_status()
    
    def stop(self):
        """Stop the background scheduler"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.status['is_running'] = False
        schedule.clear()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("PDF Scheduler stopped")
        self._save_status()
    
    def _schedule_immediate_check(self):
        """Schedule an immediate check on startup"""
        # Don't check immediately if we just checked recently
        if self.last_check_time:
            time_since_last = datetime.now() - self.last_check_time
            if time_since_last < timedelta(hours=1):
                logger.info(f"Skipping immediate check. Last check was {time_since_last} ago")
                return
        
        # Schedule immediate check
        schedule.every(1).seconds.do(self._check_for_new_pdfs).tag('immediate')
        logger.info("Scheduled immediate PDF check")
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        logger.info("Scheduler thread started")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                self._handle_error(e)
                time.sleep(300)  # Wait 5 minutes before retrying
        
        logger.info("Scheduler thread stopped")
    
    def _check_for_new_pdfs(self):
        """Check for new PDFs and download them"""
        logger.info("🔍 Starting scheduled PDF check...")
        
        try:
            # Clear immediate check tags
            schedule.clear('immediate')
            
            # Update status
            self.last_check_time = datetime.now()
            self.status['last_check'] = self.last_check_time.isoformat()
            self.status['next_check'] = (self.last_check_time + timedelta(hours=self.check_interval_hours)).isoformat()
            
            # Perform the check
            new_downloads = self._perform_pdf_check()
            
            # Update status
            self.last_download_count = new_downloads
            self.status['last_download_count'] = new_downloads
            self.status['total_downloads'] += new_downloads
            
            # Trigger callbacks
            if new_downloads > 0:
                logger.info(f"📥 Downloaded {new_downloads} new PDFs")
                if self.on_new_pdfs_downloaded:
                    self.on_new_pdfs_downloaded(new_downloads)
                
                # Update RAG system if available
                if self.rag_system:
                    self._update_rag_system()
            else:
                logger.info("✅ No new PDFs found")
            
            if self.on_check_complete:
                self.on_check_complete(new_downloads)
            
            self._save_status()
            logger.info(f"✅ PDF check completed. Next check: {self.status['next_check']}")
            
        except Exception as e:
            logger.error(f"❌ PDF check failed: {e}")
            logger.error(traceback.format_exc())
            self._handle_error(e)
    
    def _perform_pdf_check(self) -> int:
        """Perform the actual PDF checking and downloading using scraper"""
        total_new_documents = 0
        
        try:
            # Run Google search discovery with proceeding priority
            if self.document_scraper:
                logger.info("🔍 Running document discovery...")
                
                # Get proceedings in priority order (default first)
                primary_proceeding = config.DEFAULT_PROCEEDING
                prioritized_proceedings = [primary_proceeding]
                
                # Add other proceedings after primary
                for proc in self.document_scraper.proceedings:
                    if proc != primary_proceeding and proc not in prioritized_proceedings:
                        prioritized_proceedings.append(proc)
                
                logger.info(f"Scheduler proceeding priority order: {prioritized_proceedings}")
                
                for proceeding in prioritized_proceedings:
                    try:
                        priority_label = "PRIMARY" if proceeding == primary_proceeding else "secondary"
                        logger.info(f"🔍 Checking {proceeding} ({priority_label})")
                        
                        # Use simplified scraper to get new PDFs
                        scrape_result = scrape_proceeding_pdfs(proceeding, headless=True)
                        new_pdfs = scrape_result.get('total_pdfs', 0)
                        
                        if new_pdfs > 0:
                            logger.info(f"Enhanced search found {new_pdfs} documents for {proceeding} ({priority_label})")
                            
                            # Count as new documents found
                            total_new_documents += new_pdfs
                        else:
                            logger.info(f"No new documents found for {proceeding} ({priority_label})")
                            
                    except Exception as e:
                        logger.warning(f"Enhanced discovery failed for {proceeding}: {e}")

        
        except Exception as e:
            logger.error(f"Error during PDF check: {e}")
            raise
        
        return total_new_documents
    
    
    def _update_rag_system(self):
        """Update the RAG system with new PDFs"""
        try:
            if self.rag_system:
                logger.info("🔄 Updating RAG system with new PDFs...")
                
                # For URL-based processing, check if we have new URLs from scraped PDF history
                # Use proceeding-specific scraped PDF history
                from config import get_proceeding_file_paths, DEFAULT_PROCEEDING
                proceeding_paths = get_proceeding_file_paths(DEFAULT_PROCEEDING)
                download_history_path = proceeding_paths['scraped_pdf_history']
                if download_history_path.exists():
                    logger.info("Checking for new URLs in download history...")
                    
                    # Get vector store parity status
                    parity_check = self.rag_system._check_vector_store_parity()
                    
                    if not parity_check['has_parity'] or parity_check['missing_files']:
                        logger.info(f"Vector store needs update: {len(parity_check.get('missing_files', []))} missing files")
                        
                        # Load download history and update vector store
                        try:
                            with open(download_history_path, 'r') as f:
                                download_history = json.load(f)
                            
                            # Convert to URL format for missing files only
                            pdf_urls = []
                            missing_files = set(parity_check.get('missing_files', []))
                            
                            for hash_key, entry in download_history.items():
                                if isinstance(entry, dict) and entry.get('url') and entry.get('filename'):
                                    if entry['filename'] in missing_files:
                                        pdf_urls.append({
                                            'url': entry['url'],
                                            'filename': entry['filename']
                                        })
                            
                            if pdf_urls:
                                logger.info(f"Updating RAG system with {len(pdf_urls)} new URLs...")
                                self.rag_system.build_vector_store_from_urls(pdf_urls, force_rebuild=False, incremental_mode=True)
                                
                                logger.info("✅ RAG system updated with new URLs")
                                self.status['last_rag_update'] = datetime.now().isoformat()
                                self.status['rag_update_status'] = 'success'
                                
                                # Trigger callback if available
                                if hasattr(self, 'on_rag_updated') and self.on_rag_updated:
                                    self.on_rag_updated()
                            else:
                                logger.info("No new URLs to process")
                                self.status['rag_update_status'] = 'no_new_urls'
                                
                        except Exception as e:
                            logger.error(f"Failed to process download history for RAG update: {e}")
                            self.status['rag_update_status'] = f'error: {str(e)}'
                    else:
                        logger.info("Vector store is up to date")
                        self.status['rag_update_status'] = 'up_to_date'
                else:
                    logger.info("No download history found for RAG update")
                    self.status['rag_update_status'] = 'no_download_history'
                    
            else:
                logger.warning("RAG system not available for update")
                self.status['rag_update_status'] = 'rag_system_unavailable'
                
        except Exception as e:
            logger.error(f"Failed to update RAG system: {e}")
            self.status['rag_update_status'] = f'error: {str(e)}'
            raise
    
    def _handle_error(self, error: Exception):
        """Handle errors and update status"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error': str(error),
            'traceback': traceback.format_exc()
        }
        
        self.status['errors'].append(error_info)
        
        # Keep only last 10 errors
        if len(self.status['errors']) > 10:
            self.status['errors'] = self.status['errors'][-10:]
        
        self._save_status()
        
        if self.on_error:
            self.on_error(error)
    
    def get_status(self) -> Dict:
        """Get current scheduler status"""
        status = self.status.copy()
        status['is_running'] = self.is_running
        status['check_interval_hours'] = self.check_interval_hours
        
        if self.last_check_time:
            status['time_since_last_check'] = str(datetime.now() - self.last_check_time)
        
        return status
    
    def force_check(self):
        """Force an immediate PDF check"""
        logger.info("🔄 Forcing immediate PDF check...")
        
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        # Schedule immediate check
        schedule.every(1).seconds.do(self._check_for_new_pdfs).tag('manual')
        logger.info("Manual PDF check scheduled")


def create_pdf_scheduler(rag_system=None, check_interval_hours: int = 1) -> PDFScheduler:
    """Factory function to create PDF scheduler"""
    return PDFScheduler(rag_system=rag_system, check_interval_hours=check_interval_hours)