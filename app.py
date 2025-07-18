import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import hashlib
import json
from typing import List, Dict, Tuple

import streamlit as st
# import http.server
# import socketserver
import config

sys.path.append(str(Path(__file__).parent.resolve()))

from rag_core import CPUCRAGSystem
from timeline_integration import create_timeline_integration
from pdf_scheduler import create_pdf_scheduler
from cpuc_scraper import CPUCUnifiedScraper, get_new_pdfs_for_proceeding
from startup_manager import create_startup_manager
import data_processing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title="CPUC Regulatory RAG", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 5px solid #1f77b4;
        font-family: monospace;
    }
    .confidence-indicator {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# PDF Server functionality removed - now using URL-based processing
# class Handler(http.server.SimpleHTTPRequestHandler):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, directory=str(config.BASE_PDF_DIR), **kwargs)

# @st.cache_resource
# def start_pdf_server():
#     """Starts the PDF server in a background thread."""
#     if '_pdf_server_thread' in st.session_state and st.session_state._pdf_server_thread.is_alive():
#         logger.info("PDF server is already running.")
#         return

#     def run_server():
#         with socketserver.TCPServer(("", config.PDF_SERVER_PORT), Handler) as httpd:
#             logger.info(f"Starting PDF server at http://localhost:{config.PDF_SERVER_PORT}")
#             httpd.serve_forever()

#     thread = threading.Thread(target=run_server, daemon=True)
#     st.session_state._pdf_server_thread = thread
#     thread.start()
#     logger.info("PDF server thread started.")

@st.cache_resource
def execute_enhanced_startup_sequence() -> Dict:
    """
    Execute the enhanced startup sequence with progress tracking.
    
    Returns:
        Dictionary with startup results
    """
    try:
        # Check if we're already running startup
        if st.session_state.get('startup_in_progress', False):
            return {'status': 'in_progress'}
        
        # Mark startup as in progress
        st.session_state['startup_in_progress'] = True
        
        # Create progress indicators
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_ui_progress(message: str, progress: int):
            """Update UI with progress information."""
            if progress >= 0:
                progress_bar.progress(progress / 100)
            status_text.text(f"{message}")
            
            # Store progress in session state for persistence
            st.session_state['startup_progress'] = progress
            st.session_state['startup_message'] = message
        
        # Create and run startup manager
        startup_manager = create_startup_manager(progress_callback=update_ui_progress)
        results = startup_manager.execute_startup_sequence()
        
        # Clear progress indicators on completion
        if results['success']:
            progress_bar.progress(100)
            status_text.success("✅ Startup sequence completed successfully!")
            
            # Store RAG system in session state
            st.session_state['rag_system'] = results['rag_system']
            st.session_state['current_proceeding'] = results['proceeding']
        else:
            progress_bar.empty()
            status_text.error(f"❌ Startup failed: {results.get('error', 'Unknown error')}")
        
        # Mark startup as complete
        st.session_state['startup_in_progress'] = False
        st.session_state['startup_completed'] = True
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced startup sequence failed: {e}")
        st.session_state['startup_in_progress'] = False
        if 'progress_bar' in locals():
            progress_bar.empty()
        if 'status_text' in locals():
            status_text.error(f"❌ Startup failed: {str(e)}")
        
        return {
            'success': False,
            'error': str(e)
        }


def initialize_rag_system(selected_proceeding: str = None):
    """Initialize the RAG system with proper launch sequence following spec."""
    try:
        # Use selected proceeding or default
        if selected_proceeding is None:
            from config import DEFAULT_PROCEEDING
            selected_proceeding = st.session_state.get('selected_proceeding', DEFAULT_PROCEEDING)
        
        with st.spinner(f"Initializing RAG System for {selected_proceeding}..."):
            # Step 1: Initialize system (this will load existing DB if available)
            system = CPUCRAGSystem(current_proceeding=selected_proceeding)
            
            # Step 2: Check if we have a working vector store
            if system.vectordb is not None:
                try:
                    chunk_count = system.vectordb._collection.count()
                    if chunk_count > 0:
                        logger.info(f"✅ Loaded existing vector store with {chunk_count} chunks")
                        st.success(f"✅ RAG System Initialized with {chunk_count} existing chunks")
                        
                        # Step 3: Check for new PDFs for current proceeding in background
                        if not st.session_state.get('launch_hash_check_completed', False):
                            check_for_new_pdfs_on_launch(system, selected_proceeding)
                            st.session_state['launch_hash_check_completed'] = True
                        
                        return system
                    else:
                        logger.info("Vector store exists but is empty")
                        st.info("Vector store exists but is empty. Checking for documents...")
                except Exception as e:
                    logger.warning(f"Vector store validation failed: {e}")
                    st.warning("Vector store validation failed. Will rebuild if needed.")
            
            # Step 4: If no working vector store, check for scraped PDF history
            from config import get_proceeding_file_paths
            proceeding_paths = get_proceeding_file_paths(selected_proceeding)
            scraped_pdf_history_path = proceeding_paths['scraped_pdf_history']
            vector_db_path = proceeding_paths['vector_db']
            document_hashes_path = proceeding_paths['document_hashes']
            
            # Check if we have any existing data to work with
            has_scraped_history = scraped_pdf_history_path.exists() and scraped_pdf_history_path.stat().st_size > 10
            has_vector_db = vector_db_path.exists() and any(vector_db_path.iterdir()) if vector_db_path.exists() else False
            has_document_hashes = document_hashes_path.exists() and document_hashes_path.stat().st_size > 10
            
            if has_scraped_history or has_vector_db or has_document_hashes:
                st.info("Found scraped PDF history. Building vector store from existing URLs...")
                logger.info("No working vector store found, but scraped PDF history exists. Building from URLs...")
                # The system will auto-build from scraped PDF history
            else:
                # Check if auto-scraping is enabled
                if config.RUN_SCRAPER_ON_STARTUP:
                    st.info("🚀 No existing data found. Automatically starting document discovery...")
                    logger.info("No vector store or scraped PDF history found. Auto-starting scraper...")
                    
                    # Automatically trigger scraper to discover and build initial dataset
                    success = auto_initialize_with_scraper(system, selected_proceeding)
                    if success:
                        st.success("✅ Initial document discovery completed! Vector store is being built...")
                    else:
                        st.error("❌ Failed to initialize documents. Please check the logs or try manual scraping.")
                else:
                    st.warning("📋 No existing data found. Auto-scraping is disabled.")
                    
                    # Provide an option to enable auto-scraping for this session
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.info("💡 **To get started:** You can enable auto-scraping for this session or set `RUN_SCRAPER_ON_STARTUP=true` in your environment.")
                    with col2:
                        if st.button("🚀 Enable Auto-Scraping", help="Enable auto-scraping for this session only"):
                            # Temporarily enable auto-scraping for this session
                            config.RUN_SCRAPER_ON_STARTUP = True
                            st.rerun()
                    
                    logger.info("No data found and auto-scraping disabled. Manual intervention required.")
            
        return system
    except Exception as e:
        logger.error(f"Fatal error during RAG system initialization: {e}", exc_info=True)
        st.error(f"Could not initialize RAG System: {e}")
        return None

def auto_initialize_with_scraper(rag_system, selected_proceeding: str) -> bool:
    """
    Automatically initialize the system by running the scraper to discover documents.
    
    Args:
        rag_system: The RAG system instance
        selected_proceeding: The proceeding to scrape for
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info(f"🤖 Auto-initializing system for {selected_proceeding}")
        
        # Create status containers for progress feedback
        progress_container = st.container()
        with progress_container:
            st.write("### 🔍 Automatic Document Discovery")
            progress_bar = st.progress(0, text="Initializing scraper...")
            status_text = st.empty()
        
        # Step 1: Initialize scraper
        progress_bar.progress(10, text="Setting up document scraper...")
        status_text.info("🔧 Initializing CPUC document scraper...")
        
        from cpuc_scraper import CPUCUnifiedScraper
        scraper = CPUCUnifiedScraper(headless=True, proceedings=[selected_proceeding])
        
        # Add a small delay to ensure UI updates are visible
        import time
        time.sleep(1)
        
        # Step 2: Run scraper to discover documents
        progress_bar.progress(30, text="Discovering documents from CPUC website...")
        status_text.info("🔍 Searching CPUC website for documents...")
        
        # Run scraper (may take a few minutes)
        scraper_results = scraper.scrape_proceeding_pdfs(selected_proceeding)
        total_found = scraper_results.get('total_scraped', 0)
        new_pdfs = scraper_results.get('new_pdfs', [])
        
        if total_found == 0:
            logger.warning("No documents found during auto-initialization")
            status_text.warning("⚠️ No documents found. The proceeding may not have public documents yet.")
            progress_bar.progress(100, text="No documents found")
            return False
        
        progress_bar.progress(60, text=f"Found {total_found} documents, processing {len(new_pdfs)} new ones...")
        status_text.success(f"✅ Found {total_found} documents ({len(new_pdfs)} new)")
        
        # Step 3: Build vector store from discovered documents
        if len(new_pdfs) > 0:
            progress_bar.progress(80, text="Building vector database...")
            status_text.info("🔨 Building vector database from discovered documents...")
            
            # Trigger vector store build (this will process URLs from scraped_pdf_history)
            vector_build_success = rag_system.build_vector_store()
            
            if vector_build_success:
                progress_bar.progress(100, text="System ready!")
                status_text.success("🎉 Vector database built successfully!")
                logger.info(f"✅ Auto-initialization successful: {total_found} documents discovered, vector store built")
                return True
            else:
                status_text.error("❌ Failed to build vector database")
                logger.error("Vector store building failed during auto-initialization")
                return False
        else:
            progress_bar.progress(100, text="Documents already processed")
            status_text.info("📚 All discovered documents were already in the system")
            logger.info("Auto-initialization completed - all documents were already processed")
            return True
            
    except Exception as e:
        logger.error(f"Auto-initialization failed: {e}", exc_info=True)
        if 'status_text' in locals():
            status_text.error(f"❌ Auto-initialization failed: {e}")
        return False
    finally:
        # Clean up scraper
        if 'scraper' in locals():
            scraper._cleanup_driver()

@st.cache_resource
def initialize_pdf_scheduler(_rag_system):
    """Initialize the PDF scheduler with caching."""
    try:
        if not config.PDF_SCHEDULER_ENABLED:
            logger.info("PDF Scheduler is disabled in configuration")
            return None
            
        scheduler = create_pdf_scheduler(_rag_system, check_interval_hours=config.PDF_CHECK_INTERVAL_HOURS)
        
        # Set up callback for RAG system updates
        def on_rag_updated():
            """Callback when RAG system is updated - clear caches and force refresh"""
            logger.info("RAG system updated - clearing Streamlit caches")
            st.cache_resource.clear()
            # Store update flag in session state
            if 'rag_system_updated' not in st.session_state:
                st.session_state['rag_system_updated'] = 0
            st.session_state['rag_system_updated'] += 1
        
        scheduler.on_rag_updated = on_rag_updated
        scheduler.start()
        logger.info(f"PDF Scheduler started successfully (checking every {config.PDF_CHECK_INTERVAL_HOURS} hours)")
        return scheduler
    except Exception as e:
        logger.error(f"Failed to initialize PDF scheduler: {e}", exc_info=True)
        st.error(f"Could not initialize PDF Scheduler: {e}")
        return None


@st.cache_resource
def initialize_document_scraper(selected_proceeding: str = None):
    """Initialize the unified CPUC document scraper with caching."""
    try:
        if not config.PDF_SCHEDULER_ENABLED:
            logger.info("Document scraper is disabled (PDF_SCHEDULER_ENABLED=False)")
            return None
            
        # Use selected proceeding or default
        if selected_proceeding is None:
            selected_proceeding = st.session_state.get('selected_proceeding', config.DEFAULT_PROCEEDING)
            
        # Initialize unified scraper
        scraper = CPUCUnifiedScraper(
            headless=True,
            max_workers=config.SCRAPER_MAX_WORKERS,
            proceedings=[selected_proceeding]
        )
        
        logger.info(f"Unified document scraper initialized successfully for {selected_proceeding}")
        return scraper
    except Exception as e:
        logger.error(f"Failed to initialize document scraper: {e}", exc_info=True)
        st.warning(f"Document scraper unavailable: {e}")
        return None


def compare_scraper_results_with_hashes(rag_system, scraper_results, proceeding: str) -> int:
    """
    Compare fresh scraper results with existing hashes to find new PDFs.
    
    This implements the proper hash checking logic as specified:
    - Takes fresh scraper results and compares with existing document_hashes.json
    - Identifies truly new PDFs by URL hash comparison
    - Returns count of new PDFs found
    
    Args:
        rag_system: RAG system instance
        scraper_results: List of tuples from scraper (url, title, type)
        proceeding: Current proceeding being processed
        
    Returns:
        int: Number of new PDFs found
    """
    try:
        logger.info(f"🔍 Comparing scraper results with existing hashes for {proceeding}")
        
        # Get existing hashes
        existing_hashes = set(rag_system.doc_hashes.keys())
        
        # Convert scraper results to URLs and calculate hashes
        new_urls = []
        for result in scraper_results:
            url = result[0]  # URL is first element in tuple
            title = result[1] if len(result) > 1 else ""
            
            # Only process PDF URLs
            if url.endswith('.PDF') or url.endswith('.pdf'):
                url_hash = data_processing.get_url_hash(url)
                
                # Check if this URL hash is new
                if url_hash not in existing_hashes:
                    new_urls.append({
                        'url': url,
                        'title': title,
                        'hash': url_hash,
                        'proceeding': proceeding
                    })
        
        logger.info(f"Hash comparison results: {len(new_urls)} new PDFs found out of {len(scraper_results)} total results")
        
        if new_urls:
            # Store the new URLs for background processing
            if 'new_urls_found' not in st.session_state:
                st.session_state['new_urls_found'] = []
            st.session_state['new_urls_found'].extend(new_urls)
            
            # Log details
            for url_data in new_urls[:5]:  # Log first 5
                logger.info(f"  New PDF: {url_data['url']}")
            if len(new_urls) > 5:
                logger.info(f"  ... and {len(new_urls) - 5} more")
        
        return len(new_urls)
        
    except Exception as e:
        logger.error(f"Hash comparison failed: {e}")
        return 0


class BackgroundProcessor:
    """
    Background processor for real-time document processing following spec requirements.
    
    This class handles:
    - Progressive document processing without blocking UI
    - Real-time user notifications
    - Incremental DB updates
    - Graceful failure handling
    """
    
    def __init__(self, rag_system):
        self.rag_system = rag_system
        self.is_processing = False
        self.processing_thread = None
        self.current_status = "idle"
        self.processed_count = 0
        self.total_count = 0
        
    def start_processing(self, scraper_results, proceeding: str):
        """Start background processing of new documents"""
        if self.is_processing:
            logger.warning("Background processing already in progress")
            return
            
        self.is_processing = True
        self.current_status = "starting"
        
        # Get new URLs from session state
        new_urls = st.session_state.get('new_urls_found', [])
        
        if not new_urls:
            logger.info("No new URLs to process")
            self.is_processing = False
            return
            
        self.total_count = len(new_urls)
        self.processed_count = 0
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._process_documents_background,
            args=(new_urls, proceeding),
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info(f"Started background processing of {self.total_count} new documents")
        
    def _process_documents_background(self, new_urls: List[Dict], proceeding: str):
        """Process documents in background thread"""
        try:
            self.current_status = "processing"
            
            for i, url_data in enumerate(new_urls):
                try:
                    # Process single document
                    self._process_single_document(url_data)
                    
                    self.processed_count += 1
                    self.current_status = f"processed {self.processed_count}/{self.total_count}"
                    
                    # Update UI notification
                    if 'background_processing_status' not in st.session_state:
                        st.session_state['background_processing_status'] = {}
                    
                    st.session_state['background_processing_status'] = {
                        'proceeding': proceeding,
                        'processed': self.processed_count,
                        'total': self.total_count,
                        'current_doc': url_data.get('title', url_data['url']),
                        'status': 'processing'
                    }
                    
                    logger.info(f"Background processing: {self.processed_count}/{self.total_count} completed")
                    
                except Exception as e:
                    logger.error(f"Failed to process document {url_data['url']}: {e}")
                    continue
                    
            # Mark processing as complete
            self.current_status = "completed"
            self.is_processing = False
            
            # Update session state for user notification
            st.session_state['background_processing_status'] = {
                'proceeding': proceeding,
                'processed': self.processed_count,
                'total': self.total_count,
                'status': 'completed'
            }
            
            # Trigger RAG system update notification
            if 'rag_system_updated' not in st.session_state:
                st.session_state['rag_system_updated'] = 0
            st.session_state['rag_system_updated'] += 1
            
            logger.info(f"✅ Background processing completed: {self.processed_count}/{self.total_count} documents processed")
            
        except Exception as e:
            logger.error(f"Background processing failed: {e}")
            self.current_status = "failed"
            self.is_processing = False
            
    def _process_single_document(self, url_data: Dict):
        """Process a single document incrementally"""
        try:
            # Extract and chunk the document
            doc_chunks = data_processing.extract_and_chunk_with_docling_url(
                url_data['url'], 
                url_data.get('title', '')
            )
            
            if doc_chunks:
                # Use incremental write for immediate persistence
                success = self.rag_system.add_document_incrementally(
                    chunks=doc_chunks,
                    url_hash=url_data['hash'],
                    url_data=url_data,
                    immediate_persist=True
                )
                
                if success:
                    logger.info(f"✅ Background processed: {url_data.get('title', url_data['url'])}")
                else:
                    logger.error(f"❌ Background processing failed: {url_data['url']}")
            else:
                logger.warning(f"No chunks extracted from {url_data['url']}")
                
        except Exception as e:
            logger.error(f"Single document processing failed: {e}")
            raise
            
    def get_status(self) -> Dict:
        """Get current processing status"""
        return {
            'is_processing': self.is_processing,
            'status': self.current_status,
            'processed': self.processed_count,
            'total': self.total_count,
            'progress_percent': (self.processed_count / self.total_count * 100) if self.total_count > 0 else 0
        }


def get_prioritized_proceedings():
    """
    Return proceedings in priority order.
    Default proceeding is always processed first.
    """
    primary_proceeding = config.DEFAULT_PROCEEDING
    
    # Get all configured proceedings
    all_proceedings = config.SCRAPER_PROCEEDINGS if hasattr(config, 'SCRAPER_PROCEEDINGS') else [primary_proceeding]
    
    # Ensure primary proceeding is first
    prioritized = [primary_proceeding]
    
    # Add other proceedings after primary
    for proc in all_proceedings:
        if proc != primary_proceeding and proc not in prioritized:
            prioritized.append(proc)
    
    logger.info(f"Proceeding priority order: {prioritized}")
    return prioritized


def check_for_new_pdfs_on_launch(rag_system, selected_proceeding=None):
    """Check for new PDFs on launch following spec requirements with proceeding priority"""
    try:
        logger.info("🔍 Checking for new PDFs on launch...")
        
        # Step 1: Get proceedings in priority order (selected proceeding first)
        if selected_proceeding:
            proceedings = [selected_proceeding]  # Only check the selected proceeding
        else:
            proceedings = get_prioritized_proceedings()
        primary_proceeding = proceedings[0]
        
        scraper = initialize_document_scraper()
        if not scraper:
            logger.warning("No scraper available for launch hash check")
            return
            
        # Step 2: Run quick scraper check for primary proceeding first
        total_new_pdfs = 0
        
        for proceeding in proceedings:
            try:
                logger.info(f"🔍 Checking {proceeding} (priority: {'PRIMARY' if proceeding == primary_proceeding else 'secondary'})")
                
                # Get fresh URLs from unified scraper for this proceeding
                fresh_results = get_new_pdfs_for_proceeding(proceeding, headless=True)
                
                if fresh_results:
                    logger.info(f"Launch check found {len(fresh_results)} potential URLs for {proceeding}")
                    
                    # Compare with existing hashes
                    new_pdfs_found = compare_scraper_results_with_hashes(rag_system, fresh_results, proceeding)
                    total_new_pdfs += new_pdfs_found
                    
                    if new_pdfs_found > 0:
                        # Show notification to user
                        st.session_state['new_pdfs_found_on_launch'] = total_new_pdfs
                        st.session_state['current_proceeding'] = proceeding
                        
                        # Start background processing
                        if 'background_processor' not in st.session_state:
                            st.session_state['background_processor'] = BackgroundProcessor(rag_system)
                        
                        st.session_state['background_processor'].start_processing(fresh_results, proceeding)
                        
                        logger.info(f"✅ Launch check completed. Found {new_pdfs_found} new PDFs for {proceeding}")
                        
                        # Update UI
                        priority_label = "PRIMARY" if proceeding == primary_proceeding else "secondary"
                        st.info(f"🔍 Found {new_pdfs_found} new documents for {proceeding} ({priority_label}). Processing in background...")
                    else:
                        logger.info(f"✅ Launch check completed. No new PDFs found for {proceeding}")
                        
                # For primary proceeding, always process even if we found some documents
                # For secondary proceedings, only continue if we haven't found enough in primary
                if proceeding == primary_proceeding:
                    continue  # Always check all proceedings for full discovery
                    
            except Exception as e:
                logger.error(f"Launch hash check failed for {proceeding}: {e}")
        
        if total_new_pdfs > 0:
            logger.info(f"✅ Launch check completed. Found {total_new_pdfs} total new PDFs across all proceedings")
        else:
            logger.info("✅ Launch check completed. No new PDFs found across all proceedings")
            
    except Exception as e:
        logger.error(f"Launch PDF check failed: {e}")

def run_startup_scraper_check(selected_proceeding: str = None):
    """Run a quick scraper check on startup if enabled with proceeding priority"""
    try:
        if not config.RUN_SCRAPER_ON_STARTUP:
            return
            
        scraper = initialize_document_scraper(selected_proceeding)
        if not scraper:
            return
        
        # Use selected proceeding or default
        if selected_proceeding is None:
            selected_proceeding = config.DEFAULT_PROCEEDING
            
        logger.info(f"Running startup document scraper check for {selected_proceeding}...")
        with st.spinner("Running quick document discovery check..."):
            # Use selected proceeding only
            prioritized_proceedings = [selected_proceeding]
            
            for proceeding in prioritized_proceedings:
                try:
                    priority_label = "PRIMARY" if proceeding == prioritized_proceedings[0] else "secondary"
                    logger.info(f"Startup check for {proceeding} ({priority_label})")
                    
                    # Get new PDFs using unified scraper
                    new_pdfs = get_new_pdfs_for_proceeding(proceeding, headless=True)
                    if new_pdfs:
                        logger.info(f"Startup scraper found {len(new_pdfs)} new documents for {proceeding}")
                        st.success(f"📄 Found {len(new_pdfs)} new documents for {proceeding} ({priority_label})")
                    else:
                        logger.info(f"No new documents found for {proceeding} during startup check")
                except Exception as e:
                    logger.warning(f"Startup scraper check failed for {proceeding}: {e}")
                    
    except Exception as e:
        logger.error(f"Startup scraper check failed: {e}")


def handle_background_data_refresh():
    """Handle background data refresh when proceeding switches"""
    if st.session_state.get('background_data_refresh_needed', False):
        # Mark as handled
        st.session_state['background_data_refresh_needed'] = False
        
        # Show status
        st.info("🔄 Refreshing data for new proceeding in background...")
        
        # This will trigger the normal initialization process
        # which will load the appropriate vector store for the new proceeding
        logger.info("Background data refresh triggered by proceeding switch")

def show_background_notifications():
    """Display real-time background processing notifications"""
    
    # Handle background data refresh
    handle_background_data_refresh()
    
    # Check for new PDFs found on launch
    if st.session_state.get('new_pdfs_found_on_launch', 0) > 0:
        proceeding = st.session_state.get('selected_proceeding', config.DEFAULT_PROCEEDING)
        count = st.session_state['new_pdfs_found_on_launch']
        
        st.info(f"🔍 **New Documents Found**: {count} new documents discovered for {proceeding}")
        
        # Show background processing status
        bg_status = st.session_state.get('background_processing_status', {})
        if bg_status.get('status') == 'processing':
            progress = bg_status.get('processed', 0)
            total = bg_status.get('total', count)
            current_doc = bg_status.get('current_doc', 'Unknown')
            
            progress_percent = (progress / total) * 100 if total > 0 else 0
            
            st.progress(progress_percent / 100)
            st.info(f"📄 **Processing in background**: {progress}/{total} documents processed")
            st.caption(f"Currently processing: {current_doc}")
            
        elif bg_status.get('status') == 'completed':
            processed = bg_status.get('processed', 0)
            total = bg_status.get('total', count)
            
            st.success(f"✅ **Background Processing Complete**: {processed}/{total} documents processed and added to vector store")
            
            # Add "refresh for latest" button
            if st.button("🔄 Refresh page for latest retrieval"):
                st.cache_resource.clear()
                st.rerun()
                
            # Clear notification after showing completion
            if st.button("✅ Dismiss notification"):
                st.session_state['new_pdfs_found_on_launch'] = 0
                st.session_state['background_processing_status'] = {}
                st.rerun()
    
    # Check for general background processing updates
    bg_status = st.session_state.get('background_processing_status', {})
    if bg_status and not st.session_state.get('new_pdfs_found_on_launch', 0):
        # Show standalone background processing status
        if bg_status.get('status') == 'processing':
            proceeding = bg_status.get('proceeding', 'Unknown')
            progress = bg_status.get('processed', 0)
            total = bg_status.get('total', 0)
            current_doc = bg_status.get('current_doc', 'Unknown')
            
            progress_percent = (progress / total) * 100 if total > 0 else 0
            
            st.info(f"🔄 **Background Processing Active** ({proceeding})")
            st.progress(progress_percent / 100)
            st.caption(f"Processing: {progress}/{total} documents | Current: {current_doc}")


def render_proceeding_selector():
    """Render the proceeding selection dropdown."""
    from config import get_active_proceedings, get_proceeding_display_name, DEFAULT_PROCEEDING
    
    # Get active proceedings for dropdown
    active_proceedings = get_active_proceedings()
    
    # Create dropdown options
    proceeding_options = {}
    for proc_id, proc_info in active_proceedings.items():
        proceeding_options[proc_info['display_name']] = proc_id
    
    # Get current proceeding from session state or use default
    current_proceeding = st.session_state.get('selected_proceeding', DEFAULT_PROCEEDING)
    current_display_name = get_proceeding_display_name(current_proceeding)
    
    # Create dropdown
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_display = st.selectbox(
            "📋 Select Proceeding",
            options=list(proceeding_options.keys()),
            index=list(proceeding_options.keys()).index(current_display_name) if current_display_name in proceeding_options else 0,
            help="Select the CPUC proceeding to analyze. Switching will load the appropriate vector database and documents."
        )
    
    with col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", help="Refresh the current proceeding's data"):
            st.cache_resource.clear()
            st.rerun()
    
    # Get selected proceeding ID
    selected_proceeding = proceeding_options[selected_display]
    
    # Check if proceeding changed
    if selected_proceeding != current_proceeding:
        logger.info(f"Proceeding changed from {current_proceeding} to {selected_proceeding}")
        st.session_state['selected_proceeding'] = selected_proceeding
        st.session_state['proceeding_switched'] = True
        st.session_state['rag_system_needs_refresh'] = True
        st.session_state['launch_hash_check_completed'] = False  # Reset check for new proceeding
        
        # Clear cache to force re-initialization with new proceeding
        st.cache_resource.clear()
        
        # Show switching message
        st.info(f"🔄 Switching to {config.get_proceeding_display_name(selected_proceeding)}...")
        
        # Trigger background data refresh
        st.session_state['background_data_refresh_needed'] = True
        st.rerun()
    
    return selected_proceeding

def main():
    st.title("⚖️ CPUC Regulatory Document Analysis System")
    
    # Check if we need to run the enhanced startup sequence
    if not st.session_state.get('startup_completed', False):
        st.info("🚀 **Initializing CPUC RAG System** - Running enhanced startup sequence...")
        
        with st.container():
            startup_results = execute_enhanced_startup_sequence()
            
            if startup_results['success']:
                # Show success with any warnings
                if startup_results.get('startup_warnings'):
                    st.warning("⚠️ **System initialized with warnings:**")
                    for warning in startup_results['startup_warnings']:
                        st.write(f"• {warning}")
                    if startup_results.get('fallbacks_used'):
                        st.info("💡 System is using fallback components but should function normally.")
                else:
                    st.success("✅ **System initialized successfully!**")
                    st.balloons()
                
                # Store additional info in session state
                st.session_state['startup_warnings'] = startup_results.get('startup_warnings', [])
                st.session_state['fallbacks_used'] = startup_results.get('fallbacks_used', False)
                
                time.sleep(1)  # Brief pause to show success
                st.rerun()  # Refresh to show main interface
                
            elif startup_results.get('status') == 'in_progress':
                st.info("⏳ Startup in progress...")
                time.sleep(2)
                st.rerun()
            else:
                # Show detailed error information
                st.error(f"❌ **Startup failed:** {startup_results.get('error', 'Unknown error')}")
                
                # Show startup errors if available
                if startup_results.get('startup_errors'):
                    st.error("**Startup Errors:**")
                    for error in startup_results['startup_errors']:
                        st.write(f"• {error}")
                
                # Show startup warnings if available  
                if startup_results.get('startup_warnings'):
                    st.warning("**Startup Warnings:**")
                    for warning in startup_results['startup_warnings']:
                        st.write(f"• {warning}")
                
                # Provide recovery options
                st.info("**Recovery Options:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Retry Startup"):
                        st.session_state.clear()
                        st.rerun()
                
                with col2:
                    if st.button("🛠️ Initialize Minimal System"):
                        # Try to initialize with minimal components
                        try:
                            st.session_state['rag_system'] = None
                            st.session_state['current_proceeding'] = config.DEFAULT_PROCEEDING
                            st.session_state['startup_completed'] = True
                            st.session_state['minimal_mode'] = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Minimal initialization also failed: {e}")
                
                st.stop()
        
        return  # Don't continue until startup is complete
    
    # Proceeding selector (showing current proceeding from startup)
    current_proceeding = st.session_state.get('current_proceeding', config.DEFAULT_PROCEEDING)
    st.success(f"📋 **Active Proceeding:** {config.get_proceeding_display_name(current_proceeding)}")
    
    # Get RAG system from session state
    rag_system = st.session_state.get('rag_system')
    minimal_mode = st.session_state.get('minimal_mode', False)
    
    if not rag_system and not minimal_mode:
        st.error("❌ RAG system not available. Please restart the application.")
        if st.button("🔄 Restart System"):
            st.session_state.clear()
            st.rerun()
        return
    elif minimal_mode:
        st.warning("⚠️ **Running in Minimal Mode** - Limited functionality available.")
        
        # Try to initialize RAG system manually
        if st.button("🔧 Try to Initialize RAG System"):
            try:
                from rag_core import CPUCRAGSystem
                rag_system = CPUCRAGSystem(current_proceeding=current_proceeding)
                st.session_state['rag_system'] = rag_system
                st.session_state['minimal_mode'] = False
                st.success("✅ RAG system initialized successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to initialize RAG system: {e}")
        
        if not rag_system:
            st.info("Some features may not be available in minimal mode.")
            return
    
    # Enhanced notification system with background processing updates
    show_background_notifications()
    
    # Initialize PDF scheduler for the current proceeding
    scheduler = None
    try:
        scheduler = initialize_pdf_scheduler(rag_system)
    except Exception as e:
        logger.error(f"Failed to initialize PDF scheduler: {e}")
    
    # Show system status
    render_enhanced_system_status(rag_system, scheduler, current_proceeding)
    
    # Create tabs for different features
    tab1, tab2, tab3 = st.tabs(["🔍 Document Analysis", "📅 Timeline", "⚙️ System Management"])
    
    with tab1:
        render_document_analysis_tab(rag_system)
    
    with tab2:
        render_timeline_tab(rag_system)
    
    with tab3:
        render_system_management_tab(rag_system, current_proceeding)


def render_enhanced_system_status(rag_system, scheduler, current_proceeding):
    """Render enhanced system status with embedding progress."""
    try:
        from incremental_embedder import create_incremental_embedder
        
        # Get embedding status
        embedder = create_incremental_embedder(current_proceeding)
        embedding_status = embedder.get_embedding_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📋 Proceeding", current_proceeding)
        
        with col2:
            embedded_count = embedding_status.get('total_embedded', 0)
            st.metric("📚 Documents Embedded", embedded_count)
        
        with col3:
            failed_count = embedding_status.get('total_failed', 0)
            st.metric("❌ Failed Documents", failed_count)
        
        with col4:
            status = embedding_status.get('status', 'unknown')
            status_color = "🟢" if status == 'ready' else "🟡" if status == 'empty' else "🔴"
            st.metric("🔋 System Status", f"{status_color} {status.title()}")
        
        # Show last update time
        last_updated = embedding_status.get('last_updated')
        if last_updated:
            st.caption(f"Last updated: {last_updated}")
        
    except Exception as e:
        logger.error(f"Failed to render enhanced system status: {e}")
        st.error("Could not load system status")


def render_system_management_tab(rag_system, current_proceeding):
    """Render system management tab with controls."""
    st.header("⚙️ System Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Update System")
        
        if st.button("🚀 Run Scraper", help="Run the scraper to find new documents"):
            with st.spinner("Running scraper..."):
                try:
                    from cpuc_scraper import CPUCUnifiedScraper
                    
                    progress_placeholder = st.empty()
                    
                    def progress_callback(message, progress):
                        if progress >= 0:
                            st.progress(progress / 100)
                            progress_placeholder.text(f"[{progress}%] {message}")
                    
                    scraper = CPUCUnifiedScraper(headless=True)
                    results = scraper.scrape_proceeding_pdfs(current_proceeding)
                    
                    if results:
                        st.success(f"✅ Scraping completed!")
                        st.json({
                            'csv_urls': len(results.get('csv_urls', [])),
                            'google_urls': len(results.get('google_urls', [])),
                            'total_scraped': results.get('total_scraped', 0)
                        })
                    else:
                        st.error(f"❌ Scraping failed: No results returned")
                        
                except Exception as e:
                    st.error(f"❌ Failed to run scraper: {e}")
        
        if st.button("🔨 Process Embeddings", help="Process incremental embeddings for new documents"):
            with st.spinner("Processing embeddings..."):
                try:
                    from incremental_embedder import process_incremental_embeddings
                    
                    def progress_callback(message, progress):
                        st.progress(progress / 100)
                        st.text(message)
                    
                    results = process_incremental_embeddings(current_proceeding, progress_callback)
                    
                    if results['status'] == 'completed':
                        st.success(f"✅ Processed {results['documents_processed']} documents!")
                        st.json(results)
                    else:
                        st.info(f"ℹ️ Status: {results['status']}")
                        
                except Exception as e:
                    st.error(f"❌ Failed to process embeddings: {e}")
    
    with col2:
        st.subheader("📊 System Information")
        
        # Show proceeding information
        proceeding_info = config.get_proceeding_info(current_proceeding)
        if proceeding_info:
            st.write(f"**Display Name:** {proceeding_info.get('display_name', 'Unknown')}")
            st.write(f"**Description:** {proceeding_info.get('description', 'Unknown')}")
            st.write(f"**Active:** {'Yes' if proceeding_info.get('active', False) else 'No'}")
        
        # Show file paths
        paths = config.get_proceeding_file_paths(current_proceeding)
        st.write("**File Paths:**")
        for path_name, path_obj in paths.items():
            exists = "✅" if path_obj.exists() else "❌"
            st.text(f"{exists} {path_name}: {path_obj}")
        
        # Reset button
        if st.button("🔄 Reset System", help="Clear all data and restart"):
            if st.checkbox("I confirm I want to reset the entire system"):
                st.session_state.clear()
                st.rerun()


def display_system_status(rag_system, scheduler, scraper=None):
    """Display system status in the header"""
    if rag_system:
        # Get system stats and log to console
        stats = rag_system.get_system_stats()
        print("\n" + "="*60)
        print("📊 CPUC RAG SYSTEM STATISTICS")
        print("="*60)
        print(f"📁 Documents on Disk:     {stats.get('total_documents_on_disk', 'N/A')}")
        print(f"✅ Documents Processed:   {stats.get('total_documents_hashed', 'N/A')}")
        print(f"🔢 Total Chunks in DB:    {stats.get('total_chunks', 'N/A')}")
        print(f"⏳ Files Pending:         {stats.get('files_not_hashed', 'N/A')}")
        print(f"🗄️  Vector Store Status:   {stats.get('vector_store_status', 'unknown')}")
        print(f"🤖 LLM Model:             {stats.get('llm_model', 'N/A')}")
        print(f"📂 Base Directory:        {stats.get('base_directory', 'N/A')}")
        print("="*60)
        
        # Show warning if the system needs attention
        if stats.get('total_chunks', 0) == 0 and stats.get('total_documents_on_disk', 0) > 0:
            print("⚠️  WARNING: Vector store appears empty but PDFs exist")
            print("   Run: rag_system.build_vector_store() to rebuild")
        elif stats.get('vector_store_status') == 'loaded':
            print("✅ System ready for queries")
        print()
        
        # Display status info at the top of the page
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Display scheduler status
            if scheduler:
                scheduler_status = scheduler.get_status()
                if scheduler_status.get('last_check'):
                    last_check = scheduler_status['last_check']
                    st.info(f"🔄 **Auto-Update**: {last_check[:16].replace('T', ' ')}")
                else:
                    st.info("🔄 **Auto-Update**: Pending...")
            else:
                st.warning("⚠️ **Auto-Update**: Disabled")
        
        with col2:
            # Display real-time stats
            chunk_count = stats.get('total_chunks', 0)
            doc_count = stats.get('total_documents_hashed', 0)
            st.info(f"📊 **Stats**: {doc_count} docs, {chunk_count} chunks")
        
        with col3:
            # Display system health with refresh button
            if stats.get('vector_store_status') == 'loaded':
                st.success("✅ Ready")
            else:
                st.warning("⚠️ Loading")
    else:
        print("❌ System failed to initialize. Check logs.")
        st.error("System failed to initialize. Check console logs for details.")



def render_document_analysis_tab(rag_system):
    """Render the document analysis tab"""
    st.header("🔍 Ask a Question")
    with st.form("query_form"):
        query_text = st.text_input(
            "Enter your query about CPUC regulations:",
            placeholder="e.g., What are the requirements for microgrid tariffs?"
        )
        submitted = st.form_submit_button("Analyze")

    if submitted and query_text:
        
        st.markdown("---")
        final_result = None

        with st.status("Processing your query with advanced retrieval...", expanded=True) as status:
            for result in rag_system.query(query_text):
                if isinstance(result, str):
                    status.write(f"⏳ {result}")
                elif isinstance(result, dict):
                    final_result = result
                    status.update(label="Analysis Complete!", state="complete", expanded=False)

        if final_result:
            answer = final_result.get("answer", "No answer could be generated.")
            sources = final_result.get("sources", [])
            confidence = final_result.get("confidence_indicators", {})
            st.markdown(answer, unsafe_allow_html=True)

            # Enhanced Confidence Analysis
            st.subheader("🎯 Confidence Analysis")
            
            # Calculate numerical confidence score (0-100)
            score_factors = [
                confidence.get('num_sources', 0) >= 3,
                confidence.get('num_sources', 0) >= 5,
                confidence.get('source_consistency', False),
                confidence.get('question_alignment', 0) > 0.3,
                confidence.get('question_alignment', 0) > 0.5,
                confidence.get('has_citations', '❌ No') == '✅ Yes'
            ]
            confidence_score = int((sum(score_factors) / len(score_factors)) * 100)

            # Query logged (auth disabled)
            
            # Display confidence metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence Score", f"{confidence_score}/100")
            with col2:
                st.metric("Sources Found", confidence.get('num_sources', 0))
            with col3:
                st.metric("Cited in Answer", confidence.get('has_citations', 'N/A'))
            
            # Sources section (collapsed by default)
            with st.expander("📚 Retrieved Sources", expanded=False):
                if sources:
                    for source in sources:
                        st.markdown(
                            f"**Document:** `{source['document']}` (Page: {source['page']}) | **Score:** {source['relevance_score']}")
                        st.markdown(f"<div class='source-box'>{source['excerpt']}</div>", unsafe_allow_html=True)
                else:
                    st.warning("No sources were retrieved from the local corpus for this query.")


def render_timeline_tab(rag_system):
    """Render the timeline tab"""
    try:
        # Add refresh button for timeline
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Refresh Timeline", help="Refresh timeline with latest documents"):
                st.cache_resource.clear()
                st.rerun()
        
        with col2:
            # Show last update notification
            if 'rag_system_updated' in st.session_state and st.session_state['rag_system_updated'] > 0:
                st.info(f"📅 Timeline includes latest updates (Update #{st.session_state['rag_system_updated']})")
        
        # Create timeline integration
        timeline_integration = create_timeline_integration(rag_system)
        
        # Show timeline data timestamp
        st.caption(f"Timeline data as of: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get current proceeding from session state
        current_proceeding = st.session_state.get('selected_proceeding', config.DEFAULT_PROCEEDING)
        
        # Render timeline interface
        timeline_integration.render_timeline_interface(config.format_proceeding_for_search(current_proceeding))
        
    except Exception as e:
        logger.error(f"Error rendering timeline tab: {e}")
        st.error(f"Error loading timeline: {e}")
        st.info("Please ensure the vector store is built and contains timeline data.")


if __name__ == "__main__":
    main()
