import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import hashlib
import json
from typing import List, Dict

import streamlit as st
import config

sys.path.append(str(Path(__file__).parent.resolve()))

from rag_core import CPUCRAGSystem
from timeline_integration import create_timeline_integration
from pdf_scheduler import create_pdf_scheduler
# Scraper imports removed - use standalone_scraper.py for document discovery
from startup_manager import create_startup_manager

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
                        
                        # Document discovery has been moved to standalone_scraper.py
                        logger.info("Skipping launch PDF check - use standalone_scraper.py for document discovery")
                        
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
                # Auto-scraping has been moved to standalone process
                st.warning("📋 No existing data found.")
                st.info("💡 **To get started:** Run `python standalone_scraper.py` to discover documents, then restart the application.")
                st.code("python standalone_scraper.py", language="bash")
                logger.info("No data found. Use standalone_scraper.py to discover documents.")
            
        return system
    except Exception as e:
        logger.error(f"Fatal error during RAG system initialization: {e}", exc_info=True)
        st.error(f"Could not initialize RAG System: {e}")
        return None

# auto_initialize_with_scraper function removed - use standalone_scraper.py for document discovery

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
# initialize_document_scraper function removed - use standalone_scraper.py for document discovery


# compare_scraper_results_with_hashes function removed - use standalone_scraper.py for document discovery


# BackgroundProcessor removed - document processing handled by standalone_data_processor.py


# Orphaned function get_prioritized_proceedings removed - no longer used after scraper moved to standalone


# check_for_new_pdfs_on_launch function removed - use standalone_scraper.py for document discovery

# run_startup_scraper_check function removed - use standalone_scraper.py for document discovery


# Background processing functions removed - handled by standalone_data_processor.py


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
    
    # Background processing notifications removed - handled by standalone_data_processor.py
    
    # Initialize PDF scheduler for the current proceeding
    scheduler = None
    try:
        scheduler = initialize_pdf_scheduler(rag_system)
    except Exception as e:
        logger.error(f"Failed to initialize PDF scheduler: {e}")
    
    # Show system status
    render_system_status(rag_system, scheduler, current_proceeding)
    
    # Create tabs for different features
    tab1, tab2, tab3 = st.tabs(["🔍 Document Analysis", "📅 Timeline", "⚙️ System Management"])
    
    with tab1:
        render_document_analysis_tab(rag_system)
    
    with tab2:
        render_timeline_tab(rag_system)
    
    with tab3:
        render_system_management_tab(rag_system, current_proceeding)


def render_system_status(rag_system, scheduler, current_proceeding):
    """Render basic system status focused on RAG functionality."""
    try:
        # Get basic system stats
        stats = rag_system.get_system_stats() if rag_system else {}
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📋 Proceeding", current_proceeding)
        
        with col2:
            chunk_count = stats.get('total_chunks', 0)
            st.metric("📊 Chunks Available", chunk_count)
        
        with col3:
            doc_count = stats.get('total_documents_hashed', 0)
            st.metric("📚 Documents", doc_count)
        
        with col4:
            vs_status = stats.get('vector_store_status', 'unknown')
            status_color = "🟢" if vs_status == 'loaded' else "🟡" if vs_status == 'empty' else "🔴"
            st.metric("🔋 Vector Store", f"{status_color} {vs_status.title()}")
        
        # Show scheduler status if available
        if scheduler:
            scheduler_status = scheduler.get_status()
            if scheduler_status.get('last_check'):
                st.caption(f"Auto-update last check: {scheduler_status['last_check'][:16].replace('T', ' ')}")
        
    except Exception as e:
        logger.error(f"Failed to render system status: {e}")
        st.error("Could not load system status")


def render_system_management_tab(rag_system, current_proceeding):
    """Render system management tab with controls."""
    st.header("⚙️ System Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Update System")
        
        st.info("📋 **Data Processing**")
        st.write("Document discovery and processing is handled by standalone scripts:")
        st.code(f"python standalone_scraper.py {current_proceeding}", language="bash")
        st.write("For document discovery")
        st.code(f"python standalone_data_processor.py {current_proceeding}", language="bash")
        st.write("For chunking and embedding. Then restart the application to load the processed data.")
    
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


# Orphaned function display_system_status removed - replaced by render_system_status



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
