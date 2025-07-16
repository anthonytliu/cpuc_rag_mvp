import logging
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
# import http.server
# import socketserver
import config

sys.path.append(str(Path(__file__).parent.resolve()))

from rag_core import CPUCRAGSystem
from timeline_integration import create_timeline_integration
from pdf_scheduler import create_pdf_scheduler

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
def initialize_rag_system():
    """Initialize the RAG system with caching. Auto-builds vector store if needed."""
    try:
        with st.spinner("Initializing RAG System... If vector store needs rebuilding, this may take several minutes."):
            system = CPUCRAGSystem()
            # Auto-build vector store if needed - this may take several minutes
            
        st.success("RAG System Initialized.")
        return system
    except Exception as e:
        logger.error(f"Fatal error during RAG system initialization: {e}", exc_info=True)
        st.error(f"Could not initialize RAG System: {e}")
        return None

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


def main():
    st.title("⚖️ CPUC Regulatory Document Analysis System")
    
    # Check for RAG system updates and show notification
    if 'rag_system_updated' in st.session_state and st.session_state['rag_system_updated'] > 0:
        st.success(f"🔄 **System Updated!** New documents have been processed. Update #{st.session_state['rag_system_updated']}")
        if st.button("🔄 Refresh to see latest stats"):
            st.cache_resource.clear()
            st.rerun()
    
    # Initialize system and log stats to console (no sidebar)
    rag_system = initialize_rag_system()
    
    # Initialize PDF scheduler
    scheduler = None
    if rag_system:
        scheduler = initialize_pdf_scheduler(rag_system)
    
    # Display system status and scheduler info
    display_system_status(rag_system, scheduler)
    
    if not rag_system:
        st.error("System is not available. Please check console logs for errors.")
        return

    # Create tabs for different features
    tab1, tab2, tab3 = st.tabs(["🔍 Document Analysis", "📅 Timeline", "⚙️ System Status"])
    
    with tab1:
        render_document_analysis_tab(rag_system)
    
    with tab2:
        render_timeline_tab(rag_system)
    
    with tab3:
        render_system_status_tab(rag_system, scheduler)
    
    # Add a periodic refresh mechanism (only visible if auto-refresh is enabled)
    if st.sidebar.checkbox("🔄 Auto-refresh every 60 seconds", value=False, help="Automatically refresh to show latest updates"):
        time.sleep(60)
        st.rerun()


def display_system_status(rag_system, scheduler):
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


def render_system_status_tab(rag_system, scheduler):
    """Render the system status tab"""
    st.header("⚙️ System Status & PDF Updates")
    
    # Real-time refresh button
    col_refresh, col_auto = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 Refresh Stats", help="Get latest system statistics"):
            # Clear the cache to get fresh stats
            st.cache_resource.clear()
            st.rerun()
    
    with col_auto:
        # Auto-refresh checkbox
        auto_refresh = st.checkbox("🔄 Auto-refresh every 30 seconds", value=False)
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    # System Statistics
    if rag_system:
        st.subheader("📊 Real-Time System Statistics")
        
        # Get fresh stats (not cached)
        with st.spinner("Getting latest statistics..."):
            stats = rag_system.get_system_stats()
        
        # Show last updated time
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents on Disk", stats.get('total_documents_on_disk', 'N/A'))
            st.metric("Total Chunks", stats.get('total_chunks', 'N/A'))
        
        with col2:
            st.metric("Documents Processed", stats.get('total_documents_hashed', 'N/A'))
            st.metric("Vector Store Status", stats.get('vector_store_status', 'unknown'))
        
        with col3:
            st.metric("Files Pending", stats.get('files_not_hashed', 'N/A'))
            st.metric("LLM Model", stats.get('llm_model', 'N/A'))
    
    # PDF Scheduler Status
    st.subheader("🔄 Automated PDF Updates")
    
    if scheduler:
        scheduler_status = scheduler.get_status()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Scheduler Status**")
            if scheduler_status.get('is_running'):
                st.success("✅ Running")
            else:
                st.error("❌ Not Running")
            
            if scheduler_status.get('last_check'):
                st.write(f"**Last Check**: {scheduler_status['last_check'][:16].replace('T', ' ')}")
            else:
                st.write("**Last Check**: Never")
            
            if scheduler_status.get('next_check'):
                st.write(f"**Next Check**: {scheduler_status['next_check'][:16].replace('T', ' ')}")
            
            st.write(f"**Check Interval**: {scheduler_status.get('check_interval_hours', 'N/A')} hours")
        
        with col2:
            st.write("**Download Statistics**")
            st.write(f"**Last Download Count**: {scheduler_status.get('last_download_count', 0)}")
            st.write(f"**Total Downloads**: {scheduler_status.get('total_downloads', 0)}")
            
            if scheduler_status.get('time_since_last_check'):
                st.write(f"**Time Since Last Check**: {scheduler_status['time_since_last_check']}")
            
            # RAG Update Status
            if scheduler_status.get('rag_update_status'):
                st.write(f"**RAG Update Status**: {scheduler_status['rag_update_status']}")
        
        # Manual Control
        st.subheader("🔧 Manual Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Force PDF Check Now"):
                with st.spinner("Forcing PDF check..."):
                    scheduler.force_check()
                    st.success("PDF check initiated! Check logs for progress.")
        
        with col2:
            if st.button("🔄 Refresh Status"):
                st.cache_resource.clear()
                st.rerun()
        
        # Add a test button to simulate RAG system updates (for testing)
        if st.button("🧪 Test Update Notification", help="Simulate a RAG system update for testing"):
            if 'rag_system_updated' not in st.session_state:
                st.session_state['rag_system_updated'] = 0
            st.session_state['rag_system_updated'] += 1
            st.success("✅ Test update triggered! Refresh to see notification.")
            st.rerun()
        
        # Error Log
        if scheduler_status.get('errors'):
            st.subheader("⚠️ Recent Errors")
            for error in scheduler_status['errors'][-5:]:  # Show last 5 errors
                st.error(f"**{error['timestamp'][:16].replace('T', ' ')}**: {error['error']}")
    else:
        st.error("❌ PDF Scheduler is not running")
        st.info("The automated PDF update system is not available. Please check the system logs.")


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
        
        # Render timeline interface
        timeline_integration.render_timeline_interface("R.22-07-005")
        
    except Exception as e:
        logger.error(f"Error rendering timeline tab: {e}")
        st.error(f"Error loading timeline: {e}")
        st.info("Please ensure the vector store is built and contains timeline data.")


if __name__ == "__main__":
    main()
