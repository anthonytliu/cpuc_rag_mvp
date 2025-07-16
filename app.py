import logging
import sys
import threading
from pathlib import Path

import streamlit as st
import http.server
import socketserver
import config

sys.path.append(str(Path(__file__).parent.resolve()))

from rag_core import CPUCRAGSystem

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

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(config.BASE_PDF_DIR), **kwargs)

@st.cache_resource
def start_pdf_server():
    """Starts the PDF server in a background thread."""
    if '_pdf_server_thread' in st.session_state and st.session_state._pdf_server_thread.is_alive():
        logger.info("PDF server is already running.")
        return

    def run_server():
        with socketserver.TCPServer(("", config.PDF_SERVER_PORT), Handler) as httpd:
            logger.info(f"Starting PDF server at http://localhost:{config.PDF_SERVER_PORT}")
            httpd.serve_forever()

    thread = threading.Thread(target=run_server, daemon=True)
    st.session_state._pdf_server_thread = thread
    thread.start()
    logger.info("PDF server thread started.")

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with caching. Does NOT build vector store automatically."""
    try:
        with st.spinner("Initializing RAG System... This may take a moment."):
            system = CPUCRAGSystem()
            # Do NOT automatically build vector store - let it load existing one only
            
        st.success("RAG System Initialized.")
        return system
    except Exception as e:
        logger.error(f"Fatal error during RAG system initialization: {e}", exc_info=True)
        st.error(f"Could not initialize RAG System: {e}")
        return None


def main():
    # Import authentication system
    from streamlit_auth import streamlit_auth
    
    # Check authentication first
    if not streamlit_auth.is_authenticated():
        streamlit_auth.render_login_page()
        return
    
    # User is authenticated - render main app
    st.title("⚖️ CPUC Regulatory Document Analysis System")
    
    # Render user dashboard
    streamlit_auth.render_user_dashboard()
    
    # Initialize system and log stats to console (no sidebar)
    rag_system = initialize_rag_system()
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
    else:
        print("❌ System failed to initialize. Check logs.")
        st.error("System failed to initialize. Check console logs for details.")

    if not rag_system:
        st.error("System is not available. Please check console logs for errors.")
        return

    st.header("🔍 Ask a Question")
    with st.form("query_form"):
        query_text = st.text_input(
            "Enter your query about CPUC regulations:",
            placeholder="e.g., What are the requirements for microgrid tariffs?"
        )
        submitted = st.form_submit_button("Analyze")

    if submitted and query_text:
        # Check query permission
        can_query, message = streamlit_auth.check_query_permission()
        
        if not can_query:
            st.error(f"⚠️ {message}")
            return
        
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

            # Log the query
            streamlit_auth.log_query(
                query_text, 
                confidence_score, 
                confidence.get('num_sources', 0)
            )
            
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


if __name__ == "__main__":
    main()
