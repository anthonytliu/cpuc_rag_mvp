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
    st.title("⚖️ CPUC Regulatory Document Analysis System")

    with st.sidebar:
        st.header("System Controls")
        rag_system = initialize_rag_system()
        if rag_system:
            # Refresh stats button to update without cache
            col1, col2 = st.columns([3, 1])
            with col1:
                st.subheader("📊 System Stats")
            with col2:
                if st.button("🔄", help="Refresh Stats"):
                    st.rerun()
            
            # Get fresh stats (not cached)
            stats = rag_system.get_system_stats()
            
            # Show document counts
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents on Disk", stats.get('total_documents_on_disk', 'N/A'))
                st.metric("Documents Processed", stats.get('total_documents_hashed', 'N/A'))
            with col2:
                st.metric("Total Chunks in DB", stats.get('total_chunks', 'N/A'))
                st.metric("Files Pending", stats.get('files_not_hashed', 'N/A'))
            
            # Show status indicators
            vs_status = stats.get('vector_store_status', 'unknown')
            if vs_status == "loaded":
                st.success(f"✅ Vector Store: {vs_status}")
            elif vs_status == "not_loaded":
                st.warning(f"⚠️ Vector Store: {vs_status}")
            else:
                st.error(f"❌ Vector Store: {vs_status}")
                
            st.info(f"**Model**: `{stats.get('llm_model', 'N/A')}`")
            
            # Manual rebuild button (only if needed)
            if stats.get('total_chunks', 0) == 0 and stats.get('total_documents_on_disk', 0) > 0:
                st.warning("Vector store appears empty but PDFs exist")
                if st.button("🔨 Build Vector Store", type="primary"):
                    with st.spinner("Building vector store... This may take a while."):
                        rag_system.build_vector_store()
                    st.success("Vector store build complete!")
                    st.rerun()
            
            if st.button("🔄 Force Rebuild Vector Store", help="Only use if vector store is corrupted"):
                if st.button("⚠️ Confirm Force Rebuild", type="secondary"):
                    with st.spinner("Force rebuilding vector store... This may take a while."):
                        rag_system.build_vector_store(force_rebuild=True)
                    st.success("Vector store rebuild complete!")
                    st.rerun()
        else:
            st.error("System failed to initialize. Check logs.")

    if not rag_system:
        st.error("System is not available. Please check the sidebar and console logs for errors.")
        return

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
            with st.expander("Confidence & Sources Analysis", expanded=False):
                st.subheader("🎯 Confidence Analysis")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Confidence", confidence.get('overall_confidence', 'N/A'))
                with col2:
                    st.metric("Sources Found", confidence.get('num_sources', 0))
                with col3:
                    st.metric("Cited in Answer", confidence.get('has_citations', 'N/A'))

                st.subheader("📚 Retrieved Corpus Sources")
                if sources:
                    for source in sources:
                        st.markdown(
                            f"**Document:** `{source['document']}` (Page: {source['page']}) | **Score:** {source['relevance_score']}")
                        st.markdown(f"<div class='source-box'>{source['excerpt']}</div>", unsafe_allow_html=True)
                else:
                    st.warning("No sources were retrieved from the local corpus for this query.")

            with st.expander("🕵️‍♀️ Debug Information"):
                st.subheader("Raw LLM Output (Part 1)")
                st.text(final_result.get("raw_part1_answer", "Not available."))
                st.subheader("Final Rendered HTML")
                st.code(final_result.get("answer", "Not available."), language="html")


if __name__ == "__main__":
    main()
