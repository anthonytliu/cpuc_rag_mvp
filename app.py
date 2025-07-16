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
        
        # Show warning if system needs attention
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

            # Add yellow highlighting for critical information
            highlighted_answer = answer.replace(
                'Technical Analysis from Regulatory Documents',
                '<span style="background-color: yellow; padding: 2px 4px; border-radius: 3px;">📋 Technical Analysis from Regulatory Documents</span>'
            ).replace(
                'Simplified Explanation',
                '<span style="background-color: yellow; padding: 2px 4px; border-radius: 3px;">💡 Simplified Explanation</span>'
            )
            
            st.markdown(highlighted_answer, unsafe_allow_html=True)
            
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
            
            # Determine confidence level and color
            if confidence_score >= 80:
                confidence_color = "🟢 High"
                score_color = "#4CAF50"
            elif confidence_score >= 60:
                confidence_color = "🟡 Medium"
                score_color = "#FF9800"
            else:
                confidence_color = "🔴 Low"
                score_color = "#F44336"
            
            # Display confidence metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Confidence Score", f"{confidence_score}/100")
            with col2:
                st.metric("Sources Found", confidence.get('num_sources', 0))
            with col3:
                st.metric("Cited in Answer", confidence.get('has_citations', 'N/A'))
            with col4:
                st.markdown(f"**Overall:** <span style='color: {score_color}; font-weight: bold;'>{confidence_color}</span>", unsafe_allow_html=True)
            
            # Written analysis of the confidence score
            st.subheader("📝 Confidence Analysis")
            
            analysis = f"""
            **Score: {confidence_score}/100** - {confidence_color.split(' ')[1]} Confidence
            
            **Analysis:**
            """
            
            if confidence_score >= 80:
                analysis += f"""
                ✅ **Excellent reliability** - This answer is backed by {confidence.get('num_sources', 0)} sources with strong alignment to your question.
                The response includes proper citations and demonstrates high consistency across regulatory documents.
                """
            elif confidence_score >= 60:
                analysis += f"""
                ⚠️ **Good reliability** - This answer draws from {confidence.get('num_sources', 0)} sources but may have some limitations.
                While generally trustworthy, consider reviewing the source documents for complete context.
                """
            else:
                analysis += f"""
                🔴 **Limited reliability** - This answer is based on {confidence.get('num_sources', 0)} sources with potentially weak alignment.
                Use this information cautiously and verify with additional sources or direct document review.
                """
            
            # Add specific factors
            factors = []
            if confidence.get('num_sources', 0) >= 5:
                factors.append("✅ Multiple sources (5+)")
            elif confidence.get('num_sources', 0) >= 3:
                factors.append("✅ Adequate sources (3+)")
            else:
                factors.append("❌ Few sources")
                
            if confidence.get('has_citations', '❌ No') == '✅ Yes':
                factors.append("✅ Proper citations included")
            else:
                factors.append("❌ No citations found")
                
            if confidence.get('question_alignment', 0) > 0.5:
                factors.append("✅ High question alignment")
            elif confidence.get('question_alignment', 0) > 0.3:
                factors.append("⚠️ Moderate question alignment")
            else:
                factors.append("❌ Low question alignment")
            
            analysis += f"""
            
            **Key Factors:**
            {chr(10).join(f"• {factor}" for factor in factors)}
            """
            
            st.markdown(analysis)
            
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
