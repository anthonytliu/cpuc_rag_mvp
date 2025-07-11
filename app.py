# 📁 app.py
# The Streamlit web application for the CPUC RAG system.

import streamlit as st
import logging
import sys
from pathlib import Path

# Add project root to path to allow imports
sys.path.append(str(Path(__file__).parent.resolve()))

from rag_core import CPUCRAGSystem
import config

# Configure logging
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
def initialize_rag_system():
    """Initialize the RAG system with caching. This will build the DB on first run."""
    try:
        with st.spinner("Initializing RAG System... This may take a moment."):
            system = CPUCRAGSystem()
            system.build_vector_store()
        st.success("RAG System Initialized.")
        return system
    except Exception as e:
        logger.error(f"Fatal error during RAG system initialization: {e}", exc_info=True)
        st.error(f"Could not initialize RAG System: {e}")
        return None


def main():
    # ... (no changes to the top part of main or the sidebar)
    st.title("⚖️ CPUC Regulatory Document Analysis System")
    with st.sidebar:
        st.header("System Controls")
        rag_system = initialize_rag_system()
        if rag_system:
            stats = rag_system.get_system_stats()
            st.subheader("📊 System Stats")
            st.metric("Total Documents Indexed", stats.get('total_documents', 'N/A'))
            st.metric("Total Chunks in DB", stats.get('total_chunks', 'N/A'))
            st.info(f"**Model**: `{stats.get('llm_model', 'N/A')}`")
            if st.button("Rebuild Vector Store"):
                st.cache_resource.clear()
                st.rerun()
        else:
            st.error("System failed to initialize. Check logs.")
    if not rag_system:
        st.error("System is not available. Please check the sidebar and console logs for errors.")
        return

    st.header("🔍 Ask a Question")
    with st.form("query_form"):
        query_text = st.text_input("Enter your query about CPUC regulations:",
                                   placeholder="e.g., What are the requirements for microgrid tariffs?")
        submitted = st.form_submit_button("Search")

    if submitted and query_text:
        with st.spinner("Searching documents, generating answer, and searching the web..."):
            result = rag_system.query(query_text)
            answer = result.get("answer", "No answer could be generated.")
            sources = result.get("sources", [])
            confidence = result.get("confidence_indicators", {})

            st.markdown("---")

            # ### FIX: Using `unsafe_allow_html=True` to render the new styled answer correctly.
            st.markdown(answer, unsafe_allow_html=True)

            with st.expander("Confidence & Sources Analysis", expanded=False):
                # ... (no changes to the confidence/sources section)
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


if __name__ == "__main__":
    main()