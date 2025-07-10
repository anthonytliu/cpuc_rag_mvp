import logging
import os
import sys

import streamlit as st

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your RAG system
from rag_core import CPUCRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="CPUC RAG System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
        border-left: 4px solid #1f77b4;
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
    """Initialize the RAG system with caching"""
    try:
        rag = CPUCRAGSystem()

        # Check if vector store exists
        if not rag.db_dir.exists() or not any(rag.db_dir.iterdir()):
            st.info("Building vector store for the first time. This may take a few minutes...")
            with st.spinner("Processing PDFs and building vector store..."):
                vectordb = rag.build_vector_store(force_rebuild=True)
                if vectordb is None:
                    st.error("Failed to build vector store. Please check your PDF directory.")
                    return None

        # Setup QA pipeline
        rag.setup_qa_pipeline()

        if rag.retriever is None:
            st.error("Failed to setup QA pipeline.")
            return None

        return rag

    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        logger.error(f"Initialization error: {e}")
        return None


def main():
    """Main Streamlit application"""

    st.title("⚖️ CPUC Regulatory Document RAG System")
    st.markdown("Query California Public Utilities Commission documents using AI-powered search")

    # Sidebar for system information and controls
    with st.sidebar:
        st.header("System Information")

        # Initialize RAG system
        rag_system = initialize_rag_system()

        if rag_system is None:
            st.error("System not initialized properly")
            st.stop()

        # Display system stats
        stats = rag_system.get_system_stats()

        st.subheader("📊 System Stats")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Documents", stats.get("total_documents", "Unknown"))
            st.metric("Chunk Size", stats.get("chunk_size", "Unknown"))

        with col2:
            st.metric("Total Chunks", stats.get("total_chunks", "Unknown"))
            st.metric("Chunk Overlap", stats.get("chunk_overlap", "Unknown"))

        st.info(f"**Model**: {stats.get('llm_model', 'Unknown')}")
        st.info(f"**Embeddings**: {stats.get('embedding_model', 'Unknown')}")
        st.info(f"**Device**: {stats.get('device', 'Unknown')}")

        # Display base directory info
        st.subheader("📁 Data Source")
        st.text(f"Directory: {stats.get('base_directory', 'Unknown')}")
        st.text(f"Exists: {stats.get('base_directory_exists', 'Unknown')}")

        # Rebuild option
        st.subheader("🔄 Maintenance")
        if st.button("Rebuild Vector Store", help="Rebuild the entire vector store from scratch"):
            with st.spinner("Rebuilding vector store..."):
                vectordb = rag_system.build_vector_store(force_rebuild=True)
                if vectordb:
                    st.success("Vector store rebuilt successfully!")
                    st.experimental_rerun()
                else:
                    st.error("Failed to rebuild vector store")

    # Main query interface
    st.header("🔍 Query Interface")

    # Query input
    col1, col2 = st.columns([3, 1])

    with col1:
        query_text = st.text_input(
            "Enter your query:",
            placeholder="e.g., What are the rate-setting procedures for Southern California Edison?",
            help="Ask questions about CPUC regulations, procedures, or specific proceedings"
        )

    with col2:
        proceeding_filter = st.text_input(
            "Proceeding Filter (optional):",
            placeholder="e.g., R2207005",
            help="Filter results to specific proceeding"
        )

    # Query execution
    if st.button("🚀 Submit Query", type="primary") or query_text:
        if not query_text.strip():
            st.warning("Please enter a query")
            return

        with st.spinner("Searching documents..."):
            try:
                # Execute query
                result = rag_system.query(
                    query_text,
                    proceeding_filter=proceeding_filter if proceeding_filter.strip() else None
                )

                # Display results
                st.header("📋 Results")

                # Answer section
                st.subheader("💬 Answer")
                st.markdown(result["answer"])

                # Confidence indicators
                st.subheader("🎯 Confidence Indicators")
                confidence = result["confidence_indicators"]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Sources Found", confidence.get("num_sources", 0))

                with col2:
                    has_quotes = confidence.get("has_exact_quotes", False)
                    st.metric("Has Exact Quotes", "✅" if has_quotes else "❌")

                with col3:
                    has_pages = confidence.get("has_page_references", False)
                    st.metric("Has Page References", "✅" if has_pages else "❌")

                # Additional indicators
                st.markdown(f"""
                <div class="confidence-indicator">
                    <strong>Model Type:</strong> {confidence.get('model_type', 'Unknown')}<br>
                    <strong>Source Consistency:</strong> {'✅ Consistent' if confidence.get('source_consistency', False) else '⚠️ Mixed Sources'}
                </div>
                """, unsafe_allow_html=True)

                # Sources section
                if result["sources"]:
                    st.subheader("📚 Sources")

                    for i, source in enumerate(result["sources"], 1):
                        with st.expander(f"Source {i}: {source['document']} (Page {source['page']})"):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Document:</strong> {source['document']}<br>
                                <strong>Proceeding:</strong> {source['proceeding']}<br>
                                <strong>Page:</strong> {source['page']}<br>
                                <strong>Relevance Score:</strong> {source['relevance_score']}<br><br>
                                <strong>Excerpt:</strong><br>
                                {source['excerpt']}
                            </div>
                            """, unsafe_allow_html=True)

                # Download results option
                if st.button("📥 Download Results as JSON"):
                    import json
                    results_json = json.dumps(result, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=results_json,
                        file_name=f"cpuc_query_results_{query_text[:30]}.json",
                        mime="application/json"
                    )

            except Exception as e:
                st.error(f"Query failed: {e}")
                logger.error(f"Query error: {e}")

    # Example queries
    st.header("💡 Example Queries")

    examples = [
        "What are the rate-setting procedures for utility companies?",
        "What are the filing requirements for rate change applications?",
        "What are the public participation procedures?",
        "What are the deadlines for submitting testimony?",
        "What are the requirements for environmental impact assessments?",
        "What are the procedures for appeals?"
    ]

    for example in examples:
        if st.button(f"📝 {example}", key=f"example_{example[:20]}"):
            st.text_input("Enter your query:", value=example, key="example_query")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>CPUC RAG System - AI-powered regulatory document search</p>
        <p>Optimized for M4 MacBook with local models</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
