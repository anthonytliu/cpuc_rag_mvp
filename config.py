# 📁 config.py
# Centralized configuration for the RAG system

from pathlib import Path

# --- DIRECTORY SETTINGS ---
# The root directory of the project
PROJECT_ROOT = Path(__file__).parent.resolve()

# Directory containing the PDF documents.
# The system will search this directory and all its subdirectories for .pdf files.
BASE_PDF_DIR = PROJECT_ROOT / "cpuc_pdfs" / "R2207005"

# Directory to store the Chroma vector database
DB_DIR = PROJECT_ROOT / "local_chroma_db"

# --- MODEL SETTINGS ---
# Local embedding model from HuggingFace
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"

# Local LLM model to use with Ollama
LLM_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"

# --- RAG/CHUNKING SETTINGS ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200 # Increased overlap for better context continuity

# --- RETRIEVAL SETTINGS ---
# Number of documents to retrieve for context
TOP_K_DOCS = 15

# --- PROMPT TEMPLATE ---
ACCURACY_PROMPT_TEMPLATE = """You are a CPUC regulatory analyst. Your task is to provide a precise and analytical answer to the user's question based *only* on the provided context.
Current date: {current_date}

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1.  **Answer Directly:** Provide a direct and comprehensive answer to the question.
2.  **Cite Sources:** For every piece of information you use, you MUST cite the source document and page number, like this: `[Source: document_name.pdf, Page: 12]`.
3.  **Analyze Dates:** Identify any dates or deadlines and explain their significance relative to the current date ({current_date}). Note if they are past, upcoming, or urgent.
4.  **Identify Obligations:** Clearly state any regulatory requirements, obligations, or mandates mentioned in the context.
5.  **Be Factual:** Do not infer or add information not present in the context. If the context does not contain the answer, state "The provided context does not contain sufficient information to answer this question."

Response:"""