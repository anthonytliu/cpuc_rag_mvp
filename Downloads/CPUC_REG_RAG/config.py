# 📁 config.py
# Centralized configuration for the RAG system

from pathlib import Path

# --- DIRECTORY SETTINGS ---
# The root directory of the project
PROJECT_ROOT = Path(__file__).parent

# Directory containing the PDF documents
# It can point to a directory of PDFs or a directory of subdirectories
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
CHUNK_OVERLAP = 100

# --- RETRIEVAL SETTINGS ---
# Number of documents to retrieve for context
TOP_K_DOCS = 15

# --- PROMPT TEMPLATE ---
# This can be moved here if it gets very complex, or left in rag_core.py
ACCURACY_PROMPT_TEMPLATE = """You are a CPUC regulatory analyst. Current date: {current_date}

CONTEXT: {context}

QUESTION: {question}

Provide a direct, complete answer that:
1. Answers the question fully in relation to today's date
2. Cites all sources and clearly states any inferences made
3. Analyzes and contextualizes all relevant information (dates, deadlines, regulatory significance)

Response:"""
