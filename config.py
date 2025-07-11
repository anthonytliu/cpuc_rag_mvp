# 📁 config.py
# Centralized configuration for the RAG system

# Add these imports at the top
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.resolve()

# Directory containing the PDF documents.
BASE_PDF_DIR = PROJECT_ROOT / "cpuc_pdfs" / "R2207005"

# Directory to store the Chroma vector database
DB_DIR = PROJECT_ROOT / "local_chroma_db"

# --- MODEL SETTINGS ---
# Local embedding model from HuggingFace
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"

# ### FIX: Replace local model settings with OpenAI settings ###
# Local LLM model to use with Ollama - (Commented out or removed)
# LLM_MODEL = "llama3.2"
# OLLAMA_BASE_URL = "http://localhost:11434"

# New OpenAI Model Settings
OPENAI_MODEL_NAME = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- RAG/CHUNKING SETTINGS ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- RETRIEVAL SETTINGS ---
TOP_K_DOCS = 15

# --- PROMPT TEMPLATES ---
# This is the prompt for the core technical answer (Part 1)
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

### ENHANCEMENT: New prompt for the layperson's summary (Part 2)
LAYMAN_PROMPT_TEMPLATE = """You are an expert communicator. Your task is to rephrase the following complex regulatory text into simple, clear language that a non-expert can easily understand. Do not add new information or opinions.

Focus on explaining the key outcomes, requirements, or deadlines in plain English.

COMPLEX TEXT:
{technical_answer}

SIMPLIFIED EXPLANATION (in layman's terms):"""
