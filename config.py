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

PDF_SERVER_PORT = 8001

# New OpenAI Model Settings
OPENAI_MODEL_NAME = "gpt-4.1-2025-04-14"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- RAG/CHUNKING SETTINGS ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# --- RETRIEVAL SETTINGS ---
TOP_K_DOCS = 15

# --- PROMPT TEMPLATES ---
ACCURACY_PROMPT_TEMPLATE = """You are a senior CPUC regulatory analyst. Your task is to provide a precise, synthesized, and analytical answer to the user's question based *only* on the provided context, which may include structured tables.
Current date: {current_date}

CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1.  **Synthesize, Don't Just List:** Do not just list facts. Synthesize information from multiple sources to form a complete, coherent answer.
2.  **Prioritize Numerical Data:** Pay extremely close attention to any numerical data, dollar amounts ($), percentages (%), or values in tables. Explicitly cite these numbers in your answer. If the question involves costs or rates, extract the exact figures.
3.  **Perform Inference:** Based on the text, make logical inferences. For example, if one document gives a start date and another gives a duration, infer the end date. Clearly state when you are making an inference (e.g., "Based on [Source A] and [Source B], it can be inferred that...").
4.  **Cite Sources Rigorously:** For every piece of information, you MUST cite the source, like this: `[Source: document_name.pdf, Page: 12]`.
5.  **Answer Fully:** If the context does not contain the answer, state "The provided context does not contain sufficient information to answer this question."

Response:"""

### ENHANCEMENT: New prompt for the layperson's summary (Part 2)
LAYMAN_PROMPT_TEMPLATE = """You are an expert communicator. Your task is to rephrase the following complex regulatory text into a simple, clear, 1-2 paragraph explanation that a non-expert can easily understand.
Focus on explaining the key outcomes, requirements, or deadlines in plain English. Convert complex numbers or rates into understandable impacts (e.g., "This means an average customer's bill will increase by about $5 per month.").

COMPLEX TEXT:
{technical_answer}

SIMPLIFIED EXPLANATION (in layman's terms):"""
