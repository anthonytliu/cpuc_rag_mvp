# 📁 config.py
# Centralized configuration for the RAG system

# Add these imports at the top
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.resolve()

# Directory containing the PDF documents (DEPRECATED - moved to URL-based processing)
# BASE_PDF_DIR = PROJECT_ROOT / "cpuc_pdfs" / "R2207005"

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

# --- URL PROCESSING SETTINGS ---
# Enable URL-based processing (True) or file-based processing (False)
USE_URL_PROCESSING = True

# URL processing timeouts and retry settings
URL_VALIDATION_TIMEOUT = 30  # seconds
URL_PROCESSING_TIMEOUT = 300  # seconds for Docling processing

# --- MULTI-PROCEEDING SYSTEM SETTINGS ---
# Available proceedings that can be selected in the UI
AVAILABLE_PROCEEDINGS = {
    "R2207005": {
        "display_name": "R.22-07-005 - Demand Flexibility Rulemaking",
        "full_name": "R.22-07-005",
        "short_name": "R2207005",
        "description": "Demand Flexibility Rulemaking",
        "active": True
    },
    "R2108025": {
        "display_name": "R.21-08-025 - Integrated Resource Planning",
        "full_name": "R.21-08-025", 
        "short_name": "R2108025",
        "description": "Integrated Resource Planning",
        "active": False
    },
    "R2011003": {
        "display_name": "R.20-11-003 - Microgrids",
        "full_name": "R.20-11-003",
        "short_name": "R2011003", 
        "description": "Microgrids Development",
        "active": False
    }
}

# Default proceeding to load on startup (first one in AVAILABLE_PROCEEDINGS)
def get_first_proceeding() -> str:
    """Get the first proceeding from AVAILABLE_PROCEEDINGS."""
    return next(iter(AVAILABLE_PROCEEDINGS.keys()))

DEFAULT_PROCEEDING = os.getenv("DEFAULT_PROCEEDING", get_first_proceeding())

# --- DOCUMENT SCRAPER SETTINGS ---
# List of proceedings to scrape (can be overridden in environment)
SCRAPER_PROCEEDINGS = os.getenv("SCRAPER_PROCEEDINGS", "R2207005").split(",")

# Maximum number of worker threads for scraping
SCRAPER_MAX_WORKERS = int(os.getenv("SCRAPER_MAX_WORKERS", "8"))

# Run scraper check on application startup
RUN_SCRAPER_ON_STARTUP = os.getenv("RUN_SCRAPER_ON_STARTUP", "false").lower() == "true"

# Maximum Google search results to process per query
SCRAPER_MAX_GOOGLE_RESULTS = int(os.getenv("SCRAPER_MAX_GOOGLE_RESULTS", "50"))
URL_RETRY_COUNT = 3
URL_RETRY_DELAY = 5  # seconds between retries

# CPUC specific URL settings
CPUC_BASE_URL = "https://docs.cpuc.ca.gov"
CPUC_SEARCH_BASE = "https://docs.cpuc.ca.gov/SearchRes.aspx"

# --- GOOGLE SEARCH API SETTINGS ---
# Google Custom Search API credentials (more reliable than screen scraping)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Get from Google Cloud Console
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")    # Custom Search Engine ID
GOOGLE_SEARCH_FALLBACK_ENABLED = os.getenv("GOOGLE_SEARCH_FALLBACK_ENABLED", "true").lower() == "true"

# Rate limiting for Google searches
GOOGLE_SEARCH_DELAY_SECONDS = float(os.getenv("GOOGLE_SEARCH_DELAY_SECONDS", "2.0"))
GOOGLE_SEARCH_MAX_RETRIES = int(os.getenv("GOOGLE_SEARCH_MAX_RETRIES", "3"))
GOOGLE_SEARCH_RETRY_DELAY = int(os.getenv("GOOGLE_SEARCH_RETRY_DELAY", "30"))

# --- PDF SCHEDULER SETTINGS ---
# Automated PDF checking and downloading
PDF_CHECK_INTERVAL_HOURS = int(os.getenv("PDF_CHECK_INTERVAL_HOURS", "12"))  # Default: 12 hours
PDF_SCHEDULER_ENABLED = os.getenv("PDF_SCHEDULER_ENABLED", "true").lower() == "true"
PDF_SCHEDULER_HEADLESS = os.getenv("PDF_SCHEDULER_HEADLESS", "true").lower() == "true"
PDF_SCHEDULER_MAX_RETRIES = int(os.getenv("PDF_SCHEDULER_MAX_RETRIES", "3"))

# Proceeding numbers to monitor
MONITORED_PROCEEDINGS = os.getenv("MONITORED_PROCEEDINGS", "R2207005").split(",")

# Auto-update settings
AUTO_UPDATE_RAG_SYSTEM = os.getenv("AUTO_UPDATE_RAG_SYSTEM", "true").lower() == "true"
AUTO_UPDATE_DELAY_MINUTES = int(os.getenv("AUTO_UPDATE_DELAY_MINUTES", "5"))  # Wait 5 minutes after download before updating RAG

# --- PERFORMANCE OPTIMIZATION SETTINGS ---
# Parallel processing settings
URL_PARALLEL_WORKERS = 3  # Number of parallel workers for URL processing
VECTOR_STORE_BATCH_SIZE = 256  # Batch size for vector store operations (increased from 64)

# Docling performance settings
DOCLING_FAST_MODE = True  # Enable fast table processing mode
DOCLING_MAX_PAGES = None  # Limit max pages per document (None = no limit)
DOCLING_THREADS = 4  # Number of threads for Docling processing

# Embedding optimization settings
EMBEDDING_BATCH_SIZE = 32  # Batch size for embedding generation

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
1.  **Structure Your Response:** Organize your answer with clear sections using HTML formatting:
   - **Executive Summary:** Start with a brief 1-2 sentence overview
   - **Key Findings:** Present the main regulatory information with bullet points
   - **Detailed Analysis:** Provide comprehensive analysis with numbered points
   - **Regulatory Impact:** Explain implications and requirements
   - **Important Dates/Deadlines:** Highlight time-sensitive information if applicable

2.  **Use HTML Formatting:** Format your response with proper HTML tags:
   - Use `<h4>` tags for section headings
   - Use `<ul>` and `<li>` for bullet points
   - Use `<ol>` and `<li>` for numbered lists
   - Use `<strong>` for emphasis on key terms
   - Use `<em>` for regulatory requirements
   - Use `<br>` for line breaks where needed

3.  **Prioritize Numerical Data:** Pay extremely close attention to any numerical data, dollar amounts ($), percentages (%), or values in tables. Present these in formatted lists or tables when applicable.

4.  **Perform Inference:** Based on the text, make logical inferences. Clearly state when you are making an inference using <em>Based on [Source A] and [Source B], it can be inferred that...</em>

5.  **Cite Sources Rigorously:** For every piece of information, you MUST cite the source using this exact format: `[CITE:filename.pdf,page_12]` where filename.pdf is the document name and page_12 is the page number.

6.  **Answer Fully:** If the context does not contain the answer, state in a formatted box: "<div style='background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; padding: 10px; margin: 10px 0;'><strong>Note:</strong> The provided context does not contain sufficient information to answer this question.</div>"

Response:"""

### ENHANCEMENT: New prompt for the layperson's summary (Part 2)
LAYMAN_PROMPT_TEMPLATE = """You are an expert communicator. Your task is to rephrase the following complex regulatory text into a simple, clear, well-formatted explanation that a non-expert can easily understand.

**FORMATTING REQUIREMENTS:**
- Use HTML formatting with proper tags
- Start with a <strong>brief summary sentence</strong>
- Use <ul> and <li> tags for key points
- Use <strong> tags for important terms
- Use <em> tags for deadlines and requirements
- Convert complex numbers or rates into understandable impacts (e.g., "This means an average customer's bill will increase by about $5 per month")

**CONTENT REQUIREMENTS:**
- Focus on practical implications and real-world impacts
- Explain what this means for ordinary consumers or businesses
- Highlight any important deadlines or action items
- Use everyday language and avoid regulatory jargon

COMPLEX TEXT:
{technical_answer}

SIMPLIFIED EXPLANATION (in layman's terms with HTML formatting):"""

# --- PROCEEDING UTILITY FUNCTIONS ---
def get_proceeding_info(proceeding_id: str) -> dict:
    """Get information about a specific proceeding."""
    return AVAILABLE_PROCEEDINGS.get(proceeding_id, {})

def get_active_proceedings() -> dict:
    """Get all active proceedings."""
    return {k: v for k, v in AVAILABLE_PROCEEDINGS.items() if v.get('active', False)}

def get_proceeding_display_name(proceeding_id: str) -> str:
    """Get display name for a proceeding."""
    return AVAILABLE_PROCEEDINGS.get(proceeding_id, {}).get('display_name', proceeding_id)

def get_proceeding_short_name(proceeding_id: str) -> str:
    """Get short name for a proceeding."""
    return AVAILABLE_PROCEEDINGS.get(proceeding_id, {}).get('short_name', proceeding_id)

def get_proceeding_full_name(proceeding_id: str) -> str:
    """Get full name for a proceeding."""
    return AVAILABLE_PROCEEDINGS.get(proceeding_id, {}).get('full_name', proceeding_id)

def format_proceeding_for_search(proceeding_id: str) -> str:
    """Format proceeding ID for search queries (R2207005 -> R.22-07-005)."""
    if len(proceeding_id) == 8 and proceeding_id.startswith('R'):
        return f"R.{proceeding_id[1:3]}-{proceeding_id[3:5]}-{proceeding_id[5:]}"
    return proceeding_id

def get_proceeding_file_paths(proceeding_id: str, base_dir: Path = None) -> dict:
    """Get standardized file paths for a proceeding."""
    if base_dir is None:
        base_dir = PROJECT_ROOT
    
    proceeding_lower = proceeding_id.lower()
    
    return {
        'scraped_pdf_history': base_dir / "cpuc_csvs" / f"{proceeding_lower}_scraped_pdf_history.json",
        'result_csv': base_dir / "cpuc_csvs" / f"{proceeding_lower}_resultCSV.csv",
        'pdf_dir': base_dir / "cpuc_pdfs" / proceeding_id,
        'vector_db': base_dir / "local_chroma_db" / proceeding_id,
        'document_hashes': base_dir / "local_chroma_db" / proceeding_id / "document_hashes.json"
    }

def get_proceeding_urls(proceeding_id: str) -> dict:
    """Get standardized URLs for a proceeding."""
    return {
        'cpuc_apex': f"https://apps.cpuc.ca.gov/apex/f?p=401:56:::NO:RP,57,RIR:P5_PROCEEDING_SELECT:{proceeding_id}",
        'cpuc_search': f"https://docs.cpuc.ca.gov/SearchRes.aspx?category=proceeding&proceeding={proceeding_id}"
    }
