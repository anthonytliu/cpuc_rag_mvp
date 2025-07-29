# 📁 config.py
# Centralized configuration for the RAG system

# Add these imports at the top
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.resolve()

# Directory to store the LanceDB vector database
DB_DIR = PROJECT_ROOT / "local_lance_db"

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

# Debug and Logging Settings
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
VERBOSE_LOGGING = DEBUG  # Show detailed processing logs when DEBUG is enabled

# --- URL PROCESSING SETTINGS ---
# Enable URL-based processing (True) or file-based processing (False)
USE_URL_PROCESSING = True

# URL processing timeouts and retry settings
URL_VALIDATION_TIMEOUT = 30  # seconds
URL_PROCESSING_TIMEOUT = 300  # seconds for Docling processing

# --- MULTI-PROCEEDING SYSTEM SETTINGS ---
# Load proceeding titles from the generated JSON file
def load_proceeding_titles():
    """Load proceeding titles from the generated JSON file."""
    try:
        import json
        titles_file = PROJECT_ROOT / "proceeding_titles.json"
        if titles_file.exists():
            with open(titles_file, 'r') as f:
                data = json.load(f)
                return data.get('proceeding_titles', {})
    except Exception:
        pass
    
    # Fallback titles if JSON file not available
    return {
        "R2207005": "R.22-07-005 - Demand Flexibility Rulemaking",
        "R1807006": "R.18-07-006 - Affordability Rulemaking",
        "R1311005": "R.13-11-005 - Energy Efficiency Business Plan Applications"
    }

# Available proceedings that can be selected in the UI - will be populated later
AVAILABLE_PROCEEDINGS = {}


# Default proceeding to load on startup (first one in AVAILABLE_PROCEEDINGS)
def get_first_proceeding() -> str:
    """Get the first proceeding from AVAILABLE_PROCEEDINGS."""
    if not AVAILABLE_PROCEEDINGS:
        return "R2207005"  # Safe fallback
    return next(iter(AVAILABLE_PROCEEDINGS.keys()))


DEFAULT_PROCEEDING = os.getenv("DEFAULT_PROCEEDING", "R2207005")  # Use static default for now

# --- DOCUMENT SCRAPER SETTINGS ---
# Coverage threshold for skipping proceeding scraping (in percentage)
# If coverage is >= this threshold, skip full scraping process
SCRAPER_COVERAGE_THRESHOLD = 98.0  # Skip scraping if coverage is 98% or higher

# List of proceedings to scrape (single source of truth)
SCRAPER_PROCEEDINGS = [
    "R2207005",  # Advance Demand Flexibility Through Electric Rates
    "R1807006",  # Affordability Rulemaking
    "R1901011",  # Building Decarbonization
    "R1202009",  # CCA Code of Conduct
    "R0310003",  # CCA Rulemaking
    "R2008022",  # Clean Energy Financing
    "R1804019",  # Climate Change Adaptation Strategies
    "R2102014",  # COVID-19 Accumulated Debt OIR
    "R1707007",  # DER Interconnection Rulemaking
    "R1309011",  # Demand Response
    "R1408013",  # Distribution Resources Plans
    "R1803011",  # Emergency Disaster Relief Program
    "R1311005",  # Energy Efficiency, Business Plan Applications
    "R1503011",  # Energy Storage Rulemaking
    "R1812006",  # EV Rates and Infrastructure
    "R1910005",  # Electric Program Investment Charge (EPIC)
    "R2106017",  # High Distributed Energy Resources Future
    "R1410003",  # Integrated Distributed Energy Resources
    "R2005003",  # Integrated Resource Planning
    "R1909009",  # Microgrids Rulemaking
    "R1407002",  # Net Energy Metering
    "R2008020",  # Net Energy Metering
    "R1807005",  # New Approaches to Disconnections
    "R1706026",  # PCIA Rulemaking
    "R1812005",  # PG&E PSPS Order to Show Cause
    "R2103011",  # Provider of Last Resort (POLR)
    "R1206013",  # Residential Rate Rulemaking
    "R2110002",  # RA Rulemaking 2023
    "R2310011",  # OIR re RA Program
    "R2305018",  # Order Amending General Order 131-D
    "R1807003",  # RPS Program Implementation
    "R2401017",  # RPS Program Implementation
    "R1211005",  # Self-Generation Incentive Program (SGIP) Rulemaking
    "R2005012",  # Self-Generation Incentive Program (SGIP) Rulemaking
    "R2103010",  # Supplier Diversity
    "R1311007",  # Transportation Electrification – OIR Policy/Infrastructure
    "R1810007"   # Wildfire Mitigation Plans
]

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
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  # Custom Search Engine ID
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
AUTO_UPDATE_DELAY_MINUTES = int(
    os.getenv("AUTO_UPDATE_DELAY_MINUTES", "5"))  # Wait 5 minutes after download before updating RAG

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

5.  **Cite Sources Rigorously:** For every piece of information, you MUST cite the source using this exact format: `[CITE:filename.pdf,page_12,line_45]` where filename.pdf is the document name, page_12 is the page number, and line_45 is the line number within that page. If line number is not available, use: `[CITE:filename.pdf,page_12]`

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

# --- SPECIALIZED AGENT PROMPT TEMPLATES ---

# Technical Industry Expert Agent - provides precise regulatory and technical analysis
TECHNICAL_EXPERT_PROMPT_TEMPLATE = """You are a senior CPUC regulatory analyst with extensive expertise in California utility regulation. Your task is to provide a precise, authoritative, and comprehensive technical analysis based exclusively on the provided regulatory context.

Current date: {current_date}
Current proceeding: {current_proceeding}

REGULATORY CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS FOR TECHNICAL EXPERT ANALYSIS:

1. **Regulatory Framework Analysis:**
   - Identify the specific legal and regulatory framework governing this matter
   - Cite relevant Public Utilities Code sections, General Orders, and precedent decisions
   - Explain the regulatory hierarchy and authority structure

2. **Technical Requirements and Compliance:**
   - Detail specific technical requirements, standards, and specifications
   - Identify mandatory compliance obligations and deadlines
   - Explain enforcement mechanisms and penalties for non-compliance

3. **Legal Citations and References:**
   - Use precise legal citation format for all regulatory references
   - Include specific page numbers and document sections
   - Cross-reference related proceedings and decisions using format: `[CITE:filename.pdf,page_X]`

4. **Regulatory Impact Assessment:**
   - Analyze regulatory precedent and how this fits within existing framework
   - Identify potential conflicts or complementary regulations
   - Assess regulatory risk and compliance implications

5. **Technical Accuracy and Precision:**
   - Use exact regulatory terminology and defined terms
   - Provide specific numerical data, rates, percentages, and dollar amounts
   - Include technical specifications and measurement standards

6. **HTML Formatting for Professional Presentation:**
   - Use `<h4>` for main section headings
   - Use `<h5>` for subsection headings
   - Use `<strong>` for regulatory requirements and key terms
   - Use `<em>` for legal citations and cross-references
   - Use `<ul>` and `<li>` for detailed requirement lists
   - Use `<div class="regulatory-requirement">` for critical compliance items

7. **Regulatory Authority and Jurisdictional Scope:**
   - Clearly identify CPUC authority and jurisdiction limits
   - Distinguish between state and federal regulatory oversight
   - Note any joint jurisdiction or coordination requirements

RESPONSE STRUCTURE:
Begin with an Executive Summary, followed by detailed regulatory analysis organized by topic. Conclude with specific compliance requirements and deadlines if applicable.

If the regulatory context does not contain sufficient information, state: "<div style='background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; padding: 10px; margin: 10px 0;'><strong>Regulatory Notice:</strong> The provided regulatory documents do not contain sufficient information for a complete technical analysis of this matter.</div>"

TECHNICAL EXPERT ANALYSIS:"""

# Laymen Interpretation Agent - translates complex regulatory content
LAYMEN_INTERPRETATION_PROMPT_TEMPLATE = """You are an expert regulatory communications specialist. Your task is to translate complex regulatory and technical content into clear, accessible language that everyday consumers and businesses can understand.

TECHNICAL CONTENT TO TRANSLATE:
{technical_answer}

USER'S ORIGINAL QUESTION:
{question}

CURRENT PROCEEDING CONTEXT: {current_proceeding}

TRANSLATION GUIDELINES:

1. **Clear Communication Principles:**
   - Use everyday language and avoid regulatory jargon
   - Explain technical terms when first mentioned
   - Use analogies and real-world examples
   - Break complex concepts into simple steps

2. **Practical Impact Focus:**
   - Explain what this means for ordinary consumers
   - Describe how this affects businesses and organizations
   - Highlight financial impacts in understandable terms (e.g., "about $5 more per month")
   - Identify who is affected and how

3. **Actionable Information:**
   - Clearly state any deadlines or important dates
   - Explain what actions people might need to take
   - Identify who to contact for more information
   - Highlight any opportunities for public participation

4. **HTML Formatting for Accessibility:**
   - Use `<h4>` for main section headings
   - Start with a `<strong>` summary sentence
   - Use `<ul>` and `<li>` for key points and action items
   - Use `<strong>` for important terms and deadlines
   - Use `<em>` for financial impacts and specific examples
   - Use `<div class="consumer-alert">` for critical information consumers should know

5. **Real-World Context:**
   - Provide concrete examples and scenarios
   - Explain the "why" behind regulations
   - Connect to broader policy goals
   - Use familiar comparisons and analogies

6. **Simplified Structure:**
   - What this means in simple terms
   - Who is affected and how
   - Important dates and deadlines
   - What you can do or need to know
   - Where to get more information

SIMPLIFIED EXPLANATION:"""

# Further Sources Researcher Agent - provides additional research and external resources
FURTHER_SOURCES_PROMPT_TEMPLATE = """You are a regulatory research specialist with expertise in finding and synthesizing additional information sources. Your task is to provide comprehensive research resources and external context beyond the primary regulatory documents.

RESEARCH CONTEXT:
- User Question: {question}
- Current Proceeding: {proceeding}
- Primary Analysis: {documents_context}

EXTERNAL SEARCH RESULTS:
{search_results}

RESEARCH SYNTHESIS INSTRUCTIONS:

1. **Source Evaluation and Organization:**
   - Categorize sources by type (news, industry analysis, expert commentary, official resources)
   - Assess source credibility and relevance
   - Prioritize recent developments and current perspectives
   - Identify authoritative voices and expert opinions

2. **Additional Context and Perspective:**
   - Provide industry stakeholder viewpoints
   - Include recent news and developments
   - Highlight expert analysis and commentary
   - Connect to broader policy trends and implications

3. **Research Resource Categories:**
   - **Official Resources:** CPUC, CEC, CAISO, and other regulatory bodies
   - **Industry Analysis:** Trade associations, research institutions, consulting firms
   - **News and Current Events:** Recent articles, press releases, industry publications
   - **Expert Commentary:** Academic research, policy analysis, stakeholder positions
   - **Public Participation:** Comment opportunities, hearing schedules, stakeholder processes

4. **HTML Formatting for Research Presentation:**
   - Use `<h4>` for main research categories
   - Use `<h5>` for subcategories and source types
   - Use `<ul>` and `<li>` for resource lists
   - Use `<a href="URL" target="_blank">` for external links
   - Use `<strong>` for source names and titles
   - Use `<em>` for publication dates and key context

5. **Comprehensive Resource Assembly:**
   - Recent news articles and press coverage
   - Industry association positions and analysis
   - Academic and policy research
   - Related proceedings and cross-references
   - Stakeholder comment opportunities

6. **Research Quality and Reliability:**
   - Favor official and authoritative sources
   - Include publication dates and context
   - Note any conflicts of interest or bias
   - Provide balanced perspectives when available

RESPONSE STRUCTURE:
- Executive Research Summary
- Recent Developments and News
- Industry and Stakeholder Perspectives  
- Official Resources and Documents
- Expert Analysis and Commentary
- Opportunities for Further Engagement

If web search results are limited, focus on providing comprehensive official resources and explaining how to access additional information through proper channels.

ADDITIONAL RESEARCH SOURCES AND CONTEXT:"""

# --- CITATION ENHANCEMENT SETTINGS ---
# Enable line-level precision in citations
LINE_LEVEL_CITATIONS_ENABLED = True

# Estimated characters per line for line number calculation
# This is used when precise line information isn't available from the PDF parser
ESTIMATED_CHARACTERS_PER_LINE = 80

# Enable citation validation to cross-reference line numbers with content
CITATION_VALIDATION_ENABLED = True

# --- MULTI-AGENT SYSTEM CONFIGURATIONS ---
# Enable/disable multi-agent mode
MULTI_AGENT_ENABLED = True

# Agent response configuration
AGENT_RESPONSE_CONFIG = {
    "enable_question_interpretation": True,
    "enable_technical_analysis": True,
    "enable_laymen_explanation": True,
    "enable_sources_agent": True,
    "enable_response_synthesis": True
}

# --- MULTI-AGENT PROMPT TEMPLATES ---

# Question Interpretation Agent - analyzes and reformulates user queries
QUESTION_INTERPRETATION_AGENT_TEMPLATE = """You are a Question Interpretation Agent specialized in CPUC regulatory analysis. Your role is to analyze user questions and enhance them for optimal retrieval and response generation.

USER QUESTION: {question}

ANALYSIS TASKS:
1. **Intent Classification**: Identify the primary intent (factual inquiry, regulatory compliance, procedural question, timeline request, etc.)
2. **Key Entity Extraction**: Extract regulatory entities, proceeding numbers, dates, dollar amounts, and technical terms
3. **Query Enhancement**: Reformulate the question to include relevant regulatory context and synonyms
4. **Search Strategy**: Suggest optimal search terms and document types to focus on

OUTPUT FORMAT (JSON):
{{
    "original_question": "{question}",
    "intent_classification": "classification here",
    "key_entities": ["entity1", "entity2", "entity3"],
    "enhanced_query": "reformulated query with regulatory context",
    "search_terms": ["term1", "term2", "term3"],
    "document_focus": ["document_type1", "document_type2"],
    "complexity_level": "low/medium/high",
    "expected_sources": "number and type of sources likely needed"
}}

Provide only the JSON output, no additional text."""

# Technical Analysis Agent - provides detailed regulatory analysis
TECHNICAL_ANALYSIS_AGENT_TEMPLATE = """You are a Technical Analysis Agent specialized in CPUC regulatory documents. Your role is to provide comprehensive, accurate analysis based on retrieved documents.

QUERY ANALYSIS:
{query_analysis}

RETRIEVED CONTEXT:
{context}

QUESTION:
{question}

INSTRUCTIONS:
1. **Regulatory Framework**: Begin with relevant regulatory framework and authority
2. **Detailed Analysis**: Provide comprehensive analysis with specific citations
3. **Compliance Requirements**: Detail any compliance obligations, deadlines, or procedures
4. **Numerical Data**: Present all relevant numbers, rates, fees, and calculations
5. **Cross-References**: Note related proceedings, decisions, or regulatory sections
6. **Precedent Analysis**: Reference relevant precedents or similar cases if applicable

**CITATION FORMAT**: Use `[CITE:filename.pdf,page_X]` for every factual claim
**STRUCTURE**: Use HTML formatting with clear sections and professional language
**ACCURACY**: Base response strictly on provided context

TECHNICAL ANALYSIS:"""

# Laymen Explanation Agent - translates technical content to accessible language
LAYMEN_EXPLANATION_AGENT_TEMPLATE = """You are a Laymen Explanation Agent that makes complex regulatory content accessible to the general public.

TECHNICAL ANALYSIS TO TRANSLATE:
{technical_analysis}

ORIGINAL QUESTION:
{question}

TRANSLATION REQUIREMENTS:
1. **Plain Language**: Use simple, everyday terms instead of regulatory jargon
2. **Practical Impact**: Focus on real-world implications for consumers, businesses, or stakeholders
3. **Visual Structure**: Use clear HTML formatting with bullets and sections
4. **Context Setting**: Briefly explain background that average person might not know
5. **Action Items**: Highlight what people need to know or do
6. **Timeline**: Emphasize important dates and deadlines in accessible terms

**AVOID**: Technical jargon, complex legal terminology, acronyms without explanation
**INCLUDE**: Concrete examples, analogies, practical implications
**FORMAT**: HTML with clear headings and bullet points

ACCESSIBLE EXPLANATION:"""

# Sources Agent - curates and validates source information
SOURCES_AGENT_TEMPLATE = """You are a Sources Agent responsible for analyzing and presenting source information with quality assessment.

RETRIEVED DOCUMENTS:
{sources}

QUESTION:
{question}

ANALYSIS TASKS:
1. **Source Quality Assessment**: Evaluate each source's relevance, authority, and recency
2. **Citation Validation**: Ensure all cited sources are accurate and properly formatted
3. **Coverage Analysis**: Identify any gaps in source coverage for the question
4. **Source Hierarchy**: Rank sources by regulatory authority (decisions > rulings > staff reports)
5. **Cross-Reference Check**: Verify consistency across sources and flag conflicts

OUTPUT FORMAT:
- **Primary Sources**: Most authoritative and directly relevant documents
- **Supporting Sources**: Additional context and background information
- **Source Quality Summary**: Brief assessment of evidence strength
- **Coverage Assessment**: Areas well-covered vs. potential gaps
- **Reliability Score**: Overall confidence in source base (1-10)

SOURCES ANALYSIS:"""

# Response Synthesis Agent - coordinates and integrates all agent outputs
RESPONSE_SYNTHESIS_AGENT_TEMPLATE = """You are a Response Synthesis Agent responsible for coordinating multi-agent outputs into a cohesive response.

AGENT OUTPUTS:
Query Analysis: {query_analysis}
Technical Analysis: {technical_analysis}
Laymen Explanation: {laymen_explanation}
Sources Analysis: {sources_analysis}

SYNTHESIS TASKS:
1. **Consistency Check**: Verify all agents provide consistent information
2. **Completeness Assessment**: Ensure the question is fully addressed
3. **Quality Integration**: Combine insights while avoiding redundancy
4. **Confidence Scoring**: Assess overall response reliability
5. **Gap Identification**: Note any missing information or limitations

SYNTHESIS CRITERIA:
- Maintain technical accuracy from Technical Agent
- Preserve accessibility from Laymen Agent
- Ensure proper source attribution from Sources Agent
- Address query intent from Question Interpretation Agent

FINAL INTEGRATED RESPONSE:
- Present unified response maintaining both technical depth and accessibility
- Provide clear confidence indicators
- Note any limitations or areas needing additional research

SYNTHESIZED RESPONSE:"""


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
    """Get standardized file paths for a proceeding using new cpuc_proceedings structure."""
    if base_dir is None:
        base_dir = PROJECT_ROOT

    # New cpuc_proceedings structure
    cpuc_proceedings_dir = base_dir / "cpuc_proceedings"
    proceeding_dir = cpuc_proceedings_dir / proceeding_id

    return {
        # New structure paths
        'proceeding_dir': proceeding_dir,
        'documents_dir': proceeding_dir / "documents",
        'embeddings_dir': proceeding_dir / "embeddings",
        'scraped_pdf_history': proceeding_dir / "scraped_pdf_history.json",
        'scraped_pdf_history_alt': proceeding_dir / f"{proceeding_id}_scraped_pdf_history.json",
        'documents_csv': proceeding_dir / "documents" / f"{proceeding_id}_documents.csv",

        # Embeddings and processing files (in proceeding/embeddings/)
        'chunks_metadata': proceeding_dir / "embeddings" / "chunks_metadata.json",
        'processing_log': proceeding_dir / "embeddings" / "processing_log.json",
        'embedding_status': proceeding_dir / "embeddings" / "embedding_status.json",

        # Centralized vector DB (still in root for shared access)
        'vector_db': base_dir / "local_lance_db" / proceeding_id,
        'document_hashes': base_dir / "local_lance_db" / proceeding_id / "document_hashes.json",

        # Legacy paths for backward compatibility
        'legacy_scraped_pdf_history': base_dir / "cpuc_csvs" / f"{proceeding_id.lower()}_scraped_pdf_history.json",
        'legacy_result_csv': base_dir / "cpuc_csvs" / f"{proceeding_id.lower()}_resultCSV.csv"
    }


def get_proceeding_urls(proceeding_id: str) -> dict:
    """Get standardized URLs for a proceeding."""
    return {
        'cpuc_apex': f"https://apps.cpuc.ca.gov/apex/f?p=401:56:::NO:RP,57,RIR:P5_PROCEEDING_SELECT:{proceeding_id}",
        'cpuc_search': f"https://docs.cpuc.ca.gov/SearchRes.aspx?category=proceeding&proceeding={proceeding_id}"
    }


# --- CHONKIE FALLBACK SETTINGS ---
# Enable Chonkie fallback for failed PDF extractions
CHONKIE_FALLBACK_ENABLED = os.getenv("CHONKIE_FALLBACK_ENABLED", "true").lower() == "true"

# Chonkie chunking strategy priority order
CHONKIE_STRATEGIES = ["recursive", "sentence", "token"]

# Chonkie chunk size settings (should match existing settings)
CHONKIE_CHUNK_SIZE = CHUNK_SIZE  # Use existing chunk size
CHONKIE_CHUNK_OVERLAP = CHUNK_OVERLAP  # Use existing overlap

# Minimum text length required for Chonkie processing
CHONKIE_MIN_TEXT_LENGTH = 100

# Enable different PDF text extraction methods for Chonkie
CHONKIE_USE_PDFPLUMBER = True
CHONKIE_USE_PYPDF2 = True

# --- INTELLIGENT HYBRID PROCESSING SETTINGS ---
# Enable intelligent hybrid processing (Chonkie primary, Docling for tables/financial)
INTELLIGENT_HYBRID_ENABLED = os.getenv("INTELLIGENT_HYBRID_ENABLED", "true").lower() == "true"

# Keywords that trigger table/financial document detection
TABLE_FINANCIAL_KEYWORDS = [
    "compensation", "financial", "rate", "tariff", "cost", "revenue", "expense", 
    "budget", "billing", "payment", "refund", "surcharge", "fee", "charge",
    "table", "schedule", "appendix", "exhibit", "calculation", "formula",
    "compliance report", "joint filing", "annual report", "quarterly report"
]

# Minimum score to trigger hybrid evaluation (0.0-1.0)
HYBRID_TRIGGER_THRESHOLD = 0.3

# Enable agent-based evaluation for hybrid decisions
AGENT_EVALUATION_ENABLED = os.getenv("AGENT_EVALUATION_ENABLED", "true").lower() == "true"

# Directory for storing agent evaluation logs
AGENT_EVALUATION_LOG_DIR = PROJECT_ROOT / "agent_evaluations"

# Agent evaluation timeout (seconds)
AGENT_EVALUATION_TIMEOUT = 60

# --- CUDA OPTIMIZATION SETTINGS ---
# GPU memory management
CUDA_MEMORY_OPTIMIZATION = os.getenv("CUDA_MEMORY_OPTIMIZATION", "true").lower() == "true"
CUDA_MEMORY_FRACTION = float(os.getenv("CUDA_MEMORY_FRACTION", "0.8"))  # Reserve 80% of GPU memory
CUDA_ENABLE_MIXED_PRECISION = os.getenv("CUDA_ENABLE_MIXED_PRECISION", "true").lower() == "true"

# Dynamic batch sizing based on GPU memory
CUDA_DYNAMIC_BATCH_SIZE = os.getenv("CUDA_DYNAMIC_BATCH_SIZE", "true").lower() == "true"
CUDA_MAX_BATCH_SIZE = int(os.getenv("CUDA_MAX_BATCH_SIZE", "64"))
CUDA_MIN_BATCH_SIZE = int(os.getenv("CUDA_MIN_BATCH_SIZE", "8"))

# CUDA memory monitoring
CUDA_MEMORY_MONITORING = os.getenv("CUDA_MEMORY_MONITORING", "true").lower() == "true"
CUDA_MEMORY_CLEANUP_THRESHOLD = float(os.getenv("CUDA_MEMORY_CLEANUP_THRESHOLD", "0.85"))  # Cleanup when 85% full

# --- ERROR RECOVERY SETTINGS ---
# Processing timeouts and retries
PROCESSING_TIMEOUT_SECONDS = int(os.getenv("PROCESSING_TIMEOUT_SECONDS", "300"))
MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF_MULTIPLIER = float(os.getenv("RETRY_BACKOFF_MULTIPLIER", "2.0"))
INITIAL_RETRY_DELAY = float(os.getenv("INITIAL_RETRY_DELAY", "1.0"))

# Batch processing settings for recovery
PROCESSING_BATCH_SIZE = int(os.getenv("PROCESSING_BATCH_SIZE", "10"))
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "5"))  # Save progress every 5 documents
EMERGENCY_STOP_ON_CONSECUTIVE_FAILURES = int(os.getenv("EMERGENCY_STOP_ON_CONSECUTIVE_FAILURES", "10"))

# Memory cleanup intervals
MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "50"))  # Clean memory every 50 documents
FORCE_GC_INTERVAL = int(os.getenv("FORCE_GC_INTERVAL", "25"))  # Force garbage collection every 25 documents

# --- INTEGRATION VALIDATION SETTINGS ---
# Dependency validation
VALIDATE_DEPENDENCIES_ON_STARTUP = os.getenv("VALIDATE_DEPENDENCIES_ON_STARTUP", "true").lower() == "true"
REQUIRED_MODELS = ["BAAI/bge-base-en-v1.5"]  # Models that must be available
REQUIRED_PACKAGES = ["torch", "lancedb", "docling", "chonkie", "langchain"]

# Database connectivity validation
VALIDATE_DATABASE_ON_STARTUP = os.getenv("VALIDATE_DATABASE_ON_STARTUP", "true").lower() == "true"
DATABASE_HEALTH_CHECK_TIMEOUT = int(os.getenv("DATABASE_HEALTH_CHECK_TIMEOUT", "30"))

# API endpoint validation
VALIDATE_OPENAI_API_ON_STARTUP = os.getenv("VALIDATE_OPENAI_API_ON_STARTUP", "true").lower() == "true"
API_HEALTH_CHECK_TIMEOUT = int(os.getenv("API_HEALTH_CHECK_TIMEOUT", "15"))

# --- USER EXPERIENCE SETTINGS ---
# Input validation
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "2000"))
MIN_QUERY_LENGTH = int(os.getenv("MIN_QUERY_LENGTH", "5"))
MAX_DOCUMENT_TITLE_LENGTH = int(os.getenv("MAX_DOCUMENT_TITLE_LENGTH", "200"))

# Progress reporting
DETAILED_PROGRESS_REPORTING = os.getenv("DETAILED_PROGRESS_REPORTING", "true").lower() == "true"
PROGRESS_UPDATE_INTERVAL = int(os.getenv("PROGRESS_UPDATE_INTERVAL", "10"))  # Update every 10 documents

# User feedback and notifications
ENABLE_SUCCESS_NOTIFICATIONS = os.getenv("ENABLE_SUCCESS_NOTIFICATIONS", "true").lower() == "true"
ENABLE_ERROR_NOTIFICATIONS = os.getenv("ENABLE_ERROR_NOTIFICATIONS", "true").lower() == "true"
ENABLE_PERFORMANCE_METRICS = os.getenv("ENABLE_PERFORMANCE_METRICS", "true").lower() == "true"

# --- MONITORING AND LOGGING SETTINGS ---
# Performance monitoring
PERFORMANCE_LOGGING_ENABLED = os.getenv("PERFORMANCE_LOGGING_ENABLED", "true").lower() == "true"
PERFORMANCE_LOG_INTERVAL = int(os.getenv("PERFORMANCE_LOG_INTERVAL", "100"))  # Log every 100 operations

# Resource usage monitoring
RESOURCE_MONITORING_ENABLED = os.getenv("RESOURCE_MONITORING_ENABLED", "true").lower() == "true"
RESOURCE_LOG_INTERVAL = int(os.getenv("RESOURCE_LOG_INTERVAL", "60"))  # Log every 60 seconds

# Error tracking and analytics
ERROR_ANALYTICS_ENABLED = os.getenv("ERROR_ANALYTICS_ENABLED", "true").lower() == "true"
ERROR_REPORT_AGGREGATION_INTERVAL = int(os.getenv("ERROR_REPORT_AGGREGATION_INTERVAL", "300"))  # 5 minutes

# Log file management
MAX_LOG_FILE_SIZE_MB = int(os.getenv("MAX_LOG_FILE_SIZE_MB", "100"))
MAX_LOG_FILES_RETENTION = int(os.getenv("MAX_LOG_FILES_RETENTION", "7"))
LOG_COMPRESSION_ENABLED = os.getenv("LOG_COMPRESSION_ENABLED", "true").lower() == "true"


# --- UTILITY FUNCTIONS FOR PRODUCTION OPTIMIZATION ---
def get_optimal_worker_count() -> int:
    """Calculate optimal number of workers based on system resources."""
    import os
    cpu_count = os.cpu_count() or 4
    # Use 75% of available CPUs, minimum 2, maximum 8
    return max(2, min(8, int(cpu_count * 0.75)))


def get_memory_optimized_batch_size(base_size: int = 32) -> int:
    """Get memory-optimized batch size based on available system memory."""
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        if available_memory_gb >= 16:
            return min(base_size * 2, CUDA_MAX_BATCH_SIZE)
        elif available_memory_gb >= 8:
            return base_size
        else:
            return max(base_size // 2, CUDA_MIN_BATCH_SIZE)
    except ImportError:
        return base_size


def validate_production_config() -> dict:
    """Validate production configuration settings."""
    validation_results = {
        'cuda_available': False,
        'required_packages': True,
        'api_key_configured': bool(OPENAI_API_KEY),
        'database_directory': DB_DIR.exists() or DB_DIR.parent.exists(),
        'agent_log_directory': AGENT_EVALUATION_LOG_DIR.parent.exists()
    }
    
    # Check CUDA availability
    try:
        import torch
        validation_results['cuda_available'] = torch.cuda.is_available()
    except ImportError:
        pass
    
    # Check required packages
    for package in REQUIRED_PACKAGES:
        try:
            __import__(package)
        except ImportError:
            validation_results['required_packages'] = False
            break
    
    return validation_results


# Initialize AVAILABLE_PROCEEDINGS after all other definitions
def _initialize_available_proceedings():
    """Initialize AVAILABLE_PROCEEDINGS with proceeding titles and vector store status."""
    global AVAILABLE_PROCEEDINGS
    
    # Get proceeding titles
    proceeding_titles = load_proceeding_titles()
    
    # Auto-generate AVAILABLE_PROCEEDINGS from SCRAPER_PROCEEDINGS and titles
    for proc_id in SCRAPER_PROCEEDINGS:
        # Check if vector store exists for this proceeding
        vector_store_path = PROJECT_ROOT / "local_lance_db" / proc_id
        has_vector_store = vector_store_path.exists()
        
        # Get title from our generated titles or create fallback
        display_name = proceeding_titles.get(proc_id, f"{format_proceeding_for_search(proc_id)} - CPUC Proceeding")
        
        AVAILABLE_PROCEEDINGS[proc_id] = {
            "display_name": display_name,
            "full_name": format_proceeding_for_search(proc_id),
            "short_name": proc_id,
            "description": display_name.split(" - ", 1)[1] if " - " in display_name else "CPUC Proceeding",
            "active": has_vector_store  # Auto-activate if vector store exists
        }

# Initialize the proceedings
_initialize_available_proceedings()
