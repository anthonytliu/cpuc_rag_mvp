import hashlib
import logging
import os
import re
import requests
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.document import DocItem, TableItem
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain.docstore.document import Document

import config

logger = logging.getLogger(__name__)

# Set Docling threading configuration for performance
if hasattr(config, 'DOCLING_THREADS') and config.DOCLING_THREADS:
    os.environ['OMP_NUM_THREADS'] = str(config.DOCLING_THREADS)
    logger.info(f"Set OMP_NUM_THREADS to {config.DOCLING_THREADS} for optimized Docling performance")

# Configure optimized Docling converter for performance
pipeline_options = PdfPipelineOptions()
if hasattr(config, 'DOCLING_FAST_MODE') and config.DOCLING_FAST_MODE:
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST
    logger.info("Docling configured with FAST table processing mode for better performance")

if hasattr(config, 'DOCLING_MAX_PAGES') and config.DOCLING_MAX_PAGES:
    pipeline_options.page_boundary = (0, config.DOCLING_MAX_PAGES)
    logger.info(f"Docling configured with page limit: {config.DOCLING_MAX_PAGES}")

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            backend=DoclingParseV4DocumentBackend,
            pipeline_options=pipeline_options
        )
    }
)

def extract_text_with_pypdf2(pdf_path: Path) -> str:
    """Extract raw text using PyPDF2."""
    try:
        import PyPDF2
        
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        logger.debug(f"PyPDF2 extracted {len(text)} characters from {pdf_path.name}")
        return text
        
    except Exception as e:
        logger.error(f"PyPDF2 extraction failed for {pdf_path}: {e}")
        return ""

def extract_text_with_pdfplumber(pdf_path: Path) -> str:
    """Extract raw text using pdfplumber (better for tables)."""
    try:
        import pdfplumber
        
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        logger.debug(f"pdfplumber extracted {len(text)} characters from {pdf_path.name}")
        return text
        
    except Exception as e:
        logger.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
        return ""

def extract_text_from_url(pdf_url: str) -> str:
    """Extract raw text from PDF URL using multiple extraction methods."""
    try:
        # Download PDF to temporary file
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = Path(tmp_file.name)
        
        try:
            text = ""
            
            # Try pdfplumber first if enabled
            if config.CHONKIE_USE_PDFPLUMBER and not text.strip():
                logger.debug("Attempting text extraction with pdfplumber")
                text = extract_text_with_pdfplumber(tmp_path)
            
            # Try PyPDF2 if pdfplumber failed and it's enabled
            if config.CHONKIE_USE_PYPDF2 and not text.strip():
                logger.debug("Attempting text extraction with PyPDF2")
                text = extract_text_with_pypdf2(tmp_path)
            
            logger.info(f"Extracted {len(text)} characters from {pdf_url}")
            return text
            
        finally:
            # Clean up temporary file
            tmp_path.unlink()
    
    except Exception as e:
        logger.error(f"Text extraction from URL failed {pdf_url}: {e}")
        return ""

def safe_chunk_with_chonkie(text: str, chunker_type: str = "recursive") -> List[str]:
    """Safely chunk text with Chonkie, handling errors gracefully. Returns text only for backward compatibility."""
    chunks_with_positions = safe_chunk_with_chonkie_enhanced(text, chunker_type)
    return [chunk['text'] for chunk in chunks_with_positions]

def safe_chunk_with_chonkie_enhanced(text: str, chunker_type: str = "recursive") -> List[dict]:
    """
    Enhanced Chonkie chunking that preserves character positions for accurate citations.
    
    Returns:
        List of dictionaries with keys:
        - 'text': chunk text content
        - 'start_index': character position where chunk starts in original text
        - 'end_index': character position where chunk ends in original text  
        - 'token_count': number of tokens in chunk (if available)
        - 'level': recursion level (for recursive chunker)
        - 'strategy': chunker type used
    """
    if not text or not text.strip():
        logger.warning("Empty or whitespace-only text provided for chunking")
        return []
    
    if len(text.strip()) < config.CHONKIE_MIN_TEXT_LENGTH:
        logger.warning(f"Text too short for chunking: {len(text)} chars (min: {config.CHONKIE_MIN_TEXT_LENGTH})")
        return []
    
    try:
        from chonkie import TokenChunker, SentenceChunker, RecursiveChunker
        
        # Select chunker based on type (using correct Chonkie API)
        if chunker_type == "recursive":
            chunker = RecursiveChunker(
                chunk_size=config.CHONKIE_CHUNK_SIZE
            )
        elif chunker_type == "sentence":
            chunker = SentenceChunker(
                chunk_size=config.CHONKIE_CHUNK_SIZE
            )
        elif chunker_type == "token":
            chunker = TokenChunker(
                chunk_size=config.CHONKIE_CHUNK_SIZE
            )
        else:
            logger.warning(f"Unknown chunker type '{chunker_type}', defaulting to recursive")
            chunker = RecursiveChunker(chunk_size=config.CHONKIE_CHUNK_SIZE)
        
        # Perform chunking
        chunks = chunker(text)
        
        # Extract enhanced chunk information preserving all metadata
        enhanced_chunks = []
        for chunk in chunks:
            if not chunk.text.strip():
                continue
                
            chunk_info = {
                'text': chunk.text,
                'start_index': getattr(chunk, 'start_index', 0),
                'end_index': getattr(chunk, 'end_index', len(chunk.text)),
                'token_count': getattr(chunk, 'token_count', 0),
                'level': getattr(chunk, 'level', 0),
                'strategy': chunker_type
            }
            enhanced_chunks.append(chunk_info)
        
        logger.info(f"Successfully chunked text into {len(enhanced_chunks)} chunks using {chunker_type} with position tracking")
        return enhanced_chunks
        
    except Exception as e:
        logger.error(f"Chonkie chunking failed with {chunker_type}: {e}")
        # Fallback to simple splitting with estimated positions
        return _fallback_chunk_text_enhanced(text)

def _fallback_chunk_text_enhanced(text: str, chunk_size: int = None, overlap: int = None) -> List[dict]:
    """Enhanced fallback chunking with position tracking when Chonkie fails."""
    if chunk_size is None:
        chunk_size = config.CHONKIE_CHUNK_SIZE
    if overlap is None:
        overlap = config.CHONKIE_CHUNK_OVERLAP
    
    enhanced_chunks = []
    start_pos = 0
    
    while start_pos < len(text):
        end_pos = min(start_pos + chunk_size, len(text))
        chunk_text = text[start_pos:end_pos].strip()
        
        if chunk_text:
            chunk_info = {
                'text': chunk_text,
                'start_index': start_pos,
                'end_index': end_pos,
                'token_count': len(chunk_text.split()),  # Rough token estimate
                'level': 0,
                'strategy': 'fallback'
            }
            enhanced_chunks.append(chunk_info)
        
        start_pos += chunk_size - overlap
    
    logger.info(f"Fallback chunking created {len(enhanced_chunks)} chunks with position tracking")
    return enhanced_chunks

def _fallback_chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Simple fallback chunking if Chonkie fails."""
    if chunk_size is None:
        chunk_size = config.CHONKIE_CHUNK_SIZE
    if overlap is None:
        overlap = config.CHONKIE_CHUNK_OVERLAP
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap if end < text_length else end
    
    logger.info(f"Fallback chunking produced {len(chunks)} chunks")
    return chunks

def estimate_page_from_char_position(char_position: int, full_text: str, 
                                   chars_per_page: int = 2000) -> int:
    """
    Estimate page number from character position using form feed detection.
    
    Args:
        char_position: Character position in the full text
        full_text: Complete text of the document
        chars_per_page: Estimated characters per page (default: 2000)
        
    Returns:
        Estimated page number (1-indexed)
    """
    if char_position <= 0:
        return 1
    
    # Look for form feed characters (page breaks) in the full text
    page_breaks = []
    for i, char in enumerate(full_text):
        if char == '\f':  # Form feed character
            page_breaks.append(i)
    
    if page_breaks:
        # Find which page the character position falls into
        current_page = 1
        for break_pos in page_breaks:
            if char_position > break_pos:
                current_page += 1
            else:
                break
        return current_page
    
    # Fallback to character-based estimation
    return max(1, (char_position // chars_per_page) + 1)

def estimate_line_range_from_char_position(start_pos: int, end_pos: int, full_text: str) -> tuple:
    """
    Estimate line number range from character positions.
    
    Args:
        start_pos: Starting character position
        end_pos: Ending character position  
        full_text: Complete text of the document
        
    Returns:
        Tuple of (start_line, end_line) - both 1-indexed
    """
    if start_pos <= 0:
        start_line = 1
    else:
        start_line = full_text[:start_pos].count('\n') + 1
    
    if end_pos <= start_pos:
        end_line = start_line
    else:
        end_line = full_text[:end_pos].count('\n') + 1
    
    return start_line, end_line

def create_enhanced_chonkie_metadata(chunk_info: dict, source_name: str, pdf_url: str, 
                                   proceeding: str, raw_text: str) -> dict:
    """
    Create enhanced metadata leveraging Chonkie's position tracking for accurate citations.
    
    Args:
        chunk_info: Dictionary with chunk text and position information
        source_name: Name of the source document
        pdf_url: URL of the PDF document
        proceeding: Proceeding number
        raw_text: Full raw text of the document
        
    Returns:
        Enhanced metadata dictionary with character-position-based citations
    """
    start_pos = chunk_info.get('start_index', 0)
    end_pos = chunk_info.get('end_index', len(chunk_info['text']))
    
    # Estimate page number from character position
    estimated_page = estimate_page_from_char_position(start_pos, raw_text)
    
    # Estimate line range from character position  
    start_line, end_line = estimate_line_range_from_char_position(start_pos, end_pos, raw_text)
    
    # Create text snippet for citation verification
    chunk_text = chunk_info['text']
    text_snippet = chunk_text[:100].replace('\n', ' ').strip() if chunk_text else ""
    
    return {
        "source_url": pdf_url,
        "source": source_name,
        "page": estimated_page,
        "line_number": start_line,
        "line_range_end": end_line,
        "char_start": start_pos,          # NEW: Exact character position
        "char_end": end_pos,              # NEW: Exact character range
        "char_length": end_pos - start_pos,
        "content_type": f"text_chonkie_{chunk_info.get('strategy', 'unknown')}",
        "chunk_id": f"{source_name}_chonkie_{start_pos}_{end_pos}",
        "token_count": chunk_info.get('token_count', 0),
        "chunk_level": chunk_info.get('level', 0),
        "text_snippet": text_snippet,    # NEW: For citation verification
        "proceeding_number": proceeding,
        "last_checked": "",
        "document_date": "",
        "publication_date": "",
        "document_type": "unknown",
        "supersedes_priority": 0.5
    }

def create_precise_citation(filename: str, page: int, char_start: int, char_end: int, 
                          line_start: int = None, text_snippet: str = None) -> str:
    """
    Create precise citations using character positions from Chonkie.
    
    Format: [CITE:filename.pdf,page_X,chars_Y-Z,line_L,"snippet"]
    
    Args:
        filename: PDF filename
        page: Page number
        char_start: Starting character position
        char_end: Ending character position
        line_start: Starting line number (optional)
        text_snippet: Text snippet for verification (optional)
        
    Returns:
        Enhanced citation string with character-level precision
    """
    citation_parts = [f"CITE:{filename}", f"page_{page}", f"chars_{char_start}-{char_end}"]
    
    if line_start:
        citation_parts.append(f"line_{line_start}")
    
    if text_snippet:
        # First 50 chars for verification, clean up quotes and newlines
        snippet = text_snippet[:50].replace('"', "'").replace('\n', ' ').strip()
        citation_parts.append(f'"{snippet}..."')
    
    return f"[{','.join(citation_parts)}]"

def detect_table_financial_document(pdf_url: str, document_title: str = None) -> float:
    """
    Detect if a document likely contains tables or financial information.
    
    This function uses document title, URL patterns, and preliminary content analysis
    to determine if a document should use hybrid processing.
    
    Args:
        pdf_url (str): URL of the PDF document
        document_title (str, optional): Document title if available
        
    Returns:
        float: Score from 0.0 to 1.0 indicating likelihood of table/financial content
    """
    score = 0.0
    text_to_analyze = ""
    
    # Add URL for analysis
    if pdf_url:
        text_to_analyze += pdf_url.lower() + " "
    
    # Add title for analysis
    if document_title:
        text_to_analyze += document_title.lower() + " "
    
    # Try to get a small sample of the document content for analysis
    try:
        # For CPUC documents, we can make some assumptions based on URL patterns
        if 'cpuc.ca.gov' in pdf_url.lower():
            # Try to get PDF metadata or a small sample for better classification
            try:
                # Try basic text extraction for a quick sample
                import io
                response = requests.get(pdf_url, timeout=15, stream=True)
                if response.status_code == 200:
                    # Read just first chunk for analysis
                    first_chunk = next(response.iter_content(chunk_size=8192), b'')
                    # Try to extract some readable text from the PDF header/metadata
                    content_sample = first_chunk.decode('utf-8', errors='ignore')
                    text_to_analyze += content_sample.lower()
            except Exception:
                # If sampling fails, rely on URL and title analysis
                logger.debug(f"Content sampling failed for {pdf_url}, using URL/title only")
    except Exception:
        # If content sampling fails, just use URL and title
        pass
    
    # Calculate score based on keyword matches and document type patterns
    keyword_matches = 0
    total_keywords = len(config.TABLE_FINANCIAL_KEYWORDS)
    
    for keyword in config.TABLE_FINANCIAL_KEYWORDS:
        if keyword.lower() in text_to_analyze:
            keyword_matches += 1
    
    base_score = keyword_matches / total_keywords if total_keywords > 0 else 0.0
    
    # Add bonus points for document types that typically contain financial/table content
    bonus_score = 0.0
    
    # Check URL and title for document type indicators
    text_lower = text_to_analyze.lower()
    
    # High-probability indicators (add significant bonus)
    high_indicators = ['compensation', 'tariff', 'rate schedule', 'cost allocation', 
                      'revenue requirement', 'billing', 'finance', 'budget']
    for indicator in high_indicators:
        if indicator in text_lower:
            bonus_score += 0.3
            break
    
    # Medium-probability indicators
    medium_indicators = ['compliance report', 'annual report', 'quarterly', 
                        'exhibit', 'appendix', 'schedule', 'table']
    for indicator in medium_indicators:
        if indicator in text_lower:
            bonus_score += 0.2
            break
    
    # Document type patterns in URL
    if 'agenda' in text_lower and 'decision' in text_lower:
        bonus_score += 0.4  # Agenda decisions often have compensation tables
    elif 'compliance' in text_lower:
        bonus_score += 0.3  # Compliance reports often have financial data
    elif 'compensation' in text_lower:
        bonus_score += 0.5  # Compensation decisions have tables
    
    # Final score (capped at 1.0)
    final_score = min(base_score + bonus_score, 1.0)
    
    logger.debug(f"Table/financial detection for {pdf_url}: {final_score:.3f} (base: {base_score:.3f}, bonus: {bonus_score:.3f}, keywords: {keyword_matches}/{total_keywords})")
    return final_score

def create_agent_evaluation_log(pdf_url: str, document_title: str, detection_score: float, 
                               docling_result: dict, chonkie_result: dict, 
                               agent_decision: str, agent_reasoning: str) -> None:
    """
    Create a detailed log file of agent evaluation decisions for later analysis.
    
    Args:
        pdf_url (str): URL of the processed document
        document_title (str): Document title
        detection_score (float): Initial detection score
        docling_result (dict): Results from Docling processing
        chonkie_result (dict): Results from Chonkie processing
        agent_decision (str): Which method was chosen by the agent
        agent_reasoning (str): Agent's reasoning for the decision
    """
    try:
        # Ensure log directory exists
        config.AGENT_EVALUATION_LOG_DIR.mkdir(exist_ok=True)
        
        # Create filename with timestamp and URL hash
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        url_hash = get_url_hash(pdf_url)[:8]
        log_filename = f"agent_evaluation_{timestamp}_{url_hash}.txt"
        log_path = config.AGENT_EVALUATION_LOG_DIR / log_filename
        
        # Create detailed log content
        log_content = f"""AGENT EVALUATION LOG
{'=' * 80}
Timestamp: {datetime.now().isoformat()}
Document URL: {pdf_url}
Document Title: {document_title or 'Unknown'}
Detection Score: {detection_score:.3f}

PROCESSING RESULTS COMPARISON:
{'-' * 40}

DOCLING RESULTS:
- Success: {docling_result.get('success', False)}
- Processing Time: {docling_result.get('processing_time', 0):.2f}s
- Chunks Extracted: {docling_result.get('chunk_count', 0)}
- Content Length: {docling_result.get('content_length', 0)} characters
- Content Types: {docling_result.get('content_types', [])}
- Tables Found: {docling_result.get('tables_found', 0)}
- Error: {docling_result.get('error', 'None')}

CHONKIE RESULTS:
- Success: {chonkie_result.get('success', False)}
- Processing Time: {chonkie_result.get('processing_time', 0):.2f}s
- Chunks Extracted: {chonkie_result.get('chunk_count', 0)}
- Content Length: {chonkie_result.get('content_length', 0)} characters
- Strategy Used: {chonkie_result.get('strategy_used', 'Unknown')}
- Text Quality: {chonkie_result.get('text_quality', 0):.3f}
- Error: {chonkie_result.get('error', 'None')}

AGENT DECISION:
{'-' * 40}
Chosen Method: {agent_decision}

AGENT REASONING:
{'-' * 40}
{agent_reasoning}

PERFORMANCE METRICS:
{'-' * 40}
Speed Advantage: {docling_result.get('processing_time', 999) / max(chonkie_result.get('processing_time', 1), 0.001):.1f}x (Docling vs Chonkie)
Content Volume: Docling={docling_result.get('content_length', 0)}, Chonkie={chonkie_result.get('content_length', 0)}
Reliability: Docling={'Success' if docling_result.get('success') else 'Failed'}, Chonkie={'Success' if chonkie_result.get('success') else 'Failed'}

{'=' * 80}
"""
        
        # Write log file
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(log_content)
        
        logger.info(f"Agent evaluation log created: {log_path}")
        
    except Exception as e:
        logger.error(f"Failed to create agent evaluation log: {e}")

def get_url_hash(url: str) -> str:
    """
    Calculate SHA-256 hash of a URL for unique identification and change detection.
    
    This function creates a deterministic hash of a URL that can be used for
    tracking changes and avoiding duplicate processing. The hash remains
    consistent across sessions for the same URL.
    
    Args:
        url (str): The URL to hash
        
    Returns:
        str: SHA-256 hash of the URL as a hexadecimal string
        
    Examples:
        >>> get_url_hash("https://docs.cpuc.ca.gov/example.pdf")
        'a1b2c3d4e5f6...'
    """
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

def validate_pdf_url(url: str, timeout: int = 30) -> bool:
    """
    Validate that a URL is accessible and points to a PDF document.
    
    This function performs a HEAD request to check URL accessibility and
    verifies that the content type indicates a PDF document. It's used
    to validate URLs before attempting to process them with Docling.
    
    Args:
        url (str): The URL to validate
        timeout (int): Request timeout in seconds (default: 30)
        
    Returns:
        bool: True if URL is accessible and points to a PDF, False otherwise
        
    Examples:
        >>> validate_pdf_url("https://docs.cpuc.ca.gov/example.pdf")
        True
        >>> validate_pdf_url("https://invalid-url.com/not-found.pdf")
        False
    """
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        return 'application/pdf' in content_type or url.lower().endswith('.pdf')
        
    except Exception as e:
        logger.warning(f"URL validation failed for {url}: {e}")
        return False

def extract_filename_from_url(url: str) -> str:
    """
    Extract a meaningful filename from a URL.
    
    This function attempts to extract a filename from the URL path.
    If no meaningful filename can be extracted, it generates one
    based on the URL hash.
    
    Args:
        url (str): The URL to extract filename from
        
    Returns:
        str: Extracted or generated filename
        
    Examples:
        >>> extract_filename_from_url("https://docs.cpuc.ca.gov/document.pdf")
        'document.pdf'
        >>> extract_filename_from_url("https://docs.cpuc.ca.gov/SearchRes.aspx?DocID=123")
        'document_a1b2c3d4.pdf'
    """
    parsed = urlparse(url)
    
    # Try to get filename from path
    if parsed.path and parsed.path.endswith('.pdf'):
        return parsed.path.split('/')[-1]
    
    # Generate filename from URL hash
    url_hash = get_url_hash(url)[:8]
    return f"document_{url_hash}.pdf"


def extract_date_from_content(content: str) -> Optional[datetime]:
    """
    Extract date from document content using various patterns.
    
    This function searches through document content to find dates in multiple
    formats commonly used in regulatory documents. It supports both written
    and numerical date formats.
    
    Args:
        content (str): The document content to search for dates
        
    Returns:
        Optional[datetime]: The first valid date found in the content,
                           or None if no valid dates are found
                           
    Supported Formats:
        - "December 15, 2023" or "Dec 15, 2023" (month name)
        - "2023-12-15" or "2023/12/15" (ISO format)
        - "12/15/2023" or "12-15-2023" (US format)
        - "15 December 2023" (day-month-year)
        
    Examples:
        >>> extract_date_from_content("Filed on December 15, 2023")
        datetime.datetime(2023, 12, 15, 0, 0)
        
        >>> extract_date_from_content("No date here")
        None
    """
    date_patterns = [
        r'(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\s,]+(\d{1,2})[\s,]+(\d{4})',
        r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',
        r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',
        r'(\d{1,2})\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})',
    ]

    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    for pattern in date_patterns:
        matches = re.finditer(pattern, content.lower())
        for match in matches:
            try:
                groups = match.groups()
                if len(groups) == 3:
                    if pattern == date_patterns[0]:  # Month name pattern
                        month_str = content[match.start():match.end()].split()[0].lower()
                        month = month_names.get(month_str[:3])
                        if month:
                            day, year = int(groups[0]), int(groups[1])
                            return datetime(year, month, day)
                    elif pattern == date_patterns[1]:  # YYYY-MM-DD
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        return datetime(year, month, day)
                    elif pattern == date_patterns[2]:  # MM/DD/YYYY
                        month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                        return datetime(year, month, day)
                elif len(groups) == 2:  # DD Month YYYY
                    day, year = int(groups[0]), int(groups[1])
                    month_str = content[match.start():match.end()].split()[1].lower()
                    month = month_names.get(month_str[:3])
                    if month:
                        return datetime(year, month, day)
            except (ValueError, IndexError):
                continue

    return None


def extract_proceeding_number(content: str) -> Optional[str]:
    """Extract proceeding number from document content."""
    patterns = [
        r'(?:proceeding|application|rulemaking|investigation)\s+(?:no\.?\s*)?([A-Z]\.?\d{2}-\d{2}-\d{3})',
        r'([A-Z]\.?\d{2}-\d{2}-\d{3})',
        r'(?:docket|case)\s+(?:no\.?\s*)?([A-Z]\.?\d{2}-\d{2}-\d{3})',
    ]

    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def identify_document_type(content: str, filename: str) -> str:
    """Identify the type of regulatory document."""
    filename_lower = filename.lower()
    content_lower = content.lower()

    if any(term in filename_lower for term in ['decision', 'order']):
        return 'decision'
    elif any(term in filename_lower for term in ['ruling', 'resolution']):
        return 'ruling'
    elif any(term in filename_lower for term in ['application', 'petition']):
        return 'application'
    elif any(term in filename_lower for term in ['proposal', 'draft']):
        return 'proposal'

    if any(phrase in content_lower for phrase in ['it is ordered', 'this decision', 'we conclude']):
        return 'decision'
    elif any(phrase in content_lower for phrase in ['administrative law judge', 'ruling', 'prehearing']):
        return 'ruling'
    elif any(phrase in content_lower for phrase in ['application for', 'petition for']):
        return 'application'

    return 'unknown'


def _calculate_supersedes_priority(doc_type: str, doc_date: Optional[datetime]) -> float:
    """Calculate priority score for superseding logic."""
    base_scores = {
        'decision': 1.0,
        'ruling': 0.8,
        'resolution': 0.7,
        'application': 0.3,
        'proposal': 0.1,
        'unknown': 0.5
    }

    base_score = base_scores.get(doc_type, 0.5)

    if doc_date:
        days_old = (datetime.now() - doc_date).days
        recency_bonus = max(0.0, 1.0 - (days_old / 3650.0))  # Decay over 10 years
        return base_score + (recency_bonus * 0.5)

    return base_score

def _filter_superseded_documents(documents: List[Document]) -> List[Document]:
    """
    Filter out superseded documents based on document type and date.
    
    This function implements superseding logic where newer decisions
    supersede older proposals and rulings on the same topic.
    
    Args:
        documents: List of documents to filter
        
    Returns:
        List of documents with superseded ones filtered out
    """
    if not documents:
        return []
    
    # For now, implement a simple filter that removes obviously superseded documents
    # In the future, this could be enhanced with more sophisticated superseding logic
    
    # Group documents by source to find superseding relationships
    source_groups = {}
    for doc in documents:
        source = doc.metadata.get('source', 'unknown')
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(doc)
    
    # For each source group, keep the document with highest supersedes_priority
    filtered_docs = []
    for source, docs in source_groups.items():
        if len(docs) == 1:
            filtered_docs.extend(docs)
        else:
            # Find document with highest priority
            best_doc = max(docs, key=lambda d: d.metadata.get('supersedes_priority', 0.5))
            filtered_docs.append(best_doc)
    
    return filtered_docs

def get_source_url_from_filename(filename: str, proceeding: str = None) -> Optional[str]:
    """
    Maps a local PDF filename to its original CPUC URL using download history.
    
    This function provides the crucial link between local PDF files and their
    original CPUC URLs, enabling direct citation linking without guesswork.
    
    Args:
        filename (str): Local PDF filename (with or without extension)
        proceeding (str, optional): Proceeding ID (e.g., 'R2207005'). Uses default from config if not provided.
        
    Returns:
        Optional[str]: Original CPUC URL if found, None otherwise
        
    Examples:
        >>> get_source_url_from_filename("Comments filed by Microgrid Resources Coalition on 01_04_2023 Conf# 188758.pdf")
        'https://docs.cpuc.ca.gov/SearchRes.aspx?DocFormat=ALL&DocID=500762062'
        
        >>> get_source_url_from_filename("nonexistent.pdf")
        None
    """
    try:
        # Load download history
        import json
        from pathlib import Path
        
        if proceeding is None:
            from config import DEFAULT_PROCEEDING
            proceeding = DEFAULT_PROCEEDING
        
        from config import get_proceeding_file_paths
        proceeding_paths = get_proceeding_file_paths(proceeding)
        
        history_file = proceeding_paths['scraped_pdf_history']
        if not history_file.exists():
            logger.warning(f"Scraped PDF history file not found: {history_file}")
            return None
            
        with open(history_file, 'r') as f:
            scraped_pdf_history = json.load(f)
        
        # Normalize filename for comparison (remove extensions and extra spaces)
        filename_clean = filename.replace('.pdf', '').replace('.PDF', '').strip()
        
        # Search through download history
        for record in scraped_pdf_history.values():
            # New format uses 'title' instead of 'filename'
            recorded_filename = record.get('filename', record.get('title', '')).replace('.pdf', '').replace('.PDF', '').strip()
            if recorded_filename == filename_clean:
                url = record.get('url')
                if url:
                    logger.debug(f"Found source URL for {filename}: {url}")
                    return url
        
        # If exact match fails, try partial matching for common filename variations
        for record in scraped_pdf_history.values():
            # New format uses 'title' instead of 'filename'
            recorded_filename = record.get('filename', record.get('title', '')).replace('.pdf', '').replace('.PDF', '').strip()
            # Try both directions of partial matching
            if (filename_clean in recorded_filename or 
                recorded_filename in filename_clean or
                # Handle special cases like numbered files
                (filename_clean.replace('_', ' ') == recorded_filename.replace('_', ' '))):
                url = record.get('url')
                if url:
                    logger.debug(f"Found source URL via partial match for {filename}: {url}")
                    return url
        
        logger.debug(f"No source URL found for filename: {filename}")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to lookup source URL for {filename}: {e}")
        return None

def get_publication_date_from_filename(filename: str, proceeding: str = None) -> Optional[datetime]:
    """
    Extract publication date from download history based on filename.
    
    This function looks up the publication date for a given PDF filename
    using the download history records, which often contain metadata
    about when documents were actually published or filed.
    
    Args:
        filename (str): Local PDF filename (with or without extension)
        proceeding (str, optional): Proceeding ID (e.g., 'R2207005'). Uses default from config if not provided.
        
    Returns:
        Optional[datetime]: Publication date if found, None otherwise
        
    Examples:
        >>> get_publication_date_from_filename("D2504045 Order.pdf")
        datetime.datetime(2025, 4, 15, 0, 0)
        
        >>> get_publication_date_from_filename("nonexistent.pdf")
        None
    """
    try:
        # Load download history
        import json
        from pathlib import Path
        
        if proceeding is None:
            from config import DEFAULT_PROCEEDING
            proceeding = DEFAULT_PROCEEDING
        
        from config import get_proceeding_file_paths
        proceeding_paths = get_proceeding_file_paths(proceeding)
        
        history_file = proceeding_paths['scraped_pdf_history']
        if not history_file.exists():
            logger.warning(f"Scraped PDF history file not found: {history_file}")
            return None
            
        with open(history_file, 'r') as f:
            scraped_pdf_history = json.load(f)
        
        # Normalize filename for comparison
        filename_clean = filename.replace('.pdf', '').replace('.PDF', '').strip()
        
        # Search through download history
        for record in scraped_pdf_history.values():
            # New format uses 'title' instead of 'filename'
            recorded_filename = record.get('filename', record.get('title', '')).replace('.pdf', '').replace('.PDF', '').strip()
            if recorded_filename == filename_clean:
                # Check for publication date in the record (new format uses 'filing_date', 'pdf_creation_date')
                pub_date = (record.get('filing_date') or 
                           record.get('pdf_creation_date') or 
                           record.get('publication_date'))
                if pub_date:
                    try:
                        # Handle both ISO format and MM/DD/YYYY format
                        if '/' in pub_date:
                            return datetime.strptime(pub_date, '%m/%d/%Y')
                        else:
                            return datetime.fromisoformat(pub_date)
                    except ValueError:
                        logger.warning(f"Invalid date format in download history for {filename}: {pub_date}")
                
                # If no explicit publication date, try to extract from scrape_date
                scrape_date = record.get('scrape_date')
                if scrape_date:
                    try:
                        if '/' in scrape_date:
                            return datetime.strptime(scrape_date, '%m/%d/%Y')
                        else:
                            return datetime.fromisoformat(scrape_date)
                    except ValueError:
                        logger.warning(f"Invalid scrape date format for {filename}: {scrape_date}")
        
        # If exact match fails, try partial matching
        for record in scraped_pdf_history.values():
            # New format uses 'title' instead of 'filename'
            recorded_filename = record.get('filename', record.get('title', '')).replace('.pdf', '').replace('.PDF', '').strip()
            if (filename_clean in recorded_filename or 
                recorded_filename in filename_clean or
                (filename_clean.replace('_', ' ') == recorded_filename.replace('_', ' '))):
                
                # Check for publication date in the record (new format uses 'filing_date', 'pdf_creation_date')
                pub_date = (record.get('filing_date') or 
                           record.get('pdf_creation_date') or 
                           record.get('publication_date'))
                if pub_date:
                    try:
                        # Handle both ISO format and MM/DD/YYYY format
                        if '/' in pub_date:
                            return datetime.strptime(pub_date, '%m/%d/%Y')
                        else:
                            return datetime.fromisoformat(pub_date)
                    except ValueError:
                        continue
                
                # Try scrape_date for partial matches too
                scrape_date = record.get('scrape_date')
                if scrape_date:
                    try:
                        if '/' in scrape_date:
                            return datetime.strptime(scrape_date, '%m/%d/%Y')
                        else:
                            return datetime.fromisoformat(scrape_date)
                    except ValueError:
                        continue
        
        logger.debug(f"No publication date found for filename: {filename}")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to lookup publication date for {filename}: {e}")
        return None

def extract_and_chunk_with_docling(pdf_path: Path, proceeding: str = None) -> List[Document]:
    """
    Enhanced document processing that extracts content, dates, and proceeding info.
    
    This function processes a PDF file using Docling to extract structured content,
    including text, tables, and metadata. It enhances the extraction with document-level
    information like dates, proceeding numbers, and document types for better organization
    and retrieval.
    
    Args:
        pdf_path (Path): Path to the PDF file to process
        
    Returns:
        List[Document]: A list of LangChain Document objects containing:
                       - page_content: The extracted text or table content
                       - metadata: Enhanced metadata including source, page, document type,
                                 date, proceeding number, and supersedes priority
                                 
    Processing Steps:
        1. Convert PDF using Docling with enhanced parsing
        2. Extract document-level metadata from first few pages
        3. Process all content items (text, tables, etc.)
        4. Enhance each chunk with comprehensive metadata
        
    Metadata Fields:
        - source: Original filename
        - page: Page number in document
        - content_type: Type of content (text, table, etc.)
        - document_date: Extracted date from document
        - proceeding_number: CPUC proceeding identifier
        - document_type: Classified document type (decision, ruling, etc.)
        - supersedes_priority: Priority score for superseding logic
        
    Examples:
        >>> docs = extract_and_chunk_with_docling(Path("decision.pdf"))
        >>> len(docs)
        25
        >>> docs[0].metadata['document_type']
        'decision'
    """
    logger.info(f"Processing with enhanced Docling workflow: {pdf_path.name}")
    langchain_documents = []

    try:
        conv_results = doc_converter.convert_all([pdf_path], raises_on_error=False)
        conv_res = next(iter(conv_results), None)

        if not conv_res or conv_res.status == ConversionStatus.FAILURE:
            logger.error(f"Docling failed to convert document: {pdf_path.name}")
            return []

        docling_doc = conv_res.document

        first_page_content = ""
        page_count = 0

        for item, level in docling_doc.iterate_items(with_groups=False):
            if isinstance(item, DocItem) and hasattr(item, 'text') and item.text:
                page_num = item.prov[0].page_no + 1 if item.prov else 1
                if page_num <= 3:  # First 3 pages
                    first_page_content += item.text + " "
                    page_count = max(page_count, page_num)
                if page_count >= 3:
                    break

        doc_date = extract_date_from_content(first_page_content)
        proceeding_number = extract_proceeding_number(first_page_content)
        doc_type = identify_document_type(first_page_content, pdf_path.name)
        
        # Get source URL from download history for direct citation linking
        source_url = get_source_url_from_filename(pdf_path.name, proceeding)
        
        # Get publication date from download history (preferred over content extraction)
        publication_date = get_publication_date_from_filename(pdf_path.name, proceeding)
        if publication_date and not doc_date:
            doc_date = publication_date
            logger.info(f"Using publication date from download history: {publication_date}")
        elif publication_date and doc_date:
            # Use publication date if it's different from content date
            if abs((publication_date - doc_date).days) > 1:
                logger.info(f"Publication date ({publication_date}) differs from content date ({doc_date}), using publication date")
                doc_date = publication_date

        # ENHANCEMENT: Build page content map for line number estimation
        page_content_map = {}
        if config.LINE_LEVEL_CITATIONS_ENABLED:
            page_content_map = extract_page_content_map(docling_doc)
            logger.debug(f"Built page content map for line number estimation: {len(page_content_map)} pages")

        logger.info(f"Document metadata - Date: {doc_date}, Proceeding: {proceeding_number}, Type: {doc_type}, Source URL: {source_url is not None}")

        for item, level in docling_doc.iterate_items(with_groups=False):
            if not isinstance(item, DocItem):
                continue

            content = ""
            if isinstance(item, TableItem):
                content = item.export_to_markdown(doc=docling_doc)
            elif hasattr(item, 'text'):
                content = item.text

            if content and content.strip():
                page_num = item.prov[0].page_no + 1 if item.prov else 0

                # ENHANCEMENT: Calculate line number information
                line_number = 1
                line_range_end = 1
                if config.LINE_LEVEL_CITATIONS_ENABLED and page_num in page_content_map:
                    line_number = estimate_line_number_from_content(content, page_content_map.get(page_num))
                    start_line, end_line = calculate_line_range_for_chunk(content, page_content_map.get(page_num))
                    line_number = start_line
                    line_range_end = end_line

                metadata = {
                    "source": pdf_path.name,
                    "source_url": source_url,  # Direct URL linkage for perfect citations
                    "page": page_num,
                    "line_number": line_number,  # ENHANCEMENT: Line number within page
                    "line_range_end": line_range_end,  # ENHANCEMENT: End line for multi-line content
                    "content_type": item.label.value,
                    "chunk_id": f"{pdf_path.name}_{item.get_ref().cref.replace('#/', '').replace('/', '_')}",
                    "file_path": str(pdf_path),
                    "last_modified": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat(),

                    # Convert None values to empty strings to avoid PyArrow casting issues
                    "document_date": doc_date.isoformat() if doc_date else "",
                    "publication_date": publication_date.isoformat() if publication_date else "",
                    "proceeding_number": proceeding_number or "",
                    "document_type": doc_type or "unknown",
                    "supersedes_priority": _calculate_supersedes_priority(doc_type, doc_date),
                }

                langchain_documents.append(Document(page_content=content, metadata=metadata))

    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        return []

    logger.info(f"Extracted {len(langchain_documents)} chunks from {pdf_path.name}")
    return langchain_documents

def extract_and_chunk_with_docling_url(pdf_url: str, document_title: str = None, proceeding: str = None, 
                                     enable_ocr_fallback: bool = True, enable_chonkie_fallback: bool = None,
                                     use_intelligent_hybrid: bool = None) -> List[Document]:
    """
    Enhanced document processing using Docling with URL-based PDF processing.
    
    This function processes a PDF directly from a URL using Docling's URL processing
    capabilities. It extracts structured content including text, tables, and metadata
    without requiring local file storage. The function maintains all the document
    analysis features of the file-based version while working entirely with URLs.
    
    Args:
        pdf_url (str): URL of the PDF document to process
        document_title (str, optional): Document title for metadata (extracted from URL if None)
        
    Returns:
        List[Document]: A list of LangChain Document objects containing:
                       - page_content: The extracted text or table content
                       - metadata: Enhanced metadata including source URL, page, document type,
                                 date, proceeding number, and supersedes priority
                                 
    Processing Steps:
        1. Validate URL accessibility
        2. Convert PDF using Docling's URL processing
        3. Extract document-level metadata from first few pages
        4. Process all content items (text, tables, etc.)
        5. Enhance each chunk with comprehensive metadata including URL references
        
    Metadata Fields:
        - source_url: Original PDF URL
        - source: Document filename (extracted from URL or title)
        - page: Page number in document
        - content_type: Type of content (text, table, etc.)
        - document_date: Extracted date from document
        - proceeding_number: CPUC proceeding identifier
        - document_type: Classified document type (decision, ruling, etc.)
        - supersedes_priority: Priority score for superseding logic
        - url_hash: SHA-256 hash of the URL for tracking
        
    Examples:
        >>> docs = extract_and_chunk_with_docling_url("https://docs.cpuc.ca.gov/decision.pdf")
        >>> len(docs)
        25
        >>> docs[0].metadata['source_url']
        'https://docs.cpuc.ca.gov/decision.pdf'
        >>> docs[0].metadata['document_type']
        'decision'
    """
    logger.info(f"Processing PDF from URL: {pdf_url}")
    
    # Set defaults from config if not specified
    if use_intelligent_hybrid is None:
        use_intelligent_hybrid = config.INTELLIGENT_HYBRID_ENABLED
    if enable_chonkie_fallback is None:
        enable_chonkie_fallback = config.CHONKIE_FALLBACK_ENABLED
    
    # INTELLIGENT HYBRID PROCESSING DECISION
    if use_intelligent_hybrid:
        logger.info("Intelligent hybrid processing enabled - analyzing document...")
        
        # Detect if document likely contains tables/financial content
        detection_score = detect_table_financial_document(pdf_url, document_title)
        logger.info(f"Table/financial document detection score: {detection_score:.3f}")
        
        # If score is below threshold, use Chonkie as primary method
        if detection_score < config.HYBRID_TRIGGER_THRESHOLD:
            logger.info(f"Score {detection_score:.3f} < {config.HYBRID_TRIGGER_THRESHOLD} - using Chonkie as primary method")
            return _process_with_chonkie_primary(pdf_url, document_title, proceeding, 
                                               enable_ocr_fallback, enable_chonkie_fallback)
        else:
            logger.info(f"Score {detection_score:.3f} >= {config.HYBRID_TRIGGER_THRESHOLD} - using hybrid evaluation")
            return _process_with_hybrid_evaluation(pdf_url, document_title, proceeding, detection_score,
                                                 enable_ocr_fallback, enable_chonkie_fallback)
    
    # Validate URL first
    if not validate_pdf_url(pdf_url):
        logger.error(f"URL validation failed for: {pdf_url}")
        return []
    
    # If intelligent hybrid processing is disabled, use standard Docling processing
    return _process_with_standard_docling(pdf_url, document_title, proceeding, 
                                        enable_ocr_fallback, enable_chonkie_fallback)

def _extract_with_ocr_fallback(pdf_url: str, source_name: str, doc_date, publication_date, 
                              proceeding_number: str, doc_type: str, url_hash: str, proceeding: str) -> List[Document]:
    """
    OCR fallback for scanned PDFs that extract 0 chunks with standard processing.
    
    This function uses Docling with OCR enabled to extract text from image-based PDFs.
    """
    from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
    from docling.datamodel.base_models import InputFormat
    
    logger.info(f"Attempting OCR extraction for: {pdf_url}")
    
    try:
        # Configure pipeline with OCR enabled
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.ocr_options = EasyOcrOptions()
        
        # Create converter with OCR enabled
        ocr_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    backend=DoclingParseV4DocumentBackend,
                    pipeline_options=pipeline_options
                )
            }
        )
        
        # Process with OCR
        conv_results = ocr_converter.convert_all([pdf_url], raises_on_error=False)
        conv_res = next(iter(conv_results), None)

        if not conv_res or conv_res.status.name != 'SUCCESS':
            logger.warning(f"OCR conversion failed for: {pdf_url}")
            return []

        docling_doc = conv_res.document
        langchain_documents = []

        # Process OCR results
        for item, level in docling_doc.iterate_items(with_groups=False):
            from docling.datamodel.document import DocItem, TableItem
            
            if not isinstance(item, DocItem):
                continue

            content = ""
            if isinstance(item, TableItem):
                content = item.export_to_markdown(doc=docling_doc)
            elif hasattr(item, 'text'):
                content = item.text

            if content and content.strip():
                page_num = item.prov[0].page_no + 1 if item.prov else 0

                metadata = {
                    "source_url": pdf_url,
                    "source": source_name,
                    "page": page_num,
                    "content_type": f"{item.label.value}_ocr",  # Mark as OCR-extracted
                    "chunk_id": f"{source_name}_{item.get_ref().cref.replace('#/', '').replace('/', '_')}_ocr",
                    "url_hash": url_hash,
                    "last_checked": datetime.now().isoformat(),
                    "extraction_method": "ocr_fallback",

                    # Convert None values to empty strings to avoid PyArrow casting issues
                    "document_date": doc_date.isoformat() if doc_date else "",
                    "publication_date": publication_date.isoformat() if publication_date else "",
                    "proceeding_number": proceeding_number or "",
                    "document_type": doc_type or "unknown",
                    "supersedes_priority": _calculate_supersedes_priority(doc_type, doc_date),
                }

                langchain_documents.append(Document(page_content=content, metadata=metadata))

        logger.info(f"OCR extraction completed: {len(langchain_documents)} chunks")
        return langchain_documents

    except Exception as e:
        logger.error(f"OCR fallback failed for {pdf_url}: {e}")
        return []

def _extract_with_chonkie_fallback(pdf_url: str, source_name: str, doc_date, publication_date, 
                                  proceeding_number: str, doc_type: str, url_hash: str, proceeding: str) -> List[Document]:
    """
    Chonkie fallback for PDFs that fail both standard and OCR processing.
    
    This function attempts to extract raw text from PDFs and chunk it using Chonkie
    when structured parsing with Docling fails completely.
    """
    logger.info(f"Attempting Chonkie fallback extraction for: {pdf_url}")
    
    try:
        # Extract raw text from PDF
        raw_text = extract_text_from_url(pdf_url)
        
        if not raw_text or len(raw_text.strip()) < config.CHONKIE_MIN_TEXT_LENGTH:
            logger.warning(f"Insufficient text extracted for Chonkie processing: {len(raw_text)} chars")
            return []
        
        logger.info(f"Extracted {len(raw_text)} characters of raw text for Chonkie processing")
        
        # Try multiple chunking strategies in order of preference
        for strategy in config.CHONKIE_STRATEGIES:
            try:
                logger.info(f"Attempting Chonkie {strategy} chunking")
                enhanced_chunks = safe_chunk_with_chonkie_enhanced(raw_text, strategy)
                
                if enhanced_chunks and len(enhanced_chunks) > 0:
                    logger.info(f"Chonkie {strategy} chunking successful: {len(enhanced_chunks)} chunks")
                    
                    # Convert to LangChain Documents with enhanced position-based metadata
                    langchain_documents = []
                    for chunk_info in enhanced_chunks:
                        # Create enhanced metadata using character positions
                        enhanced_metadata = create_enhanced_chonkie_metadata(
                            chunk_info, source_name, pdf_url, proceeding, raw_text
                        )
                        
                        # Add document-specific metadata
                        enhanced_metadata.update({
                            "url_hash": url_hash,
                            "last_checked": datetime.now().isoformat(),
                            "document_date": doc_date.isoformat() if doc_date else "",
                            "publication_date": publication_date.isoformat() if publication_date else "",
                            "proceeding_number": proceeding_number or "",
                            "document_type": doc_type or "unknown",
                            "supersedes_priority": _calculate_supersedes_priority(doc_type, doc_date),
                        })
                        
                        langchain_documents.append(Document(
                            page_content=chunk_info['text'], 
                            metadata=enhanced_metadata
                        ))
                    
                    logger.info(f"Created {len(langchain_documents)} LangChain documents via Chonkie {strategy}")
                    return langchain_documents
                    
            except Exception as strategy_error:
                logger.warning(f"Chonkie {strategy} chunking failed: {strategy_error}")
                continue
        
        logger.warning("All Chonkie chunking strategies failed")
        return []
        
    except Exception as e:
        logger.error(f"Chonkie fallback completely failed for {pdf_url}: {e}")
        return []

def _process_with_chonkie_primary(pdf_url: str, document_title: str = None, proceeding: str = None,
                                enable_ocr_fallback: bool = True, enable_chonkie_fallback: bool = True) -> List[Document]:
    """
    Process document using Chonkie as the primary method.
    
    For documents with low table/financial content scores, use Chonkie directly
    as it's faster and produces comparable results for text-heavy documents.
    """
    logger.info(f"Processing with Chonkie as primary method: {pdf_url}")
    
    # Extract document metadata first
    source_name = document_title or extract_filename_from_url(pdf_url)
    url_hash = get_url_hash(pdf_url)
    
    # Try to get some basic document info for metadata
    doc_date = None
    proceeding_number = proceeding or ""
    doc_type = "unknown"
    publication_date = None
    
    try:
        # Quick content sample for metadata extraction
        response = requests.get(pdf_url, timeout=30, headers={'Range': 'bytes=0-16384'})
        if response.status_code in [200, 206]:
            # Try basic text extraction for metadata
            sample_text = extract_text_from_url(pdf_url)[:1000] if response.status_code == 200 else ""
            if sample_text:
                doc_date = extract_date_from_content(sample_text)
                proceeding_number = extract_proceeding_number(sample_text) or proceeding or ""
                doc_type = identify_document_type(sample_text, source_name)
                
        # Try to get publication date from download history
        publication_date = get_publication_date_from_filename(source_name, proceeding)
        if publication_date and not doc_date:
            doc_date = publication_date
            
    except Exception as e:
        logger.warning(f"Failed to extract basic metadata for Chonkie processing: {e}")
    
    # Process with Chonkie fallback method
    chonkie_result = _extract_with_chonkie_fallback(pdf_url, source_name, doc_date, publication_date,
                                                   proceeding_number, doc_type, url_hash, proceeding)
    
    if chonkie_result:
        logger.info(f"Chonkie primary processing successful: {len(chonkie_result)} chunks")
        return chonkie_result
    else:
        logger.warning("Chonkie primary processing failed - falling back to enhanced Docling")
        # Fallback to enhanced Docling processing with character position metadata
        from enhanced_docling_fallback import _process_with_enhanced_docling_fallback
        enhanced_result = _process_with_enhanced_docling_fallback(pdf_url, document_title, proceeding, enable_ocr_fallback)
        
        if enhanced_result:
            logger.info(f"Enhanced Docling fallback successful: {len(enhanced_result)} chunks")
            return enhanced_result
        else:
            logger.warning("Enhanced Docling fallback also failed - using standard Docling as final fallback")
            # Final fallback to standard Docling processing
            return _process_with_standard_docling(pdf_url, document_title, proceeding, 
                                                enable_ocr_fallback, enable_chonkie_fallback)


def _process_with_hybrid_evaluation(pdf_url: str, document_title: str = None, proceeding: str = None, 
                                   detection_score: float = 0.0, enable_ocr_fallback: bool = True, 
                                   enable_chonkie_fallback: bool = True) -> List[Document]:
    """
    Process document using hybrid evaluation with agent comparison.
    
    For documents with high table/financial content scores, run both Docling and Chonkie,
    then use an agent to evaluate which result is better.
    """
    logger.info(f"Processing with hybrid evaluation: {pdf_url}")
    
    if not config.AGENT_EVALUATION_ENABLED:
        logger.info("Agent evaluation disabled - falling back to standard Docling")
        return _process_with_standard_docling(pdf_url, document_title, proceeding, 
                                            enable_ocr_fallback, enable_chonkie_fallback)
    
    # Run both processing methods
    start_time = time.time()
    
    # Process with Docling
    logger.info("Running Docling processing for hybrid evaluation...")
    docling_start = time.time()
    docling_result = _process_with_standard_docling(pdf_url, document_title, proceeding, 
                                                  enable_ocr_fallback, False)  # Disable Chonkie fallback for Docling
    docling_time = time.time() - docling_start
    
    # Process with Chonkie
    logger.info("Running Chonkie processing for hybrid evaluation...")
    chonkie_start = time.time()
    chonkie_result = _process_with_chonkie_primary(pdf_url, document_title, proceeding, 
                                                 enable_ocr_fallback, enable_chonkie_fallback)
    chonkie_time = time.time() - chonkie_start
    
    # Analyze results
    docling_analysis = {
        'success': len(docling_result) > 0,
        'processing_time': docling_time,
        'chunk_count': len(docling_result),
        'content_length': sum(len(doc.page_content) for doc in docling_result),
        'content_types': _analyze_content_types(docling_result),
        'tables_found': _count_tables(docling_result),
        'error': 'None' if len(docling_result) > 0 else 'No chunks extracted'
    }
    
    chonkie_analysis = {
        'success': len(chonkie_result) > 0,
        'processing_time': chonkie_time,
        'chunk_count': len(chonkie_result),
        'content_length': sum(len(doc.page_content) for doc in chonkie_result),
        'strategy_used': _get_chonkie_strategy(chonkie_result[0]) if chonkie_result else 'None',
        'text_quality': _estimate_text_quality(chonkie_result),
        'error': 'None' if len(chonkie_result) > 0 else 'No chunks extracted'
    }
    
    # Use agent to evaluate which result is better
    agent_decision, agent_reasoning = _evaluate_with_agent(docling_analysis, chonkie_analysis, detection_score)
    
    # Log the evaluation decision
    create_agent_evaluation_log(pdf_url, document_title or "Unknown", detection_score,
                               docling_analysis, chonkie_analysis, agent_decision, agent_reasoning)
    
    # Return the chosen result
    if agent_decision.lower().startswith('docling'):
        logger.info(f"Agent chose Docling: {agent_reasoning[:100]}...")
        return docling_result
    else:
        logger.info(f"Agent chose Chonkie: {agent_reasoning[:100]}...")
        return chonkie_result


def _process_with_standard_docling(pdf_url: str, document_title: str = None, proceeding: str = None,
                                 enable_ocr_fallback: bool = True, enable_chonkie_fallback: bool = True) -> List[Document]:
    """
    Standard Docling processing (the original implementation).
    """
    # Validate URL first
    if not validate_pdf_url(pdf_url):
        logger.error(f"URL validation failed for: {pdf_url}")
        return []
    
    langchain_documents = []
    
    try:
        # Use Docling's URL processing capability
        conv_results = doc_converter.convert_all([pdf_url], raises_on_error=False)
        conv_res = next(iter(conv_results), None)

        if not conv_res or conv_res.status == ConversionStatus.FAILURE:
            logger.error(f"Docling failed to convert document from URL: {pdf_url}")
            return []

        docling_doc = conv_res.document
        
        # Extract filename from URL or use provided title
        if document_title:
            source_name = f"{document_title}.pdf"
        else:
            source_name = extract_filename_from_url(pdf_url)

        # Extract document-level metadata from first few pages
        first_page_content = ""
        page_count = 0

        for item, level in docling_doc.iterate_items(with_groups=False):
            if isinstance(item, DocItem) and hasattr(item, 'text') and item.text:
                page_num = item.prov[0].page_no + 1 if item.prov else 1
                if page_num <= 3:  # First 3 pages
                    first_page_content += item.text + " "
                    page_count = max(page_count, page_num)
                if page_count >= 3:
                    break

        # Extract document-level information
        doc_date = extract_date_from_content(first_page_content)
        proceeding_number = extract_proceeding_number(first_page_content)
        doc_type = identify_document_type(first_page_content, source_name)
        url_hash = get_url_hash(pdf_url)
        
        # Try to get publication date from download history using the source name
        publication_date = get_publication_date_from_filename(source_name, proceeding)
        if publication_date and not doc_date:
            doc_date = publication_date
            logger.info(f"Using publication date from download history: {publication_date}")
        elif publication_date and doc_date:
            # Use publication date if it's different from content date
            if abs((publication_date - doc_date).days) > 1:
                logger.info(f"Publication date ({publication_date}) differs from content date ({doc_date}), using publication date")
                doc_date = publication_date

        # ENHANCEMENT: Build page content map for line number estimation
        page_content_map = {}
        if config.LINE_LEVEL_CITATIONS_ENABLED:
            page_content_map = extract_page_content_map(docling_doc)
            logger.debug(f"Built page content map for line number estimation: {len(page_content_map)} pages")

        logger.info(f"Document metadata - Date: {doc_date}, Proceeding: {proceeding_number}, Type: {doc_type}")

        # Process all content items
        for item, level in docling_doc.iterate_items(with_groups=False):
            if not isinstance(item, DocItem):
                continue

            content = ""
            if isinstance(item, TableItem):
                content = item.export_to_markdown(doc=docling_doc)
            elif hasattr(item, 'text'):
                content = item.text

            if content and content.strip():
                page_num = item.prov[0].page_no + 1 if item.prov else 0

                metadata = {
                    "source_url": pdf_url,
                    "source": source_name,
                    "page": page_num,
                    "content_type": item.label.value,
                    "chunk_id": f"{source_name}_{item.get_ref().cref.replace('#/', '').replace('/', '_')}",
                    "url_hash": url_hash,
                    "last_checked": datetime.now().isoformat(),

                    # Convert None values to empty strings to avoid PyArrow casting issues
                    "document_date": doc_date.isoformat() if doc_date else "",
                    "publication_date": publication_date.isoformat() if publication_date else "",
                    "proceeding_number": proceeding_number or "",
                    "document_type": doc_type or "unknown",
                    "supersedes_priority": _calculate_supersedes_priority(doc_type, doc_date),
                }

                langchain_documents.append(Document(page_content=content, metadata=metadata))

    except Exception as e:
        logger.error(f"Error processing URL {pdf_url}: {e}", exc_info=True)
        return []

    # OCR fallback for scanned PDFs with 0 chunks
    if len(langchain_documents) == 0 and enable_ocr_fallback:
        logger.warning(f"PDF extracted 0 chunks, attempting OCR fallback: {pdf_url}")
        try:
            source_name = document_title or extract_filename_from_url(pdf_url)
            url_hash = get_url_hash(pdf_url)
            ocr_chunks = _extract_with_ocr_fallback(pdf_url, source_name, doc_date, publication_date, 
                                                   proceeding_number, doc_type, url_hash, proceeding)
            if ocr_chunks:
                logger.info(f"OCR fallback successful: {len(ocr_chunks)} chunks extracted")
                return ocr_chunks
            else:
                logger.warning(f"OCR fallback also extracted 0 chunks")
        except Exception as ocr_e:
            logger.warning(f"OCR fallback failed: {ocr_e}")

    # Enhanced fallback strategy for PDFs that fail both standard and OCR processing
    if len(langchain_documents) == 0 and enable_chonkie_fallback:
        logger.warning(f"OCR fallback also failed, attempting enhanced fallback methods: {pdf_url}")
        
        # First try Chonkie text extraction fallback
        try:
            source_name = document_title or extract_filename_from_url(pdf_url)
            url_hash = get_url_hash(pdf_url)
            chonkie_chunks = _extract_with_chonkie_fallback(pdf_url, source_name, doc_date, publication_date,
                                                           proceeding_number, doc_type, url_hash, proceeding)
            if chonkie_chunks:
                logger.info(f"Chonkie fallback successful: {len(chonkie_chunks)} chunks extracted")
                return chonkie_chunks
            else:
                logger.warning(f"Chonkie fallback also extracted 0 chunks")
        except Exception as chonkie_e:
            logger.warning(f"Chonkie fallback failed: {chonkie_e}")
        
        # If Chonkie fallback fails, try enhanced Docling fallback
        logger.warning(f"Chonkie fallback failed, attempting enhanced Docling fallback: {pdf_url}")
        try:
            from enhanced_docling_fallback import _process_with_enhanced_docling_fallback
            enhanced_chunks = _process_with_enhanced_docling_fallback(pdf_url, document_title, proceeding, False)
            if enhanced_chunks:
                logger.info(f"Enhanced Docling fallback successful: {len(enhanced_chunks)} chunks extracted")
                return enhanced_chunks
            else:
                logger.warning(f"Enhanced Docling fallback also extracted 0 chunks - PDF may be completely malformed")
        except Exception as enhanced_e:
            logger.warning(f"Enhanced Docling fallback failed: {enhanced_e}")

    # Final fallback - create placeholder if everything fails
    if len(langchain_documents) == 0:
        logger.warning(f"All extraction methods failed - creating placeholder: {pdf_url}")
        source_name = document_title or extract_filename_from_url(pdf_url)
        url_hash = get_url_hash(pdf_url)
        return _create_placeholder_document(pdf_url, source_name, doc_date, publication_date,
                                          proceeding_number, doc_type, url_hash)

    logger.info(f"Extracted {len(langchain_documents)} chunks from URL: {pdf_url}")
    return langchain_documents


def _analyze_content_types(documents: List[Document]) -> List[str]:
    """Analyze content types found in a list of documents."""
    content_types = {}
    for doc in documents:
        content_type = doc.metadata.get('content_type', 'unknown')
        content_types[content_type] = content_types.get(content_type, 0) + 1
    return list(content_types.keys())


def _count_tables(documents: List[Document]) -> int:
    """Count table content in a list of documents."""
    table_count = 0
    for doc in documents:
        content_type = doc.metadata.get('content_type', '')
        if 'table' in content_type.lower():
            table_count += 1
    return table_count


def _get_chonkie_strategy(document: Document) -> str:
    """Extract Chonkie strategy from document metadata."""
    content_type = document.metadata.get('content_type', '')
    if 'chonkie_' in content_type:
        return content_type.split('chonkie_')[1] if 'chonkie_' in content_type else 'unknown'
    return 'unknown'


def _estimate_text_quality(documents: List[Document]) -> float:
    """Estimate text quality of Chonkie-processed documents."""
    if not documents:
        return 0.0
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    if total_chars == 0:
        return 0.0
    
    # Simple heuristic: ratio of alphanumeric characters to total characters
    alphanumeric_chars = sum(sum(c.isalnum() for c in doc.page_content) for doc in documents)
    return min(alphanumeric_chars / total_chars, 1.0)


def _evaluate_with_agent(docling_analysis: dict, chonkie_analysis: dict, detection_score: float) -> tuple[str, str]:
    """
    Use an agent to evaluate which processing method produced better results.
    
    Returns:
        tuple: (decision, reasoning) - decision is either 'docling' or 'chonkie'
    """
    try:
        # Create evaluation prompt for the agent
        evaluation_prompt = f"""
You are an expert document processing evaluator. Your task is to determine which of two PDF processing methods produced better results for a document that has a table/financial content detection score of {detection_score:.3f}.

DOCLING RESULTS:
- Success: {docling_analysis['success']}
- Processing Time: {docling_analysis['processing_time']:.2f} seconds
- Chunks Extracted: {docling_analysis['chunk_count']}
- Content Length: {docling_analysis['content_length']} characters
- Content Types Found: {docling_analysis['content_types']}
- Tables Detected: {docling_analysis['tables_found']}
- Error: {docling_analysis['error']}

CHONKIE RESULTS:
- Success: {chonkie_analysis['success']}
- Processing Time: {chonkie_analysis['processing_time']:.2f} seconds
- Chunks Extracted: {chonkie_analysis['chunk_count']}
- Content Length: {chonkie_analysis['content_length']} characters
- Strategy Used: {chonkie_analysis['strategy_used']}
- Text Quality Score: {chonkie_analysis['text_quality']:.3f}
- Error: {chonkie_analysis['error']}

EVALUATION CRITERIA:
1. **Success Rate**: Did the method successfully extract content?
2. **Content Volume**: How much content was extracted?
3. **Structure Preservation**: For high detection scores, does the method preserve tables/financial data well?
4. **Processing Speed**: How efficiently was the document processed?
5. **Content Quality**: Is the extracted text coherent and useful?

DETECTION SCORE CONTEXT:
- Score {detection_score:.3f} indicates {"HIGH" if detection_score >= 0.3 else "LOW"} likelihood of table/financial content
- For high scores (≥0.3), prioritize structure preservation and table extraction
- For low scores (<0.3), prioritize speed and text quality

Please respond with:
DECISION: [DOCLING or CHONKIE]
REASONING: [2-3 sentences explaining your choice based on the evaluation criteria]
"""

        # Use Task tool to spawn an agent for evaluation
        from anthropic import Client
        import os
        
        # For now, implement a simple heuristic-based decision until agent integration is fully set up
        # This will be replaced with actual agent evaluation in future iterations
        
        # Decision logic based on detection score and results
        if not docling_analysis['success'] and chonkie_analysis['success']:
            decision = "CHONKIE"
            reasoning = f"Docling failed to extract content while Chonkie successfully extracted {chonkie_analysis['chunk_count']} chunks. Chonkie provides the only viable result."
        elif docling_analysis['success'] and not chonkie_analysis['success']:
            decision = "DOCLING"
            reasoning = f"Chonkie failed to extract content while Docling successfully extracted {docling_analysis['chunk_count']} chunks with {docling_analysis['tables_found']} tables detected."
        elif not docling_analysis['success'] and not chonkie_analysis['success']:
            decision = "DOCLING"
            reasoning = "Both methods failed, but Docling is chosen as the default fallback method."
        else:
            # Both succeeded - make decision based on detection score and content
            if detection_score >= 0.5:  # High confidence table/financial content
                if docling_analysis['tables_found'] > 0:
                    decision = "DOCLING"
                    reasoning = f"High table/financial content detected (score: {detection_score:.3f}). Docling found {docling_analysis['tables_found']} tables and preserves structure better for financial documents."
                elif chonkie_analysis['chunk_count'] > docling_analysis['chunk_count'] * 1.5:
                    decision = "CHONKIE"
                    reasoning = f"Despite high detection score, Chonkie extracted significantly more content ({chonkie_analysis['chunk_count']} vs {docling_analysis['chunk_count']} chunks) suggesting better text extraction."
                else:
                    decision = "DOCLING"
                    reasoning = f"High detection score ({detection_score:.3f}) suggests structured content where Docling's table extraction capabilities are preferred."
            else:  # Low confidence table/financial content
                if chonkie_analysis['processing_time'] < docling_analysis['processing_time'] * 0.5:
                    decision = "CHONKIE"
                    reasoning = f"Low table/financial score ({detection_score:.3f}) with Chonkie being {docling_analysis['processing_time']/chonkie_analysis['processing_time']:.1f}x faster makes it ideal for text-heavy documents."
                elif chonkie_analysis['content_length'] > docling_analysis['content_length'] * 1.2:
                    decision = "CHONKIE"
                    reasoning = f"Chonkie extracted {(chonkie_analysis['content_length']/docling_analysis['content_length']-1)*100:.0f}% more content, indicating better text extraction for this document type."
                else:
                    decision = "DOCLING"
                    reasoning = "Docling provides comparable results with better structure preservation, making it the safer choice."
        
        return decision, reasoning
        
    except Exception as e:
        logger.error(f"Agent evaluation failed: {e}")
        # Fallback to simple heuristic
        if docling_analysis['success'] and chonkie_analysis['success']:
            if detection_score >= 0.5:
                return "DOCLING", f"Agent evaluation failed, defaulting to Docling for high detection score ({detection_score:.3f})"
            else:
                return "CHONKIE", f"Agent evaluation failed, defaulting to Chonkie for low detection score ({detection_score:.3f})"
        elif docling_analysis['success']:
            return "DOCLING", "Agent evaluation failed, Docling succeeded while Chonkie failed"
        else:
            return "CHONKIE", "Agent evaluation failed, Chonkie succeeded while Docling failed"


def _create_placeholder_document(pdf_url: str, source_name: str, doc_date, publication_date,
                               proceeding_number: str, doc_type: str, url_hash: str) -> List[Document]:
    """
    Create a placeholder document for PDFs that fail to extract content.
    
    This ensures the PDF is tracked as processed but marked as failed extraction,
    preventing it from being repeatedly attempted.
    """
    placeholder_content = f"""
PDF Processing Failed: {source_name}

This PDF document could not be processed for text extraction:
- URL: {pdf_url}
- Document Type: {doc_type}
- Date: {doc_date.isoformat() if doc_date else 'Unknown'}
- Proceeding: {proceeding_number or 'Unknown'}

Possible reasons:
- Malformed PDF structure
- Scanned images without extractable text
- Corrupted or non-standard PDF format
- Unsupported PDF encoding

Status: Failed extraction but tracked for processing completeness.
""".strip()
    
    metadata = {
        "source_url": pdf_url,
        "source": source_name,
        "page": 0,
        "content_type": "extraction_failure",
        "chunk_id": f"{source_name}_failed_extraction",
        "url_hash": url_hash,
        "last_checked": datetime.now().isoformat(),
        "extraction_method": "failed",
        "processing_status": "failed_extraction",

        # Convert None values to empty strings to avoid PyArrow casting issues
        "document_date": doc_date.isoformat() if doc_date else "",
        "publication_date": publication_date.isoformat() if publication_date else "",
        "proceeding_number": proceeding_number or "",
        "document_type": doc_type or "unknown",
        "supersedes_priority": 0.0,  # Low priority for failed extractions
    }
    
    placeholder_doc = Document(page_content=placeholder_content, metadata=metadata)
    logger.info(f"Created placeholder document for failed PDF: {source_name}")
    
    return [placeholder_doc]

def estimate_line_number_from_content(content: str, page_content: str = None, config_chars_per_line: int = None) -> int:
    """
    Estimate line number within a page based on content position and character density.
    
    This function provides approximate line numbers when precise line information
    isn't available from the PDF parser. It uses content analysis to estimate
    where text appears within a page.
    
    Args:
        content (str): The content chunk for which to estimate line number
        page_content (str, optional): Full page content for context
        config_chars_per_line (int, optional): Estimated characters per line from config
        
    Returns:
        int: Estimated line number within the page (starting from 1)
        
    Algorithm:
        1. If page_content is provided, find content position within page
        2. Estimate line number based on character position and line density
        3. Apply heuristics for common PDF formatting patterns
        4. Return a reasonable line number estimate
    """
    if config_chars_per_line is None:
        config_chars_per_line = config.ESTIMATED_CHARACTERS_PER_LINE
    
    if not content or not content.strip():
        return 1
    
    # If we have page content, find the position of our content within it
    if page_content:
        # Normalize both strings for better matching
        content_normalized = content.strip()[:100]  # Use first 100 chars for matching
        page_normalized = page_content
        
        # Find the position of our content in the page
        try:
            content_position = page_normalized.find(content_normalized)
            if content_position >= 0:
                # Count newlines before this position to get approximate line
                lines_before = page_normalized[:content_position].count('\n')
                estimated_line = lines_before + 1
                
                # Add some intelligence based on content analysis
                if any(header_word in content.lower() for header_word in ['chapter', 'section', 'title', 'heading']):
                    # Headers tend to be near the top of sections
                    estimated_line = max(1, estimated_line - 2)
                elif 'table' in content.lower() or '|' in content:
                    # Tables often have multiple lines
                    estimated_line = max(1, estimated_line + 2)
                
                return max(1, estimated_line)
        except Exception as e:
            logger.debug(f"Error finding content position for line estimation: {e}")
    
    # Fallback: estimate based on content length and typical formatting
    content_length = len(content)
    
    # Heuristic: shorter content tends to be higher on page, longer content lower
    if content_length < 100:
        return 1  # Short content, likely at top
    elif content_length < 300:
        return 5  # Medium content, likely in upper portion
    elif content_length < 600:
        return 15  # Longer content, likely in middle
    else:
        return 25  # Very long content, likely spans multiple lines
    
def calculate_line_range_for_chunk(content: str, page_content: str = None) -> tuple[int, int]:
    """
    Calculate estimated line range (start, end) for a content chunk.
    
    This provides a more precise range for longer content that spans multiple lines.
    
    Args:
        content (str): The content chunk
        page_content (str, optional): Full page content for context
        
    Returns:
        tuple[int, int]: (start_line, end_line) estimated range
    """
    start_line = estimate_line_number_from_content(content, page_content)
    
    # Estimate how many lines this content spans
    content_lines = content.count('\n') + 1
    
    # For table content, estimate more lines
    if '|' in content or 'table' in content.lower():
        content_lines = max(content_lines, len(content) // config.ESTIMATED_CHARACTERS_PER_LINE + 2)
    
    # For regular text, estimate based on character count
    else:
        estimated_lines = max(1, len(content) // config.ESTIMATED_CHARACTERS_PER_LINE)
        content_lines = max(content_lines, estimated_lines)
    
    end_line = start_line + max(0, content_lines - 1)
    
    return start_line, end_line

def extract_page_content_map(docling_doc) -> dict:
    """
    Extract a mapping of page numbers to their full content for line number calculation.
    
    This function builds a comprehensive map of each page's content to enable
    accurate line number estimation when processing individual chunks.
    
    Args:
        docling_doc: Docling document object
        
    Returns:
        dict: Mapping of page_number -> full_page_content
    """
    page_content_map = {}
    
    try:
        # Iterate through all items to build page content map
        for item, level in docling_doc.iterate_items(with_groups=False):
            if not isinstance(item, DocItem):
                continue
            
            # Get page number
            page_num = item.prov[0].page_no + 1 if item.prov else 1
            
            # Get content
            content = ""
            if isinstance(item, TableItem):
                content = item.export_to_markdown(doc=docling_doc)
            elif hasattr(item, 'text'):
                content = item.text
            
            if content and content.strip():
                if page_num not in page_content_map:
                    page_content_map[page_num] = ""
                page_content_map[page_num] += content + "\n"
        
        logger.debug(f"Built page content map for {len(page_content_map)} pages")
        return page_content_map
        
    except Exception as e:
        logger.warning(f"Failed to build page content map: {e}")
        return {}

def get_file_hash(file_path: Path) -> str:
    """Calculates the SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return ""



def rerank_documents_with_recency(self, question: str, documents: List[Document]) -> List[Document]:
    """
    Enhanced reranking that considers both relevance and recency.
    
    This function implements a sophisticated document reranking system that combines
    multiple factors to prioritize the most relevant and current documents. It first
    filters out superseded documents, then applies LLM-based relevance ranking,
    and finally adds recency and document type bonuses.
    
    Args:
        self: The RAG system instance (contains _rerank_documents method)
        question (str): The user's question for relevance scoring
        documents (List[Document]): List of documents to rerank
        
    Returns:
        List[Document]: Reranked documents ordered by combined relevance,
                       recency, and document type priority scores
                       
    Ranking Factors:
        1. Superseding logic: Filters out outdated documents
        2. LLM relevance: Uses language model to assess question relevance
        3. Recency bonus: Prioritizes newer documents (5-year decay)
        4. Document type: Prioritizes decisions over proposals
        
    Scoring Components:
        - Base score: 1.0 (from LLM ranking)
        - Recency bonus: 0.3 weight (max 1.0, decays over 5 years)
        - Type priority: 0.2 weight (based on supersedes_priority)
        
    Examples:
        >>> docs = rerank_documents_with_recency(self, "rate requirements", doc_list)
        >>> docs[0].metadata['document_type']
        'decision'  # Recent decisions ranked higher
    """
    if not documents:
        return []

    filtered_docs = _filter_superseded_documents(documents)

    reranked_docs = self._rerank_documents(question, filtered_docs)

    current_time = datetime.now()
    scored_docs = []

    for doc in reranked_docs:
        base_score = 1.0  # Base relevance score from LLM ranking

        if doc.metadata.get('document_date'):
            doc_date = datetime.fromisoformat(doc.metadata['document_date'])
            days_old = (current_time - doc_date).days
            recency_score = max(0.0, 1.0 - (days_old / 1825.0))  # Decay over 5 years
        else:
            recency_score = 0.1  # Low score for undated documents

        type_priority = doc.metadata.get('supersedes_priority', 0.5)

        final_score = base_score + (recency_score * 0.3) + (type_priority * 0.2)
        scored_docs.append((doc, final_score))

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs]