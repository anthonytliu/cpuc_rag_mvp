import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.document import DocItem, TableItem
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(backend=DoclingParseV4DocumentBackend)
    }
)


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

def extract_and_chunk_with_docling(pdf_path: Path) -> List[Document]:
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

        logger.info(f"Document metadata - Date: {doc_date}, Proceeding: {proceeding_number}, Type: {doc_type}")

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
                    "source": pdf_path.name,
                    "page": page_num,
                    "content_type": item.label.value,
                    "chunk_id": f"{pdf_path.name}_{item.get_ref().cref.replace('#/', '').replace('/', '_')}",
                    "file_path": str(pdf_path),
                    "last_modified": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat(),

                    "document_date": doc_date.isoformat() if doc_date else None,
                    "proceeding_number": proceeding_number,
                    "document_type": doc_type,
                    "supersedes_priority": _calculate_supersedes_priority(doc_type, doc_date),
                }

                langchain_documents.append(Document(page_content=content, metadata=metadata))

    except Exception as e:
        logger.error(f"Error processing {pdf_path.name}: {e}", exc_info=True)
        return []

    logger.info(f"Extracted {len(langchain_documents)} chunks from {pdf_path.name}")
    return langchain_documents

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

def _filter_superseded_documents(documents: List[Document]) -> List[Document]:
    """Filter out documents that have been superseded by more recent ones."""
    if not documents:
        return documents

    proceeding_groups = {}
    ungrouped_docs = []

    for doc in documents:
        proceeding = doc.metadata.get('proceeding_number')
        if proceeding:
            if proceeding not in proceeding_groups:
                proceeding_groups[proceeding] = []
            proceeding_groups[proceeding].append(doc)
        else:
            ungrouped_docs.append(doc)

    filtered_docs = []

    for proceeding, group_docs in proceeding_groups.items():
        group_docs.sort(key=lambda x: (
            x.metadata.get('supersedes_priority', 0),
            datetime.fromisoformat(x.metadata['document_date']) if x.metadata.get('document_date') else datetime.min
        ), reverse=True)

        top_doc = group_docs[0]
        if top_doc.metadata.get('document_type') in ['decision', 'ruling']:
            logger.info(
                f"Keeping most recent {top_doc.metadata.get('document_type')} for proceeding {proceeding}: {top_doc.metadata.get('source')}")
            filtered_docs.append(top_doc)

            for doc in group_docs[1:]:
                if doc.metadata.get('document_type') in ['application', 'proposal']:
                    filtered_docs.append(doc)
        else:
            filtered_docs.extend(group_docs)

    filtered_docs.extend(ungrouped_docs)

    return filtered_docs


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