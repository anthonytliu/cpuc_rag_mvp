#!/usr/bin/env python3
"""
Docling to Chonkie Metadata Converter

Converts Docling-processed documents to use the proven Chonkie metadata schema.
This ensures all documents (whether processed by Chonkie or Docling) have identical
metadata structures that work without ArrowSchema recursion issues.
"""

import logging
from typing import List, Dict, Any
from langchain.schema import Document
from .chonkie_schema import ChonkieSchema, create_chonkie_document

logger = logging.getLogger(__name__)


def convert_docling_to_chonkie_schema(docling_documents: List[Document], 
                                    pdf_url: str, 
                                    source_name: str, 
                                    proceeding: str,
                                    full_text: str = "") -> List[Document]:
    """
    Convert Docling-processed documents to use Chonkie metadata schema.
    
    Args:
        docling_documents: List of documents processed by Docling
        pdf_url: URL of the source PDF
        source_name: Name of the source document  
        proceeding: Proceeding identifier
        full_text: Full text of document for position estimation
        
    Returns:
        List of documents with Chonkie-compatible metadata
    """
    logger.info(f"Converting {len(docling_documents)} Docling documents to Chonkie schema")
    
    converted_documents = []
    
    for i, doc in enumerate(docling_documents):
        try:
            # Extract Docling metadata
            docling_metadata = doc.metadata
            
            # Create chunk_info structure that mimics Chonkie output
            chunk_info = _create_chunk_info_from_docling_metadata(
                docling_metadata, doc.page_content, i, full_text
            )
            
            # Create new document with Chonkie schema
            chonkie_doc = create_chonkie_document(
                text=doc.page_content,
                chunk_info=chunk_info,
                pdf_url=pdf_url,
                source_name=source_name,
                proceeding=proceeding,
                raw_text=full_text
            )
            
            converted_documents.append(chonkie_doc)
            
        except Exception as e:
            logger.warning(f"Failed to convert Docling document {i}: {e}")
            # Use fallback conversion
            try:
                fallback_doc = _fallback_docling_conversion(doc, pdf_url, source_name, proceeding, i)
                converted_documents.append(fallback_doc)
            except Exception as fallback_error:
                logger.error(f"Fallback conversion also failed for document {i}: {fallback_error}")
                continue
    
    logger.info(f"Successfully converted {len(converted_documents)} documents to Chonkie schema")
    return converted_documents


def _create_chunk_info_from_docling_metadata(docling_metadata: Dict[str, Any], 
                                           content: str, 
                                           chunk_index: int,
                                           full_text: str) -> Dict[str, Any]:
    """
    Create Chonkie-style chunk_info from Docling metadata.
    
    Args:
        docling_metadata: Original Docling metadata
        content: Text content of the chunk
        chunk_index: Index of this chunk
        full_text: Full document text for position estimation
        
    Returns:
        Dictionary with chunk information in Chonkie format
    """
    # Extract page information
    page = docling_metadata.get('page', 1)
    
    # Estimate character positions based on content and page
    char_start, char_end = _estimate_char_positions_from_docling(
        content, page, chunk_index, full_text
    )
    
    # Map Docling content_type to Chonkie strategy
    docling_content_type = docling_metadata.get('content_type', 'text')
    chonkie_strategy = _map_docling_content_type_to_chonkie_strategy(docling_content_type)
    
    # Extract or estimate other metadata
    return {
        'text': content,
        'start_index': char_start,
        'end_index': char_end,
        'token_count': len(content.split()),
        'level': _extract_docling_level(docling_content_type),
        'strategy': f"docling_{chonkie_strategy}",
        'page': page,
        'line_number': docling_metadata.get('line_number', 1),
        'line_range_end': docling_metadata.get('line_range_end', 1),
        'document_type': docling_metadata.get('document_type', 'unknown'),
        'last_checked': docling_metadata.get('last_checked', ''),
        'document_date': docling_metadata.get('document_date', ''),
        'publication_date': docling_metadata.get('publication_date', ''),
        'supersedes_priority': docling_metadata.get('supersedes_priority', 0.5)
    }


def _estimate_char_positions_from_docling(content: str, page: int, chunk_index: int, 
                                        full_text: str) -> tuple:
    """
    Estimate character positions for Docling content.
    
    Args:
        content: Text content of the chunk
        page: Page number from Docling
        chunk_index: Index of the chunk
        full_text: Full document text
        
    Returns:
        Tuple of (start_pos, end_pos)
    """
    content_length = len(content)
    
    if full_text and content in full_text:
        # Try to find exact position in full text
        start_pos = full_text.find(content)
        if start_pos != -1:
            return (start_pos, start_pos + content_length)
    
    # Fallback: estimate based on page and chunk index
    # Assume ~2000 chars per page, plus chunk_index * average chunk size
    chars_per_page = 2000
    estimated_start = max(0, (page - 1) * chars_per_page + chunk_index * 500)
    estimated_end = estimated_start + content_length
    
    return (estimated_start, estimated_end)


def _map_docling_content_type_to_chonkie_strategy(docling_content_type: str) -> str:
    """
    Map Docling content types to Chonkie strategy names.
    
    Args:
        docling_content_type: Docling content type (e.g., 'table', 'text', 'title')
        
    Returns:
        Chonkie-compatible strategy name
    """
    # Clean up the content type
    clean_type = str(docling_content_type).lower().strip()
    
    # Map common Docling types to Chonkie strategies
    docling_to_chonkie_map = {
        'table': 'table',
        'text': 'recursive',
        'paragraph': 'sentence', 
        'title': 'recursive',
        'subtitle': 'recursive',
        'list': 'recursive',
        'figure': 'token',
        'caption': 'sentence'
    }
    
    # Find matching strategy
    for docling_type, chonkie_strategy in docling_to_chonkie_map.items():
        if docling_type in clean_type:
            return chonkie_strategy
    
    # Default fallback
    return 'recursive'


def _extract_docling_level(content_type: str) -> int:
    """
    Extract hierarchical level from Docling content type.
    
    Args:
        content_type: Docling content type
        
    Returns:
        Hierarchical level (0-3)
    """
    clean_type = str(content_type).lower()
    
    # Map content types to levels
    if 'title' in clean_type:
        return 0  # Top level
    elif 'subtitle' in clean_type:
        return 1  # Second level
    elif 'table' in clean_type or 'figure' in clean_type:
        return 2  # Structured content
    else:
        return 3  # Regular text


def _fallback_docling_conversion(doc: Document, pdf_url: str, source_name: str, 
                               proceeding: str, chunk_index: int) -> Document:
    """
    Fallback conversion when primary conversion fails.
    
    Args:
        doc: Original Docling document
        pdf_url: URL of source PDF
        source_name: Name of source document
        proceeding: Proceeding identifier
        chunk_index: Index of this chunk
        
    Returns:
        Document with basic Chonkie schema
    """
    logger.info(f"Using fallback conversion for chunk {chunk_index}")
    
    # Create minimal chunk_info
    chunk_info = {
        'text': doc.page_content,
        'start_index': chunk_index * 1000,  # Rough estimation
        'end_index': chunk_index * 1000 + len(doc.page_content),
        'token_count': len(doc.page_content.split()),
        'level': 3,
        'strategy': 'docling_fallback',
        'page': doc.metadata.get('page', 1),
        'line_number': 1,
        'document_type': 'unknown',
        'last_checked': '',
        'document_date': '',
        'publication_date': '',
        'supersedes_priority': 0.5
    }
    
    return create_chonkie_document(
        text=doc.page_content,
        chunk_info=chunk_info,
        pdf_url=pdf_url,
        source_name=source_name,
        proceeding=proceeding
    )


def convert_single_docling_document(doc: Document, pdf_url: str, source_name: str,
                                  proceeding: str, chunk_index: int = 0,
                                  full_text: str = "") -> Document:
    """
    Convert a single Docling document to Chonkie schema.
    
    Args:
        doc: Docling document to convert
        pdf_url: URL of source PDF
        source_name: Name of source document
        proceeding: Proceeding identifier  
        chunk_index: Index of this chunk
        full_text: Full document text
        
    Returns:
        Document with Chonkie schema
    """
    try:
        return convert_docling_to_chonkie_schema([doc], pdf_url, source_name, proceeding, full_text)[0]
    except (IndexError, Exception) as e:
        logger.warning(f"Single document conversion failed: {e}")
        return _fallback_docling_conversion(doc, pdf_url, source_name, proceeding, chunk_index)


def validate_chonkie_conversion(converted_docs: List[Document]) -> bool:
    """
    Validate that all converted documents have proper Chonkie schema.
    
    Args:
        converted_docs: List of converted documents
        
    Returns:
        True if all documents are valid, False otherwise
    """
    for i, doc in enumerate(converted_docs):
        if not ChonkieSchema.validate_metadata(doc.metadata):
            logger.error(f"Document {i} failed Chonkie schema validation")
            logger.error(f"Metadata keys: {list(doc.metadata.keys())}")
            return False
    
    logger.info(f"All {len(converted_docs)} converted documents passed Chonkie schema validation")
    return True