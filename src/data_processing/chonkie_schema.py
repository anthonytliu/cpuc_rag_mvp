#!/usr/bin/env python3
"""
Chonkie Schema for CPUC RAG System

This module defines the proven Chonkie metadata schema that works without ArrowSchema recursion.
All processing methods (Chonkie primary, Docling fallback) will use this exact schema structure.

Based on the working create_enhanced_chonkie_metadata function that successfully processes
documents without causing ArrowSchema recursion errors.
"""

from datetime import datetime
from typing import Dict, Any, List
from langchain.schema import Document


class ChonkieSchema:
    """
    Proven Chonkie metadata schema that works without ArrowSchema recursion issues.
    
    This schema has been tested and verified to work correctly with LanceDB
    without causing recursion errors during embedding.
    """
    
    # Exact field structure from working create_enhanced_chonkie_metadata
    SCHEMA_FIELDS = {
        # Core document identification (strings)
        'source_url': str,
        'source': str,
        'proceeding_number': str,
        'chunk_id': str,
        
        # Content classification (strings)
        'content_type': str,
        'document_type': str,
        'text_snippet': str,
        
        # Positional information (integers)
        'page': int,
        'line_number': int,
        'line_range_end': int,
        'char_start': int,
        'char_end': int,
        'char_length': int,
        'token_count': int,
        'chunk_level': int,
        
        # Temporal metadata (strings - ISO format)
        'last_checked': str,
        'document_date': str,
        'publication_date': str,
        
        # Priority scoring (float)
        'supersedes_priority': float
    }
    
    @classmethod
    def create_base_metadata(cls, 
                           pdf_url: str,
                           source_name: str,
                           proceeding: str,
                           chunk_info: Dict[str, Any],
                           raw_text: str = "") -> Dict[str, Any]:
        """
        Create base Chonkie metadata using the proven working structure.
        
        Args:
            pdf_url: URL of the source PDF
            source_name: Name of the source document
            proceeding: Proceeding identifier
            chunk_info: Dictionary with chunk text and position information
            raw_text: Full raw text of the document for position estimation
            
        Returns:
            Dictionary with Chonkie-compatible metadata
        """
        start_pos = chunk_info.get('start_index', 0)
        end_pos = chunk_info.get('end_index', len(chunk_info.get('text', '')))
        chunk_text = chunk_info.get('text', '')
        
        # Estimate page number from character position or use provided page
        estimated_page = chunk_info.get('page', 1)
        if raw_text and start_pos > 0:
            estimated_page = cls._estimate_page_from_char_position(start_pos, raw_text)
        
        # Estimate line range
        start_line = chunk_info.get('line_number', 1)
        end_line = chunk_info.get('line_range_end', start_line)
        if raw_text:
            start_line, end_line = cls._estimate_line_range_from_char_position(
                start_pos, end_pos, raw_text
            )
        
        # Create text snippet for citation verification
        text_snippet = chunk_text[:100].replace('\n', ' ').strip() if chunk_text else ""
        
        # Create chunk ID using consistent format
        strategy = chunk_info.get('strategy', 'chonkie')
        chunk_id = f"{source_name}_{strategy}_{start_pos}_{end_pos}"
        
        return {
            # Core document identification
            "source_url": str(pdf_url),
            "source": str(source_name),
            "proceeding_number": str(proceeding),
            "chunk_id": str(chunk_id),
            
            # Content classification
            "content_type": f"text_{strategy}_{chunk_info.get('level', 0)}",
            "document_type": str(chunk_info.get('document_type', 'unknown')),
            "text_snippet": str(text_snippet),
            
            # Positional information
            "page": int(estimated_page),
            "line_number": int(start_line),
            "line_range_end": int(end_line),
            "char_start": int(start_pos),
            "char_end": int(end_pos),
            "char_length": int(end_pos - start_pos),
            "token_count": int(chunk_info.get('token_count', len(chunk_text.split()))),
            "chunk_level": int(chunk_info.get('level', 0)),
            
            # Temporal metadata (as strings to avoid datetime issues)
            "last_checked": str(chunk_info.get('last_checked', datetime.now().isoformat())),
            "document_date": str(chunk_info.get('document_date', '')),
            "publication_date": str(chunk_info.get('publication_date', '')),
            
            # Priority scoring
            "supersedes_priority": float(chunk_info.get('supersedes_priority', 0.5))
        }
    
    @classmethod
    def _estimate_page_from_char_position(cls, char_position: int, full_text: str,
                                        chars_per_page: int = 2000) -> int:
        """
        Estimate page number from character position.
        
        Args:
            char_position: Character position in the full text
            full_text: Complete text of the document
            chars_per_page: Estimated characters per page
            
        Returns:
            Estimated page number (1-based)
        """
        if not full_text or char_position <= 0:
            return 1
        
        # Look for form feed characters first
        text_up_to_pos = full_text[:char_position]
        form_feeds = text_up_to_pos.count('\f')
        if form_feeds > 0:
            return form_feeds + 1
        
        # Fallback to character-based estimation
        return max(1, (char_position // chars_per_page) + 1)
    
    @classmethod
    def _estimate_line_range_from_char_position(cls, start_pos: int, end_pos: int, 
                                              full_text: str) -> tuple:
        """
        Estimate line range from character positions.
        
        Args:
            start_pos: Starting character position
            end_pos: Ending character position
            full_text: Complete text of the document
            
        Returns:
            Tuple of (start_line, end_line) - 1-based line numbers
        """
        if not full_text:
            return (1, 1)
        
        # Count newlines up to start and end positions
        text_up_to_start = full_text[:start_pos]
        text_up_to_end = full_text[:end_pos]
        
        start_line = text_up_to_start.count('\n') + 1
        end_line = text_up_to_end.count('\n') + 1
        
        return (start_line, end_line)
    
    @classmethod
    def validate_metadata(cls, metadata: Dict[str, Any]) -> bool:
        """
        Validate that metadata conforms to Chonkie schema.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        for field_name, field_type in cls.SCHEMA_FIELDS.items():
            if field_name not in metadata:
                return False
            
            # Check type compatibility (allowing for type coercion)
            value = metadata[field_name]
            if not isinstance(value, (field_type, type(None))):
                # Allow string representations of numbers
                if field_type in (int, float) and isinstance(value, str):
                    try:
                        field_type(value)
                    except (ValueError, TypeError):
                        return False
                else:
                    return False
        
        return True
    
    @classmethod
    def normalize_document(cls, document: Document) -> Document:
        """
        Normalize any document to use Chonkie schema.
        
        Args:
            document: Document to normalize
            
        Returns:
            Document with Chonkie-compatible metadata
        """
        metadata = document.metadata.copy()
        
        # Extract basic information
        pdf_url = metadata.get('source_url', metadata.get('source', ''))
        source_name = metadata.get('source', 'unknown')
        proceeding = metadata.get('proceeding_number', metadata.get('proceeding', ''))
        
        # Create chunk_info from existing metadata
        chunk_info = {
            'text': document.page_content,
            'start_index': metadata.get('char_start', 0),
            'end_index': metadata.get('char_end', len(document.page_content)),
            'token_count': metadata.get('token_count', len(document.page_content.split())),
            'level': metadata.get('chunk_level', 0),
            'strategy': metadata.get('content_type', 'unknown').replace('text_', '').split('_')[0],
            'page': metadata.get('page', 1),
            'line_number': metadata.get('line_number', 1),
            'document_type': metadata.get('document_type', 'unknown'),
            'last_checked': metadata.get('last_checked', ''),
            'document_date': metadata.get('document_date', ''),
            'publication_date': metadata.get('publication_date', ''),
            'supersedes_priority': metadata.get('supersedes_priority', 0.5)
        }
        
        # Create normalized metadata using Chonkie schema
        normalized_metadata = cls.create_base_metadata(
            pdf_url=pdf_url,
            source_name=source_name,
            proceeding=proceeding,
            chunk_info=chunk_info
        )
        
        return Document(
            page_content=document.page_content,
            metadata=normalized_metadata
        )


def convert_to_chonkie_schema(documents: List[Document]) -> List[Document]:
    """
    Convert a list of documents to use Chonkie schema.
    
    Args:
        documents: List of documents to convert
        
    Returns:
        List of documents with Chonkie-compatible metadata
    """
    return [ChonkieSchema.normalize_document(doc) for doc in documents]


def create_chonkie_document(text: str, chunk_info: Dict[str, Any], 
                          pdf_url: str, source_name: str, proceeding: str,
                          raw_text: str = "") -> Document:
    """
    Create a new document with Chonkie schema metadata.
    
    Args:
        text: Content of the document chunk
        chunk_info: Dictionary with chunk position and metadata
        pdf_url: URL of the source PDF
        source_name: Name of the source document
        proceeding: Proceeding identifier
        raw_text: Full raw text for position estimation
        
    Returns:
        Document with Chonkie-compatible metadata
    """
    metadata = ChonkieSchema.create_base_metadata(
        pdf_url=pdf_url,
        source_name=source_name,
        proceeding=proceeding,
        chunk_info=chunk_info,
        raw_text=raw_text
    )
    
    return Document(page_content=text, metadata=metadata)