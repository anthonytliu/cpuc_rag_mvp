#!/usr/bin/env python3
"""
Minimal Metadata Schema for CPUC RAG System

A drastically simplified metadata schema that prevents ArrowSchema recursion issues
while maintaining compatibility across Docling, Chonkie, and Hybrid processing.

This addresses the critical ArrowSchema recursion errors that were causing 
every document embedding to fail.
"""

from datetime import datetime
from typing import Dict, Any
from langchain.schema import Document


class MinimalMetadataSchema:
    """
    Minimal metadata schema to prevent ArrowSchema recursion issues.
    
    Uses only essential fields that LanceDB can handle without hitting recursion limits.
    """
    
    # Only the most essential fields to prevent ArrowSchema recursion
    ESSENTIAL_FIELDS = {
        'source',           # PDF URL (string)
        'title',            # Document title (string) 
        'proceeding',       # Proceeding ID (string)
        'content_type',     # Content type (string)
        'processing_method' # How it was processed (string)
    }
    
    @classmethod
    def create_minimal_metadata(cls, 
                              pdf_url: str,
                              document_title: str, 
                              proceeding: str,
                              processing_method: str = 'docling') -> Dict[str, Any]:
        """
        Create minimal metadata that won't cause ArrowSchema recursion.
        
        Args:
            pdf_url: Source PDF URL
            document_title: Title of the document
            proceeding: Proceeding identifier
            processing_method: Processing method used
            
        Returns:
            Minimal metadata dictionary
        """
        return {
            'source': str(pdf_url),
            'title': str(document_title),
            'proceeding': str(proceeding),
            'content_type': 'text',
            'processing_method': str(processing_method)
        }
    
    @classmethod
    def normalize_to_minimal(cls, document: Document) -> Document:
        """
        Convert any document to use minimal metadata schema.
        
        Args:
            document: Document to normalize
            
        Returns:
            Document with minimal metadata
        """
        metadata = document.metadata
        
        # Extract essential information only
        minimal_metadata = cls.create_minimal_metadata(
            pdf_url=metadata.get('source', metadata.get('url', '')),
            document_title=metadata.get('title', 'Unknown Document'),
            proceeding=metadata.get('proceeding', ''),
            processing_method=metadata.get('processing_method', 'unknown')
        )
        
        return Document(
            page_content=document.page_content,
            metadata=minimal_metadata
        )
    
    @classmethod
    def validate_schema(cls, metadata: Dict[str, Any]) -> bool:
        """
        Validate that metadata conforms to minimal schema.
        
        Args:
            metadata: Metadata to validate
            
        Returns:
            True if valid, False otherwise
        """
        for field in cls.ESSENTIAL_FIELDS:
            if field not in metadata:
                return False
        return True


def create_minimal_documents(documents: list, processing_method: str = 'docling') -> list:
    """
    Convert a list of documents to use minimal metadata schema.
    
    Args:
        documents: List of Document objects
        processing_method: Processing method identifier
        
    Returns:
        List of documents with minimal metadata
    """
    minimal_docs = []
    
    for doc in documents:
        # Override processing method if specified
        doc.metadata['processing_method'] = processing_method
        
        # Normalize to minimal schema
        minimal_doc = MinimalMetadataSchema.normalize_to_minimal(doc)
        minimal_docs.append(minimal_doc)
    
    return minimal_docs


def normalize_document_metadata_minimal(document: Document, processing_method: str = None) -> Document:
    """
    Normalize document to minimal metadata schema to prevent ArrowSchema recursion.
    
    Args:
        document: Document to normalize
        processing_method: Override processing method if needed
        
    Returns:
        Document with minimal metadata schema
    """
    if processing_method:
        document.metadata['processing_method'] = processing_method
    
    return MinimalMetadataSchema.normalize_to_minimal(document)