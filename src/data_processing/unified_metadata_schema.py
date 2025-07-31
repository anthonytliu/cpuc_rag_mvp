#!/usr/bin/env python3
"""
Unified Metadata Schema for CPUC RAG System

Creates a standardized metadata schema that works consistently across:
- Docling processing (tables, images, text)
- Chonkie processing (text chunking)
- Hybrid processing (combination)
- Schema migrations

This prevents schema compatibility issues and data loss.
"""

from datetime import datetime
from typing import Dict, Any, Optional, List
from langchain.schema import Document


class UnifiedMetadataSchema:
    """
    Defines the complete metadata schema used across all processing methods.
    
    This ensures that all documents (Docling, Chonkie, Hybrid) have consistent
    metadata structures that are compatible with LanceDB schema migrations.
    """
    
    # Core fields that MUST be present in all documents
    REQUIRED_FIELDS = {
        'source', 'url', 'title', 'chunk_id', 'proceeding', 
        'content_type', 'last_checked'
    }
    
    # Extended fields for enhanced functionality
    EXTENDED_FIELDS = {
        # Document-level metadata
        'document_date', 'publication_date', 'document_type', 
        'proceeding_number', 'url_hash', 'file_size',
        
        # Chunk-level positioning
        'page', 'page_number', 'chunk_index', 'total_chunks',
        'char_start', 'char_end', 'char_length', 'line_number',
        
        # Processing metadata  
        'processing_method', 'extraction_method', 'extraction_confidence',
        'chunk_level', 'chunk_overlap', 'source_section',
        
        # Citation support
        'creation_date', 'last_modified',
        'supersedes_priority'
    }
    
    # All possible fields (required + extended)
    ALL_FIELDS = REQUIRED_FIELDS | EXTENDED_FIELDS
    
    @classmethod
    def create_base_metadata(cls, 
                           pdf_url: str,
                           document_title: str, 
                           proceeding: str,
                           processing_method: str) -> Dict[str, Any]:
        """
        Create base metadata structure with all required fields.
        
        Args:
            pdf_url: Source PDF URL
            document_title: Title of the document
            proceeding: Proceeding identifier (e.g., R1311007)
            processing_method: Method used (e.g., 'docling', 'chonkie', 'hybrid')
            
        Returns:
            Dictionary with base metadata structure
        """
        return {
            # Required fields (always present)
            'source': pdf_url,
            'url': pdf_url, 
            'title': document_title,
            'chunk_id': '',  # Will be set by specific processors
            'proceeding': proceeding,
            'content_type': 'text',  # Default, can be overridden
            'last_checked': datetime.now().isoformat(),
            
            # Extended fields with safe defaults
            'document_date': '',
            'publication_date': '',
            'document_type': 'unknown',
            'proceeding_number': proceeding or '',
            'url_hash': '',
            'file_size': 0,
            
            'page': 0,
            'page_number': 0, 
            'chunk_index': 0,
            'total_chunks': 0,
            'char_start': 0,
            'char_end': 0,
            'char_length': 0,
            'line_number': 1,
            
            'processing_method': processing_method,
            'extraction_method': processing_method,
            'extraction_confidence': 1.0,
            'chunk_level': 'document',
            'chunk_overlap': 0,
            'source_section': '',
            
            'creation_date': datetime.now().isoformat(),
            'last_modified': datetime.now().isoformat(),
            'supersedes_priority': 0
        }
    
    @classmethod 
    def normalize_docling_metadata(cls, docling_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Docling-generated metadata to unified schema.
        
        Args:
            docling_metadata: Raw metadata from Docling processing
            
        Returns:
            Normalized metadata conforming to unified schema
        """
        # Start with safe defaults
        base = cls.create_base_metadata(
            pdf_url=docling_metadata.get('source_url', docling_metadata.get('source', '')),
            document_title=docling_metadata.get('title', 'Unknown Document'),
            proceeding=docling_metadata.get('proceeding', ''),
            processing_method='docling'
        )
        
        # Map Docling-specific fields to unified schema
        field_mapping = {
            'source_url': 'url',
            'source': 'source', 
            'page': 'page_number',
            'content_type': 'content_type',
            'chunk_id': 'chunk_id',
            'url_hash': 'url_hash',
            'last_checked': 'last_checked',
            'document_date': 'document_date',
            'publication_date': 'publication_date',
            'proceeding_number': 'proceeding_number',
            'document_type': 'document_type',
            'supersedes_priority': 'supersedes_priority'
        }
        
        # Apply mappings
        for docling_field, unified_field in field_mapping.items():
            if docling_field in docling_metadata:
                base[unified_field] = docling_metadata[docling_field]
        
        # Ensure content_type is set correctly for Docling
        if 'content_type' in docling_metadata:
            base['content_type'] = docling_metadata['content_type']
        
        return base
    
    @classmethod
    def normalize_chonkie_metadata(cls, chonkie_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Chonkie-generated metadata to unified schema.
        
        Args:
            chonkie_metadata: Raw metadata from Chonkie processing
            
        Returns:
            Normalized metadata conforming to unified schema
        """
        # Start with safe defaults
        base = cls.create_base_metadata(
            pdf_url=chonkie_metadata.get('url', chonkie_metadata.get('source', '')),
            document_title=chonkie_metadata.get('title', 'Unknown Document'),
            proceeding=chonkie_metadata.get('proceeding', ''),
            processing_method='chonkie'
        )
        
        # Map Chonkie-specific fields to unified schema
        field_mapping = {
            'url': 'url',
            'source': 'source',
            'title': 'title',
            'chunk_id': 'chunk_id',
            'proceeding': 'proceeding',
            'char_start': 'char_start',
            'char_end': 'char_end', 
            'char_length': 'char_length',
            'chunk_index': 'chunk_index',
            'extraction_confidence': 'extraction_confidence',
            'document_date': 'document_date',
            'document_type': 'document_type',
            'proceeding_number': 'proceeding_number'
        }
        
        # Apply mappings
        for chonkie_field, unified_field in field_mapping.items():
            if chonkie_field in chonkie_metadata:
                base[unified_field] = chonkie_metadata[chonkie_field]
        
        # Set Chonkie-specific defaults
        base['content_type'] = chonkie_metadata.get('chunk_level', 'text')
        base['chunk_level'] = chonkie_metadata.get('chunk_level', 'sentence')
        
        return base
    
    @classmethod
    def create_hybrid_metadata(cls,
                             pdf_url: str,
                             document_title: str,
                             proceeding: str,
                             docling_tables: List[Document] = None,
                             chonkie_text: List[Document] = None) -> Dict[str, Any]:
        """
        Create metadata for hybrid processing that combines Docling and Chonkie.
        
        Args:
            pdf_url: Source PDF URL
            document_title: Title of the document
            proceeding: Proceeding identifier
            docling_tables: Table documents from Docling
            chonkie_text: Text documents from Chonkie
            
        Returns:
            Hybrid metadata structure
        """
        base = cls.create_base_metadata(
            pdf_url=pdf_url,
            document_title=document_title,
            proceeding=proceeding,
            processing_method='hybrid'
        )
        
        # Add hybrid-specific information
        base.update({
            'extraction_method': 'hybrid_docling_chonkie',
            'docling_tables': len(docling_tables) if docling_tables else 0,
            'chonkie_chunks': len(chonkie_text) if chonkie_text else 0,
            'total_chunks': (len(docling_tables) if docling_tables else 0) +
                           (len(chonkie_text) if chonkie_text else 0)
        })
        
        return base
    
    @classmethod
    def ensure_compatibility(cls, document: Document) -> Document:
        """
        Ensure a document has all required metadata fields for schema compatibility.
        
        Args:
            document: LangChain Document to normalize
            
        Returns:
            Document with complete, compatible metadata
        """
        # Start with existing metadata
        metadata = document.metadata.copy()
        
        # Create base template
        base = cls.create_base_metadata(
            pdf_url=metadata.get('url', metadata.get('source', '')),
            document_title=metadata.get('title', 'Unknown Document'),
            proceeding=metadata.get('proceeding', ''),
            processing_method=metadata.get('processing_method', 'unknown')
        )
        
        # Merge existing metadata with base, preserving existing values
        for field in cls.ALL_FIELDS:
            if field in metadata:
                base[field] = metadata[field]
        
        # Return new document with normalized metadata
        return Document(
            page_content=document.page_content,
            metadata=base
        )
    
    @classmethod
    def get_schema_fields(cls) -> Dict[str, str]:
        """
        Get the complete schema field definitions for LanceDB.
        
        Returns:
            Dictionary mapping field names to their types
        """
        return {
            # String fields
            'source': 'string',
            'url': 'string',
            'title': 'string', 
            'chunk_id': 'string',
            'proceeding': 'string',
            'content_type': 'string',
            'last_checked': 'string',
            'document_date': 'string',
            'publication_date': 'string',
            'document_type': 'string',
            'proceeding_number': 'string',
            'url_hash': 'string',
            'processing_method': 'string',
            'extraction_method': 'string',
            'chunk_level': 'string',
            'source_section': 'string',
            'creation_date': 'string',
            'last_modified': 'string',
            
            # Numeric fields
            'file_size': 'int64',
            'page': 'int64',
            'page_number': 'int64',
            'chunk_index': 'int64',
            'total_chunks': 'int64',
            'char_start': 'int64',
            'char_end': 'int64',
            'char_length': 'int64',
            'line_number': 'int64',
            'chunk_overlap': 'int64',
            'supersedes_priority': 'int64',
            
            # Float fields
            'extraction_confidence': 'float64'
        }


def normalize_document_metadata(document: Document, processing_method: str = None) -> Document:
    """
    Normalize any document to use the unified metadata schema.
    
    Args:
        document: Document to normalize
        processing_method: Override processing method if needed
        
    Returns:
        Document with unified metadata schema
    """
    if processing_method:
        document.metadata['processing_method'] = processing_method
    
    return UnifiedMetadataSchema.ensure_compatibility(document)


def create_unified_documents(documents: List[Document], processing_method: str) -> List[Document]:
    """
    Convert a list of documents to use unified metadata schema.
    
    Args:
        documents: List of documents to normalize
        processing_method: Processing method identifier
        
    Returns:
        List of documents with unified metadata
    """
    return [normalize_document_metadata(doc, processing_method) for doc in documents]