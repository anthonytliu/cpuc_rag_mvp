#!/usr/bin/env python3
"""
Hybrid Processing System

Combines Docling and Chonkie processing to get the best of both:
- Docling: Excellent table extraction and structured content
- Chonkie: Superior text chunking and positioning metadata

This prevents data collision and schema issues by:
1. Using Docling for tables, forms, and structured content
2. Using Chonkie for text-heavy content with better chunking
3. Ensuring unified metadata schema across all outputs
4. Preventing data loss during schema migrations
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from langchain.schema import Document
from .minimal_metadata_schema import MinimalMetadataSchema, normalize_document_metadata_minimal

logger = logging.getLogger(__name__)


class HybridProcessor:
    """
    Intelligently combines Docling and Chonkie processing.
    
    Strategy:
    - Use Docling for tables, structured content, images
    - Use Chonkie for text-heavy content with better chunking
    - Merge outputs with unified metadata schema
    - Prevent schema conflicts and data loss
    """
    
    def __init__(self, proceeding: str):
        self.proceeding = proceeding
        
    def process_document(self, 
                        pdf_url: str, 
                        document_title: str, 
                        table_score: float = 0.0) -> List[Document]:
        """
        Process document using optimal hybrid strategy.
        
        Args:
            pdf_url: URL of PDF to process
            document_title: Title of the document
            table_score: Score indicating table/financial content (0.0-1.0)
            
        Returns:
            List of documents with unified metadata schema
        """
        logger.info(f"🔄 Starting hybrid processing for: {pdf_url}")
        logger.info(f"📊 Table/financial score: {table_score:.3f}")
        
        try:
            # Step 1: Run Docling to extract structured content (tables, images)
            docling_results = self._extract_structured_content(pdf_url, document_title)
            
            # Step 2: Run Chonkie for text chunking if needed
            chonkie_results = self._extract_text_content(pdf_url, document_title, table_score)
            
            # Step 3: Combine and deduplicate results
            combined_results = self._combine_results(docling_results, chonkie_results, pdf_url, document_title)
            
            # Step 4: If no results, fall back to standard Docling processing
            if not combined_results:
                logger.warning("🔄 No hybrid results - falling back to standard Docling processing")
                return self._fallback_processing(pdf_url, document_title)
            
            # Step 5: Ensure all documents have unified metadata
            unified_results = self._ensure_unified_schema(combined_results)
            
            logger.info(f"✅ Hybrid processing completed: {len(unified_results)} total chunks")
            logger.info(f"   📊 Docling structured: {len(docling_results)} chunks")
            logger.info(f"   📝 Chonkie text: {len(chonkie_results)} chunks") 
            logger.info(f"   🔄 Combined total: {len(unified_results)} chunks")
            
            return unified_results
            
        except Exception as e:
            logger.error(f"❌ Hybrid processing failed for {pdf_url}: {e}")
            # Fallback to simple processing
            return self._fallback_processing(pdf_url, document_title)
    
    def _extract_structured_content(self, pdf_url: str, document_title: str) -> List[Document]:
        """
        Use Docling to extract tables, images, and structured content.
        
        Args:
            pdf_url: URL of PDF to process
            document_title: Title of the document
            
        Returns:
            List of documents with structured content
        """
        logger.info("📊 Extracting structured content with Docling...")
        
        try:
            from .data_processing import _process_with_standard_docling
            
            # Get all Docling results
            all_docling = _process_with_standard_docling(
                pdf_url, document_title, self.proceeding,
                enable_ocr_fallback=False,  # Disable fallbacks for structured extraction
                enable_chonkie_fallback=False
            )
            
            # Filter for structured content (tables, images, forms)
            structured_content = []
            structured_types = {'table', 'figure', 'form', 'list'}
            text_types = {'text', 'paragraph', 'title', 'subtitle'}
            
            # Separate structured vs text content
            for doc in all_docling:
                content_type = doc.metadata.get('content_type', '').lower()
                
                # Check if this is structured content
                is_structured = any(struct_type in content_type for struct_type in structured_types)
                
                # For hybrid processing, we want both structured content AND some text
                # But we'll prioritize structured content
                if is_structured or table_score > 0.5:
                    # Mark as structured content
                    doc.metadata['processing_method'] = 'hybrid_docling_structured'
                    doc.metadata['extraction_method'] = 'docling_structured'
                    structured_content.append(doc)
                elif len(structured_content) < 10:  # Include some text content too
                    doc.metadata['processing_method'] = 'hybrid_docling_text'
                    doc.metadata['extraction_method'] = 'docling_text'
                    structured_content.append(doc)
            
            logger.info(f"📊 Extracted {len(structured_content)} structured content chunks")
            return structured_content
            
        except Exception as e:
            logger.warning(f"Structured content extraction failed: {e}")
            return []
    
    def _extract_text_content(self, pdf_url: str, document_title: str, table_score: float) -> List[Document]:
        """
        Use Chonkie for superior text chunking when appropriate.
        
        Args:
            pdf_url: URL of PDF to process
            document_title: Title of document
            table_score: Table/financial content score
            
        Returns:
            List of documents with text content
        """
        # Only use Chonkie for text-heavy documents (low table score)
        if table_score > 0.3:
            logger.info("📊 High table score - skipping Chonkie text extraction")
            return []
            
        logger.info("📝 Extracting text content with Chonkie...")
        
        try:
            from .data_processing import _extract_with_chonkie_fallback, extract_filename_from_url, get_url_hash
            from datetime import datetime
            
            # Extract basic metadata
            source_name = document_title or extract_filename_from_url(pdf_url)
            url_hash = get_url_hash(pdf_url)
            
            # Use Chonkie for text extraction
            chonkie_results = _extract_with_chonkie_fallback(
                pdf_url, source_name, 
                doc_date=datetime.now(),
                publication_date=None,
                proceeding_number=self.proceeding,
                doc_type='proceeding',
                url_hash=url_hash,
                proceeding=self.proceeding
            )
            
            # Mark as Chonkie text content
            for doc in chonkie_results:
                doc.metadata['processing_method'] = 'hybrid_chonkie_text'
                doc.metadata['extraction_method'] = 'chonkie_text'
            
            logger.info(f"📝 Extracted {len(chonkie_results)} text chunks")
            return chonkie_results
            
        except Exception as e:
            logger.warning(f"Text content extraction failed: {e}")
            return []
    
    def _combine_results(self, 
                        docling_results: List[Document], 
                        chonkie_results: List[Document],
                        pdf_url: str,
                        document_title: str) -> List[Document]:
        """
        Combine Docling and Chonkie results, removing duplicates.
        
        Args:
            docling_results: Structured content from Docling
            chonkie_results: Text content from Chonkie  
            pdf_url: Source PDF URL
            document_title: Document title
            
        Returns:
            Combined list of documents
        """
        logger.info("🔄 Combining Docling and Chonkie results...")
        
        combined = []
        
        # Add all structured content from Docling
        combined.extend(docling_results)
        
        # Add text content from Chonkie, avoiding overlap with structured content
        for chonkie_doc in chonkie_results:
            # Simple deduplication: check if content significantly overlaps
            is_duplicate = False
            chonkie_content = chonkie_doc.page_content.strip().lower()
            
            for docling_doc in docling_results:
                docling_content = docling_doc.page_content.strip().lower()
                
                # Check for significant overlap (>80% of shorter content)
                if len(chonkie_content) > 0 and len(docling_content) > 0:
                    shorter_len = min(len(chonkie_content), len(docling_content))
                    
                    # Simple substring check for major overlap
                    if (chonkie_content in docling_content or 
                        docling_content in chonkie_content or
                        self._calculate_overlap(chonkie_content, docling_content) > 0.8):
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                combined.append(chonkie_doc)
        
        # Update chunk indices for combined results
        for i, doc in enumerate(combined):
            doc.metadata.update({
                'chunk_index': i,
                'total_chunks': len(combined),
                'processing_method': f"hybrid_{doc.metadata.get('extraction_method', 'unknown')}"
            })
        
        logger.info(f"🔄 Combined results: {len(docling_results)} structured + {len(chonkie_results)} text → {len(combined)} total")
        return combined
    
    def _calculate_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate overlap between two texts.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        if not text1 or not text2:
            return 0.0
            
        # Simple word-based overlap calculation
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _ensure_unified_schema(self, documents: List[Document]) -> List[Document]:
        """
        Ensure all documents use unified metadata schema.
        
        Args:
            documents: List of documents to normalize
            
        Returns:
            List of documents with unified schema
        """
        logger.info("🔧 Ensuring unified metadata schema...")
        
        unified_documents = []
        
        for doc in documents:
            # Normalize to minimal schema to prevent ArrowSchema recursion
            normalized_doc = normalize_document_metadata_minimal(doc, 'hybrid')
            
            # Only update essential fields in minimal schema
            normalized_doc.metadata.update({
                'processing_method': 'hybrid'
            })
            
            unified_documents.append(normalized_doc)
        
        return unified_documents
    
    def _fallback_processing(self, pdf_url: str, document_title: str) -> List[Document]:
        """
        Fallback to simple processing if hybrid fails.
        
        Args:
            pdf_url: URL of PDF to process
            document_title: Title of document
            
        Returns:
            List of documents from fallback processing
        """
        logger.warning("⚠️ Using fallback processing...")
        
        try:
            from .data_processing import _process_with_standard_docling
            
            results = _process_with_standard_docling(
                pdf_url, document_title, self.proceeding
            )
            
            # Ensure unified schema
            return self._ensure_unified_schema(results)
            
        except Exception as e:
            logger.error(f"❌ Fallback processing also failed: {e}")
            return []


def create_hybrid_processor(proceeding: str) -> HybridProcessor:
    """
    Create a hybrid processor for the given proceeding.
    
    Args:
        proceeding: Proceeding identifier (e.g., R1311007)
        
    Returns:
        Configured HybridProcessor instance
    """
    return HybridProcessor(proceeding)


def process_with_intelligent_hybrid(pdf_url: str, 
                                   document_title: str, 
                                   proceeding: str,
                                   table_score: float = 0.0) -> List[Document]:
    """
    Process document using intelligent hybrid processing.
    
    Args:
        pdf_url: URL of PDF to process
        document_title: Title of the document
        proceeding: Proceeding identifier
        table_score: Score indicating table/financial content (0.0-1.0)
        
    Returns:
        List of documents with unified metadata schema
    """
    processor = create_hybrid_processor(proceeding)
    return processor.process_document(pdf_url, document_title, table_score)