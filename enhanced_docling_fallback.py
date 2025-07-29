#!/usr/bin/env python3
"""
Enhanced Docling Fallback with Citation Metadata

This module provides enhanced Docling processing that creates character position
metadata similar to what Chonkie provides, ensuring consistent citation format
regardless of which processing method is used.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.document import DocItem, TableItem
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain.docstore.document import Document

import config
from data_processing import (
    validate_pdf_url, extract_filename_from_url, get_url_hash,
    extract_date_from_content, extract_proceeding_number, identify_document_type,
    get_source_url_from_filename, get_publication_date_from_filename,
    _calculate_supersedes_priority, estimate_line_range_from_char_position,
    create_enhanced_chonkie_metadata, create_precise_citation
)

logger = logging.getLogger(__name__)

# Configure optimized Docling converter for enhanced processing
pipeline_options = PdfPipelineOptions()
if hasattr(config, 'DOCLING_FAST_MODE') and config.DOCLING_FAST_MODE:
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST

enhanced_doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            backend=DoclingParseV4DocumentBackend,
            pipeline_options=pipeline_options
        )
    }
)


def _process_with_enhanced_docling_fallback(pdf_url: str, document_title: str = None, 
                                          proceeding: str = None, enable_ocr_fallback: bool = True) -> List[Document]:
    """
    Enhanced Docling fallback that provides character position metadata.
    
    This function processes PDFs with Docling but creates enhanced citation metadata
    similar to what Chonkie provides, including character positions, line ranges,
    and text snippets for improved citation accuracy.
    
    Args:
        pdf_url: URL of the PDF to process
        document_title: Optional document title
        proceeding: Proceeding identifier
        enable_ocr_fallback: Whether to enable OCR if standard processing fails
        
    Returns:
        List of Document objects with enhanced citation metadata
    """
    logger.info(f"Processing with enhanced Docling fallback: {pdf_url}")
    
    # Validate URL first
    if not validate_pdf_url(pdf_url):
        logger.error(f"URL validation failed for: {pdf_url}")
        return []
    
    langchain_documents = []
    
    try:
        # Use Docling's URL processing capability
        conv_results = enhanced_doc_converter.convert_all([pdf_url], raises_on_error=False)
        conv_res = next(iter(conv_results), None)
        
        if not conv_res or conv_res.status == ConversionStatus.FAILURE:
            logger.error(f"Enhanced Docling failed to convert document from URL: {pdf_url}")
            return []
        
        docling_doc = conv_res.document
        if not docling_doc:
            logger.error(f"Enhanced Docling conversion returned empty document: {pdf_url}")
            return []
        
        # Extract metadata from first few pages
        source_name = document_title or extract_filename_from_url(pdf_url)
        url_hash = get_url_hash(pdf_url)
        
        # Build full text for character position calculation
        full_text = ""
        content_items = []
        
        # First pass: collect all content and build full text
        for item, level in docling_doc.iterate_items(with_groups=False):
            if not isinstance(item, DocItem):
                continue
                
            content = ""
            if isinstance(item, TableItem):
                content = item.export_to_markdown(doc=docling_doc)
            elif hasattr(item, 'text'):
                content = item.text
            
            if content and content.strip():
                page_num = item.prov[0].page_no + 1 if item.prov else 1
                
                # Calculate character positions in full text
                char_start = len(full_text)
                char_end = char_start + len(content)
                
                content_items.append({
                    'content': content,
                    'page_num': page_num,
                    'char_start': char_start,
                    'char_end': char_end,
                    'item': item,
                    'level': level
                })
                
                # Add content to full text with newline separator
                full_text += content + "\n"
        
        # Extract document metadata from first page content
        first_page_content = ""
        for item_data in content_items[:10]:  # First 10 items for metadata
            first_page_content += item_data['content'] + " "
        
        doc_date = extract_date_from_content(first_page_content)
        proceeding_number = extract_proceeding_number(first_page_content) or proceeding or ""
        doc_type = identify_document_type(first_page_content, source_name)
        source_url = get_source_url_from_filename(source_name, proceeding)
        publication_date = get_publication_date_from_filename(source_name, proceeding)
        
        logger.info(f"Enhanced Docling metadata - Date: {doc_date}, Proceeding: {proceeding_number}, Type: {doc_type}")
        
        # Second pass: create enhanced documents with character positions
        for item_data in content_items:
            content = item_data['content']
            page_num = item_data['page_num']
            char_start = item_data['char_start']
            char_end = item_data['char_end']
            item = item_data['item']
            
            # Calculate line range from character positions
            line_start, line_end = estimate_line_range_from_char_position(char_start, char_end, full_text)
            
            # Create text snippet for citation verification
            text_snippet = content[:100].replace('\n', ' ').strip()
            
            # Create enhanced metadata similar to Chonkie
            base_metadata = {
                "source": source_name,
                "source_url": source_url,
                "page": page_num,
                "content_type": item.label.value if hasattr(item, 'label') else "text",
                "chunk_id": f"{source_name}_{url_hash}_{char_start}_{char_end}",
                "url_hash": url_hash,
                "last_checked": datetime.now().isoformat(),
                "document_date": doc_date.isoformat() if doc_date else "",
                "publication_date": publication_date.isoformat() if publication_date else "",
                "proceeding_number": proceeding_number,
                "document_type": doc_type or "unknown",
                "supersedes_priority": _calculate_supersedes_priority(doc_type, doc_date),
            }
            
            # Add enhanced citation metadata
            enhanced_metadata = create_enhanced_chonkie_metadata(
                chunk_info={
                    'text': content,
                    'start_index': char_start,
                    'end_index': char_end,
                    'token_count': len(content.split()),
                    'level': 0,
                    'strategy': 'enhanced_docling'
                },
                source_name=source_name,
                pdf_url=source_url or pdf_url,
                proceeding=proceeding_number,
                raw_text=full_text
            )
            
            # Merge base metadata with enhanced metadata
            final_metadata = {**base_metadata, **enhanced_metadata}
            
            # Override certain fields to ensure consistency
            final_metadata.update({
                "char_start": char_start,
                "char_end": char_end,
                "char_length": char_end - char_start,
                "line_number": line_start,
                "line_range_end": line_end,
                "text_snippet": text_snippet,
                "token_count": len(content.split()),
                "chunk_level": 0
            })
            
            langchain_documents.append(Document(page_content=content, metadata=final_metadata))
        
        logger.info(f"Enhanced Docling fallback successful: {len(langchain_documents)} chunks with character positions")
        return langchain_documents
        
    except Exception as e:
        logger.error(f"Enhanced Docling fallback failed for {pdf_url}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def test_enhanced_docling_fallback():
    """Test the enhanced Docling fallback with a sample PDF."""
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M566/K886/566886171.PDF"
    
    logger.info("Testing enhanced Docling fallback...")
    
    result = _process_with_enhanced_docling_fallback(test_url, proceeding="R2207005")
    
    if result:
        print(f"✅ Enhanced Docling fallback successful: {len(result)} chunks")
        
        # Check enhanced metadata
        sample_doc = result[0]
        metadata = sample_doc.metadata
        
        enhanced_fields = ['char_start', 'char_end', 'char_length', 'line_number', 'text_snippet']
        found_enhanced = [field for field in enhanced_fields if field in metadata]
        
        print(f"Enhanced fields found: {found_enhanced}")
        
        if len(found_enhanced) == len(enhanced_fields):
            print("🎉 All enhanced metadata fields present!")
            print(f"Sample: chars {metadata['char_start']}-{metadata['char_end']}, line {metadata['line_number']}")
            print(f"Snippet: '{metadata['text_snippet']}'")
        else:
            print(f"❌ Missing enhanced fields: {set(enhanced_fields) - set(found_enhanced)}")
    else:
        print("❌ Enhanced Docling fallback failed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_enhanced_docling_fallback()