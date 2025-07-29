#!/usr/bin/env python3
"""
Debug script to explore Docling provenance information to understand
what line-level data is available for citation enhancement.
"""

import logging
from pathlib import Path
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.document import DocItem, TableItem
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def explore_docling_provenance(pdf_url: str = "https://docs.cpuc.ca.gov/SearchRes.aspx?DocFormat=ALL&DocID=500762062"):
    """
    Explore what provenance information Docling provides for each content item.
    This will help us understand if line numbers are available.
    """
    print(f"🔍 Exploring Docling provenance for: {pdf_url}")
    
    # Configure Docling converter
    pipeline_options = PdfPipelineOptions()
    pipeline_options.table_structure_options.mode = TableFormerMode.FAST
    
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                backend=DoclingParseV4DocumentBackend,
                pipeline_options=pipeline_options
            )
        }
    )
    
    try:
        # Convert PDF
        print("📄 Converting PDF with Docling...")
        conv_results = doc_converter.convert_all([pdf_url], raises_on_error=False)
        conv_res = next(iter(conv_results), None)

        if not conv_res or conv_res.status == ConversionStatus.FAILURE:
            print(f"❌ Docling failed to convert document: {pdf_url}")
            return

        docling_doc = conv_res.document
        print(f"✅ Successfully converted document")
        
        # Explore first few items to understand provenance structure
        print("\n🔬 Analyzing provenance structure:")
        print("=" * 60)
        
        item_count = 0
        for item, level in docling_doc.iterate_items(with_groups=False):
            if not isinstance(item, DocItem) or item_count >= 5:  # Limit to first 5 items
                continue
            
            item_count += 1
            print(f"\nItem #{item_count}:")
            print(f"  Type: {type(item).__name__}")
            print(f"  Label: {item.label.value if hasattr(item, 'label') else 'N/A'}")
            
            # Explore provenance information
            if hasattr(item, 'prov') and item.prov:
                print(f"  Provenance available: {len(item.prov)} prov entries")
                for i, prov_entry in enumerate(item.prov):
                    print(f"    Prov #{i}:")
                    print(f"      Type: {type(prov_entry).__name__}")
                    
                    # Check all available attributes
                    for attr_name in dir(prov_entry):
                        if not attr_name.startswith('_'):
                            try:
                                attr_value = getattr(prov_entry, attr_name)
                                if not callable(attr_value):
                                    print(f"      {attr_name}: {attr_value}")
                            except Exception as e:
                                print(f"      {attr_name}: <error: {e}>")
            else:
                print("  No provenance information available")
            
            # Show content preview
            if hasattr(item, 'text') and item.text:
                content_preview = item.text[:100].replace('\n', ' ')
                print(f"  Content preview: {content_preview}...")
            
            print("-" * 40)
            
            if item_count >= 5:
                break
        
        # Check if we can access bbox or coordinate information
        print(f"\n📊 Summary: Analyzed {item_count} items for provenance information")
        
    except Exception as e:
        print(f"❌ Error exploring Docling provenance: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explore_docling_provenance()