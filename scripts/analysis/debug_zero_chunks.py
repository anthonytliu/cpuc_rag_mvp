#!/usr/bin/env python3
"""
Debug script to understand why certain PDFs extract 0 chunks from Docling.
"""

import logging
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.backend.docling_parse_v4_backend import DoclingParseV4DocumentBackend
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import PdfFormatOption

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_docling_extraction():
    """Debug Docling extraction process in detail"""
    
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/EFILE/CM/162841.PDF"
    
    print("🔬 DEBUGGING DOCLING ZERO CHUNKS ISSUE")
    print("=" * 60)
    print(f"📄 URL: {test_url}")
    
    # Configure Docling converter with same settings as production
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
        print("\n🔄 Converting document with Docling...")
        conv_results = doc_converter.convert_all([test_url], raises_on_error=False)
        conv_res = next(iter(conv_results), None)
        
        if not conv_res:
            print("❌ No conversion result returned")
            return
        
        print(f"✅ Conversion status: {conv_res.status}")
        
        if hasattr(conv_res, 'document'):
            docling_doc = conv_res.document
            print(f"✅ Document object created: {type(docling_doc)}")
            
            # Analyze document structure
            print(f"\n📊 Document Analysis:")
            print(f"   Document has {len(list(docling_doc.iterate_items()))} total items")
            
            # Count different types of items
            item_types = {}
            text_items = 0
            table_items = 0
            content_items = 0
            
            for item, level in docling_doc.iterate_items(with_groups=False):
                item_type = type(item).__name__
                item_types[item_type] = item_types.get(item_type, 0) + 1
                
                if hasattr(item, 'text') and item.text:
                    text_items += 1
                    if item.text.strip():
                        content_items += 1
                        
                from docling.datamodel.document import TableItem
                if isinstance(item, TableItem):
                    table_items += 1
            
            print(f"   Item types: {item_types}")
            print(f"   Items with text: {text_items}")
            print(f"   Items with content: {content_items}")
            print(f"   Table items: {table_items}")
            
            # Examine first few items in detail
            print(f"\n🔍 Detailed Item Analysis:")
            for i, (item, level) in enumerate(docling_doc.iterate_items(with_groups=False)):
                if i >= 10:  # Look at first 10 items
                    break
                    
                print(f"   Item {i+1}:")
                print(f"      Type: {type(item).__name__}")
                print(f"      Label: {getattr(item, 'label', 'No label')}")
                
                if hasattr(item, 'text'):
                    text_preview = (item.text[:100] + "...") if len(item.text) > 100 else item.text
                    print(f"      Text: {repr(text_preview)}")
                    print(f"      Text length: {len(item.text) if item.text else 0}")
                    print(f"      Has content: {bool(item.text and item.text.strip())}")
                
                if hasattr(item, 'prov') and item.prov:
                    page_num = item.prov[0].page_no + 1 if item.prov else 0
                    print(f"      Page: {page_num}")
                
                print()
            
            # Now simulate the actual chunk creation logic
            print(f"\n🧪 Simulating Chunk Creation Logic:")
            chunk_count = 0
            empty_content_count = 0
            
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
                    chunk_count += 1
                    if chunk_count <= 5:  # Show first 5 chunks
                        print(f"   Chunk {chunk_count}: {len(content)} chars - {repr(content[:100])}")
                else:
                    empty_content_count += 1
            
            print(f"\n📈 Chunk Creation Results:")
            print(f"   Total chunks created: {chunk_count}")
            print(f"   Empty content items skipped: {empty_content_count}")
            
            if chunk_count == 0:
                print(f"\n❌ PROBLEM IDENTIFIED: All document items have empty content!")
                print(f"   This explains why 0 chunks are extracted.")
                print(f"   The PDF content might be:")
                print(f"   - Images/scanned content without OCR")
                print(f"   - Corrupted or malformed text")
                print(f"   - Non-standard PDF structure")
            else:
                print(f"\n✅ Chunks should be extracted successfully")
        
        else:
            print("❌ No document object in conversion result")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        logger.exception("Detailed debug error:")

if __name__ == '__main__':
    debug_docling_extraction()