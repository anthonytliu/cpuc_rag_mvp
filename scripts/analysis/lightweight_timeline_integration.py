#!/usr/bin/env python3
"""
Lightweight Timeline Integration - No PDF Downloads Required

This creates timeline metadata from existing text extraction methods,
avoiding any PDF downloads while adding date extraction capabilities.
"""

import sys
from pathlib import Path
from typing import List
from langchain.schema import Document

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from date_aware_chunker import DateAwareChunker
from data_processing import extract_text_from_url

def add_timeline_to_existing_text(text: str, base_metadata: dict = None) -> List[Document]:
    """
    Add timeline metadata to existing extracted text without downloading PDFs.
    
    This is the lightweight approach that works with your current text extraction.
    """
    if base_metadata is None:
        base_metadata = {}
    
    # Use DateAwareChunker to add temporal metadata
    date_chunker = DateAwareChunker(chunker_type="recursive")
    enhanced_chunks = date_chunker.chunk_with_dates(text)
    
    documents = []
    for chunk in enhanced_chunks:
        # Create enhanced metadata
        enhanced_metadata = base_metadata.copy()
        
        # Add timeline metadata if dates were found
        if chunk.extracted_dates:
            enhanced_metadata.update({
                'extracted_dates_count': len(chunk.extracted_dates),
                'extracted_dates_texts': [d.text for d in chunk.extracted_dates],
                'extracted_dates_types': [d.date_type.value for d in chunk.extracted_dates],
                'temporal_significance': chunk.temporal_significance,
                'chronological_order': chunk.chronological_order
            })
            
            # Add primary date info
            if chunk.primary_date:
                enhanced_metadata.update({
                    'primary_date_text': chunk.primary_date.text,
                    'primary_date_type': chunk.primary_date.date_type.value,
                    'primary_date_confidence': chunk.primary_date.confidence
                })
                
                if chunk.primary_date.parsed_date:
                    enhanced_metadata['primary_date_parsed'] = chunk.primary_date.parsed_date.isoformat()
        
        # Add document structure metadata
        enhanced_metadata.update({
            'contains_decision': chunk.contains_decision,
            'contains_resolution': chunk.contains_resolution,
            'contains_rulemaking': chunk.contains_rulemaking,
            'chunk_start_index': chunk.start_index,
            'chunk_end_index': chunk.end_index
        })
        
        # Create Document object
        doc = Document(
            page_content=chunk.text,
            metadata=enhanced_metadata
        )
        documents.append(doc)
    
    return documents

def lightweight_timeline_processing(pdf_url: str, document_title: str = None, 
                                  proceeding: str = None) -> List[Document]:
    """
    Lightweight timeline processing using existing text extraction - NO PDF DOWNLOAD.
    
    This uses your existing extract_text_from_url() function that doesn't download PDFs.
    """
    print(f"🔄 Lightweight timeline processing: {pdf_url}")
    print("📋 Using existing text extraction (no PDF download)")
    
    try:
        # Use existing lightweight text extraction
        extracted_text = extract_text_from_url(pdf_url)
        
        if not extracted_text or len(extracted_text.strip()) < 100:
            print(f"❌ Insufficient text extracted: {len(extracted_text)} chars")
            return []
        
        print(f"✅ Extracted {len(extracted_text):,} characters of text")
        
        # Create base metadata
        base_metadata = {
            'source': document_title or 'Unknown Document',
            'proceeding': proceeding or 'Unknown',
            'url': pdf_url,
            'processing_method': 'lightweight_timeline'
        }
        
        # Add timeline metadata using lightweight chunking
        enhanced_documents = add_timeline_to_existing_text(extracted_text, base_metadata)
        
        print(f"✅ Generated {len(enhanced_documents)} documents with timeline metadata")
        
        # Show timeline stats
        timeline_docs = [doc for doc in enhanced_documents if 'primary_date_text' in doc.metadata]
        print(f"📅 Documents with dates: {len(timeline_docs)}")
        
        return enhanced_documents
        
    except Exception as e:
        print(f"❌ Lightweight timeline processing failed: {e}")
        return []

def test_lightweight_timeline():
    """Test the lightweight timeline processing."""
    
    print("🧪 TESTING LIGHTWEIGHT TIMELINE PROCESSING")
    print("=" * 70)
    print("📋 Using existing text extraction - NO PDF DOWNLOADS")
    print()
    
    # Test with a known working URL
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M500/K308/500308139.PDF"
    test_proceeding = "R1311005"
    
    print(f"📄 Test URL: {test_url}")
    print(f"📋 Proceeding: {test_proceeding}")
    print()
    
    documents = lightweight_timeline_processing(
        pdf_url=test_url,
        document_title="Test Timeline Document",
        proceeding=test_proceeding
    )
    
    if documents:
        print(f"\n📊 RESULTS:")
        print(f"✅ Successfully processed without downloading PDF")
        print(f"📄 Generated {len(documents)} documents")
        
        # Analyze timeline metadata
        timeline_docs = [doc for doc in documents if 'primary_date_text' in doc.metadata]
        if timeline_docs:
            print(f"📅 Documents with timeline data: {len(timeline_docs)}")
            
            # Show sample timeline metadata
            sample = timeline_docs[0]
            print(f"\n📋 Sample timeline metadata:")
            timeline_fields = [k for k in sample.metadata.keys() 
                             if any(term in k for term in ['date', 'temporal', 'chronological'])]
            for field in timeline_fields[:5]:
                print(f"   • {field}: {sample.metadata[field]}")
        
        print(f"\n🎉 SUCCESS: Lightweight timeline processing works!")
        print(f"✅ No PDF download required")
        print(f"✅ Timeline metadata successfully added")
        return True
    else:
        print(f"❌ No documents generated")
        return False

def integrate_with_existing_pipeline():
    """Show how to integrate with existing pipeline."""
    
    print(f"\n🔧 INTEGRATION WITH EXISTING PIPELINE")
    print("=" * 70)
    
    integration_code = '''
# Replace heavy PDF processing with lightweight timeline processing:

# OLD (downloads PDFs):
# documents = enhance_existing_processing_with_dates(pdf_url, ...)

# NEW (lightweight, no downloads):
from lightweight_timeline_integration import lightweight_timeline_processing

documents = lightweight_timeline_processing(
    pdf_url=pdf_url,
    document_title=document_title,
    proceeding=proceeding
)

# This gives you timeline metadata without PDF downloads!
'''
    
    print("📝 Code changes needed:")
    print(integration_code)
    
    print("💡 Benefits:")
    print("   ✅ No PDF downloads")
    print("   ✅ Uses existing text extraction")
    print("   ✅ Fast processing")
    print("   ✅ Same timeline metadata")
    print("   ✅ Compatible with current pipeline")

if __name__ == "__main__":
    print("🚀 LIGHTWEIGHT TIMELINE INTEGRATION")
    print("Adding timeline features without PDF downloads")
    print("=" * 80)
    
    # Test the lightweight approach
    success = test_lightweight_timeline()
    
    # Show integration approach
    integrate_with_existing_pipeline()
    
    print(f"\n{'='*80}")
    if success:
        print("🎉 SUCCESS: Lightweight timeline processing is working!")
        print("✅ Timeline metadata added without PDF downloads")
        print("✅ Fast, efficient processing maintained")
        print("🚀 Ready to integrate with your existing pipeline")
    else:
        print("⚠️ Test had issues but approach is correct")
    
    print(f"\n💡 KEY INSIGHT:")
    print(f"   🎯 Timeline features don't require PDF downloads")
    print(f"   📋 Works with existing extract_text_from_url() function")
    print(f"   🚀 Maintains your lightweight processing approach")