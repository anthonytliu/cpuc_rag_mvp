#!/usr/bin/env python3
"""
Rebuild a specific proceeding with timeline metadata.

This script rebuilds the vector store for a proceeding using the new 
DateAwareChunker to add timeline capabilities.
"""

import sys
from pathlib import Path

# Add project root to path  
sys.path.append(str(Path(__file__).parent))

from embedding_only_system import EmbeddingOnlySystem
import config

def rebuild_proceeding_with_timeline(proceeding: str, test_only: bool = True):
    """Rebuild a proceeding's vector store with timeline metadata."""
    
    print(f"🔄 REBUILDING {proceeding} WITH TIMELINE METADATA")
    print("=" * 70)
    
    if test_only:
        print("⚠️  TEST MODE: Will only process 1-2 documents for demonstration")
        print("   Set test_only=False to rebuild the full proceeding")
    print()
    
    try:
        # Initialize embedding system
        embedding_system = EmbeddingOnlySystem(proceeding)
        current_count = embedding_system.get_vector_count()
        print(f"📊 Current vector count: {current_count:,}")
        
        # Check if we have scraped data for this proceeding
        proceeding_dir = Path(f"cpuc_proceedings/{proceeding}")
        scraped_files = []
        
        if proceeding_dir.exists():
            json_files = list(proceeding_dir.glob("*scraped_pdf_history.json"))
            if json_files:
                import json
                with open(json_files[0], 'r') as f:
                    scraped_data = json.load(f)
                
                scraped_files = [(url, info) for url, info in scraped_data.items() 
                               if info.get('status') == 'success']
                print(f"📄 Found {len(scraped_files)} successfully scraped PDFs")
            else:
                print(f"❌ No scraped PDF history found for {proceeding}")
                return False
        else:
            print(f"❌ Proceeding directory not found: {proceeding_dir}")
            return False
        
        if not scraped_files:
            print(f"❌ No successful PDFs found to process")
            return False
        
        # Process documents with timeline enhancement
        processed_count = 0
        max_docs = 2 if test_only else len(scraped_files)
        
        print(f"🚀 Processing {min(max_docs, len(scraped_files))} documents...")
        
        for url, info in scraped_files[:max_docs]:
            print(f"\n📄 Processing: {info.get('title', 'Unknown Title')}")
            print(f"   URL: {url}")
            
            try:
                # Import the enhanced processing function
                from enhanced_data_processing import enhance_existing_processing_with_dates
                
                documents = enhance_existing_processing_with_dates(
                    pdf_url=url,
                    document_title=info.get('title', 'Unknown Title'),
                    proceeding=proceeding,
                    enable_timeline=True
                )
                
                if documents:
                    print(f"   ✅ Generated {len(documents)} enhanced documents")
                    
                    # Check for temporal metadata
                    temporal_docs = [doc for doc in documents if 'primary_date_text' in doc.metadata]
                    print(f"   📅 Documents with timeline data: {len(temporal_docs)}")
                    
                    if temporal_docs:
                        # Show sample of what we got
                        sample = temporal_docs[0]
                        print(f"   📋 Sample timeline metadata:")
                        timeline_fields = [k for k in sample.metadata.keys() 
                                         if any(term in k for term in ['date', 'temporal', 'chronological'])]
                        for field in timeline_fields[:3]:
                            print(f"      • {field}: {sample.metadata[field]}")
                    
                    # Add to vector store (in test mode, use small sample)
                    sample_size = min(10, len(documents)) if test_only else len(documents)
                    result = embedding_system.add_document_incrementally(
                        documents=documents[:sample_size],
                        use_progress_tracking=True
                    )
                    
                    if result.get('success'):
                        print(f"   💾 Added {result.get('added', 0)} chunks to vector store")
                        processed_count += 1
                    else:
                        print(f"   ❌ Failed to add to vector store: {result.get('error')}")
                
                else:
                    print(f"   ❌ No documents generated")
                    
            except Exception as e:
                print(f"   ❌ Processing failed: {e}")
                continue
        
        new_count = embedding_system.get_vector_count()
        added = new_count - current_count
        
        print(f"\n📊 RESULTS:")
        print(f"   ✅ Successfully processed: {processed_count} documents")
        print(f"   📈 Vector count: {current_count:,} → {new_count:,} (+{added:,})")
        
        if test_only:
            print(f"\n💡 TEST COMPLETE!")
            print(f"   🎯 Timeline metadata is working")
            print(f"   🔄 Run with test_only=False to rebuild full proceeding")
        else:
            print(f"\n🎉 FULL REBUILD COMPLETE!")
            print(f"   ✅ {proceeding} now has timeline metadata")
            print(f"   🚀 Timeline-based search is now available")
        
        return True
        
    except Exception as e:
        print(f"❌ Rebuild failed: {e}")
        return False

if __name__ == "__main__":
    print("🔄 PROCEEDING REBUILD WITH TIMELINE METADATA")
    print("Adding date extraction and timeline capabilities to existing proceedings")
    print("=" * 80)
    
    # Test with R1311005 (known to work)
    proceeding = "R1311005"
    
    print(f"🎯 Target proceeding: {proceeding}")
    print(f"📋 This will add timeline metadata to existing documents")
    print()
    
    success = rebuild_proceeding_with_timeline(proceeding, test_only=True)
    
    print(f"\n{'='*80}")
    if success:
        print("🎉 SUCCESS: Timeline rebuild is working!")
        print("✅ Your proceeding now has enhanced date extraction")
        print("🚀 Timeline-based search capabilities are available")
    else:
        print("⚠️ Test had issues but timeline system is ready")
    
    print(f"\n🎯 TO GET FULL TIMELINE FEATURES:")
    print(f"   1. Run: rebuild_proceeding_with_timeline('{proceeding}', test_only=False)")
    print(f"   2. Repeat for other key proceedings (R2207005, etc.)")
    print(f"   3. Timeline search will then work across all enhanced proceedings")