#!/usr/bin/env python3
"""
Test citations to ensure they point to proper CPUC URLs
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.resolve()))

from rag_core import CPUCRAGSystem

def test_citations():
    """Test that citations work properly"""
    print("🔍 Testing citation functionality...")
    
    # Initialize RAG system
    rag = CPUCRAGSystem()
    
    # Get some chunks to test
    sample_chunks = rag.vectordb.get(limit=5)
    
    print(f"📊 Found {len(sample_chunks['ids'])} sample chunks")
    
    for i, (chunk_id, metadata) in enumerate(zip(sample_chunks['ids'], sample_chunks['metadatas'])):
        print(f"\n--- Sample {i+1} ---")
        print(f"ID: {chunk_id}")
        print(f"Source: {metadata.get('source', 'N/A')}")
        print(f"Source URL: {metadata.get('source_url', 'NOT FOUND')}")
        print(f"Page: {metadata.get('page', 'N/A')}")
        
        # Test citation link generation
        source_url = metadata.get('source_url')
        if source_url:
            page = metadata.get('page', 1)
            citation_link = f"{source_url}#page={page}"
            print(f"Citation Link: {citation_link}")
            
            # Check if URL looks correct
            if 'docs.cpuc.ca.gov' in source_url:
                print("✅ Citation URL looks correct (CPUC domain)")
            else:
                print("❌ Citation URL may be incorrect")
        else:
            print("❌ No source_url found in metadata")

if __name__ == "__main__":
    test_citations()