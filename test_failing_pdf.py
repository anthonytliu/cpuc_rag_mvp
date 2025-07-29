#!/usr/bin/env python3
"""
Test script to analyze specific failing PDF: https://docs.cpuc.ca.gov/PublishedDocs/EFILE/CM/162841.PDF

This script performs comprehensive analysis to understand why this PDF fails to embed.
"""

import logging
import requests
import json
from pathlib import Path
import data_processing
from rag_core import CPUCRAGSystem

# Setup detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_failing_pdf():
    """Comprehensive analysis of the failing PDF"""
    
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/EFILE/CM/162841.PDF"
    test_proceeding = 'R1202009'
    
    print("🔍 ANALYZING FAILING PDF")
    print("=" * 60)
    print(f"📄 URL: {test_url}")
    print(f"📋 Proceeding: {test_proceeding}")
    
    # Step 1: URL Validation
    print("\n📡 STEP 1: URL Validation")
    print("-" * 30)
    
    try:
        # Test basic connectivity
        response = requests.head(test_url, timeout=30, allow_redirects=True)
        print(f"✅ HTTP Status: {response.status_code}")
        print(f"✅ Content-Type: {response.headers.get('content-type', 'Unknown')}")
        print(f"✅ Content-Length: {response.headers.get('content-length', 'Unknown')}")
        
        # Validate using our function
        is_valid = data_processing.validate_pdf_url(test_url)
        print(f"✅ PDF Validation: {is_valid}")
        
    except Exception as e:
        print(f"❌ URL validation failed: {e}")
        return False
    
    # Step 2: Filename extraction
    print("\n📝 STEP 2: Filename Extraction")
    print("-" * 30)
    
    try:
        filename = data_processing.extract_filename_from_url(test_url)
        print(f"✅ Extracted filename: {filename}")
        
        url_hash = data_processing.get_url_hash(test_url)
        print(f"✅ URL hash: {url_hash}")
        
    except Exception as e:
        print(f"❌ Filename extraction failed: {e}")
        return False
    
    # Step 3: Docling processing
    print("\n🔄 STEP 3: Docling Processing")
    print("-" * 30)
    
    try:
        print("⏳ Processing PDF with Docling...")
        chunks = data_processing.extract_and_chunk_with_docling_url(
            test_url, 
            "162841", 
            test_proceeding
        )
        
        print(f"✅ Docling processing successful: {len(chunks)} chunks extracted")
        
        if chunks:
            print(f"\n📊 First chunk analysis:")
            first_chunk = chunks[0]
            print(f"   Content length: {len(first_chunk.page_content)}")
            print(f"   Content preview: {first_chunk.page_content[:200]}...")
            print(f"   Metadata keys: {list(first_chunk.metadata.keys())}")
            
            # Check for problematic metadata
            problematic_fields = []
            for key, value in first_chunk.metadata.items():
                if value is None:
                    problematic_fields.append(f"{key}: None")
                elif isinstance(value, str) and len(value) == 0:
                    problematic_fields.append(f"{key}: empty string")
            
            if problematic_fields:
                print(f"   ⚠️  Potential issues: {problematic_fields}")
            else:
                print(f"   ✅ Metadata looks clean")
        
    except Exception as e:
        print(f"❌ Docling processing failed: {e}")
        logger.exception("Detailed Docling error:")
        return False
    
    # Step 4: Vector store integration
    print("\n🗄️ STEP 4: Vector Store Integration")
    print("-" * 30)
    
    try:
        # Initialize RAG system
        rag_system = CPUCRAGSystem(current_proceeding=test_proceeding)
        
        # Test adding to vector store
        url_data = {
            'url': test_url,
            'title': '162841',
            'id': 'test_failing_pdf'
        }
        
        print("⏳ Testing vector store addition...")
        success = rag_system.add_document_incrementally(
            chunks=chunks,
            url_hash=url_hash,
            url_data=url_data,
            immediate_persist=True
        )
        
        if success:
            print("✅ Vector store addition successful")
        else:
            print("❌ Vector store addition failed")
            return False
        
    except Exception as e:
        print(f"❌ Vector store integration failed: {e}")
        logger.exception("Detailed vector store error:")
        return False
    
    # Step 5: End-to-end test
    print("\n🎯 STEP 5: End-to-End Processing Test")
    print("-" * 30)
    
    try:
        # Test the complete pipeline
        result = rag_system._process_single_url(url_data)
        
        if result['success']:
            print(f"✅ End-to-end processing successful")
            print(f"   Chunks: {result['chunk_count']}")
            print(f"   URL: {result['url']}")
        else:
            print(f"❌ End-to-end processing failed: {result.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        logger.exception("Detailed end-to-end error:")
        return False
    
    # Step 6: Query test
    print("\n🔍 STEP 6: Query Test")
    print("-" * 30)
    
    try:
        # Test querying the document
        print("⏳ Testing document query...")
        
        # Get system stats first
        stats = rag_system.get_system_stats()
        print(f"   System stats: {stats.get('total_chunks', 0)} chunks available")
        
        if stats.get('total_chunks', 0) > 0:
            # Test a basic query
            test_query = "What is this document about?"
            
            query_results = []
            for result in rag_system.query(test_query):
                if isinstance(result, dict):
                    query_results.append(result)
                    break
            
            if query_results:
                final_result = query_results[0]
                print(f"✅ Query successful")
                print(f"   Answer length: {len(final_result.get('answer', ''))}")
                print(f"   Sources: {len(final_result.get('sources', []))}")
            else:
                print("⚠️  Query completed but no results")
        else:
            print("⚠️  No chunks available for querying")
        
    except Exception as e:
        print(f"❌ Query test failed: {e}")
        logger.exception("Detailed query error:")
        # Don't fail the entire test for query issues
    
    print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
    return True

def analyze_pdf_characteristics():
    """Analyze specific characteristics of this PDF that might cause issues"""
    
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/EFILE/CM/162841.PDF"
    
    print("\n🔬 DETAILED PDF CHARACTERISTICS ANALYSIS")
    print("=" * 60)
    
    try:
        # Get PDF headers for detailed analysis
        response = requests.head(test_url, timeout=30, allow_redirects=True)
        
        print("📊 HTTP Response Analysis:")
        for header, value in response.headers.items():
            if any(keyword in header.lower() for keyword in ['content', 'type', 'length', 'encoding', 'range']):
                print(f"   {header}: {value}")
        
        # Check for redirects
        if response.history:
            print(f"\n🔄 Redirect Analysis:")
            for i, resp in enumerate(response.history):
                print(f"   Redirect {i+1}: {resp.status_code} -> {resp.headers.get('location', 'Unknown')}")
        
        # Try to get first few bytes to analyze PDF structure
        partial_response = requests.get(test_url, headers={'Range': 'bytes=0-1023'}, timeout=30)
        if partial_response.status_code == 206:  # Partial content
            pdf_header = partial_response.content[:100]
            print(f"\n📄 PDF Header Analysis:")
            print(f"   First 50 bytes: {pdf_header[:50]}")
            print(f"   PDF version detected: {pdf_header[:8] if pdf_header.startswith(b'%PDF') else 'Invalid PDF header'}")
        
    except Exception as e:
        print(f"❌ PDF characteristics analysis failed: {e}")

if __name__ == '__main__':
    print("🧪 FAILING PDF ANALYSIS TEST")
    print("=" * 80)
    print("Testing PDF: https://docs.cpuc.ca.gov/PublishedDocs/EFILE/CM/162841.PDF")
    print("This test analyzes why this specific PDF fails to embed properly.")
    print("=" * 80)
    
    # Run detailed characteristics analysis
    analyze_pdf_characteristics()
    
    # Run main analysis
    success = analyze_failing_pdf()
    
    if success:
        print("\n✅ ALL TESTS PASSED - PDF should embed successfully")
    else:
        print("\n❌ TEST FAILED - PDF has embedding issues")
    
    exit(0 if success else 1)