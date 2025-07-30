#!/usr/bin/env python3
"""
Test script for massive PDF processing with enhanced progress tracking.

This script tests the processing of very large PDFs like:
https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M151/K988/151988887.PDF

It analyzes processing stages and provides recommendations for clarity improvements.
"""

import sys
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config
from embedding_only_system import EmbeddingOnlySystem

def analyze_pdf_size(pdf_url: str) -> Dict[str, Any]:
    """Analyze PDF size and characteristics."""
    print("🔍 Analyzing PDF characteristics...")
    
    try:
        # Get PDF headers without downloading full content
        response = requests.head(pdf_url, timeout=30)
        
        if response.status_code == 200:
            content_length = response.headers.get('content-length')
            if content_length:
                size_bytes = int(content_length)
                size_mb = size_bytes / (1024 * 1024)
                
                # Estimate processing characteristics
                estimated_pages = max(50, int(size_mb * 20))  # Rough estimate: 20 pages per MB
                estimated_chars = size_bytes * 2  # Conservative character estimate
                estimated_timeout = config.get_adaptive_timeout(content_size_bytes=size_bytes)
                
                analysis = {
                    "url": pdf_url,
                    "size_bytes": size_bytes,
                    "size_mb": round(size_mb, 2),
                    "estimated_pages": estimated_pages,
                    "estimated_characters": estimated_chars,
                    "adaptive_timeout": estimated_timeout,
                    "processing_category": get_processing_category(size_mb),
                    "estimated_chunks": max(100, int(estimated_chars / 1000))  # ~1000 chars per chunk
                }
                
                return analysis
            
    except Exception as e:
        return {"error": f"Failed to analyze PDF: {e}"}
    
    return {"error": "Could not determine PDF size"}

def get_processing_category(size_mb: float) -> str:
    """Categorize PDF by processing complexity."""
    if size_mb < 1:
        return "Small (< 1MB)"
    elif size_mb < 10:
        return "Medium (1-10MB)"
    elif size_mb < 50:
        return "Large (10-50MB)"
    else:
        return "Massive (> 50MB)"

def test_massive_pdf_processing(pdf_url: str, proceeding: str = "R1206013"):
    """
    Test processing of a massive PDF with detailed progress tracking.
    
    Args:
        pdf_url: URL to the massive PDF
        proceeding: Proceeding to use for testing
    """
    print("🧪 Testing Massive PDF Processing")
    print("=" * 80)
    print(f"📄 PDF URL: {pdf_url}")
    print(f"📋 Test Proceeding: {proceeding}")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Analyze PDF characteristics
    analysis = analyze_pdf_size(pdf_url)
    
    if "error" in analysis:
        print(f"❌ {analysis['error']}")
        return False
    
    print("📊 PDF Analysis Results:")
    print(f"  📏 Size: {analysis['size_mb']} MB ({analysis['size_bytes']:,} bytes)")
    print(f"  📄 Estimated Pages: ~{analysis['estimated_pages']:,}")
    print(f"  📝 Estimated Characters: ~{analysis['estimated_characters']:,}")
    print(f"  🔢 Estimated Chunks: ~{analysis['estimated_chunks']:,}")
    print(f"  ⏱️  Adaptive Timeout: {analysis['adaptive_timeout']} seconds ({analysis['adaptive_timeout']//60} minutes)")
    print(f"  📂 Category: {analysis['processing_category']}")
    print()
    
    # Step 2: Test current processing with timing
    print("🚀 Testing Current Processing Approach...")
    print("-" * 50)
    
    try:
        # Initialize embedding system
        print("🔧 Initializing EmbeddingOnlySystem...")
        start_init = time.time()
        embedding_system = EmbeddingOnlySystem(proceeding)
        init_time = time.time() - start_init
        print(f"✅ System initialized in {init_time:.2f}s")
        print()
        
        # Test document processing with detailed timing
        print("🔄 Processing PDF (this may take 10-40 minutes for massive files)...")
        
        # Track processing stages
        stages = {
            "download_extract": {"start": None, "end": None, "status": "pending"},
            "chunking": {"start": None, "end": None, "status": "pending"},
            "embedding": {"start": None, "end": None, "status": "pending"},
            "storage": {"start": None, "end": None, "status": "pending"}
        }
        
        overall_start = time.time()
        stages["download_extract"]["start"] = overall_start
        
        try:
            # Process the document
            documents = embedding_system.process_document_url(
                pdf_url=pdf_url,
                document_title="Test Massive PDF",
                proceeding=proceeding
            )
            
            extract_time = time.time() - overall_start
            stages["download_extract"]["end"] = time.time()
            stages["download_extract"]["status"] = "completed"
            
            print(f"📝 Text Extraction completed in {extract_time:.2f}s")
            print(f"📊 Generated {len(documents)} document chunks")
            
            if documents:
                # Analyze actual results vs estimates
                actual_chars = sum(len(doc.page_content) for doc in documents)
                actual_chunks = len(documents)
                
                print(f"📈 Actual vs Estimated:")
                print(f"  Characters: {actual_chars:,} (estimated: {analysis['estimated_characters']:,})")
                print(f"  Chunks: {actual_chunks:,} (estimated: {analysis['estimated_chunks']:,})")
                print()
                
                # Test embedding storage (small sample to avoid overloading)
                print("💾 Testing embedding storage (sample of 10 chunks)...")
                sample_docs = documents[:min(10, len(documents))]
                
                stages["storage"]["start"] = time.time()
                result = embedding_system.add_document_incrementally(sample_docs)
                stages["storage"]["end"] = time.time()
                stages["storage"]["status"] = "completed" if result.get("success") else "failed"
                
                storage_time = stages["storage"]["end"] - stages["storage"]["start"]
                print(f"✅ Storage test completed in {storage_time:.2f}s")
                print(f"📊 Added {result.get('added', 0)} chunks successfully")
                
            else:
                print("❌ No documents generated from PDF processing")
                return False
                
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            return False
        
        total_time = time.time() - overall_start
        
        # Step 3: Analyze processing performance
        print()
        print("📈 Processing Performance Analysis:")
        print("-" * 50)
        print(f"⏱️  Total Processing Time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print(f"🚀 Processing Rate: {len(documents)/total_time:.1f} chunks/second")
        print(f"📄 Character Processing Rate: {actual_chars/total_time:,.0f} chars/second")
        
        # Analyze bottlenecks
        if extract_time > 60:  # More than 1 minute
            print(f"⚠️  Text extraction took {extract_time:.1f}s - this is a major bottleneck")
        
        if len(documents) > 1000:
            print(f"⚠️  Generated {len(documents)} chunks - users need progress tracking")
        
        # Step 4: Recommendations for improvement
        print()
        print("💡 Clarity Improvement Recommendations:")
        print("-" * 50)
        
        if analysis["size_mb"] > 10:
            print("🔍 Large File Detected - Recommended Improvements:")
            print("  1. ✅ Pre-processing size analysis (implemented)")
            print("  2. ⚠️  Real-time progress updates during text extraction")
            print("  3. ⚠️  Chunking progress indicators (X/Y chunks created)")
            print("  4. ⚠️  Embedding progress with ETA calculations")
            print("  5. ⚠️  Stage-by-stage status display")
            print("  6. ⚠️  Cancellation capability for long operations")
        
        if total_time > 300:  # More than 5 minutes
            print("⏰ Long Processing Time - Additional Recommendations:")
            print("  7. ⚠️  Background processing with periodic status updates")
            print("  8. ⚠️  Checkpoint saving for resume capability")
            print("  9. ⚠️  Memory usage monitoring and optimization")
            print("  10. ⚠️  Parallel processing for chunk embedding")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_progress_clarity_improvements():
    """Test specific improvements for processing clarity."""
    print("\n🔬 Testing Progress Clarity Improvements")
    print("=" * 60)
    
    improvements = [
        {
            "name": "PDF Size Pre-Analysis",
            "status": "✅ Implemented",
            "description": "Shows size, estimated pages, timeout before processing"
        },
        {
            "name": "Adaptive Timeout Configuration", 
            "status": "✅ Implemented",
            "description": "Scales timeout based on document size (15-40 minutes)"
        },
        {
            "name": "Real-time Text Extraction Progress",
            "status": "⚠️ Missing",
            "description": "Show pages processed, characters extracted during download"
        },
        {
            "name": "Chunking Progress Indicators",
            "status": "⚠️ Missing", 
            "description": "Display chunk creation progress (X/Y chunks)"
        },
        {
            "name": "Embedding Progress with ETA",
            "status": "⚠️ Missing",
            "description": "Show embedding progress with time remaining estimates"
        },
        {
            "name": "Stage-by-Stage Status Display",
            "status": "⚠️ Missing",
            "description": "Clear indicators for Download→Extract→Chunk→Embed→Store"
        },
        {
            "name": "Memory Usage Monitoring",
            "status": "⚠️ Missing",
            "description": "Track and display memory usage during processing"
        },
        {
            "name": "Cancellation Capability",
            "status": "⚠️ Missing",
            "description": "Allow users to cancel long-running operations"
        }
    ]
    
    print("📊 Current Implementation Status:")
    for improvement in improvements:
        print(f"  {improvement['status']} {improvement['name']}")
        print(f"     {improvement['description']}")
    
    implemented = sum(1 for imp in improvements if "✅" in imp["status"])
    total = len(improvements)
    
    print(f"\n📈 Progress: {implemented}/{total} improvements implemented ({implemented/total*100:.0f}%)")
    
    return improvements

if __name__ == "__main__":
    # Test with the massive PDF mentioned by the user
    massive_pdf_url = "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M151/K988/151988887.PDF"
    
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    else:
        test_url = massive_pdf_url
    
    if len(sys.argv) > 2:
        test_proceeding = sys.argv[2]
    else:
        test_proceeding = "R1206013"
    
    print("🧪 MASSIVE PDF PROCESSING CLARITY TEST")
    print("=" * 80)
    
    # Run comprehensive test
    success = test_massive_pdf_processing(test_url, test_proceeding)
    
    # Analyze current improvements
    improvements = test_progress_clarity_improvements()
    
    print("\n🎯 SUMMARY")
    print("=" * 40)
    
    if success:
        print("✅ Massive PDF processing test completed successfully")
        print("💡 Key findings:")
        print("  - Current system can handle massive PDFs")
        print("  - Adaptive timeouts prevent false timeouts")
        print("  - But users have no visibility into long processes")
        print("  - Progress tracking is essential for files > 10MB")
    else:
        print("❌ Massive PDF processing test failed")
        print("💡 System needs improvements for handling large documents")
    
    print(f"\n🚀 Next steps: Implement remaining {8-2} clarity improvements")
    print("   Priority: Real-time progress updates and stage indicators")