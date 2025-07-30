#!/usr/bin/env python3
"""
Quick analysis of the massive PDF to understand processing requirements
"""

import requests
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import config

def quick_pdf_analysis(pdf_url: str):
    """Quick analysis without downloading the full PDF."""
    print("🔍 Quick PDF Analysis")
    print("=" * 50)
    print(f"🌐 URL: {pdf_url}")
    
    try:
        # Get headers to understand size
        print("📡 Fetching PDF headers...")
        response = requests.head(pdf_url, timeout=30)
        
        if response.status_code == 200:
            content_length = response.headers.get('content-length')
            content_type = response.headers.get('content-type', 'unknown')
            
            print(f"✅ PDF accessible (Status: {response.status_code})")
            print(f"📄 Content Type: {content_type}")
            
            if content_length:
                size_bytes = int(content_length)
                size_mb = size_bytes / (1024 * 1024)
                
                print(f"📏 File Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
                
                # Processing estimates
                estimated_pages = max(50, int(size_mb * 15))  # Conservative: 15 pages per MB
                estimated_chars = size_bytes * 1.5  # Conservative character estimate
                
                # Get adaptive timeout
                timeout_seconds = config.get_adaptive_timeout(content_size_bytes=size_bytes)
                timeout_minutes = timeout_seconds / 60
                
                print(f"📄 Estimated Pages: ~{estimated_pages:,}")
                print(f"📝 Estimated Characters: ~{estimated_chars:,.0f}")
                print(f"⏱️  Adaptive Timeout: {timeout_seconds}s ({timeout_minutes:.1f} minutes)")
                
                # Processing category
                if size_mb < 1:
                    category = "Small (< 1MB) - Quick processing"
                    urgency = "🟢 Low"
                elif size_mb < 10:
                    category = "Medium (1-10MB) - Some delay expected"
                    urgency = "🟡 Medium" 
                elif size_mb < 50:
                    category = "Large (10-50MB) - Long processing, needs progress"
                    urgency = "🟠 High"
                else:
                    category = "Massive (> 50MB) - Very long processing, CRITICAL need for progress"
                    urgency = "🔴 Critical"
                
                print(f"📂 Category: {category}")
                print(f"🚨 Progress Clarity Urgency: {urgency}")
                
                # Specific recommendations
                print(f"\n💡 Recommendations for this file:")
                if size_mb > 10:
                    print("  🔄 Pre-processing analysis: Essential")
                    print("  📊 Real-time progress updates: Critical")
                    print("  ⏰ ETA calculations: Very helpful")
                    print("  🛑 Cancellation capability: Recommended")
                    
                if size_mb > 50:
                    print("  🔀 Background processing: Consider")
                    print("  💾 Checkpoint saving: Recommended")
                    print("  🔧 Memory optimization: Critical")
                
                # Current processing time estimate
                print(f"\n⏰ Expected Processing Time:")
                print(f"  Download + Extract: {max(2, size_mb/10):.1f} - {max(5, size_mb/5):.1f} minutes")
                print(f"  Chunking: {max(1, estimated_chars/100000):.1f} - {max(2, estimated_chars/50000):.1f} minutes")
                print(f"  Embedding: {max(3, estimated_chars/200000):.1f} - {max(10, estimated_chars/100000):.1f} minutes")
                print(f"  Total Estimate: {timeout_minutes/2:.1f} - {timeout_minutes:.1f} minutes")
                
                print(f"\n⚠️  User Experience Issue:")
                if timeout_minutes > 5:
                    print(f"  Users will see NO FEEDBACK for {timeout_minutes:.1f} minutes!")
                    print(f"  This appears as a frozen/broken system")
                    print(f"  CRITICAL: Implement progress tracking immediately")
                
                return {
                    'size_mb': size_mb,
                    'estimated_pages': estimated_pages,
                    'timeout_minutes': timeout_minutes,
                    'needs_progress': size_mb > 10,
                    'critical_clarity': size_mb > 50
                }
            else:
                print("⚠️ Could not determine file size from headers")
        else:
            print(f"❌ Failed to access PDF (Status: {response.status_code})")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        
    return None

if __name__ == "__main__":
    # The massive PDF mentioned by the user
    pdf_url = "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M151/K988/151988887.PDF"
    
    if len(sys.argv) > 1:
        pdf_url = sys.argv[1]
    
    analysis = quick_pdf_analysis(pdf_url)
    
    if analysis and analysis.get('critical_clarity'):
        print(f"\n🚨 CRITICAL FINDING:")
        print(f"This {analysis['size_mb']:.1f}MB file needs immediate clarity improvements!")
        print(f"Users will wait {analysis['timeout_minutes']:.1f} minutes with no feedback!")