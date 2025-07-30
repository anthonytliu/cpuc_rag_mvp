#!/usr/bin/env python3
"""
Quick test to verify Chonkie API fixes
"""

import logging
import data_processing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_chonkie_api_fix():
    """Test the fixed Chonkie API usage"""
    
    print("🧪 TESTING CHONKIE API FIXES")
    print("=" * 50)
    
    # Test with sample text
    sample_text = """
    This is a test document for Chonkie chunking. It contains multiple sentences 
    and paragraphs to test the chunking functionality. The goal is to verify that 
    the API parameters are correct and the chunking works properly.
    
    This is another paragraph to provide more content for testing. We want to ensure
    that the chunking strategies work without parameter errors.
    """ * 10  # Make it longer
    
    strategies = ["recursive", "sentence", "token"]
    
    for strategy in strategies:
        try:
            print(f"🔄 Testing {strategy} chunking...")
            chunks = data_processing.safe_chunk_with_chonkie(sample_text, strategy)
            print(f"   ✅ {strategy}: {len(chunks)} chunks created")
            
            if chunks:
                print(f"      Sample chunk: {chunks[0][:100]}...")
            
        except Exception as e:
            print(f"   ❌ {strategy} failed: {e}")
    
    print("\n✅ Chonkie API test completed!")

if __name__ == '__main__':
    test_chonkie_api_fix()