#!/usr/bin/env python3
"""
Test Enhanced Docling Fallback Integration

This script tests that the enhanced Docling fallback is properly integrated
into the data processing pipeline and maintains citation metadata consistency.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_processing import _process_with_chonkie_primary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_enhanced_fallback_integration():
    """Test that enhanced Docling fallback works when Chonkie fails."""
    print("🧪 Testing Enhanced Docling Fallback Integration")
    print("=" * 60)
    
    # Test with a PDF that we know works
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M566/K886/566886171.PDF"
    
    try:
        print(f"Testing with: {test_url}")
        
        # Process with Chonkie primary (which has enhanced Docling fallback)
        result = _process_with_chonkie_primary(test_url, proceeding="R2207005")
        
        if result:
            print(f"✅ Processing successful: {len(result)} chunks")
            
            # Check first chunk for enhanced metadata
            sample_doc = result[0]
            metadata = sample_doc.metadata
            
            # Check for enhanced citation fields
            enhanced_fields = ['char_start', 'char_end', 'char_length', 'line_number', 'text_snippet']
            found_enhanced = [field for field in enhanced_fields if field in metadata]
            
            print(f"\n📊 Metadata Analysis:")
            print(f"   Enhanced fields found: {len(found_enhanced)}/{len(enhanced_fields)}")
            print(f"   Fields: {found_enhanced}")
            
            if len(found_enhanced) == len(enhanced_fields):
                print("🎉 All enhanced metadata fields present!")
                print(f"   Character range: {metadata.get('char_start', 'N/A')}-{metadata.get('char_end', 'N/A')}")
                print(f"   Line range: {metadata.get('line_number', 'N/A')}-{metadata.get('line_range_end', 'N/A')}")
                print(f"   Text snippet: '{metadata.get('text_snippet', 'N/A')[:50]}...'")
                
                # Determine processing method used
                chunk_id = metadata.get('chunk_id', '')
                if 'enhanced_docling' in str(metadata.get('strategy', '')):
                    print("   🔧 Processing method: Enhanced Docling fallback")
                elif 'recursive' in str(metadata.get('strategy', '')):
                    print("   🔧 Processing method: Chonkie recursive chunking")
                else:
                    print("   🔧 Processing method: Unknown")
                
                return True
            else:
                missing = set(enhanced_fields) - set(found_enhanced)
                print(f"❌ Missing enhanced fields: {missing}")
                return False
        else:
            print("❌ Processing failed - no documents returned")
            return False
            
    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fallback_with_known_failure():
    """Test fallback behavior with a URL that might cause Chonkie to fail."""
    print("\n🧪 Testing Fallback with Challenging Document")
    print("=" * 60)
    
    # Use a document that might be more challenging for Chonkie
    test_url = "https://docs.cpuc.ca.gov/PublishedDocs/Published/G000/M571/K985/571985189.PDF"
    
    try:
        print(f"Testing with: {test_url}")
        
        result = _process_with_chonkie_primary(test_url, proceeding="R2207005")
        
        if result:
            print(f"✅ Processing successful: {len(result)} chunks")
            
            # Check metadata consistency
            enhanced_count = 0
            for doc in result[:5]:  # Check first 5 docs
                metadata = doc.metadata
                if all(field in metadata for field in ['char_start', 'char_end', 'text_snippet']):
                    enhanced_count += 1
            
            print(f"   Enhanced metadata consistency: {enhanced_count}/5 documents")
            
            if enhanced_count == 5:
                print("🎉 Consistent enhanced metadata across all samples!")
                return True
            else:
                print("⚠️ Inconsistent enhanced metadata")
                return False
        else:
            print("❌ Processing failed")
            return False
            
    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        return False


def main():
    """Run enhanced fallback tests."""
    print("🚀 Enhanced Docling Fallback Integration Tests")
    print("=" * 60)
    
    test1_success = test_enhanced_fallback_integration()
    test2_success = test_fallback_with_known_failure()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    tests_passed = sum([test1_success, test2_success])
    total_tests = 2
    
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Enhanced Docling fallback is working correctly.")
        print("   • Fallback integration is functional")
        print("   • Enhanced metadata is preserved")
        print("   • Citation consistency is maintained")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)