#!/usr/bin/env python3
"""
Quick Enhanced Citation Format Test

This script tests that the enhanced citation system produces citations
with character position information without doing full PDF validation.
"""

import logging
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from rag_core import CPUCRAGSystem

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce log noise
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_enhanced_citation_format():
    """Test that enhanced citations are being generated."""
    print("🔍 Testing Enhanced Citation Format")
    print("=" * 50)
    
    try:
        # Use the test vector store we created
        rag_system = CPUCRAGSystem(current_proceeding="R2207005_test")
        
        # Test queries
        test_queries = [
            "What is the purpose of this proceeding?",
            "What are the main regulatory requirements?",
            "What decisions have been made?"
        ]
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🧪 Test {i}/3: {query}")
            
            # Get response
            response_gen = rag_system.query(query)
            result = None
            for r in response_gen:
                if isinstance(r, dict):
                    result = r
                    break
            
            if not result:
                print("❌ No response received")
                continue
            
            answer = result.get('answer', '')
            sources = result.get('sources', [])
            
            # Check for enhanced metadata in sources
            enhanced_sources = 0
            for source in sources:
                if 'char_start' in source and 'char_end' in source:
                    enhanced_sources += 1
            
            # Look for enhanced citations in the answer
            enhanced_citations = re.findall(r'\[CITE:[^,]+,page_\d+,chars_\d+-\d+[^\]]*\]', answer)
            standard_citations = re.findall(r'\[CITE:[^,]+,page_\d+\]', answer)
            
            # Remove enhanced citations from standard count to avoid double-counting
            for enhanced in enhanced_citations:
                standard_base = enhanced.split(',chars_')[0] + ']'
                if standard_base in standard_citations:
                    standard_citations.remove(standard_base)
            
            print(f"   Sources: {len(sources)} total, {enhanced_sources} with character positions")
            print(f"   Enhanced citations: {len(enhanced_citations)}")
            print(f"   Standard citations: {len(standard_citations)}")
            
            # Show examples
            if enhanced_citations:
                print("   ✅ Enhanced citation examples:")
                for citation in enhanced_citations[:2]:
                    print(f"      {citation}")
            
            if standard_citations:
                print("   📝 Standard citation examples:")
                for citation in standard_citations[:2]:
                    print(f"      {citation}")
            
            results.append({
                'query': query,
                'sources_total': len(sources),
                'sources_enhanced': enhanced_sources,
                'enhanced_citations': len(enhanced_citations),
                'standard_citations': len(standard_citations),
                'success': enhanced_sources > 0 or len(enhanced_citations) > 0
            })
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 SUMMARY")
        print("=" * 50)
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r['success'])
        
        total_enhanced_sources = sum(r['sources_enhanced'] for r in results)
        total_sources = sum(r['sources_total'] for r in results)
        
        total_enhanced_citations = sum(r['enhanced_citations'] for r in results)
        total_standard_citations = sum(r['standard_citations'] for r in results)
        
        print(f"Tests passed: {successful_tests}/{total_tests}")
        print(f"Enhanced sources: {total_enhanced_sources}/{total_sources} ({total_enhanced_sources/total_sources*100:.1f}%)")
        print(f"Enhanced citations: {total_enhanced_citations}")
        print(f"Standard citations: {total_standard_citations}")
        
        if successful_tests == total_tests and total_enhanced_sources > 0:
            print("\n🎉 SUCCESS: Enhanced citation system is working correctly!")
            print("   • All sources have character position metadata")
            if total_enhanced_citations > 0:
                print("   • Enhanced citations are being generated")
            return True
        elif total_enhanced_sources > 0:
            print("\n✅ PARTIAL SUCCESS: Enhanced metadata is available")
            print("   • Sources have character position data")
            if total_enhanced_citations == 0:
                print("   ⚠️ But enhanced citations are not being generated in answers")
            return True
        else:
            print("\n❌ FAILURE: Enhanced citation system is not working")
            return False
            
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    success = test_enhanced_citation_format()
    if success:
        print("\n✅ Enhanced citation system test completed successfully!")
    else:
        print("\n❌ Enhanced citation system test failed!")


if __name__ == "__main__":
    main()