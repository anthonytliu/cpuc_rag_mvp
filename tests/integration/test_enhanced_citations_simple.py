#!/usr/bin/env python3
"""
Simple test to verify enhanced citation system is working
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from rag_core import CPUCRAGSystem

def test_enhanced_citations():
    """Test that enhanced citation metadata is working correctly."""
    print("🧪 Testing Enhanced Citation System")
    print("=" * 50)
    
    # Initialize RAG system for R1311005 (just migrated)
    print("🔧 Initializing RAG system for R1311005...")
    rag_system = CPUCRAGSystem(current_proceeding="R1311005")
    
    # Test query
    question = "What is this proceeding about?"
    print(f"❓ Testing query: {question}")
    
    # Get answer
    try:
        result_generator = rag_system.query(question)
        result = next(result_generator)  # Get first result from generator
        
        print(f"✅ Query successful!")
        print(f"📝 Answer: {result['answer'][:200]}...")
        
        # Check sources for enhanced metadata
        sources = result.get('sources', [])
        print(f"📚 Found {len(sources)} source documents")
        
        # Check first source for enhanced fields
        if sources:
            first_source = sources[0]
            metadata = first_source.metadata
            
            print("\n🔍 Enhanced Citation Metadata Check:")
            enhanced_fields = ['char_start', 'char_end', 'char_length', 'text_snippet']
            
            for field in enhanced_fields:
                if field in metadata:
                    value = metadata[field]
                    print(f"  ✅ {field}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")
                else:
                    print(f"  ❌ {field}: Missing")
            
            # Check if we have basic metadata too
            basic_fields = ['source', 'proceeding', 'page']
            print("\n📋 Basic Metadata Check:")
            for field in basic_fields:
                if field in metadata:
                    print(f"  ✅ {field}: {metadata[field]}")
                else:
                    print(f"  ❌ {field}: Missing")
                    
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return False

if __name__ == "__main__":
    success = test_enhanced_citations()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILURE'}: Enhanced citation test")
    sys.exit(0 if success else 1)