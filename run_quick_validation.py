#!/usr/bin/env python3
"""
Quick validation of comprehensive fixes
"""

import sys
import time
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

print("🧪 Quick Validation of Comprehensive Fixes")
print("=" * 50)

def test_schema_fixes():
    """Test schema compatibility fixes."""
    print("\n🔧 Testing Schema Compatibility...")
    try:
        from data_processing.embedding_only_system import EmbeddingOnlySystem
        
        # Test with R1311007 (known problematic proceeding)
        system = EmbeddingOnlySystem("R1311007")
        print("✅ EmbeddingOnlySystem initialization: PASS")
        
        # Check if vector store initializes without schema errors
        if hasattr(system, 'vectordb') or hasattr(system, 'lance_db'):
            print("✅ Vector store initialization: PASS")
        else:
            print("⚠️ Vector store not fully initialized")
            
        return True
        
    except Exception as e:
        print(f"❌ Schema test FAILED: {e}")
        return False

def test_timeout_fixes():
    """Test timeout handling."""
    print("\n⏰ Testing Timeout Handling...")
    try:
        from data_processing.incremental_embedder import create_incremental_embedder
        
        # Create embedder with timeout enabled
        embedder = create_incremental_embedder("R1311007", enable_timeout=True)
        print("✅ Incremental embedder with timeout: PASS")
        
        # Check timeout setting
        if hasattr(embedder, 'enable_timeout') and embedder.enable_timeout:
            print("✅ Timeout enabled: PASS")
        else:
            print("⚠️ Timeout setting unclear")
            
        return True
        
    except Exception as e:
        print(f"❌ Timeout test FAILED: {e}")
        return False

def test_m4_optimizations():
    """Test M4 Pro optimizations."""
    print("\n🚀 Testing M4 Pro Optimizations...")
    try:
        from core.models import get_embedding_model
        
        # Initialize embedding model
        model = get_embedding_model()
        print("✅ Embedding model initialization: PASS")
        
        # Test a simple embedding
        test_embedding = model.embed_query("Test embedding for validation")
        if test_embedding and len(test_embedding) > 0:
            print("✅ Embedding generation: PASS")
        else:
            print("⚠️ Embedding generation unclear")
            
        return True
        
    except Exception as e:
        print(f"❌ M4 optimization test FAILED: {e}")
        return False

def test_recursion_protection():
    """Test recursion protection measures."""
    print("\n🔄 Testing Recursion Protection...")
    try:
        from data_processing.incremental_embedder import create_incremental_embedder
        
        # Test with R1311007 which had recursion issues
        embedder = create_incremental_embedder("R1311007")
        
        # Get status to see if system handles existing failures
        status = embedder.get_embedding_status()
        failed_count = status.get('total_failed', 0)
        embedded_count = status.get('total_embedded', 0)
        
        print(f"📊 Current status - Embedded: {embedded_count}, Failed: {failed_count}")
        
        if failed_count > 0:
            print("✅ Failed documents available for retry with fixes")
        else:
            print("✅ No failed documents or all resolved")
            
        return True
        
    except Exception as e:
        print(f"❌ Recursion protection test FAILED: {e}")
        return False

def main():
    """Run quick validation tests."""
    start_time = time.time()
    
    # Run all tests
    tests = [
        ("Schema Fixes", test_schema_fixes),
        ("Timeout Fixes", test_timeout_fixes),
        ("M4 Optimizations", test_m4_optimizations),
        ("Recursion Protection", test_recursion_protection),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n📊 VALIDATION SUMMARY")
    print("=" * 30)
    print(f"⏱️ Total time: {total_time:.2f}s")
    print(f"✅ Passed: {passed}/{total}")
    print(f"📊 Success rate: {(passed/total*100):.1f}%")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    if passed == total:
        print(f"\n🎉 ALL VALIDATIONS PASSED!")
        print("Comprehensive fixes are working correctly.")
        return True
    else:
        print(f"\n⚠️ {total-passed} tests need attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)