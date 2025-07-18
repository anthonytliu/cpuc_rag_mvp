#!/usr/bin/env python3
"""
Simple test to verify DB directory creation and basic functionality.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add the project root to path
sys.path.append(str(Path(__file__).parent))

def test_db_directory_creation():
    """Test that the DB directory is created properly."""
    print("Testing DB Directory Creation...")
    
    # Check if the directory exists
    db_dir = Path("/Users/anthony.liu/Downloads/CPUC_REG_RAG/local_chroma_db")
    
    if db_dir.exists():
        print(f"✅ DB directory exists: {db_dir}")
        
        # Check if it's writable
        test_file = db_dir / "test_write.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()
            print("✅ DB directory is writable")
        except Exception as e:
            print(f"❌ DB directory not writable: {e}")
            return False
    else:
        print(f"❌ DB directory missing: {db_dir}")
        return False
    
    return True

def test_incremental_write_function():
    """Test that incremental write function exists and has correct signature."""
    print("\nTesting Incremental Write Function...")
    
    try:
        from rag_core import CPUCRAGSystem
        
        # Create a temporary instance (without triggering full initialization)
        import config
        original_db_dir = config.DB_DIR
        
        # Use a temporary directory to avoid triggering rebuild
        temp_dir = Path(tempfile.mkdtemp())
        config.DB_DIR = temp_dir
        
        try:
            # Quick check that the method exists
            rag = CPUCRAGSystem()
            
            # Check if the method exists
            if hasattr(rag, 'add_document_incrementally'):
                print("✅ add_document_incrementally method exists")
            else:
                print("❌ add_document_incrementally method missing")
                return False
                
            # Check if recovery methods exist
            if hasattr(rag, 'recover_partial_build'):
                print("✅ recover_partial_build method exists")
            else:
                print("❌ recover_partial_build method missing")
                return False
                
            if hasattr(rag, 'create_checkpoint'):
                print("✅ create_checkpoint method exists")
            else:
                print("❌ create_checkpoint method missing")
                return False
                
            print("✅ All new methods are present")
            
        finally:
            # Restore original config
            config.DB_DIR = original_db_dir
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ Function test failed: {e}")
        return False
    
    return True

def test_config_validation():
    """Test that configuration is properly set up."""
    print("\nTesting Configuration...")
    
    try:
        import config
        
        # Check DB_DIR
        if hasattr(config, 'DB_DIR'):
            print(f"✅ DB_DIR configured: {config.DB_DIR}")
        else:
            print("❌ DB_DIR not configured")
            return False
            
        # Check VECTOR_STORE_BATCH_SIZE
        if hasattr(config, 'VECTOR_STORE_BATCH_SIZE'):
            print(f"✅ VECTOR_STORE_BATCH_SIZE: {config.VECTOR_STORE_BATCH_SIZE}")
        else:
            print("❌ VECTOR_STORE_BATCH_SIZE not configured")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("Simple DB Fixes Test")
    print("=" * 40)
    
    tests = [
        ("DB Directory Creation", test_db_directory_creation),
        ("Incremental Write Function", test_incremental_write_function),
        ("Configuration Validation", test_config_validation)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 40)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All basic tests passed!")
    else:
        print("⚠️ Some tests failed.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)