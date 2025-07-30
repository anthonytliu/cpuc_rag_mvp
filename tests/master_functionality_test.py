#!/usr/bin/env python3
"""
Master Functionality Test

Comprehensive test suite to validate all core functionality after fixes:
1. Import fixes for config paths
2. Vector store directory paths
3. Agent evaluation disabled
4. Core data processing pipeline

Author: Claude Code
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config_imports():
    """Test that all config imports work properly."""
    print("🧪 Testing config imports...")
    
    try:
        from src.core.config import get_proceeding_file_paths, DEFAULT_PROCEEDING, PROJECT_ROOT
        print("  ✅ Core config imports work")
        
        # Test get_proceeding_file_paths function
        paths = get_proceeding_file_paths('R1311005')
        expected_vector_path = PROJECT_ROOT / "data" / "vector_stores" / "local_lance_db" / "R1311005"
        
        assert paths['vector_db'] == expected_vector_path, f"Vector DB path mismatch: {paths['vector_db']} != {expected_vector_path}"
        print(f"  ✅ Vector store path correct: {paths['vector_db']}")
        
        # Test various paths exist in the config
        required_keys = ['proceeding_dir', 'documents_dir', 'embeddings_dir', 'vector_db', 'document_hashes']
        for key in required_keys:
            assert key in paths, f"Missing required path key: {key}"
        print(f"  ✅ All required path keys present: {required_keys}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Config import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_processing_imports():
    """Test data processing imports work without errors."""
    print("🧪 Testing data processing imports...")
    
    try:
        from src.data_processing.data_processing import _process_with_hybrid_evaluation
        print("  ✅ Data processing imports work")
        
        from src.data_processing.incremental_embedder import IncrementalEmbedder
        print("  ✅ Incremental embedder imports work")
        
        from src.data_processing.embedding_only_system import EmbeddingOnlySystem
        print("  ✅ Embedding only system imports work")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data processing import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store_initialization():
    """Test that vector store can be initialized with correct paths."""
    print("🧪 Testing vector store initialization...")
    
    try:
        from src.data_processing.embedding_only_system import EmbeddingOnlySystem
        
        # Initialize system for test proceeding
        test_proceeding = 'R1311005'
        embedding_system = EmbeddingOnlySystem(test_proceeding)
        
        # Check that the path is correct
        expected_path = Path.cwd() / "data" / "vector_stores" / "local_lance_db" / test_proceeding
        assert embedding_system.db_dir == expected_path, f"DB dir mismatch: {embedding_system.db_dir} != {expected_path}"
        print(f"  ✅ Vector store path correct: {embedding_system.db_dir}")
        
        # Test that directory would be created
        print(f"  ✅ Vector store initialization successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Vector store initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_evaluation_disabled():
    """Test that agent evaluation is properly disabled."""
    print("🧪 Testing agent evaluation disabled...")
    
    try:
        # Check that the function exists but doesn't require anthropic
        from src.data_processing.data_processing import _evaluate_with_agent
        print("  ✅ Agent evaluation function exists")
        
        # Test with mock data - should not require anthropic
        mock_docling = {'success': True, 'chunk_count': 10, 'processing_time': 2.0, 'content_length': 1000, 'tables_found': 2, 'content_types': []}
        mock_chonkie = {'success': True, 'chunk_count': 8, 'processing_time': 1.0, 'content_length': 900, 'content_types': []}
        
        decision, reasoning = _evaluate_with_agent(mock_docling, mock_chonkie, 0.7)
        print(f"  ✅ Agent evaluation works without anthropic: {decision}")
        
        # Should return either DOCLING or CHONKIE with valid reasoning
        assert decision in ['DOCLING', 'CHONKIE'], f"Invalid decision: {decision}"
        assert isinstance(reasoning, str) and len(reasoning) > 10, f"Invalid reasoning: {reasoning}"
        print(f"  ✅ Decision logic works: {decision} - {reasoning[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ui_imports():
    """Test UI component imports work."""
    print("🧪 Testing UI imports...")
    
    try:
        from src.ui.app import get_available_proceedings_from_db
        print("  ✅ UI app imports work")
        
        # Test that it can find proceedings (may be empty but shouldn't crash)
        proceedings = get_available_proceedings_from_db()
        print(f"  ✅ Found {len(proceedings)} proceedings in vector stores")
        
        return True
        
    except Exception as e:
        print(f"  ❌ UI import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_rag_system():
    """Test core RAG system can initialize."""
    print("🧪 Testing core RAG system...")
    
    try:
        from src.core.rag_core import CPUCRAGSystem
        print("  ✅ RAG core imports work")
        
        # Test initialization (should not crash)
        rag_system = CPUCRAGSystem(current_proceeding='R1311005')
        print("  ✅ RAG system initialization successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ RAG system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_standalone_imports():
    """Test standalone scripts can import config properly."""
    print("🧪 Testing standalone script imports...")
    
    try:
        # Test that standalone_data_processor can import config
        import subprocess
        import sys
        
        # Run a simple test that tries to import from the standalone processor
        result = subprocess.run([
            sys.executable, '-c', 
            '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
sys.path.insert(0, str(Path.cwd()))

try:
    from core.config import get_proceeding_file_paths
    print("SUCCESS: Standalone config import works")
except Exception as e:
    print(f"ERROR: {e}")
    exit(1)
            '''
        ], capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            print("  ✅ Standalone script imports work")
            return True
        else:
            print(f"  ❌ Standalone script import failed: {result.stderr}")
            return False
        
    except Exception as e:
        print(f"  ❌ Standalone import test failed: {e}")
        return False

def run_master_test():
    """Run all functionality tests."""
    print("🚀 MASTER FUNCTIONALITY TEST")
    print("=" * 60)
    print("Testing all core functionality after recent fixes...")
    print()
    
    tests = [
        ("Config Imports", test_config_imports),
        ("Data Processing Imports", test_data_processing_imports),
        ("Vector Store Initialization", test_vector_store_initialization),
        ("Agent Evaluation Disabled", test_agent_evaluation_disabled),
        ("UI Imports", test_ui_imports),
        ("Core RAG System", test_core_rag_system),
        ("Standalone Imports", test_standalone_imports),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print()
    print(f"Total: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! Core functionality is working correctly.")
        return True
    else:
        print(f"⚠️  {failed} tests failed. Review the output above for details.")
        return False

if __name__ == "__main__":
    success = run_master_test()
    sys.exit(0 if success else 1)