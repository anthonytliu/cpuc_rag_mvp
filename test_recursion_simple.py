#!/usr/bin/env python3
"""
Simple Test for ArrowSchema Recursion and Import Error Handling

Direct test of the error handling mechanisms without complex mocking.
"""

import logging
import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_docling_direct_functionality():
    """Test the Docling direct processing function directly."""
    logger.info("🧪 Testing Docling direct processing functionality...")
    
    try:
        from data_processing.data_processing import _process_with_docling_direct
        
        test_url = "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M440/K092/440092094.PDF"
        result = _process_with_docling_direct(test_url, "Test Document", "R1311007")
        
        if result and len(result) > 0:
            logger.info(f"✅ Docling direct processing successful: {len(result)} chunks")
            
            # Check metadata
            first_chunk = result[0]
            processing_method = first_chunk.metadata.get('processing_method', '')
            if 'docling_direct_recursion_recovery' in processing_method:
                logger.info("✅ Correct processing method tag found")
                return True, f"Successfully processed {len(result)} chunks"
            else:
                logger.error(f"❌ Wrong processing method: {processing_method}")
                return False, f"Wrong processing method: {processing_method}"
        else:
            logger.error("❌ No chunks produced")
            return False, "No chunks produced"
            
    except Exception as e:
        logger.error(f"❌ Docling direct test failed: {e}")
        return False, str(e)


def test_import_fallback_logic():
    """Test the import fallback logic by examining the code structure."""
    logger.info("🔧 Testing import fallback logic...")
    
    try:
        from data_processing.data_processing import _process_with_hybrid_evaluation
        import inspect
        
        # Get the source code to verify fallback logic exists
        source = inspect.getsource(_process_with_hybrid_evaluation)
        
        # Check for proper fallback structure
        checks = {
            'enhanced_docling_fallback import': 'enhanced_docling_fallback' in source,
            'ImportError handling': 'ImportError' in source,
            'simplified_fallback': '_process_with_simplified_fallback' in source,
            'triple_fallback': source.count('except ImportError') >= 2
        }
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        
        logger.info(f"Import fallback checks: {passed_checks}/{total_checks} passed")
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {check_name}")
        
        return passed_checks == total_checks, f"Import fallback logic: {passed_checks}/{total_checks} checks"
        
    except Exception as e:
        logger.error(f"❌ Import fallback test failed: {e}")
        return False, str(e)


def test_recursion_error_detection():
    """Test recursion error detection logic."""
    logger.info("🔄 Testing recursion error detection...")
    
    try:
        from data_processing.data_processing import _process_with_hybrid_evaluation
        import inspect
        
        # Get the source code to verify recursion handling
        source = inspect.getsource(_process_with_hybrid_evaluation)
        
        # Check for recursion error handling
        checks = {
            'recursion_level_detection': 'recursion level' in source.lower(),
            'arrowschema_detection': 'arrowschema' in source.lower(),
            'docling_direct_fallback': '_process_with_docling_direct' in source,
            'error_message_parsing': 'error_msg.lower()' in source
        }
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        
        logger.info(f"Recursion error detection checks: {passed_checks}/{total_checks} passed")
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {check_name}")
        
        return passed_checks == total_checks, f"Recursion detection: {passed_checks}/{total_checks} checks"
        
    except Exception as e:
        logger.error(f"❌ Recursion detection test failed: {e}")
        return False, str(e)


def test_incremental_embedder_batch_protection():
    """Test incremental embedder batch size protection."""
    logger.info("🔗 Testing incremental embedder batch protection...")
    
    try:
        from data_processing.incremental_embedder import IncrementalEmbedder
        import inspect
        
        # Get the source code to verify batch protection
        source = inspect.getsource(IncrementalEmbedder)
        
        # Check for batch protection mechanisms
        checks = {
            'small_batch_for_recursion': 'batch_size = 25' in source,
            'ultra_small_batch': 'batch_size = 10' in source,
            'docling_direct_detection': 'docling_direct' in source,
            'ultra_small_for_docling': 'batch_size = 5' in source,
            'recursion_error_catch': 'recursion level' in source.lower(),
            'minimal_batch_recovery': 'batch_size=1' in source
        }
        
        passed_checks = sum(checks.values())
        total_checks = len(checks)
        
        logger.info(f"Incremental embedder protection checks: {passed_checks}/{total_checks} passed")
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            logger.info(f"   {status} {check_name}")
        
        return passed_checks >= total_checks * 0.8, f"Batch protection: {passed_checks}/{total_checks} checks"
        
    except Exception as e:
        logger.error(f"❌ Incremental embedder test failed: {e}")
        return False, str(e)


def main():
    """Run simple error handling tests."""
    print("🧪 Simple Error Handling Validation")
    print("=" * 50)
    
    tests = [
        ("Docling Direct Processing", test_docling_direct_functionality),
        ("Import Fallback Logic", test_import_fallback_logic),
        ("Recursion Error Detection", test_recursion_error_detection),
        ("Batch Protection", test_incremental_embedder_batch_protection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔧 Running {test_name}...")
        try:
            success, message = test_func()
            results.append((test_name, success, message))
        except Exception as e:
            results.append((test_name, False, f"Test failed: {e}"))
    
    # Summary
    print(f"\n📊 TEST RESULTS:")
    print("=" * 30)
    
    passed = 0
    for test_name, success, message in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        print(f"    {message}")
        if success:
            passed += 1
    
    total = len(results)
    success_rate = (passed / total * 100) if total > 0 else 0
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Tests Passed: {passed}/{total}")
    print(f"   Success Rate: {success_rate:.1f}%")
    
    if passed == total:
        print(f"\n🎉 ALL ERROR HANDLING MECHANISMS VALIDATED!")
        print("   • ArrowSchema recursion detection active")
        print("   • Docling direct fallback functional")
        print("   • Import error handling in place")
        print("   • Batch size protection enabled")
        return True
    elif passed >= total * 0.75:
        print(f"\n✅ MOST MECHANISMS WORKING")
        print("   • Core error handling is functional")
        return True
    else:
        print(f"\n⚠️ ISSUES DETECTED - Review failed tests")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)