#!/usr/bin/env python3
"""
Test Error Handling for ArrowSchema Recursion and Import Issues

Tests the enhanced error handling for:
1. ArrowSchema recursion level exceeded errors
2. Missing enhanced_docling_fallback module errors
3. Docling direct processing fallback
"""

import logging
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ErrorHandlingTester:
    """Tests enhanced error handling scenarios."""
    
    def __init__(self):
        self.test_proceeding = "R1311007"
        self.test_url = "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M440/K092/440092094.PDF"
        logger.info("🧪 Error Handling Tester initialized")
    
    def test_import_error_handling(self) -> bool:
        """Test handling of missing enhanced_docling_fallback module."""
        logger.info("🔧 Testing import error handling...")
        
        try:
            from data_processing.data_processing import _process_with_hybrid_evaluation
            
            # Mock the import to raise ImportError
            with patch('data_processing.data_processing.__import__', side_effect=ImportError("No module named 'enhanced_docling_fallback'")):
                with patch('data_processing.data_processing._process_with_simplified_fallback') as mock_fallback:
                    mock_fallback.return_value = [Mock()]  # Mock successful fallback
                    
                    # This should trigger the import error and fallback
                    result = _process_with_hybrid_evaluation(
                        self.test_url, 
                        "Test Document", 
                        self.test_proceeding
                    )
                    
                    if mock_fallback.called:
                        logger.info("✅ Import error handling: Fallback called correctly")
                        return True
                    else:
                        logger.error("❌ Import error handling: Fallback not called")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Import error handling test failed: {e}")
            return False
    
    def test_arrowschema_recursion_handling(self) -> bool:
        """Test handling of ArrowSchema recursion errors."""
        logger.info("🔄 Testing ArrowSchema recursion handling...")
        
        try:
            from data_processing.data_processing import _process_with_hybrid_evaluation
            
            # Create a mock that raises ArrowSchema recursion error
            def mock_process_with_error(*args, **kwargs):
                raise RuntimeError("Recursion level in ArrowSchema struct exceeded")
            
            with patch('data_processing.data_processing._process_with_chonkie') as mock_chonkie:
                mock_chonkie.side_effect = mock_process_with_error
                
                with patch('data_processing.data_processing._process_with_docling_direct') as mock_docling_direct:
                    mock_docling_direct.return_value = [Mock()]  # Mock successful Docling direct
                    
                    # This should trigger ArrowSchema error and use Docling direct
                    result = _process_with_hybrid_evaluation(
                        self.test_url,
                        "Test Document", 
                        self.test_proceeding
                    )
                    
                    if mock_docling_direct.called:
                        logger.info("✅ ArrowSchema recursion handling: Docling direct called correctly")
                        return True
                    else:
                        logger.error("❌ ArrowSchema recursion handling: Docling direct not called")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ ArrowSchema recursion handling test failed: {e}")
            return False
    
    def test_docling_direct_processing(self) -> bool:
        """Test Docling direct processing functionality."""
        logger.info("📄 Testing Docling direct processing...")
        
        try:
            from data_processing.data_processing import _process_with_docling_direct
            
            # Mock Docling converter and result
            mock_result = Mock()
            mock_result.status = Mock()
            mock_result.status.name = "SUCCESS"
            mock_result.document.export_to_markdown.return_value = "Test content for processing.\n\nThis is a test document with multiple paragraphs.\n\nEach paragraph should be processed correctly."
            
            with patch('data_processing.data_processing.DocumentConverter') as mock_converter_class:
                mock_converter = Mock()
                mock_converter.convert.return_value = mock_result
                mock_converter_class.return_value = mock_converter
                
                # Mock ConversionStatus.SUCCESS
                with patch('data_processing.data_processing.ConversionStatus') as mock_status:
                    mock_status.SUCCESS = mock_result.status
                    
                    result = _process_with_docling_direct(
                        self.test_url,
                        "Test Document",
                        self.test_proceeding
                    )
                    
                    if result and len(result) > 0:
                        logger.info(f"✅ Docling direct processing: {len(result)} chunks generated")
                        
                        # Verify chunk metadata
                        first_chunk = result[0]
                        if hasattr(first_chunk, 'metadata'):
                            processing_method = first_chunk.metadata.get('processing_method', '')
                            if 'docling_direct_recursion_recovery' in processing_method:
                                logger.info("✅ Docling direct processing: Correct processing method set")
                                return True
                            else:
                                logger.error(f"❌ Docling direct processing: Wrong processing method: {processing_method}")
                                return False
                        else:
                            logger.error("❌ Docling direct processing: No metadata found")
                            return False
                    else:
                        logger.error("❌ Docling direct processing: No chunks generated")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ Docling direct processing test failed: {e}")
            return False
    
    def test_incremental_embedder_recursion_protection(self) -> bool:
        """Test incremental embedder ArrowSchema recursion protection."""
        logger.info("🔗 Testing incremental embedder recursion protection...")
        
        try:
            from data_processing.incremental_embedder import create_incremental_embedder
            from langchain.schema import Document
            
            # Create test documents with docling_direct processing method
            test_docs = [
                Document(
                    page_content="Test content",
                    metadata={
                        'url': self.test_url,
                        'title': 'Test Doc',
                        'processing_method': 'docling_direct_recursion_recovery',
                        'proceeding': self.test_proceeding
                    }
                )
            ]
            
            embedder = create_incremental_embedder(self.test_proceeding)
            
            # Mock the embedding system to detect ultra-small batch sizes
            original_add_method = embedder.embedding_system.add_document_incrementally
            
            def mock_add_with_batch_detection(documents, batch_size, **kwargs):
                if batch_size == 5:  # Ultra-small batch size for docling_direct
                    logger.info("✅ Incremental embedder: Ultra-small batch size detected for docling_direct")
                    return {'success': True, 'added': len(documents)}
                else:
                    return original_add_method(documents, batch_size, **kwargs)
            
            embedder.embedding_system.add_document_incrementally = mock_add_with_batch_detection
            
            result = embedder._process_single_document(
                self.test_url,
                "Test Document",
                self.test_proceeding,
                test_docs
            )
            
            if result.get('success'):
                logger.info("✅ Incremental embedder recursion protection: Test passed")
                return True
            else:
                logger.error("❌ Incremental embedder recursion protection: Test failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Incremental embedder recursion protection test failed: {e}")
            return False
    
    def run_all_tests(self) -> dict:
        """Run all error handling tests."""
        logger.info("🚀 Running all error handling tests...")
        
        start_time = time.time()
        
        tests = {
            'import_error_handling': self.test_import_error_handling(),
            'arrowschema_recursion_handling': self.test_arrowschema_recursion_handling(),
            'docling_direct_processing': self.test_docling_direct_processing(),
            'incremental_embedder_protection': self.test_incremental_embedder_recursion_protection()
        }
        
        total_time = time.time() - start_time
        
        return {
            'tests': tests,
            'total_time': total_time,
            'success_count': sum(tests.values()),
            'total_count': len(tests)
        }
    
    def generate_test_report(self, results: dict) -> str:
        """Generate comprehensive test report."""
        success_count = results['success_count']
        total_count = results['total_count']
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        
        report = f"""
🧪 ERROR HANDLING TEST REPORT
{'='*50}

📊 SUMMARY:
   Test Duration: {results['total_time']:.2f}s
   Tests Passed: {success_count}/{total_count}
   Success Rate: {success_rate:.1f}%

🔧 TEST RESULTS:
"""
        
        for test_name, passed in results['tests'].items():
            status = "✅ PASS" if passed else "❌ FAIL"
            test_display = test_name.replace('_', ' ').title()
            report += f"   {status} {test_display}\n"
        
        report += f"""
🎯 OVERALL ASSESSMENT:
"""
        
        if success_count == total_count:
            report += "   🎉 ALL TESTS PASSED!\n"
            report += "   • Import error handling working correctly\n"
            report += "   • ArrowSchema recursion protection active\n"
            report += "   • Docling direct fallback functional\n"
            report += "   • Incremental embedder protection enabled\n"
            report += "   • System ready for production error scenarios\n"
        elif success_count >= total_count * 0.75:
            report += "   ✅ MOST TESTS PASSED - Minor issues\n"
            report += "   • Core error handling working\n"
            report += "   • Some refinements may be needed\n"
        else:
            report += "   ⚠️ SIGNIFICANT ISSUES DETECTED\n"
            report += "   • Error handling needs attention\n"
            report += "   • Review failed tests and fix issues\n"
        
        return report


def main():
    """Run error handling tests."""
    print("🧪 Error Handling Test Suite")
    print("=" * 40)
    
    tester = ErrorHandlingTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Generate and display report
    report = tester.generate_test_report(results)
    print(report)
    
    # Save report
    report_file = Path('error_handling_test_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📋 Test report saved to: {report_file}")
    
    # Return success
    all_passed = results['success_count'] == results['total_count']
    if all_passed:
        print(f"\n🎉 ALL ERROR HANDLING TESTS PASSED!")
        print("The system is ready to handle ArrowSchema recursion and import errors.")
        return True
    else:
        print(f"\n⚠️ Some tests failed - check report for details.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)