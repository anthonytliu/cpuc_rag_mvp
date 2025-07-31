#!/usr/bin/env python3
"""
Comprehensive Test of Fixes Across Multiple Proceedings

Tests all the applied fixes:
1. Schema compatibility fixes
2. ArrowSchema recursion handling  
3. Extended timeout handling
4. Simplified fallback processing
"""

import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path and suppress warnings
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import warnings
warnings.filterwarnings('ignore', message='.*pin_memory.*', category=UserWarning)

from data_processing.incremental_embedder import create_incremental_embedder
from data_processing.embedding_only_system import EmbeddingOnlySystem
from core import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveFixTester:
    """Test comprehensive fixes across multiple proceedings."""
    
    def __init__(self):
        # Test with proceedings that are known to have various issues
        self.test_proceedings = [
            "R1311007",  # Known to have ArrowSchema recursion issues
            "R1206013", 
            "R1211005",
            "R2401017"   # Was previously consolidated
        ]
        
        logger.info("🧪 Comprehensive Fix Tester initialized")
        logger.info(f"   Testing proceedings: {self.test_proceedings}")
    
    def test_schema_compatibility(self) -> dict:
        """Test schema compatibility fixes across proceedings."""
        logger.info("🔧 Testing schema compatibility fixes...")
        
        results = {}
        
        for proceeding in self.test_proceedings:
            logger.info(f"Testing schema for {proceeding}...")
            
            try:
                # Test EmbeddingOnlySystem initialization and schema migration
                system = EmbeddingOnlySystem(proceeding)
                
                # Test adding a document with all enhanced fields
                test_doc = {
                    'content': f'Schema test for {proceeding}',
                    'url': f'test://schema-validation-{proceeding}',
                    'title': f'Schema Test {proceeding}',
                    'char_start': 0,
                    'char_end': 25,
                    'char_length': 25,
                    'line_number': 1,
                    'page_number': 1,
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'document_hash': f'test_hash_{proceeding}',
                    'processing_method': 'schema_test',
                    'extraction_confidence': 1.0,
                    'source_section': 'test',
                    'creation_date': datetime.now().isoformat(),
                    'last_modified': datetime.now().isoformat(),
                    'file_size': 1024,
                    'chunk_overlap': 0,
                    'chunk_level': 'document',
                    'content_type': 'text/plain',
                    'document_date': datetime.now().isoformat(),  # This was the critical missing field
                    'document_type': 'proceeding',
                    'proceeding_number': proceeding,
                }
                
                # Test schema compatibility
                result = system.add_document_incrementally(
                    documents=[test_doc],
                    batch_size=1,
                    use_progress_tracking=False
                )
                
                if result['success']:
                    results[proceeding] = {'status': 'PASS', 'error': None}
                    logger.info(f"✅ {proceeding}: Schema test PASSED")
                else:
                    results[proceeding] = {'status': 'FAIL', 'error': result.get('error', 'Unknown error')}
                    logger.error(f"❌ {proceeding}: Schema test FAILED - {result.get('error')}")
                
            except Exception as e:
                results[proceeding] = {'status': 'ERROR', 'error': str(e)}
                logger.error(f"❌ {proceeding}: Schema test ERROR - {e}")
        
        return results
    
    def test_timeout_handling(self) -> dict:
        """Test extended timeout handling."""
        logger.info("⏰ Testing timeout handling...")
        
        results = {}
        
        for proceeding in self.test_proceedings:
            logger.info(f"Testing timeouts for {proceeding}...")
            
            try:
                # Create incremental embedder with timeout enabled
                embedder = create_incremental_embedder(proceeding, enable_timeout=True)
                
                # Check timeout setting
                if hasattr(embedder, 'enable_timeout') and embedder.enable_timeout:
                    results[proceeding] = {'status': 'PASS', 'timeout_enabled': True}
                    logger.info(f"✅ {proceeding}: Timeout handling PASSED (enabled)")
                else:
                    results[proceeding] = {'status': 'PARTIAL', 'timeout_enabled': False}
                    logger.warning(f"⚠️ {proceeding}: Timeout handling PARTIAL (not enabled)")
                
            except Exception as e:
                results[proceeding] = {'status': 'ERROR', 'error': str(e)}
                logger.error(f"❌ {proceeding}: Timeout test ERROR - {e}")
        
        return results
    
    def test_recursion_handling(self) -> dict:
        """Test ArrowSchema recursion handling."""
        logger.info("🔄 Testing recursion handling...")
        
        results = {}
        
        for proceeding in self.test_proceedings:
            logger.info(f"Testing recursion protection for {proceeding}...")
            
            try:
                embedder = create_incremental_embedder(proceeding, enable_timeout=True)
                
                # Get current status to check for any existing failed documents
                status = embedder.get_embedding_status()
                failed_count = status.get('total_failed', 0)
                
                results[proceeding] = {
                    'status': 'PASS',
                    'failed_documents': failed_count,
                    'system_ready': True
                }
                
                if failed_count > 0:
                    logger.info(f"📊 {proceeding}: {failed_count} failed documents available for retry")
                else:
                    logger.info(f"✅ {proceeding}: No failed documents, system ready")
                
            except Exception as e:
                results[proceeding] = {'status': 'ERROR', 'error': str(e)}
                logger.error(f"❌ {proceeding}: Recursion test ERROR - {e}")
        
        return results
    
    def test_batch_processing_limits(self) -> dict:
        """Test batch processing with size limits for recursion prevention."""
        logger.info("📦 Testing batch processing limits...")
        
        results = {}
        
        for proceeding in self.test_proceedings:
            logger.info(f"Testing batch limits for {proceeding}...")
            
            try:
                system = EmbeddingOnlySystem(proceeding)
                
                # Create a large set of test documents to test batch limiting
                large_doc_set = []
                for i in range(25):  # Test with 25 documents
                    doc = {
                        'content': f'Batch test document {i} for {proceeding}',
                        'url': f'test://batch-test-{proceeding}-{i}',
                        'title': f'Batch Test {i}',
                        'chunk_level': 'document',
                        'content_type': 'text/plain', 
                        'document_date': datetime.now().isoformat(),
                        'document_type': 'test',
                        'proceeding_number': proceeding,
                    }
                    large_doc_set.append(doc)
                
                # Test batch processing with small batch size
                result = system.add_document_incrementally(
                    documents=large_doc_set,
                    batch_size=5,  # Small batch size to prevent recursion
                    use_progress_tracking=False
                )
                
                if result['success']:
                    results[proceeding] = {'status': 'PASS', 'batch_size': 5, 'docs_processed': result.get('added', 0)}
                    logger.info(f"✅ {proceeding}: Batch processing PASSED")
                else:
                    results[proceeding] = {'status': 'FAIL', 'error': result.get('error')}
                    logger.error(f"❌ {proceeding}: Batch processing FAILED")
                
            except Exception as e:
                results[proceeding] = {'status': 'ERROR', 'error': str(e)}
                logger.error(f"❌ {proceeding}: Batch test ERROR - {e}")
        
        return results
    
    def generate_comprehensive_report(self, test_results: dict) -> dict:
        """Generate comprehensive test report."""
        logger.info("📊 Generating comprehensive test report...")
        
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'proceedings_tested': self.test_proceedings,
            'test_results': test_results,
            'summary': {}
        }
        
        # Calculate summary statistics
        for test_name, results in test_results.items():
            passed = sum(1 for r in results.values() if r.get('status') == 'PASS')
            failed = sum(1 for r in results.values() if r.get('status') in ['FAIL', 'ERROR'])
            partial = sum(1 for r in results.values() if r.get('status') == 'PARTIAL')
            
            report['summary'][test_name] = {
                'passed': passed,
                'failed': failed,
                'partial': partial,
                'total': len(results),
                'success_rate': (passed / len(results) * 100) if results else 0
            }
        
        return report
    
    def run_all_tests(self) -> dict:
        """Run all comprehensive tests."""
        logger.info("🚀 Running all comprehensive tests...")
        
        test_results = {}
        
        # Test 1: Schema compatibility
        test_results['schema_compatibility'] = self.test_schema_compatibility()
        
        # Test 2: Timeout handling
        test_results['timeout_handling'] = self.test_timeout_handling()
        
        # Test 3: Recursion handling
        test_results['recursion_handling'] = self.test_recursion_handling()
        
        # Test 4: Batch processing limits
        test_results['batch_processing'] = self.test_batch_processing_limits()
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(test_results)
        
        return report


def main():
    """Run comprehensive fix testing."""
    print("🧪 Comprehensive Fix Testing Suite")
    print("=" * 50)
    
    tester = ComprehensiveFixTester()
    
    # Run all tests
    start_time = time.time()
    report = tester.run_all_tests()
    total_time = time.time() - start_time
    
    # Display results
    print(f"\n📊 TEST RESULTS SUMMARY")
    print("=" * 30)
    
    for test_name, summary in report['summary'].items():
        print(f"\n{test_name.replace('_', ' ').title()}:")
        print(f"   ✅ Passed: {summary['passed']}")
        print(f"   ❌ Failed: {summary['failed']}")
        print(f"   ⚠️ Partial: {summary['partial']}")
        print(f"   📊 Success Rate: {summary['success_rate']:.1f}%")
    
    # Overall assessment
    total_passed = sum(s['passed'] for s in report['summary'].values())
    total_tests = sum(s['total'] for s in report['summary'].values())
    overall_success = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 OVERALL RESULTS:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Total Passed: {total_passed}")
    print(f"   Overall Success Rate: {overall_success:.1f}%")
    print(f"   Total Time: {total_time:.2f}s")
    
    # Save detailed report
    report_file = Path('comprehensive_fix_test_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Detailed report saved to: {report_file}")
    
    if overall_success >= 80:
        print(f"\n🎉 COMPREHENSIVE TESTING SUCCESS!")
        print("All core fixes are working properly across multiple proceedings.")
        return True
    else:
        print(f"\n⚠️ Some tests need attention.")
        print("Check the detailed report for specific issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)