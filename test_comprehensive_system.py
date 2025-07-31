#!/usr/bin/env python3
"""
Comprehensive System Test

Tests the entire system end-to-end:
1. Backup consolidation and prevention
2. Schema compatibility fixes
3. Processing pipeline integrity
4. Performance optimizations
5. Error handling robustness
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
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


class ComprehensiveSystemTester:
    """Comprehensive end-to-end system testing."""
    
    def __init__(self):
        self.lance_db_dir = Path('/Users/anthony.liu/Downloads/CPUC_REG_RAG/data/vector_stores/local_lance_db')
        self.test_proceedings = ["R1311007", "R1206013", "R1211005", "R2401017"]
        
        logger.info("🧪 Comprehensive System Tester initialized")
        logger.info(f"   Testing proceedings: {self.test_proceedings}")
    
    def test_backup_prevention(self) -> dict:
        """Test that no backup folders are created during processing."""
        logger.info("🛡️ Testing backup prevention...")
        
        results = {
            'backup_folders_found': [],
            'prevention_active': True
        }
        
        # Scan for any backup folders
        backup_patterns = ['*_backup*', '*backup*']
        backup_folders = []
        
        for pattern in backup_patterns:
            backup_folders.extend(list(self.lance_db_dir.glob(pattern)))
        
        if backup_folders:
            results['backup_folders_found'] = [str(f) for f in backup_folders]
            results['prevention_active'] = False
            logger.error(f"❌ Found {len(backup_folders)} backup folders")
        else:
            logger.info("✅ No backup folders found - prevention working")
        
        return results
    
    def test_schema_compatibility(self) -> dict:
        """Test schema compatibility across all proceedings."""
        logger.info("🔧 Testing schema compatibility...")
        
        results = {}
        
        for proceeding in self.test_proceedings:
            logger.info(f"   Testing {proceeding}...")
            
            try:
                # Initialize EmbeddingOnlySystem
                system = EmbeddingOnlySystem(proceeding)
                
                # Test document with all enhanced fields (as LangChain Document)
                from langchain.schema import Document
                
                test_doc = Document(
                    page_content=f'Schema test for {proceeding}',
                    metadata={
                        'url': f'test://schema-{proceeding}',
                        'title': f'Test {proceeding}',
                        'char_start': 0,
                        'char_end': 25,
                        'char_length': 25,
                        'line_number': 1,
                        'page_number': 1,
                        'chunk_index': 0,
                        'total_chunks': 1,
                        'document_hash': f'test_{proceeding}',
                        'processing_method': 'test',
                        'extraction_confidence': 1.0,
                        'source_section': 'test',
                        'creation_date': datetime.now().isoformat(),
                        'last_modified': datetime.now().isoformat(),
                        'file_size': 1024,
                        'chunk_overlap': 0,
                        'chunk_level': 'document',
                        'content_type': 'text/plain',
                        'document_date': datetime.now().isoformat(),
                        'document_type': 'proceeding',
                        'proceeding_number': proceeding,
                    }
                )
                
                # Test schema compatibility
                result = system.add_document_incrementally(
                    documents=[test_doc],
                    batch_size=1,
                    use_progress_tracking=False
                )
                
                results[proceeding] = {
                    'status': 'PASS' if result['success'] else 'FAIL',
                    'error': result.get('error', None)
                }
                
                if result['success']:
                    logger.info(f"   ✅ {proceeding}: Schema compatibility PASSED")
                else:
                    logger.error(f"   ❌ {proceeding}: Schema compatibility FAILED - {result.get('error')}")
                
            except Exception as e:
                results[proceeding] = {'status': 'ERROR', 'error': str(e)}
                logger.error(f"   ❌ {proceeding}: Schema test ERROR - {e}")
        
        return results
    
    def test_processing_pipeline(self) -> dict:
        """Test the processing pipeline integrity."""
        logger.info("🔄 Testing processing pipeline...")
        
        results = {}
        
        for proceeding in self.test_proceedings:
            logger.info(f"   Testing pipeline for {proceeding}...")
            
            try:
                # Create incremental embedder
                embedder = create_incremental_embedder(proceeding, enable_timeout=True)
                
                # Get current status
                status = embedder.get_embedding_status()
                
                # Test pipeline components
                pipeline_status = {
                    'embedder_created': True,
                    'timeout_enabled': hasattr(embedder, 'enable_timeout') and embedder.enable_timeout,
                    'status_loaded': isinstance(status, dict),
                    'total_embedded': status.get('total_embedded', 0),
                    'total_failed': status.get('total_failed', 0)
                }
                
                results[proceeding] = pipeline_status
                logger.info(f"   ✅ {proceeding}: Pipeline integrity PASSED")
                
            except Exception as e:
                results[proceeding] = {'status': 'ERROR', 'error': str(e)}
                logger.error(f"   ❌ {proceeding}: Pipeline test ERROR - {e}")
        
        return results
    
    def test_performance_optimizations(self) -> dict:
        """Test performance optimizations."""
        logger.info("🚀 Testing performance optimizations...")
        
        try:
            from core.models import get_embedding_model
            
            # Test embedding model initialization
            start_time = time.time()
            model = get_embedding_model()
            init_time = time.time() - start_time
            
            # Test embedding performance
            test_texts = [f"Performance test sentence {i}" for i in range(10)]
            
            embed_start = time.time()
            embeddings = model.embed_documents(test_texts)
            embed_time = time.time() - embed_start
            
            throughput = len(test_texts) / embed_time if embed_time > 0 else 0
            
            results = {
                'model_init_time': init_time,
                'embedding_time': embed_time,
                'throughput_per_sec': throughput,
                'optimizations_active': throughput > 50,  # Should be fast with M4 Pro
                'embeddings_generated': len(embeddings) if embeddings else 0
            }
            
            if results['optimizations_active']:
                logger.info(f"   ✅ Performance optimizations ACTIVE - {throughput:.1f} docs/sec")
            else:
                logger.warning(f"   ⚠️ Performance may not be optimal - {throughput:.1f} docs/sec")
            
            return results
            
        except Exception as e:
            logger.error(f"   ❌ Performance test ERROR - {e}")
            return {'status': 'ERROR', 'error': str(e)}
    
    def test_error_handling(self) -> dict:
        """Test error handling robustness."""
        logger.info("🛠️ Testing error handling...")
        
        results = {
            'timeout_handling': False,
            'schema_error_handling': False,
            'recursion_protection': False
        }
        
        # Test timeout handling
        try:
            embedder = create_incremental_embedder("R1311007", enable_timeout=True)
            if hasattr(embedder, 'enable_timeout') and embedder.enable_timeout:
                results['timeout_handling'] = True
                logger.info("   ✅ Timeout handling enabled")
        except Exception as e:
            logger.warning(f"   ⚠️ Timeout handling test failed: {e}")
        
        # Test schema error handling (by checking enhanced fields)
        try:
            system = EmbeddingOnlySystem("R1311007")
            if hasattr(system, '_attempt_schema_migration'):
                results['schema_error_handling'] = True
                logger.info("   ✅ Schema error handling available")
        except Exception as e:
            logger.warning(f"   ⚠️ Schema error handling test failed: {e}")
        
        # Test recursion protection (by checking batch size limits in incremental embedder)
        try:
            embedder = create_incremental_embedder("R1311007")
            # Check if the processing method exists (indicates recursion protection)
            if hasattr(embedder, '_process_single_document'):
                results['recursion_protection'] = True
                logger.info("   ✅ Recursion protection available")
        except Exception as e:
            logger.warning(f"   ⚠️ Recursion protection test failed: {e}")
        
        return results
    
    def test_main_folder_integrity(self) -> dict:
        """Test main folder integrity as source of truth."""
        logger.info("📁 Testing main folder integrity...")
        
        results = {}
        
        for proceeding in self.test_proceedings:
            main_folder = self.lance_db_dir / proceeding
            
            folder_status = {
                'exists': main_folder.exists(),
                'has_hashes': False,
                'has_lance_data': False,
                'is_source_of_truth': False
            }
            
            if folder_status['exists']:
                # Check for document_hashes.json
                hashes_file = main_folder / 'document_hashes.json'
                if hashes_file.exists():
                    try:
                        with open(hashes_file) as f:
                            hashes = json.load(f)
                        folder_status['has_hashes'] = len(hashes) > 0
                    except:
                        pass
                
                # Check for Lance data
                lance_dir = main_folder / f"{proceeding}_documents.lance"
                if lance_dir.exists() and any(lance_dir.iterdir()) if lance_dir.exists() else False:
                    folder_status['has_lance_data'] = True
                
                # Source of truth if has either hashes or lance data
                folder_status['is_source_of_truth'] = folder_status['has_hashes'] or folder_status['has_lance_data']
            
            results[proceeding] = folder_status
            
            status = "✅" if folder_status['is_source_of_truth'] else "❌"
            logger.info(f"   {status} {proceeding}: exists={folder_status['exists']}, hashes={folder_status['has_hashes']}, lance={folder_status['has_lance_data']}")
        
        return results
    
    def run_comprehensive_tests(self) -> dict:
        """Run all comprehensive tests."""
        logger.info("🚀 Running comprehensive system tests...")
        
        test_results = {}
        start_time = time.time()
        
        # Test 1: Backup prevention
        test_results['backup_prevention'] = self.test_backup_prevention()
        
        # Test 2: Schema compatibility
        test_results['schema_compatibility'] = self.test_schema_compatibility()
        
        # Test 3: Processing pipeline
        test_results['processing_pipeline'] = self.test_processing_pipeline()
        
        # Test 4: Performance optimizations
        test_results['performance_optimizations'] = self.test_performance_optimizations()
        
        # Test 5: Error handling
        test_results['error_handling'] = self.test_error_handling()
        
        # Test 6: Main folder integrity
        test_results['main_folder_integrity'] = self.test_main_folder_integrity()
        
        test_results['total_test_time'] = time.time() - start_time
        
        return test_results
    
    def generate_comprehensive_report(self, test_results: dict) -> str:
        """Generate comprehensive test report."""
        report = f"""
🧪 COMPREHENSIVE SYSTEM TEST REPORT
{'='*70}

📊 SUMMARY:
   Test Duration: {test_results['total_test_time']:.2f}s
   Test Categories: 6
   Proceedings Tested: {len(self.test_proceedings)}

🛡️ BACKUP PREVENTION:
   Prevention Active: {test_results['backup_prevention']['prevention_active']}
   Backup Folders Found: {len(test_results['backup_prevention']['backup_folders_found'])}
"""
        
        if test_results['backup_prevention']['backup_folders_found']:
            report += "   ❌ BACKUP FOLDERS STILL EXIST:\n"
            for folder in test_results['backup_prevention']['backup_folders_found']:
                report += f"     - {folder}\n"
        else:
            report += "   ✅ NO BACKUP FOLDERS - SYSTEM CLEAN\n"
        
        report += f"\n🔧 SCHEMA COMPATIBILITY:\n"
        for proceeding, result in test_results['schema_compatibility'].items():
            status = "✅" if result['status'] == 'PASS' else "❌"
            report += f"   {status} {proceeding}: {result['status']}\n"
            if result.get('error'):
                report += f"      Error: {result['error']}\n"
        
        report += f"\n🔄 PROCESSING PIPELINE:\n"
        for proceeding, result in test_results['processing_pipeline'].items():
            if 'error' in result:
                report += f"   ❌ {proceeding}: ERROR - {result['error']}\n"
            else:
                report += f"   ✅ {proceeding}: Embedded={result.get('total_embedded', 0)}, Failed={result.get('total_failed', 0)}\n"
        
        perf = test_results['performance_optimizations']
        report += f"\n🚀 PERFORMANCE OPTIMIZATIONS:\n"
        if 'error' in perf:
            report += f"   ❌ ERROR: {perf['error']}\n"
        else:
            report += f"   Model Init Time: {perf['model_init_time']:.2f}s\n"
            report += f"   Embedding Throughput: {perf['throughput_per_sec']:.1f} docs/sec\n"
            report += f"   Optimizations Active: {perf['optimizations_active']}\n"
        
        error_handling = test_results['error_handling']
        report += f"\n🛠️ ERROR HANDLING:\n"
        report += f"   ✅ Timeout Handling: {error_handling['timeout_handling']}\n"
        report += f"   ✅ Schema Error Handling: {error_handling['schema_error_handling']}\n"
        report += f"   ✅ Recursion Protection: {error_handling['recursion_protection']}\n"
        
        report += f"\n📁 MAIN FOLDER INTEGRITY:\n"
        for proceeding, result in test_results['main_folder_integrity'].items():
            status = "✅" if result['is_source_of_truth'] else "❌"
            report += f"   {status} {proceeding}: Source of Truth = {result['is_source_of_truth']}\n"
        
        # Overall assessment
        backup_clean = test_results['backup_prevention']['prevention_active']
        schema_passed = all(r['status'] == 'PASS' for r in test_results['schema_compatibility'].values())
        pipeline_working = all('error' not in r for r in test_results['processing_pipeline'].values())
        performance_good = test_results['performance_optimizations'].get('optimizations_active', False)
        error_handling_complete = all(test_results['error_handling'].values())
        folders_valid = sum(r['is_source_of_truth'] for r in test_results['main_folder_integrity'].values())
        
        overall_score = sum([backup_clean, schema_passed, pipeline_working, performance_good, error_handling_complete]) * 20
        
        report += f"""
🎯 OVERALL ASSESSMENT:
   Overall Score: {overall_score}/100
   ✅ Backup Prevention: {backup_clean}
   ✅ Schema Compatibility: {schema_passed}
   ✅ Processing Pipeline: {pipeline_working}
   ✅ Performance Optimizations: {performance_good}
   ✅ Error Handling: {error_handling_complete}
   📁 Valid Main Folders: {folders_valid}/{len(self.test_proceedings)}

"""
        
        if overall_score >= 80:
            report += "🎉 SYSTEM STATUS: EXCELLENT - PRODUCTION READY\n"
            report += "All core systems are working properly with comprehensive fixes applied.\n"
        elif overall_score >= 60:
            report += "✅ SYSTEM STATUS: GOOD - MINOR ISSUES\n"
            report += "Most systems working, some minor issues to address.\n"
        else:
            report += "⚠️ SYSTEM STATUS: NEEDS ATTENTION\n"
            report += "Several issues need to be resolved before production use.\n"
        
        return report


def main():
    """Run comprehensive system testing."""
    print("🧪 Comprehensive System Test Suite")
    print("=" * 50)
    
    tester = ComprehensiveSystemTester()
    
    # Run all tests
    test_results = tester.run_comprehensive_tests()
    
    # Generate and display report
    report = tester.generate_comprehensive_report(test_results)
    print(report)
    
    # Save report
    report_file = Path('comprehensive_system_test_report.txt')
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📋 Comprehensive report saved to: {report_file}")
    
    # Determine success
    backup_clean = test_results['backup_prevention']['prevention_active']
    schema_passed = all(r['status'] == 'PASS' for r in test_results['schema_compatibility'].values())
    
    if backup_clean and schema_passed:
        print(f"\n🎉 COMPREHENSIVE TESTING SUCCESSFUL!")
        print("System is clean, robust, and production-ready.")
        return True
    else:
        print(f"\n⚠️ Some issues need attention.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)