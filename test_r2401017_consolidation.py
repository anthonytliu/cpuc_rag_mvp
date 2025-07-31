#!/usr/bin/env python3
"""
Comprehensive End-to-End Test for R2401017 Consolidation

Tests the full consolidation and validates the R2401017 system is working properly.
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from data_processing.embedding_only_system import EmbeddingOnlySystem
from data_processing.incremental_embedder import create_incremental_embedder
from core import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class R2401017ValidationTest:
    """Comprehensive validation test for R2401017 consolidation."""
    
    def __init__(self):
        self.proceeding = "R2401017"
        self.test_results = {}
        self.start_time = time.time()
        
    def run_full_validation(self):
        """Run complete validation test suite."""
        print("🚀 Starting R2401017 Comprehensive Validation Test")
        print("="*70)
        
        try:
            # Test 1: System Health Check
            self.test_system_health()
            
            # Test 2: Data Integrity Check
            self.test_data_integrity()
            
            # Test 3: Vector Database Validation
            self.test_vector_database()
            
            # Test 4: Incremental Processing Test
            self.test_incremental_processing()
            
            # Test 5: Check for Ongoing Processes
            self.test_for_ongoing_processes()
            
            # Test 6: Backup Folder Verification
            self.test_backup_cleanup()
            
            # Generate final report
            self.generate_report()
            
        except Exception as e:
            logger.error(f"❌ Validation test failed: {e}")
            self.test_results['overall_status'] = 'FAILED'
            self.test_results['error'] = str(e)
            self.generate_report()
    
    def test_system_health(self):
        """Test 1: System Health Check."""
        print("\n🩺 Test 1: System Health Check")
        print("-" * 40)
        
        try:
            system = EmbeddingOnlySystem(self.proceeding)
            health = system.health_check()
            
            self.test_results['health_check'] = {
                'status': 'PASSED' if health['healthy'] else 'FAILED',
                'proceeding': health['proceeding'],
                'embedding_model_ready': health['embedding_model_ready'],
                'vector_store_ready': health['vector_store_ready'],
                'lance_db_connected': health['lance_db_connected'],
                'vector_count': health['vector_count'],
                'db_path': health['db_path'],
                'timestamp': health['timestamp']
            }
            
            print(f"✅ Proceeding: {health['proceeding']}")
            print(f"✅ Embedding Model: {'Ready' if health['embedding_model_ready'] else 'Not Ready'}")
            print(f"✅ Vector Store: {'Ready' if health['vector_store_ready'] else 'Not Ready'}")
            print(f"✅ LanceDB: {'Connected' if health['lance_db_connected'] else 'Not Connected'}")
            print(f"✅ Vector Count: {health['vector_count']}")
            print(f"✅ Database Path: {health['db_path']}")
            print(f"✅ Overall Health: {'HEALTHY' if health['healthy'] else 'UNHEALTHY'}")
            
        except Exception as e:
            self.test_results['health_check'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Health check failed: {e}")
    
    def test_data_integrity(self):
        """Test 2: Data Integrity Check."""
        print("\n📊 Test 2: Data Integrity Check")
        print("-" * 40)
        
        try:
            base_path = Path("/Users/anthony.liu/Downloads/CPUC_REG_RAG/data/vector_stores/local_lance_db")
            main_folder = base_path / self.proceeding
            hash_file = main_folder / "document_hashes.json"
            
            # Check file existence
            files_exist = {
                'main_folder': main_folder.exists(),
                'document_hashes': hash_file.exists(),
                'vector_db': (main_folder / f"{self.proceeding}_documents.lance").exists()
            }
            
            # Load and validate document hashes
            document_data = {}
            if hash_file.exists():
                with open(hash_file, 'r') as f:
                    document_data = json.load(f)
            
            # Calculate statistics
            total_docs = len(document_data)
            total_chunks = sum(doc.get('chunk_count', 0) for doc in document_data.values())
            unique_urls = len(set(doc.get('url', '') for doc in document_data.values()))
            
            self.test_results['data_integrity'] = {
                'status': 'PASSED' if all(files_exist.values()) else 'FAILED',
                'files_exist': files_exist,
                'total_documents': total_docs,
                'total_expected_chunks': total_chunks,
                'unique_urls': unique_urls,
                'average_chunks_per_doc': round(total_chunks / total_docs, 2) if total_docs > 0 else 0
            }
            
            print(f"✅ Main Folder: {'Exists' if files_exist['main_folder'] else 'Missing'}")
            print(f"✅ Document Hashes: {'Exists' if files_exist['document_hashes'] else 'Missing'}")
            print(f"✅ Vector Database: {'Exists' if files_exist['vector_db'] else 'Missing'}")
            print(f"✅ Total Documents: {total_docs}")
            print(f"✅ Expected Chunks: {total_chunks:,}")
            print(f"✅ Unique URLs: {unique_urls}")
            print(f"✅ Avg Chunks/Doc: {total_chunks / total_docs:.1f}" if total_docs > 0 else "✅ Avg Chunks/Doc: N/A")
            
        except Exception as e:
            self.test_results['data_integrity'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Data integrity check failed: {e}")
    
    def test_vector_database(self):
        """Test 3: Vector Database Validation."""
        print("\n🔢 Test 3: Vector Database Validation")
        print("-" * 40)
        
        try:
            system = EmbeddingOnlySystem(self.proceeding)
            
            # Get vector count
            vector_count = system.get_vector_count()
            
            # Test database connectivity
            table_name = f"{self.proceeding}_documents"
            try:
                table = system.lance_db.open_table(table_name)
                table_accessible = True
                schema_info = str(table.schema)
                row_count = table.count_rows()
            except Exception as e:
                table_accessible = False
                schema_info = f"Error: {e}"
                row_count = 0
            
            self.test_results['vector_database'] = {
                'status': 'PASSED' if table_accessible and vector_count > 0 else 'FAILED',
                'vector_count': vector_count,
                'table_accessible': table_accessible,
                'row_count': row_count,
                'table_name': table_name,
                'schema_available': 'Error' not in schema_info
            }
            
            print(f"✅ Vector Count: {vector_count}")
            print(f"✅ Table Accessible: {'Yes' if table_accessible else 'No'}")
            print(f"✅ Row Count: {row_count}")
            print(f"✅ Table Name: {table_name}")
            print(f"✅ Schema Available: {'Yes' if 'Error' not in schema_info else 'No'}")
            
        except Exception as e:
            self.test_results['vector_database'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Vector database test failed: {e}")
    
    def test_incremental_processing(self):
        """Test 4: Incremental Processing Test."""
        print("\n🔄 Test 4: Incremental Processing Test")
        print("-" * 40)
        
        try:
            # Create incremental embedder
            embedder = create_incremental_embedder(self.proceeding, enable_timeout=True)
            
            # Get embedding status
            status = embedder.get_embedding_status()
            
            # Test basic functionality without actually processing
            embedder_ready = hasattr(embedder, 'embedding_system') and embedder.embedding_system is not None
            
            self.test_results['incremental_processing'] = {
                'status': 'PASSED' if embedder_ready else 'FAILED',
                'embedder_ready': embedder_ready,
                'total_embedded': status.get('total_embedded', 0),
                'total_failed': status.get('total_failed', 0),
                'last_updated': status.get('last_updated', 'Never'),
                'embedding_status': status.get('status', 'Unknown')
            }
            
            print(f"✅ Embedder Ready: {'Yes' if embedder_ready else 'No'}")
            print(f"✅ Total Embedded: {status.get('total_embedded', 0)}")
            print(f"✅ Total Failed: {status.get('total_failed', 0)}")
            print(f"✅ Last Updated: {status.get('last_updated', 'Never')}")
            print(f"✅ Status: {status.get('status', 'Unknown')}")
            
        except Exception as e:
            self.test_results['incremental_processing'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Incremental processing test failed: {e}")
    
    def test_for_ongoing_processes(self):
        """Test 5: Check for Ongoing Processes."""
        print("\n🔍 Test 5: Ongoing Process Check")
        print("-" * 40)
        
        try:
            import subprocess
            import psutil
            
            # Check for Python processes that might be processing R2401017
            python_processes = []
            r2401017_processes = []
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['name'] and 'python' in proc.info['name'].lower():
                        cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
                        python_processes.append({
                            'pid': proc.info['pid'],
                            'cmdline': cmdline
                        })
                        
                        if 'R2401017' in cmdline or 'r2401017' in cmdline.lower():
                            r2401017_processes.append({
                                'pid': proc.info['pid'],
                                'cmdline': cmdline
                            })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            self.test_results['ongoing_processes'] = {
                'status': 'PASSED' if len(r2401017_processes) == 0 else 'WARNING',
                'python_processes_count': len(python_processes),
                'r2401017_processes_count': len(r2401017_processes),
                'r2401017_processes': r2401017_processes
            }
            
            print(f"✅ Python Processes: {len(python_processes)} total")
            print(f"✅ R2401017 Processes: {len(r2401017_processes)}")
            
            if r2401017_processes:
                print("⚠️ WARNING: Found R2401017-related processes:")
                for proc in r2401017_processes:
                    print(f"   PID {proc['pid']}: {proc['cmdline'][:100]}...")
            else:
                print("✅ No R2401017-specific processes found")
                
        except Exception as e:
            self.test_results['ongoing_processes'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Process check failed: {e}")
    
    def test_backup_cleanup(self):
        """Test 6: Backup Folder Verification."""
        print("\n🧹 Test 6: Backup Cleanup Verification")
        print("-" * 40)
        
        try:
            base_dir = Path("/Users/anthony.liu/Downloads/CPUC_REG_RAG/data/vector_stores/local_lance_db")
            
            # Look for any remaining backup folders
            backup_folders = list(base_dir.glob(f"{self.proceeding}_backup_*"))
            consolidated_backups = list(base_dir.glob(f"{self.proceeding}_consolidated_backup*"))
            
            # Check if main folder exists and is properly structured
            main_folder = base_dir / self.proceeding
            main_folder_proper = main_folder.exists() and (main_folder / "document_hashes.json").exists()
            
            self.test_results['backup_cleanup'] = {
                'status': 'PASSED' if len(backup_folders) == 0 and main_folder_proper else 'WARNING',
                'remaining_backup_folders': len(backup_folders),
                'consolidated_backups': len(consolidated_backups),
                'main_folder_proper': main_folder_proper,
                'backup_folder_names': [f.name for f in backup_folders]
            }
            
            print(f"✅ Main Folder Proper: {'Yes' if main_folder_proper else 'No'}")
            print(f"✅ Remaining Backup Folders: {len(backup_folders)}")
            print(f"✅ Consolidated Backups: {len(consolidated_backups)}")
            
            if backup_folders:
                print("⚠️ WARNING: Found remaining backup folders:")
                for folder in backup_folders:
                    print(f"   {folder.name}")
            else:
                print("✅ All individual backup folders cleaned up")
                
        except Exception as e:
            self.test_results['backup_cleanup'] = {'status': 'FAILED', 'error': str(e)}
            print(f"❌ Backup cleanup verification failed: {e}")
    
    def generate_report(self):
        """Generate final validation report."""
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "="*70)
        print("📋 R2401017 CONSOLIDATION VALIDATION REPORT")
        print("="*70)
        
        # Overall status
        all_tests = [result.get('status', 'FAILED') for result in self.test_results.values() if isinstance(result, dict) and 'status' in result]
        failed_tests = [status for status in all_tests if status == 'FAILED']
        warning_tests = [status for status in all_tests if status == 'WARNING']
        
        if len(failed_tests) == 0 and len(warning_tests) == 0:
            overall_status = "✅ ALL TESTS PASSED"
        elif len(failed_tests) == 0:
            overall_status = "⚠️ PASSED WITH WARNINGS"
        else:
            overall_status = "❌ SOME TESTS FAILED"
        
        print(f"Overall Status: {overall_status}")
        print(f"Proceeding: {self.proceeding}")
        print(f"Test Duration: {elapsed_time:.2f} seconds")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        print("\nTest Summary:")
        print("-" * 40)
        
        test_names = {
            'health_check': '🩺 System Health Check',
            'data_integrity': '📊 Data Integrity Check', 
            'vector_database': '🔢 Vector Database Validation',
            'incremental_processing': '🔄 Incremental Processing Test',
            'ongoing_processes': '🔍 Ongoing Process Check',
            'backup_cleanup': '🧹 Backup Cleanup Verification'
        }
        
        for test_key, test_name in test_names.items():
            if test_key in self.test_results:
                result = self.test_results[test_key]
                status = result.get('status', 'UNKNOWN')
                status_emoji = "✅" if status == "PASSED" else "⚠️" if status == "WARNING" else "❌"
                print(f"{status_emoji} {test_name}: {status}")
                
                if status == "FAILED" and 'error' in result:
                    print(f"    Error: {result['error']}")
        
        # Key metrics
        if 'data_integrity' in self.test_results and self.test_results['data_integrity']['status'] != 'FAILED':
            data = self.test_results['data_integrity']
            print(f"\nKey Metrics:")
            print(f"📄 Documents: {data.get('total_documents', 'N/A')}")
            print(f"🔢 Expected Chunks: {data.get('total_expected_chunks', 'N/A'):,}")
            
        if 'vector_database' in self.test_results and self.test_results['vector_database']['status'] != 'FAILED':
            vector = self.test_results['vector_database']
            print(f"🗄️ Current Vectors: {vector.get('vector_count', 'N/A')}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if len(failed_tests) > 0:
            print("❌ Address failed tests before proceeding with R2401017 operations")
        elif len(warning_tests) > 0:
            print("⚠️ Review warnings - system functional but may need attention")
        else:
            print("✅ R2401017 consolidation is complete and system is fully operational")
        
        print("="*70)
        
        # Save detailed results to file
        report_file = Path(f"r2401017_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        try:
            with open(report_file, 'w') as f:
                json.dump({
                    'overall_status': overall_status,
                    'proceeding': self.proceeding,
                    'test_duration': elapsed_time,
                    'timestamp': datetime.now().isoformat(),
                    'test_results': self.test_results
                }, f, indent=2)
            print(f"📋 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"⚠️ Could not save detailed report: {e}")


if __name__ == "__main__":
    validator = R2401017ValidationTest()
    validator.run_full_validation()