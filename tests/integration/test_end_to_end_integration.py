#!/usr/bin/env python3
"""
Comprehensive End-to-End Integration Test

This test performs a complete pipeline validation:
1. Builds a full vector database using R1202009 data (75 PDFs)
2. Validates chunking, embedding, and vector store creation
3. Tests RAG system functionality and query processing
4. Launches Streamlit app and validates UI components
5. Verifies all data is properly loaded and accessible

This is the definitive test that proves the entire system works correctly.

Author: Claude Code
"""

import unittest
import subprocess
import time
import requests
import signal
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import tempfile
import shutil

# Setup test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEndToEndIntegration(unittest.TestCase):
    """Comprehensive end-to-end integration test"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.test_proceeding = 'R1202009'  # Smallest proceeding with 75 PDFs
        cls.streamlit_port = 8502  # Use different port to avoid conflicts
        cls.streamlit_process = None
        cls.max_processing_time = 1800  # 30 minutes max for full processing
        
        print(f"\n🧪 End-to-End Integration Test Setup")
        print(f"📋 Test Proceeding: {cls.test_proceeding}")
        print(f"🔗 Streamlit Port: {cls.streamlit_port}")
        print("=" * 60)
    
    @classmethod
    def tearDownClass(cls):
        """Cleanup after tests"""
        if cls.streamlit_process:
            try:
                cls.streamlit_process.terminate()
                cls.streamlit_process.wait(timeout=10)
                print(f"✅ Streamlit process terminated")
            except subprocess.TimeoutExpired:
                cls.streamlit_process.kill()
                print(f"⚠️ Streamlit process killed (timeout)")
            except Exception as e:
                print(f"⚠️ Error terminating Streamlit: {e}")

    def test_01_validate_prerequisites(self):
        """Test that all prerequisites are in place"""
        print(f"\n📋 Step 1: Validating Prerequisites")
        
        # Check proceeding directory exists
        proceeding_dir = Path('cpuc_proceedings') / self.test_proceeding
        self.assertTrue(proceeding_dir.exists(), 
                       f"Proceeding directory should exist: {proceeding_dir}")
        print(f"   ✅ Proceeding directory: {proceeding_dir}")
        
        # Check scraped PDF history exists
        import config
        paths = config.get_proceeding_file_paths(self.test_proceeding)
        
        scraped_history_exists = (paths['scraped_pdf_history'].exists() or 
                                paths['scraped_pdf_history_alt'].exists())
        self.assertTrue(scraped_history_exists, "Scraped PDF history should exist")
        
        # Load and validate scraped data
        from standalone_data_processor import load_scraped_pdf_data
        scraped_data = load_scraped_pdf_data(self.test_proceeding)
        self.assertIsNotNone(scraped_data, "Should be able to load scraped data")
        self.assertGreater(len(scraped_data), 0, "Should have scraped PDFs")
        
        pdf_count = len(scraped_data)
        print(f"   ✅ Scraped PDF data: {pdf_count} PDFs found")
        
        # Validate it's around 75 PDFs (allowing some variance)
        self.assertGreater(pdf_count, 50, "Should have more than 50 PDFs")
        self.assertLess(pdf_count, 100, "Should have less than 100 PDFs")
        print(f"   ✅ PDF count validation: {pdf_count} PDFs (expected ~75)")

    def test_02_full_data_processing_pipeline(self):
        """Test complete data processing from scraped PDFs to vector store"""
        print(f"\n🔄 Step 2: Full Data Processing Pipeline")
        
        start_time = datetime.now()
        
        # Clean up any existing processed data to ensure fresh test
        import config
        paths = config.get_proceeding_file_paths(self.test_proceeding)
        
        # Clean vector DB directory
        if paths['vector_db'].exists():
            shutil.rmtree(paths['vector_db'])
            print(f"   🧹 Cleaned existing vector DB: {paths['vector_db']}")
        
        # Clean embeddings directory
        if paths['embeddings_dir'].exists():
            shutil.rmtree(paths['embeddings_dir'])
            print(f"   🧹 Cleaned existing embeddings: {paths['embeddings_dir']}")
        
        # Run standalone data processor
        print(f"   🚀 Starting data processing (max {self.max_processing_time/60:.1f} minutes)...")
        
        try:
            result = subprocess.run([
                'python', 'standalone_data_processor.py', self.test_proceeding,
                '--batch-size', '5',  # Smaller batches for more stable processing
                '--verbose'
            ], 
            capture_output=True, 
            text=True, 
            timeout=self.max_processing_time,
            cwd=Path.cwd()
            )
            
            print(f"   📊 Processing return code: {result.returncode}")
            
            if result.returncode != 0:
                print(f"   ❌ Processing stderr: {result.stderr}")
                print(f"   📝 Processing stdout: {result.stdout}")
                self.fail(f"Data processing failed with return code {result.returncode}")
            
            processing_time = datetime.now() - start_time
            print(f"   ✅ Data processing completed in {processing_time}")
            
            # Show some output for debugging
            if result.stdout:
                stdout_lines = result.stdout.split('\n')
                print(f"   📝 Last few lines of output:")
                for line in stdout_lines[-5:]:
                    if line.strip():
                        print(f"      {line}")
        
        except subprocess.TimeoutExpired:
            self.fail(f"Data processing timed out after {self.max_processing_time/60:.1f} minutes")
        except Exception as e:
            self.fail(f"Data processing failed with exception: {e}")

    def test_03_validate_vector_store_creation(self):
        """Test that vector store was properly created"""
        print(f"\n🗄️ Step 3: Validate Vector Store Creation")
        
        import config
        paths = config.get_proceeding_file_paths(self.test_proceeding)
        
        # Check vector DB directory exists
        self.assertTrue(paths['vector_db'].exists(), 
                       f"Vector DB directory should exist: {paths['vector_db']}")
        print(f"   ✅ Vector DB directory: {paths['vector_db']}")
        
        # Check document hashes file exists
        self.assertTrue(paths['document_hashes'].exists(),
                       f"Document hashes should exist: {paths['document_hashes']}")
        
        # Load and validate document hashes
        with open(paths['document_hashes'], 'r') as f:
            doc_hashes = json.load(f)
        
        self.assertGreater(len(doc_hashes), 0, "Should have processed documents")
        print(f"   ✅ Document hashes: {len(doc_hashes)} documents processed")
        
        # Validate that we have LanceDB data
        lance_db_files = list(paths['vector_db'].rglob("*"))
        self.assertGreater(len(lance_db_files), 0, "Should have LanceDB files")
        print(f"   ✅ LanceDB files: {len(lance_db_files)} files created")

    def test_04_rag_system_initialization(self):
        """Test RAG system can be initialized and works correctly"""
        print(f"\n🤖 Step 4: RAG System Initialization")
        
        # Initialize RAG system
        from rag_core import CPUCRAGSystem
        
        rag_system = CPUCRAGSystem(current_proceeding=self.test_proceeding)
        self.assertIsNotNone(rag_system, "RAG system should initialize")
        print(f"   ✅ RAG system initialized")
        
        # Check vector store is loaded
        self.assertIsNotNone(rag_system.vectordb, "Vector store should be loaded")
        print(f"   ✅ Vector store loaded")
        
        # Get system stats
        stats = rag_system.get_system_stats()
        self.assertIsInstance(stats, dict, "Should return stats dictionary")
        self.assertGreater(stats.get('total_chunks', 0), 0, "Should have chunks")
        print(f"   ✅ System stats: {stats.get('total_chunks', 0)} chunks")
        
        # Test query functionality
        print(f"   🔍 Testing query functionality...")
        test_query = "What is this proceeding about?"
        
        query_results = []
        try:
            for result in rag_system.query(test_query):
                if isinstance(result, dict):
                    query_results.append(result)
                    break  # We just need to verify it works
            
            self.assertGreater(len(query_results), 0, "Should get query results")
            
            final_result = query_results[0]
            self.assertIn('answer', final_result, "Should have answer field")
            self.assertIn('sources', final_result, "Should have sources field")
            
            print(f"   ✅ Query successful: {len(final_result['sources'])} sources")
            
        except Exception as e:
            self.fail(f"Query failed: {e}")

    def test_05_streamlit_app_launch(self):
        """Test Streamlit app can be launched and responds"""
        print(f"\n🌐 Step 5: Streamlit App Launch & Validation")
        
        # Launch Streamlit app
        print(f"   🚀 Launching Streamlit on port {self.streamlit_port}...")
        
        try:
            self.streamlit_process = subprocess.Popen([
                'streamlit', 'run', 'app.py',
                '--server.port', str(self.streamlit_port),
                '--server.headless', 'true',
                '--server.runOnSave', 'false',
                '--browser.gatherUsageStats', 'false'
            ], 
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path.cwd()
            )
            
            # Wait for Streamlit to start
            print(f"   ⏳ Waiting for Streamlit to start...")
            max_wait = 60  # 60 seconds max wait
            wait_time = 0
            
            while wait_time < max_wait:
                try:
                    response = requests.get(
                        f'http://localhost:{self.streamlit_port}',
                        timeout=5
                    )
                    if response.status_code == 200:
                        print(f"   ✅ Streamlit app responding on port {self.streamlit_port}")
                        break
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(2)
                wait_time += 2
            else:
                self.fail(f"Streamlit app failed to start within {max_wait} seconds")
            
            # Test app health endpoint
            try:
                health_response = requests.get(
                    f'http://localhost:{self.streamlit_port}/healthz',
                    timeout=5
                )
                print(f"   ✅ Health check: {health_response.status_code}")
            except requests.exceptions.RequestException:
                print(f"   ℹ️ Health endpoint not available (expected)")
            
            # Basic HTML content check
            try:
                page_response = requests.get(
                    f'http://localhost:{self.streamlit_port}',
                    timeout=10
                )
                
                self.assertEqual(page_response.status_code, 200, "Should get 200 response")
                
                page_content = page_response.text
                self.assertIn('CPUC', page_content, "Should contain CPUC in title")
                print(f"   ✅ Page content validation passed")
                
            except requests.exceptions.RequestException as e:
                self.fail(f"Failed to fetch Streamlit page: {e}")
        
        except Exception as e:
            self.fail(f"Failed to launch Streamlit: {e}")

    def test_06_system_integration_validation(self):
        """Test complete system integration with real data"""
        print(f"\n🔗 Step 6: System Integration Validation")
        
        # Test that the app can load the proceeding data
        from app import get_available_proceedings_from_db
        
        available_proceedings = get_available_proceedings_from_db()
        self.assertIn(self.test_proceeding, available_proceedings,
                     f"{self.test_proceeding} should be available in app")
        print(f"   ✅ Proceeding available in app: {available_proceedings}")
        
        # Test RAG system integration with full stats
        from rag_core import CPUCRAGSystem
        
        rag_system = CPUCRAGSystem(current_proceeding=self.test_proceeding)
        stats = rag_system.get_system_stats()
        
        # Validate comprehensive stats
        expected_stats = ['total_chunks', 'total_documents_hashed', 'vector_store_status']
        for stat in expected_stats:
            self.assertIn(stat, stats, f"Should have {stat} in system stats")
        
        print(f"   ✅ System stats complete:")
        print(f"      📊 Total chunks: {stats.get('total_chunks', 0)}")
        print(f"      📚 Total documents: {stats.get('total_documents_hashed', 0)}")
        print(f"      🔋 Vector store status: {stats.get('vector_store_status', 'unknown')}")
        
        # Verify data consistency
        import config
        paths = config.get_proceeding_file_paths(self.test_proceeding)
        
        with open(paths['document_hashes'], 'r') as f:
            doc_hashes = json.load(f)
        
        self.assertEqual(len(doc_hashes), stats.get('total_documents_hashed', 0),
                        "Document hashes count should match system stats")
        
        print(f"   ✅ Data consistency validated")

    def generate_test_summary(self):
        """Generate a comprehensive test summary"""
        print(f"\n📊 COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        print(f"🧪 Test Proceeding: {self.test_proceeding}")
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            import config
            paths = config.get_proceeding_file_paths(self.test_proceeding)
            
            # Document statistics
            if paths['document_hashes'].exists():
                with open(paths['document_hashes'], 'r') as f:
                    doc_hashes = json.load(f)
                print(f"📚 Documents Processed: {len(doc_hashes)}")
            
            # Vector store statistics
            if paths['vector_db'].exists():
                db_files = list(paths['vector_db'].rglob("*"))
                print(f"🗄️ Vector DB Files: {len(db_files)}")
            
            # RAG system statistics
            from rag_core import CPUCRAGSystem
            rag_system = CPUCRAGSystem(current_proceeding=self.test_proceeding)
            stats = rag_system.get_system_stats()
            print(f"📊 Total Chunks: {stats.get('total_chunks', 0)}")
            print(f"🔋 Vector Store: {stats.get('vector_store_status', 'unknown')}")
            
        except Exception as e:
            print(f"⚠️ Error generating summary: {e}")
        
        print("=" * 60)
        print("✅ END-TO-END INTEGRATION TEST COMPLETED SUCCESSFULLY!")


def run_comprehensive_test():
    """Run the comprehensive end-to-end test"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEndToEndIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary if all tests passed
    if result.wasSuccessful():
        test_instance = TestEndToEndIntegration()
        test_instance.generate_test_summary()
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print(f"🚀 Starting Comprehensive End-to-End Integration Test")
    print(f"📋 This test will build a complete database using R1202009 data")
    print(f"⏱️ Expected duration: 15-30 minutes depending on system performance")
    print("=" * 80)
    
    success = run_comprehensive_test()
    exit_code = 0 if success else 1
    
    print(f"\n🎯 Test Result: {'PASSED' if success else 'FAILED'}")
    exit(exit_code)