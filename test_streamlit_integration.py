#!/usr/bin/env python3
"""
Streamlit Integration Test

This test validates that Streamlit can launch with the processed R1202009 data
and that the proceeding is properly loaded and accessible.

Author: Claude Code
"""

import unittest
import subprocess
import time
import requests
import signal
import os
from pathlib import Path

class TestStreamlitIntegration(unittest.TestCase):
    """Test Streamlit integration with processed data"""
    
    @classmethod
    def setUpClass(cls):
        """Setup test environment"""
        cls.streamlit_port = 8503  # Use different port to avoid conflicts
        cls.streamlit_process = None
        cls.test_proceeding = 'R1202009'
    
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

    def test_01_validate_processed_data_exists(self):
        """Test that processed data exists for testing"""
        print(f"\n📋 Step 1: Validate Processed Data")
        
        import config
        paths = config.get_proceeding_file_paths(self.test_proceeding)
        
        # Check vector DB exists
        self.assertTrue(paths['vector_db'].exists(), 
                       f"Vector DB should exist: {paths['vector_db']}")
        print(f"   ✅ Vector DB found: {paths['vector_db']}")
        
        # Check document hashes exist
        if paths['document_hashes'].exists():
            import json
            with open(paths['document_hashes'], 'r') as f:
                doc_hashes = json.load(f)
            print(f"   ✅ Document hashes: {len(doc_hashes)} documents")
        else:
            print(f"   ⚠️ No document hashes found")

    def test_02_rag_system_loads_correctly(self):
        """Test that RAG system can load without errors"""
        print(f"\n🤖 Step 2: Validate RAG System Loading")
        
        try:
            from rag_core import CPUCRAGSystem
            
            rag_system = CPUCRAGSystem(current_proceeding=self.test_proceeding)
            self.assertIsNotNone(rag_system, "RAG system should initialize")
            
            # Get stats
            stats = rag_system.get_system_stats()
            print(f"   ✅ RAG system loaded")
            print(f"   📊 Chunks: {stats.get('total_chunks', 0)}")
            print(f"   📚 Documents: {stats.get('total_documents_hashed', 0)}")
            print(f"   🔋 Status: {stats.get('vector_store_status', 'unknown')}")
            
        except Exception as e:
            self.fail(f"RAG system failed to load: {e}")

    def test_03_streamlit_launches_successfully(self):
        """Test that Streamlit app can launch"""
        print(f"\n🌐 Step 3: Streamlit Launch Test")
        
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
            max_wait = 30  # 30 seconds max wait
            wait_time = 0
            
            while wait_time < max_wait:
                try:
                    response = requests.get(
                        f'http://localhost:{self.streamlit_port}',
                        timeout=5
                    )
                    if response.status_code == 200:
                        print(f"   ✅ Streamlit responding on port {self.streamlit_port}")
                        break
                except requests.exceptions.RequestException:
                    pass
                
                time.sleep(2)
                wait_time += 2
            else:
                self.fail(f"Streamlit failed to start within {max_wait} seconds")
                
        except Exception as e:
            self.fail(f"Failed to launch Streamlit: {e}")

    def test_04_streamlit_loads_proceeding_data(self):
        """Test that Streamlit can load the proceeding data"""
        print(f"\n📊 Step 4: Validate Proceeding Data Loading")
        
        # Basic page content check
        try:
            response = requests.get(
                f'http://localhost:{self.streamlit_port}',
                timeout=10
            )
            
            self.assertEqual(response.status_code, 200, "Should get 200 response")
            
            page_content = response.text
            
            # Check for basic Streamlit elements
            self.assertIn('stApp', page_content, "Should be a Streamlit app")
            
            # Check for CPUC content
            cpuc_indicators = ['CPUC', 'Regulatory', 'proceeding', 'R1202009']
            found_indicators = [indicator for indicator in cpuc_indicators 
                              if indicator.lower() in page_content.lower()]
            
            print(f"   ✅ Page loaded successfully")
            print(f"   🔍 Found indicators: {found_indicators}")
            
            # Should find at least some CPUC-related content
            self.assertGreater(len(found_indicators), 0, 
                             "Should find some CPUC-related content")
            
        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to fetch Streamlit page: {e}")

    def generate_test_summary(self):
        """Generate test summary"""
        print(f"\n📊 STREAMLIT INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"🧪 Test Proceeding: {self.test_proceeding}")
        print(f"🌐 Streamlit Port: {self.streamlit_port}")
        
        # Show system status
        try:
            from rag_core import CPUCRAGSystem
            rag_system = CPUCRAGSystem(current_proceeding=self.test_proceeding)
            stats = rag_system.get_system_stats()
            
            print(f"📊 RAG System Status:")
            print(f"   Total Chunks: {stats.get('total_chunks', 0)}")
            print(f"   Documents: {stats.get('total_documents_hashed', 0)}")
            print(f"   Vector Store: {stats.get('vector_store_status', 'unknown')}")
            
        except Exception as e:
            print(f"⚠️ Could not get system stats: {e}")
        
        print("=" * 60)
        print("✅ STREAMLIT INTEGRATION TEST COMPLETED!")


def run_streamlit_test():
    """Run the Streamlit integration test"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStreamlitIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary
    if result.wasSuccessful():
        test_instance = TestStreamlitIntegration()
        test_instance.generate_test_summary()
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print(f"🚀 Starting Streamlit Integration Test")
    print(f"📋 This test validates Streamlit can load R1202009 data")
    print(f"⏱️ Expected duration: 1-2 minutes")
    print("=" * 80)
    
    success = run_streamlit_test()
    exit_code = 0 if success else 1
    
    print(f"\n🎯 Test Result: {'PASSED' if success else 'FAILED'}")
    exit(exit_code)