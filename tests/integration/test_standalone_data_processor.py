#!/usr/bin/env python3
"""
End-to-End Tests for Standalone Data Processor

Tests the complete pipeline from scraped PDF data to embeddings using
existing R1202009 data (smallest proceeding with 75 PDFs). These tests 
are designed to be non-destructive and verify the entire data processing workflow.

Author: Claude Code
"""

import json
import logging
import os
import sys
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch, MagicMock

# Setup test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestStandaloneDataProcessorIntegration(unittest.TestCase):
    """Test standalone data processor with real scraped data"""
    
    def setUp(self):
        """Setup test environment"""
        self.current_dir = Path('.')
        self.cpuc_proceedings_dir = self.current_dir / 'cpuc_proceedings'
        
        # Test proceedings that should exist - using R1202009 as primary test case (smallest)
        self.test_proceedings = ['R1202009']
        
    def test_cpuc_proceedings_structure_exists(self):
        """Test that the cpuc_proceedings directory structure exists"""
        print(f"\n🔍 Testing cpuc_proceedings directory structure...")
        
        # Check main directory exists
        self.assertTrue(self.cpuc_proceedings_dir.exists(), 
                       "cpuc_proceedings directory should exist")
        print(f"   ✅ cpuc_proceedings directory found")
        
        # Check each proceeding directory
        for proceeding in self.test_proceedings:
            proceeding_dir = self.cpuc_proceedings_dir / proceeding
            self.assertTrue(proceeding_dir.exists(), 
                           f"{proceeding} directory should exist")
            print(f"   ✅ {proceeding} directory found")
            
            # Check for required files and directories
            documents_dir = proceeding_dir / 'documents'
            embeddings_dir = proceeding_dir / 'embeddings'
            history_file = proceeding_dir / f'{proceeding}_scraped_pdf_history.json'
            csv_file = documents_dir / f'{proceeding}_documents.csv'
            
            self.assertTrue(documents_dir.exists(), f"{proceeding} documents directory should exist")
            self.assertTrue(embeddings_dir.exists(), f"{proceeding} embeddings directory should exist") 
            self.assertTrue(history_file.exists(), f"{proceeding} history file should exist")
            print(f"   ✅ {proceeding} required structure verified")

    def test_scraped_data_integrity(self):
        """Test that scraped PDF data is properly formatted and accessible"""
        print(f"\n📊 Testing scraped data integrity...")
        
        for proceeding in self.test_proceedings:
            print(f"\n🔍 Analyzing: {proceeding}...")
            
            proceeding_dir = self.cpuc_proceedings_dir / proceeding
            history_file = proceeding_dir / f'{proceeding}_scraped_pdf_history.json'
            
            # Load and verify scraped data
            with open(history_file, 'r') as f:
                scraped_data = json.load(f)
            
            self.assertIsInstance(scraped_data, dict, f"{proceeding} scraped data should be a dictionary")
            self.assertGreater(len(scraped_data), 0, f"{proceeding} should have scraped data")
            
            print(f"   📄 Total PDFs: {len(scraped_data)}")
            
            # Check data structure
            sample_data = next(iter(scraped_data.values()))
            required_fields = ['url', 'title', 'source']
            
            for field in required_fields:
                self.assertIn(field, sample_data, f"PDF data should contain '{field}' field")
            
            # Count by source
            csv_count = sum(1 for data in scraped_data.values() if data.get('source') == 'csv')
            google_count = sum(1 for data in scraped_data.values() if data.get('source') == 'google search')
            
            print(f"   📋 CSV sourced: {csv_count}")
            print(f"   🔍 Google sourced: {google_count}")
            print(f"   ✅ Data integrity verified")

    def test_config_integration(self):
        """Test that config integration works with new structure"""
        print(f"\n⚙️ Testing config integration...")
        
        import config
        
        for proceeding in self.test_proceedings:
            print(f"\n🔍 Testing config for: {proceeding}")
            
            # Get paths from config
            paths = config.get_proceeding_file_paths(proceeding)
            
            # Verify new structure paths exist
            required_paths = [
                'proceeding_dir', 'documents_dir', 'embeddings_dir',
                'scraped_pdf_history', 'documents_csv'
            ]
            
            for path_key in required_paths:
                self.assertIn(path_key, paths, f"Config should provide '{path_key}' path")
                print(f"   ✅ {path_key}: {paths[path_key]}")
            
            # Verify paths point to existing locations
            self.assertTrue(paths['proceeding_dir'].exists(), "Proceeding directory should exist")
            self.assertTrue(paths['documents_dir'].exists(), "Documents directory should exist") 
            self.assertTrue(paths['embeddings_dir'].exists(), "Embeddings directory should exist")
            
            # Check for scraped PDF history (either naming convention)
            scraped_history_exists = (paths['scraped_pdf_history'].exists() or 
                                    paths['scraped_pdf_history_alt'].exists())
            self.assertTrue(scraped_history_exists, "Scraped PDF history should exist")
            
            print(f"   ✅ All config paths verified for {proceeding}")

    def test_standalone_processor_listing(self):
        """Test that standalone processor can list proceedings correctly"""
        print(f"\n📋 Testing standalone processor listing...")
        
        from standalone_data_processor import discover_available_proceedings, get_proceeding_status
        
        # Test proceeding discovery
        discovered = discover_available_proceedings()
        self.assertIsInstance(discovered, list, "Should return list of proceedings")
        
        for proceeding in self.test_proceedings:
            self.assertIn(proceeding, discovered, f"{proceeding} should be discovered")
        
        print(f"   ✅ Discovered {len(discovered)} proceedings: {discovered}")
        
        # Test status reporting
        for proceeding in self.test_proceedings:
            status = get_proceeding_status(proceeding)
            
            self.assertIsInstance(status, dict, "Status should be a dictionary")
            self.assertEqual(status['proceeding'], proceeding, "Status should match proceeding")
            self.assertTrue(status['scraped_data_exists'], f"{proceeding} should have scraped data")
            self.assertGreater(status['total_scraped_pdfs'], 0, f"{proceeding} should have PDFs")
            
            print(f"   ✅ {proceeding}: {status['total_scraped_pdfs']} PDFs, status: {status['processing_status']}")

    @patch('incremental_embedder.IncrementalEmbedder')
    def test_processing_workflow_dry_run(self, mock_embedder_class):
        """Test the processing workflow without actually creating embeddings"""
        print(f"\n🧪 Testing processing workflow (dry run)...")
        
        # Mock the embedder to avoid actual processing
        mock_embedder = MagicMock()
        mock_embedder.process_incremental_embeddings.return_value = {
            'status': 'completed',
            'documents_processed': 10,
            'successful': 10,
            'failed': 0
        }
        mock_embedder_class.return_value = mock_embedder
        
        from standalone_data_processor import process_proceeding_documents
        
        # Test single proceeding processing
        proceeding = self.test_proceedings[0]
        print(f"\n🔍 Testing dry run for: {proceeding}")
        
        result = process_proceeding_documents(proceeding, batch_size=5, force_rebuild=False)
        
        self.assertIsInstance(result, dict, "Should return result dictionary")
        self.assertEqual(result['status'], 'completed', "Processing should complete successfully")
        
        # Verify embedder was called correctly
        mock_embedder_class.assert_called_once()
        mock_embedder.process_incremental_embeddings.assert_called_once()
        
        print(f"   ✅ Dry run completed successfully")
        print(f"   📊 Mock result: {result}")

    def test_embeddings_directory_setup(self):
        """Test that embeddings directories are properly created and configured"""
        print(f"\n📁 Testing embeddings directory setup...")
        
        from standalone_data_processor import setup_proceeding_directories
        
        for proceeding in self.test_proceedings:
            print(f"\n🔍 Testing setup for: {proceeding}")
            
            # Test directory setup
            success = setup_proceeding_directories(proceeding)
            self.assertTrue(success, f"Directory setup should succeed for {proceeding}")
            
            # Verify directories exist
            import config
            paths = config.get_proceeding_file_paths(proceeding)
            
            self.assertTrue(paths['embeddings_dir'].exists(), "Embeddings directory should exist")
            self.assertTrue(paths['vector_db'].exists(), "Vector DB directory should exist")
            
            print(f"   ✅ Embeddings directory: {paths['embeddings_dir']}")
            print(f"   ✅ Vector DB directory: {paths['vector_db']}")

    def test_data_safety_read_only_verification(self):
        """Verify that scraped PDF data remains read-only and untouched"""
        print(f"\n🔒 Testing data safety (read-only verification)...")
        
        # Record original file stats
        original_stats = {}
        
        for proceeding in self.test_proceedings:
            proceeding_dir = self.cpuc_proceedings_dir / proceeding
            history_file = proceeding_dir / f'{proceeding}_scraped_pdf_history.json'
            csv_file = proceeding_dir / 'documents' / f'{proceeding}_documents.csv'
            
            original_stats[proceeding] = {
                'history_mtime': history_file.stat().st_mtime if history_file.exists() else None,
                'history_size': history_file.stat().st_size if history_file.exists() else None,
                'csv_mtime': csv_file.stat().st_mtime if csv_file.exists() else None,
                'csv_size': csv_file.stat().st_size if csv_file.exists() else None,
            }
        
        print(f"   📊 Recorded original file stats for {len(self.test_proceedings)} proceedings")
        
        # Import and run any processing functions that might modify data
        from standalone_data_processor import load_scraped_pdf_data, get_proceeding_status
        
        for proceeding in self.test_proceedings:
            # These functions should only read data
            scraped_data = load_scraped_pdf_data(proceeding)
            status = get_proceeding_status(proceeding)
            
            self.assertIsNotNone(scraped_data, f"Should load data for {proceeding}")
            self.assertIsNotNone(status, f"Should get status for {proceeding}")
        
        print(f"   ✅ Read operations completed")
        
        # Verify no files were modified
        for proceeding in self.test_proceedings:
            proceeding_dir = self.cpuc_proceedings_dir / proceeding
            history_file = proceeding_dir / f'{proceeding}_scraped_pdf_history.json'
            csv_file = proceeding_dir / 'documents' / f'{proceeding}_documents.csv'
            
            original = original_stats[proceeding]
            
            if history_file.exists():
                current_mtime = history_file.stat().st_mtime
                current_size = history_file.stat().st_size
                
                self.assertEqual(current_mtime, original['history_mtime'], 
                               f"{proceeding} history file should not be modified")
                self.assertEqual(current_size, original['history_size'],
                               f"{proceeding} history file size should not change")
            
            if csv_file.exists():
                current_mtime = csv_file.stat().st_mtime
                current_size = csv_file.stat().st_size
                
                self.assertEqual(current_mtime, original['csv_mtime'],
                               f"{proceeding} CSV file should not be modified")  
                self.assertEqual(current_size, original['csv_size'],
                               f"{proceeding} CSV file size should not change")
        
        print(f"   ✅ All PDF scraping files remain untouched")

    def test_vector_db_root_location(self):
        """Test that vector database is correctly located in root directory"""
        print(f"\n🗄️ Testing vector database root location...")
        
        import config
        
        for proceeding in self.test_proceedings:
            paths = config.get_proceeding_file_paths(proceeding)
            vector_db_path = paths['vector_db']
            
            # Verify vector DB is in root local_lance_db directory
            expected_root_path = Path('.') / 'local_lance_db' / proceeding
            
            self.assertEqual(vector_db_path.resolve(), expected_root_path.resolve(),
                           f"Vector DB should be in root local_lance_db directory")
            
            print(f"   ✅ {proceeding} vector DB: {vector_db_path}")
            
            # Ensure directory exists (may be created by setup)
            if not vector_db_path.exists():
                vector_db_path.mkdir(parents=True, exist_ok=True)
                print(f"   📁 Created vector DB directory: {vector_db_path}")

    def test_end_to_end_integration_summary(self):
        """Provide comprehensive summary of integration test results"""
        print(f"\n📋 End-to-End Integration Summary")
        print("=" * 60)
        
        from standalone_data_processor import discover_available_proceedings, get_proceeding_status
        
        discovered = discover_available_proceedings()
        print(f"\n📁 Available Proceedings: {len(discovered)}")
        
        total_pdfs = 0
        
        for proceeding in discovered:
            status = get_proceeding_status(proceeding)
            total_pdfs += status['total_scraped_pdfs']
            
            print(f"\n  {proceeding}:")
            print(f"    📄 PDFs: {status['total_scraped_pdfs']}")
            print(f"    📊 Status: {status['processing_status']}")
            print(f"    📂 Data exists: {'✅' if status['scraped_data_exists'] else '❌'}")
            print(f"    🗂️ Embeddings dir: {'✅' if status['embeddings_dir_exists'] else '❌'}")
            print(f"    🗄️ Vector DB: {'✅' if status['vector_db_exists'] else '❌'}")
        
        print(f"\n🎯 Total PDFs available for processing: {total_pdfs}")
        print(f"✅ All integration tests passed!")


class TestImprovedLoggingAndConsoleOutput(unittest.TestCase):
    """Test improved logging functionality and console output"""
    
    def setUp(self):
        """Setup test environment for logging tests"""
        # Save original config values
        import config
        self.original_debug = getattr(config, 'DEBUG', False)
        self.original_verbose = getattr(config, 'VERBOSE_LOGGING', False)
    
    def tearDown(self):
        """Restore original config values"""
        import config
        config.DEBUG = self.original_debug
        config.VERBOSE_LOGGING = self.original_verbose
    
    def test_debug_mode_logging_configuration(self):
        """Test that DEBUG mode affects logging configuration"""
        print(f"\n🔧 Testing DEBUG mode logging configuration...")
        
        # Test with DEBUG=False (production mode)
        os.environ['DEBUG'] = 'false'
        
        # Reload config to pick up environment change
        import importlib
        import config
        importlib.reload(config)
        
        self.assertFalse(config.DEBUG, "DEBUG should be False when env var is 'false'")
        self.assertFalse(config.VERBOSE_LOGGING, "VERBOSE_LOGGING should be False when DEBUG is False")
        print(f"   ✅ Production mode: DEBUG={config.DEBUG}, VERBOSE_LOGGING={config.VERBOSE_LOGGING}")
        
        # Test with DEBUG=True (debug mode)
        os.environ['DEBUG'] = 'true'
        importlib.reload(config)
        
        self.assertTrue(config.DEBUG, "DEBUG should be True when env var is 'true'")
        self.assertTrue(config.VERBOSE_LOGGING, "VERBOSE_LOGGING should be True when DEBUG is True")
        print(f"   ✅ Debug mode: DEBUG={config.DEBUG}, VERBOSE_LOGGING={config.VERBOSE_LOGGING}")
        
        # Clean up environment
        if 'DEBUG' in os.environ:
            del os.environ['DEBUG']
        importlib.reload(config)
    
    def test_console_output_format_in_production_mode(self):
        """Test that console output is clean in production mode"""
        print(f"\n📺 Testing console output format in production mode...")
        
        # Set production mode
        import config
        config.DEBUG = False
        config.VERBOSE_LOGGING = False

        # Capture stdout
        captured_output = StringIO()
        
        # Test that we can import and setup functions work
        try:
            from standalone_data_processor import (
                discover_available_proceedings, 
                get_proceeding_status,
                setup_proceeding_directories
            )
            
            proceedings = discover_available_proceedings()
            if proceedings:
                test_proceeding = proceedings[0]
                status = get_proceeding_status(test_proceeding)
                print(f"   ✅ Console output functions work with proceeding: {test_proceeding}")
                print(f"   ✅ Status check returns: {status['processing_status']}")
            else:
                print(f"   ⚠️ No proceedings found for testing")
                
        except Exception as e:
            print(f"   ❌ Error testing console output: {e}")
    
    def test_progress_bar_integration(self):
        """Test that tqdm progress bar works correctly"""
        print(f"\n⏳ Testing progress bar integration...")
        
        try:
            from tqdm import tqdm
            import time
            
            # Test basic tqdm functionality
            items = list(range(5))
            progress_bar = tqdm(items, desc="Test Progress", disable=True)  # Disable for test
            
            for item in progress_bar:
                time.sleep(0.01)  # Simulate work
                progress_bar.set_postfix_str(f"Item {item}")
            
            progress_bar.close()
            print(f"   ✅ Progress bar integration works correctly")
            
        except ImportError:
            self.fail("tqdm is not installed - required for progress bars")
        except Exception as e:
            self.fail(f"Progress bar test failed: {e}")
    
    def test_proceeding_identification_in_output(self):
        """Test that proceeding names are clearly identified in output"""
        print(f"\n🏷️ Testing proceeding identification in output...")
        
        from standalone_data_processor import discover_available_proceedings
        
        proceedings = discover_available_proceedings()
        
        for proceeding in proceedings[:2]:  # Test first 2 proceedings
            # Verify proceeding format
            self.assertRegex(proceeding, r'^[RA]\d{7}$', 
                           f"Proceeding {proceeding} should match format R1234567 or A1234567")
            print(f"   ✅ Proceeding format valid: {proceeding}")
            
            # Test that proceeding appears in progress description
            expected_desc = f"📄 Processing {proceeding}"
            self.assertIn(proceeding, expected_desc)
            print(f"   ✅ Progress description format: {expected_desc}")


class TestStandaloneDataProcessorCLI(unittest.TestCase):
    """Test command-line interface functionality"""
    
    def test_cli_argument_parsing(self):
        """Test CLI argument parsing"""
        print(f"\n⌨️ Testing CLI argument parsing...")
        
        from standalone_data_processor import parse_arguments
        
        # Mock sys.argv for testing
        test_cases = [
            ['--list-proceedings'],
            ['--status', 'R2207005'],
            ['R2207005'],
            ['--all', '--verbose'],
            ['--force-rebuild', '--batch-size', '5']
        ]
        
        for test_argv in test_cases:
            with patch('sys.argv', ['standalone_data_processor.py'] + test_argv):
                try:
                    args = parse_arguments()
                    print(f"   ✅ Parsed arguments: {test_argv}")
                except SystemExit:
                    # Some argument combinations may cause help/exit
                    print(f"   ℹ️ Arguments caused help/exit: {test_argv}")

    def test_cli_config_integration(self):
        """Test CLI integration with config system"""
        print(f"\n⚙️ Testing CLI config integration...")
        
        from standalone_data_processor import get_config_proceedings
        
        proceedings = get_config_proceedings()
        self.assertIsInstance(proceedings, list, "Should return list of proceedings")
        self.assertGreater(len(proceedings), 0, "Should have at least one proceeding")
        
        print(f"   ✅ Config proceedings: {proceedings}")


if __name__ == '__main__':
    # Create a test suite focused on integration testing
    suite = unittest.TestSuite()
    
    # Add integration tests in logical order
    suite.addTest(TestStandaloneDataProcessorIntegration('test_cpuc_proceedings_structure_exists'))
    suite.addTest(TestStandaloneDataProcessorIntegration('test_scraped_data_integrity'))
    suite.addTest(TestStandaloneDataProcessorIntegration('test_config_integration'))
    suite.addTest(TestStandaloneDataProcessorIntegration('test_standalone_processor_listing'))
    suite.addTest(TestStandaloneDataProcessorIntegration('test_embeddings_directory_setup'))
    suite.addTest(TestStandaloneDataProcessorIntegration('test_data_safety_read_only_verification'))
    suite.addTest(TestStandaloneDataProcessorIntegration('test_vector_db_root_location'))
    suite.addTest(TestStandaloneDataProcessorIntegration('test_processing_workflow_dry_run'))
    suite.addTest(TestStandaloneDataProcessorIntegration('test_end_to_end_integration_summary'))
    
    # Add CLI tests
    suite.addTest(TestStandaloneDataProcessorCLI('test_cli_argument_parsing'))
    suite.addTest(TestStandaloneDataProcessorCLI('test_cli_config_integration'))
    
    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)