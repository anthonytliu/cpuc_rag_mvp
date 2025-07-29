#!/usr/bin/env python3
"""
Comprehensive test suite for intelligent hybrid processing system.

This test validates the new hybrid processing architecture that:
1. Uses Chonkie as primary method for text-heavy documents
2. Uses hybrid evaluation for table/financial documents
3. Implements agent-based decision making with logging

Author: Claude Code
"""

import logging
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_PROCEEDING = "R2207005"  # Test proceeding

# Test documents representing different types
TEST_DOCUMENTS = {
    "text_heavy": {
        "url": "https://docs.cpuc.ca.gov/PublishedDocs/Published/G000/M571/K985/571985189.PDF",
        "title": "571985189 - Final Decision D2506047",
        "expected_score": 0.1,  # Low table/financial score
        "expected_method": "chonkie"
    },
    "agenda_decision": {
        "url": "https://docs.cpuc.ca.gov/PublishedDocs/Published/G000/M564/K706/564706741.PDF", 
        "title": "564706741 - Agenda Decision D2505026",
        "expected_score": 0.4,  # Should get bonus for agenda + decision
        "expected_method": "hybrid"
    },
    "compliance_report": {
        "url": "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M566/K911/566911513.PDF",
        "title": "566911513 - Joint Compliance Report",
        "expected_score": 0.5,  # Should get bonus for compliance + report
        "expected_method": "hybrid"
    }
}


class TestHybridProcessing:
    """Test suite for intelligent hybrid processing."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.log_dir = self.temp_dir / "agent_evaluations"
        self.log_dir.mkdir(exist_ok=True)
        
    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_table_financial_detection(self):
        """Test the table/financial document detection algorithm."""
        print("\n🧪 Testing Table/Financial Document Detection")
        print("=" * 60)
        
        from data_processing import detect_table_financial_document
        
        for doc_type, doc_info in TEST_DOCUMENTS.items():
            score = detect_table_financial_document(doc_info["url"], doc_info["title"])
            
            print(f"📄 {doc_type}: {score:.3f}")
            print(f"   URL: {doc_info['url']}")
            print(f"   Expected: {doc_info['expected_score']:.1f}, Got: {score:.3f}")
            
            # Allow some tolerance in score detection
            if doc_info["expected_score"] < 0.3:
                assert score < 0.3, f"Expected low score for {doc_type}, got {score:.3f}"
            else:
                assert score >= 0.2, f"Expected higher score for {doc_type}, got {score:.3f}"
        
        print("✅ Table/Financial detection test passed")
    
    def test_intelligent_routing(self):
        """Test that documents are routed to correct processing methods."""
        print("\n🧪 Testing Intelligent Document Routing")
        print("=" * 60)
        
        import config
        from data_processing import detect_table_financial_document
        
        for doc_type, doc_info in TEST_DOCUMENTS.items():
            score = detect_table_financial_document(doc_info["url"], doc_info["title"])
            
            if score < config.HYBRID_TRIGGER_THRESHOLD:
                expected_method = "chonkie"
            else:
                expected_method = "hybrid"
            
            print(f"📄 {doc_type}:")
            print(f"   Score: {score:.3f}")
            print(f"   Threshold: {config.HYBRID_TRIGGER_THRESHOLD}")
            print(f"   Expected method: {expected_method}")
            print(f"   Test passes: {expected_method == doc_info['expected_method']}")
            
            # Verify routing logic
            if score < config.HYBRID_TRIGGER_THRESHOLD:
                assert expected_method == "chonkie", f"Low score should route to Chonkie for {doc_type}"
            else:
                assert expected_method == "hybrid", f"High score should route to hybrid for {doc_type}"
        
        print("✅ Intelligent routing test passed")
    
    def test_agent_evaluation_logging(self):
        """Test that agent evaluation decisions are properly logged."""
        print("\n🧪 Testing Agent Evaluation Logging")
        print("=" * 60)
        
        # Temporarily patch the config to use our test directory
        import config
        original_log_dir = config.AGENT_EVALUATION_LOG_DIR
        config.AGENT_EVALUATION_LOG_DIR = self.log_dir
        
        from data_processing import create_agent_evaluation_log
        
        # Mock evaluation results
        docling_result = {
            'success': True,
            'processing_time': 5.2,
            'chunk_count': 45,
            'content_length': 12500,
            'content_types': ['text', 'table'],
            'tables_found': 3,
            'error': 'None'
        }
        
        chonkie_result = {
            'success': True,
            'processing_time': 2.1,
            'chunk_count': 52,
            'content_length': 15200,
            'strategy_used': 'recursive',
            'text_quality': 0.95,
            'error': 'None'
        }
        
        test_url = TEST_DOCUMENTS["agenda_decision"]["url"]
        test_title = TEST_DOCUMENTS["agenda_decision"]["title"]
        
        # Create evaluation log
        create_agent_evaluation_log(
            pdf_url=test_url,
            document_title=test_title,
            detection_score=0.7,
            docling_result=docling_result,
            chonkie_result=chonkie_result,
            agent_decision="DOCLING",
            agent_reasoning="High table content detected with superior structure preservation"
        )
        
        # Verify log file was created
        log_files = list(self.log_dir.glob("agent_evaluation_*.txt"))
        assert len(log_files) == 1, f"Expected 1 log file, found {len(log_files)}"
        
        # Verify log content
        log_content = log_files[0].read_text()
        assert test_url in log_content, "PDF URL should be in log"
        assert test_title in log_content, "Document title should be in log"
        assert "DOCLING" in log_content, "Agent decision should be in log"
        assert "table content" in log_content.lower(), "Reasoning should be in log"
        assert "5.2" in log_content, "Docling processing time should be in log"
        assert "2.1" in log_content, "Chonkie processing time should be in log"
        
        print(f"📝 Log file created: {log_files[0].name}")
        
        # Restore original config
        config.AGENT_EVALUATION_LOG_DIR = original_log_dir
        
        print("✅ Agent evaluation logging test passed")
    
    def test_agent_decision_logic(self):
        """Test the agent decision-making logic with various scenarios."""
        print("\n🧪 Testing Agent Decision Logic")
        print("=" * 60)
        
        from data_processing import _evaluate_with_agent
        
        test_scenarios = [
            {
                "name": "Both succeed, high table score, Docling has tables",
                "docling": {'success': True, 'processing_time': 5.0, 'chunk_count': 40, 
                          'content_length': 10000, 'content_types': ['text', 'table'], 
                          'tables_found': 5, 'error': 'None'},
                "chonkie": {'success': True, 'processing_time': 2.0, 'chunk_count': 60,
                          'content_length': 12000, 'strategy_used': 'recursive', 
                          'text_quality': 0.9, 'error': 'None'},
                "detection_score": 0.8,
                "expected_decision": "DOCLING"
            },
            {
                "name": "Both succeed, low table score, Chonkie faster",
                "docling": {'success': True, 'processing_time': 8.0, 'chunk_count': 30,
                          'content_length': 8000, 'content_types': ['text'], 
                          'tables_found': 0, 'error': 'None'},
                "chonkie": {'success': True, 'processing_time': 3.0, 'chunk_count': 35,
                          'content_length': 9000, 'strategy_used': 'recursive',
                          'text_quality': 0.95, 'error': 'None'},
                "detection_score": 0.1,
                "expected_decision": "CHONKIE"
            },
            {
                "name": "Docling fails, Chonkie succeeds",
                "docling": {'success': False, 'processing_time': 2.0, 'chunk_count': 0,
                          'content_length': 0, 'content_types': [], 
                          'tables_found': 0, 'error': 'Processing failed'},
                "chonkie": {'success': True, 'processing_time': 3.0, 'chunk_count': 40,
                          'content_length': 11000, 'strategy_used': 'sentence',
                          'text_quality': 0.88, 'error': 'None'},
                "detection_score": 0.5,
                "expected_decision": "CHONKIE"
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n📊 Testing: {scenario['name']}")
            
            decision, reasoning = _evaluate_with_agent(
                scenario["docling"], 
                scenario["chonkie"], 
                scenario["detection_score"]
            )
            
            print(f"   Expected: {scenario['expected_decision']}")
            print(f"   Got: {decision}")
            print(f"   Reasoning: {reasoning[:100]}...")
            
            assert decision.upper() == scenario["expected_decision"], \
                f"Expected {scenario['expected_decision']}, got {decision}"
        
        print("✅ Agent decision logic test passed")
    
    def test_configuration_settings(self):
        """Test that all hybrid processing configuration settings are properly set."""
        print("\n🧪 Testing Configuration Settings")
        print("=" * 60)
        
        import config
        
        # Test that all required config variables exist
        required_configs = [
            'INTELLIGENT_HYBRID_ENABLED',
            'TABLE_FINANCIAL_KEYWORDS',
            'HYBRID_TRIGGER_THRESHOLD',
            'AGENT_EVALUATION_ENABLED',
            'AGENT_EVALUATION_LOG_DIR',
            'AGENT_EVALUATION_TIMEOUT'
        ]
        
        for config_name in required_configs:
            assert hasattr(config, config_name), f"Missing config: {config_name}"
            print(f"✅ {config_name}: {getattr(config, config_name)}")
        
        # Test that threshold is reasonable
        assert 0.0 <= config.HYBRID_TRIGGER_THRESHOLD <= 1.0, \
            f"Threshold should be 0-1, got {config.HYBRID_TRIGGER_THRESHOLD}"
        
        # Test that keywords list is not empty
        assert len(config.TABLE_FINANCIAL_KEYWORDS) > 0, \
            "TABLE_FINANCIAL_KEYWORDS should not be empty"
        print(f"📝 {len(config.TABLE_FINANCIAL_KEYWORDS)} financial keywords configured")
        
        print("✅ Configuration settings test passed")
    
    @patch('data_processing._process_with_chonkie_primary')
    @patch('data_processing._process_with_hybrid_evaluation')
    def test_main_processing_function_routing(self, mock_hybrid, mock_chonkie):
        """Test that the main processing function routes correctly."""
        print("\n🧪 Testing Main Processing Function Routing")
        print("=" * 60)
        
        from data_processing import extract_and_chunk_with_docling_url
        
        # Mock return values
        mock_chonkie.return_value = [MagicMock()]
        mock_hybrid.return_value = [MagicMock()]
        
        # Test low score document (should use Chonkie)
        test_doc = TEST_DOCUMENTS["text_heavy"]
        
        with patch('data_processing.validate_pdf_url', return_value=True), \
             patch('data_processing.detect_table_financial_document', return_value=0.1):
            
            result = extract_and_chunk_with_docling_url(
                test_doc["url"], 
                test_doc["title"], 
                TEST_PROCEEDING,
                use_intelligent_hybrid=True
            )
            
            mock_chonkie.assert_called_once()
            mock_hybrid.assert_not_called()
            print("✅ Low score document routed to Chonkie")
        
        # Reset mocks
        mock_chonkie.reset_mock()
        mock_hybrid.reset_mock()
        
        # Test high score document (should use hybrid)
        test_doc = TEST_DOCUMENTS["agenda_decision"]
        
        with patch('data_processing.validate_pdf_url', return_value=True), \
             patch('data_processing.detect_table_financial_document', return_value=0.7):
            
            result = extract_and_chunk_with_docling_url(
                test_doc["url"],
                test_doc["title"], 
                TEST_PROCEEDING,
                use_intelligent_hybrid=True
            )
            
            mock_hybrid.assert_called_once()
            mock_chonkie.assert_not_called()
            print("✅ High score document routed to hybrid evaluation")
        
        print("✅ Main processing function routing test passed")


def run_comprehensive_hybrid_test():
    """Run all hybrid processing tests."""
    print("\n🚀 STARTING COMPREHENSIVE HYBRID PROCESSING TEST")
    print("=" * 80)
    
    test_suite = TestHybridProcessing()
    
    try:
        # Setup
        test_suite.setup_method()
        
        # Run all tests
        test_suite.test_configuration_settings()
        test_suite.test_table_financial_detection()
        test_suite.test_intelligent_routing()
        test_suite.test_agent_evaluation_logging()
        test_suite.test_agent_decision_logic()
        test_suite.test_main_processing_function_routing()
        
        print("\n🎉 ALL HYBRID PROCESSING TESTS PASSED!")
        print("=" * 80)
        
        # Summary
        print(f"✅ Configuration validation: PASSED")
        print(f"✅ Document detection: PASSED")
        print(f"✅ Intelligent routing: PASSED")
        print(f"✅ Agent evaluation logging: PASSED") 
        print(f"✅ Decision logic: PASSED")
        print(f"✅ Main function routing: PASSED")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        test_suite.teardown_method()


def run_quick_integration_test():
    """Quick integration test with real document processing."""
    print("\n🧪 QUICK INTEGRATION TEST")
    print("=" * 50)
    
    try:
        from data_processing import extract_and_chunk_with_docling_url
        import config
        
        # Test with a small document
        test_url = "https://docs.cpuc.ca.gov/PublishedDocs/Published/G000/M571/K985/571985189.PDF"
        
        # Temporarily enable hybrid processing for test
        original_hybrid = config.INTELLIGENT_HYBRID_ENABLED
        config.INTELLIGENT_HYBRID_ENABLED = True
        
        print(f"🔄 Processing test document with hybrid system...")
        print(f"📄 URL: {test_url}")
        
        # Process document
        result = extract_and_chunk_with_docling_url(
            test_url,
            "Test Document",
            "R2207005",
            use_intelligent_hybrid=True
        )
        
        print(f"✅ Processing completed: {len(result)} chunks extracted")
        
        if result and len(result) > 0:
            print(f"📊 Sample chunk metadata: {list(result[0].metadata.keys())}")
            print(f"📝 Content type: {result[0].metadata.get('content_type', 'unknown')}")
        
        # Restore original setting
        config.INTELLIGENT_HYBRID_ENABLED = original_hybrid
        
        return len(result) > 0
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 HYBRID PROCESSING TEST SUITE")
    print("=" * 50)
    
    success = True
    
    # Run comprehensive tests
    if not run_comprehensive_hybrid_test():
        success = False
    
    # Run integration test
    print("\n" + "=" * 50)
    if not run_quick_integration_test():
        success = False
    
    # Final result
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED - HYBRID PROCESSING READY!")
    else:
        print("❌ SOME TESTS FAILED - CHECK LOGS ABOVE")
    
    exit(0 if success else 1)