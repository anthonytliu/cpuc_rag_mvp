#!/usr/bin/env python3
"""
QA Testing Agent for CPUC RAG System
Tests R2207005 proceeding with comprehensive evaluation
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

# Import CPUC RAG System
from rag_core import CPUCRAGSystem
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CPUCRAGTestAgent:
    """QA Testing Agent for CPUC RAG System"""
    
    def __init__(self, proceeding: str = "R2207005"):
        self.proceeding = proceeding
        self.rag_system = None
        self.test_results = []
        
    def initialize_rag_system(self) -> bool:
        """Initialize the RAG system for testing"""
        try:
            logger.info(f"Initializing RAG system for proceeding {self.proceeding}...")
            self.rag_system = CPUCRAGSystem(current_proceeding=self.proceeding)
            
            # Check if vector database exists
            if not self.rag_system.vectordb:
                logger.error("Vector database not initialized")
                return False
                
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def test_basic_query(self) -> Dict[str, Any]:
        """Test basic RAG functionality with a simple question"""
        test_name = "Basic Query Test"
        question = "What is the purpose of proceeding R.22-07-005?"
        
        logger.info(f"Running {test_name}: {question}")
        
        try:
            start_time = time.time()
            result = None
            
            # Collect streaming responses
            for response in self.rag_system.query(question):
                if isinstance(response, dict):
                    result = response
                    break
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if result:
                test_result = {
                    "test_name": test_name,
                    "question": question,
                    "success": True,
                    "response_time": response_time,
                    "answer_length": len(result.get("answer", "")),
                    "num_sources": len(result.get("sources", [])),
                    "confidence_score": result.get("confidence_indicators", {}).get("overall_score", 0),
                    "answer_preview": result.get("answer", "")[:200] + "..." if len(result.get("answer", "")) > 200 else result.get("answer", ""),
                    "sources_preview": [source.get("url", "") for source in result.get("sources", [])[:3]]
                }
            else:
                test_result = {
                    "test_name": test_name,
                    "question": question,
                    "success": False,
                    "error": "No result returned",
                    "response_time": response_time
                }
                
        except Exception as e:
            test_result = {
                "test_name": test_name,
                "question": question,
                "success": False,
                "error": str(e),
                "response_time": 0
            }
        
        self.test_results.append(test_result)
        return test_result
    
    def test_demand_flexibility_query(self) -> Dict[str, Any]:
        """Test technical question about demand flexibility"""
        test_name = "Demand Flexibility Query"
        question = "What are the key features of demand flexibility programs discussed in R.22-07-005?"
        
        logger.info(f"Running {test_name}: {question}")
        
        try:
            start_time = time.time()
            result = None
            
            for response in self.rag_system.query(question):
                if isinstance(response, dict):
                    result = response
                    break
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if result:
                # Check for technical accuracy indicators
                answer = result.get("answer", "").lower()
                technical_terms_found = sum([
                    1 for term in ["demand flexibility", "rate design", "time-of-use", "peak demand", "grid reliability"]
                    if term in answer
                ])
                
                test_result = {
                    "test_name": test_name,
                    "question": question,
                    "success": True,
                    "response_time": response_time,
                    "answer_length": len(result.get("answer", "")),
                    "num_sources": len(result.get("sources", [])),
                    "technical_terms_found": technical_terms_found,
                    "confidence_score": result.get("confidence_indicators", {}).get("overall_score", 0),
                    "answer_preview": result.get("answer", "")[:300] + "..." if len(result.get("answer", "")) > 300 else result.get("answer", ""),
                    "sources_preview": [source.get("url", "") for source in result.get("sources", [])[:3]]
                }
            else:
                test_result = {
                    "test_name": test_name,
                    "question": question,
                    "success": False,
                    "error": "No result returned",
                    "response_time": response_time
                }
                
        except Exception as e:
            test_result = {
                "test_name": test_name,
                "question": question,
                "success": False,
                "error": str(e),
                "response_time": 0
            }
        
        self.test_results.append(test_result)
        return test_result
    
    def test_fixed_charge_query(self) -> Dict[str, Any]:
        """Test question about residential fixed charges"""
        test_name = "Fixed Charge Query"
        question = "What are the proposed residential fixed charges for different income levels in this proceeding?"
        
        logger.info(f"Running {test_name}: {question}")
        
        try:
            start_time = time.time()
            result = None
            
            for response in self.rag_system.query(question):
                if isinstance(response, dict):
                    result = response
                    break
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if result:
                # Check for fixed charge related content
                answer = result.get("answer", "").lower()
                fixed_charge_terms = sum([
                    1 for term in ["fixed charge", "income graduated", "residential", "monthly charge", "assembly bill 205"]
                    if term in answer
                ])
                
                test_result = {
                    "test_name": test_name,
                    "question": question,
                    "success": True,
                    "response_time": response_time,
                    "answer_length": len(result.get("answer", "")),
                    "num_sources": len(result.get("sources", [])),
                    "fixed_charge_terms": fixed_charge_terms,
                    "confidence_score": result.get("confidence_indicators", {}).get("overall_score", 0),
                    "answer_preview": result.get("answer", "")[:300] + "..." if len(result.get("answer", "")) > 300 else result.get("answer", ""),
                    "sources_preview": [source.get("url", "") for source in result.get("sources", [])[:3]]
                }
            else:
                test_result = {
                    "test_name": test_name,
                    "question": question,
                    "success": False,
                    "error": "No result returned",
                    "response_time": response_time
                }
                
        except Exception as e:
            test_result = {
                "test_name": test_name,
                "question": question,
                "success": False,
                "error": str(e),
                "response_time": 0
            }
        
        self.test_results.append(test_result)
        return test_result
    
    def test_source_citations(self) -> Dict[str, Any]:
        """Test source citation functionality"""
        test_name = "Source Citation Test"
        question = "What utilities are involved in this proceeding?"
        
        logger.info(f"Running {test_name}: {question}")
        
        try:
            start_time = time.time()
            result = None
            
            for response in self.rag_system.query(question):
                if isinstance(response, dict):
                    result = response
                    break
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if result:
                sources = result.get("sources", [])
                valid_citations = 0
                cpuc_urls = 0
                
                for source in sources:
                    if source.get("url", "").startswith("https://docs.cpuc.ca.gov"):
                        cpuc_urls += 1
                    if source.get("title") and source.get("url"):
                        valid_citations += 1
                
                test_result = {
                    "test_name": test_name,
                    "question": question,
                    "success": True,
                    "response_time": response_time,
                    "num_sources": len(sources),
                    "valid_citations": valid_citations,
                    "cpuc_urls": cpuc_urls,
                    "citation_quality": valid_citations / len(sources) if sources else 0,
                    "sources_sample": sources[:3] if sources else []
                }
            else:
                test_result = {
                    "test_name": test_name,
                    "question": question,
                    "success": False,
                    "error": "No result returned",
                    "response_time": response_time
                }
                
        except Exception as e:
            test_result = {
                "test_name": test_name,
                "question": question,
                "success": False,
                "error": str(e),
                "response_time": 0
            }
        
        self.test_results.append(test_result)
        return test_result
    
    def evaluate_lancedb_performance(self) -> Dict[str, Any]:
        """Evaluate LanceDB integration performance"""
        test_name = "LanceDB Performance Test"
        
        logger.info(f"Running {test_name}")
        
        try:
            # Test vector database connectivity
            if not self.rag_system.vectordb:
                return {
                    "test_name": test_name,
                    "success": False,
                    "error": "Vector database not initialized"
                }
            
            # Test retrieval performance with a simple query
            start_time = time.time()
            retriever = self.rag_system.vectordb.as_retriever(search_kwargs={"k": 10})
            docs = retriever.get_relevant_documents("demand flexibility")
            retrieval_time = time.time() - start_time
            
            # Check document quality
            valid_docs = sum([1 for doc in docs if doc.page_content and len(doc.page_content) > 50])
            
            test_result = {
                "test_name": test_name,
                "success": True,
                "retrieval_time": retrieval_time,
                "documents_retrieved": len(docs),
                "valid_documents": valid_docs,
                "document_quality": valid_docs / len(docs) if docs else 0,
                "average_doc_length": sum([len(doc.page_content) for doc in docs]) / len(docs) if docs else 0
            }
            
        except Exception as e:
            test_result = {
                "test_name": test_name,
                "success": False,
                "error": str(e)
            }
        
        self.test_results.append(test_result)
        return test_result
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run all tests and compile comprehensive report"""
        logger.info("Starting comprehensive CPUC RAG system test suite...")
        
        # Initialize system
        if not self.initialize_rag_system():
            return {
                "overall_success": False,
                "error": "Failed to initialize RAG system",
                "tests_run": 0
            }
        
        # Run all tests
        tests = [
            self.test_basic_query,
            self.test_demand_flexibility_query,
            self.test_fixed_charge_query,
            self.test_source_citations,
            self.evaluate_lancedb_performance
        ]
        
        successful_tests = 0
        for test_func in tests:
            try:
                result = test_func()
                if result.get("success", False):
                    successful_tests += 1
                logger.info(f"Completed {result.get('test_name', 'Unknown Test')}: {'✓' if result.get('success') else '✗'}")
            except Exception as e:
                logger.error(f"Test failed with exception: {e}")
        
        # Compile overall results
        overall_results = {
            "proceeding": self.proceeding,
            "total_tests": len(tests),
            "successful_tests": successful_tests,
            "success_rate": successful_tests / len(tests),
            "overall_success": successful_tests >= len(tests) * 0.8,  # 80% success threshold
            "detailed_results": self.test_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return overall_results
    
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        report = f"""
# CPUC RAG System - QA Test Report
## Proceeding: {results['proceeding']}
## Timestamp: {results['timestamp']}

### Summary
- **Total Tests**: {results['total_tests']}
- **Successful Tests**: {results['successful_tests']}
- **Success Rate**: {results['success_rate']:.2%}
- **Overall Status**: {'✓ PASS' if results['overall_success'] else '✗ FAIL'}

### Detailed Test Results

"""
        
        for test in results['detailed_results']:
            report += f"#### {test['test_name']}\n"
            report += f"- **Status**: {'✓ PASS' if test.get('success') else '✗ FAIL'}\n"
            
            if test.get('success'):
                report += f"- **Response Time**: {test.get('response_time', 0):.2f}s\n"
                
                if 'answer_length' in test:
                    report += f"- **Answer Length**: {test['answer_length']} characters\n"
                if 'num_sources' in test:
                    report += f"- **Sources Found**: {test['num_sources']}\n"
                if 'confidence_score' in test:
                    report += f"- **Confidence Score**: {test['confidence_score']:.2f}\n"
                if 'answer_preview' in test:
                    report += f"- **Answer Preview**: {test['answer_preview']}\n"
                if 'sources_preview' in test and test['sources_preview']:
                    report += f"- **Source URLs**: {', '.join(test['sources_preview'])}\n"
            else:
                report += f"- **Error**: {test.get('error', 'Unknown error')}\n"
            
            report += "\n"
        
        return report

def main():
    """Main test execution"""
    test_agent = CPUCRAGTestAgent("R2207005")
    
    print("🔍 CPUC RAG System QA Testing Agent")
    print("=" * 50)
    
    # Run comprehensive test suite
    results = test_agent.run_comprehensive_test_suite()
    
    # Generate and display report
    report = test_agent.generate_test_report(results)
    print(report)
    
    # Save report to file
    report_file = project_root / "cpuc_rag_test_report.md"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    exit_code = 0 if results['overall_success'] else 1
    print(f"\n🎯 Test Suite {'PASSED' if results['overall_success'] else 'FAILED'}")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()