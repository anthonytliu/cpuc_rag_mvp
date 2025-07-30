#!/usr/bin/env python3
"""
Comprehensive Citation Accuracy Test Suite for Proceeding R2207005

This test suite validates that citations in RAG responses map correctly to actual PDF content,
detects hallucinated citations, and provides detailed accuracy metrics.

Features:
- Citation extraction and parsing from RAG responses
- PDF content validation against citations
- Citation accuracy metrics and reporting
- Automated citation checker with debugging
- Test data generation for different query types
- Integration with existing LanceDB vector store
"""

import json
import logging
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Set
from urllib.parse import urlparse

import lancedb
import pandas as pd
import requests
from langchain.docstore.document import Document

# Add project root to path
sys.path.append(str(Path(__file__).parent.resolve()))

import config
from rag_core import CPUCRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Represents a single citation extracted from text."""
    filename: str
    page: int
    line: Optional[int] = None
    source_url: Optional[str] = None
    raw_text: str = ""
    context: str = ""


@dataclass
class CitationValidationResult:
    """Results of validating a citation against actual PDF content."""
    citation: Citation
    is_valid: bool
    pdf_accessible: bool
    content_matches: bool
    match_score: float
    pdf_text_sample: str
    error_message: Optional[str] = None
    validation_time: float = 0.0


@dataclass
class CitationAccuracyMetrics:
    """Citation accuracy metrics for a set of responses."""
    total_responses: int
    responses_with_citations: int
    total_citations: int
    valid_citations: int
    accessible_pdfs: int
    content_matches: int
    
    # Calculated metrics
    citation_coverage: float = 0.0      # % of responses with citations
    citation_accuracy: float = 0.0      # % of citations that are valid
    citation_precision: float = 0.0     # % of citations with matching content
    false_citation_rate: float = 0.0    # % of citations that are incorrect
    
    def calculate_metrics(self):
        """Calculate derived metrics."""
        self.citation_coverage = (self.responses_with_citations / self.total_responses * 100) if self.total_responses > 0 else 0
        self.citation_accuracy = (self.valid_citations / self.total_citations * 100) if self.total_citations > 0 else 0
        self.citation_precision = (self.content_matches / self.total_citations * 100) if self.total_citations > 0 else 0
        self.false_citation_rate = ((self.total_citations - self.valid_citations) / self.total_citations * 100) if self.total_citations > 0 else 0


@dataclass
class TestQuery:
    """Represents a test query with expected characteristics."""
    question: str
    category: str  # 'factual', 'procedural', 'timeline', 'technical'
    expected_doc_types: List[str]
    complexity: str  # 'simple', 'medium', 'complex'
    description: str


class CitationExtractor:
    """Extracts and parses citations from RAG responses."""
    
    def __init__(self):
        # Support multiple citation formats
        self.citation_patterns = [
            # Primary format: [CITE:filename.pdf,page_X,line_Y]
            re.compile(r'\[CITE:\s*([^,]+),\s*page_(\d+),\s*line_(\d+)\]', re.IGNORECASE),
            # Secondary format: [CITE:filename.pdf,page_X]
            re.compile(r'\[CITE:\s*([^,]+),\s*page_(\d+)\]', re.IGNORECASE),
            # Alternative formats that might exist
            re.compile(r'\[CITE:\s*([^,]+)\s*,\s*p\.?\s*(\d+)\]', re.IGNORECASE),
        ]
    
    def extract_citations(self, text: str) -> List[Citation]:
        """Extract all citations from the given text."""
        citations = []
        
        for pattern in self.citation_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                groups = match.groups()
                filename = groups[0].strip()
                page = int(groups[1])
                line = int(groups[2]) if len(groups) > 2 and groups[2] else None
                
                # Extract context around the citation
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                citation = Citation(
                    filename=filename,
                    page=page,
                    line=line,
                    raw_text=match.group(0),
                    context=context
                )
                citations.append(citation)
        
        return citations
    
    def get_citation_statistics(self, text: str) -> Dict[str, Any]:
        """Get statistics about citations in the text."""
        citations = self.extract_citations(text)
        
        return {
            'total_citations': len(citations),
            'unique_documents': len(set(c.filename for c in citations)),
            'page_range': (min(c.page for c in citations), max(c.page for c in citations)) if citations else (0, 0),
            'has_line_numbers': sum(1 for c in citations if c.line is not None),
            'citation_density': len(citations) / len(text.split()) if text else 0
        }


class PDFContentValidator:
    """Validates citations against actual PDF content."""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def validate_citation(self, citation: Citation, vectordb_table) -> CitationValidationResult:
        """Validate a single citation against PDF content."""
        start_time = time.time()
        
        try:
            # Step 1: Find the source URL from vectordb metadata
            citation.source_url = self._find_source_url(citation.filename, vectordb_table)
            
            if not citation.source_url:
                return CitationValidationResult(
                    citation=citation,
                    is_valid=False,
                    pdf_accessible=False,
                    content_matches=False,
                    match_score=0.0,
                    pdf_text_sample="",
                    error_message="Source URL not found in vectordb",
                    validation_time=time.time() - start_time
                )
            
            # Step 2: Try to access the PDF
            pdf_accessible = self._check_pdf_accessibility(citation.source_url)
            
            if not pdf_accessible:
                return CitationValidationResult(
                    citation=citation,
                    is_valid=False,
                    pdf_accessible=False,
                    content_matches=False,
                    match_score=0.0,
                    pdf_text_sample="",
                    error_message="PDF not accessible",
                    validation_time=time.time() - start_time
                )
            
            # Step 3: Get text from vectordb for the specific page
            pdf_text_sample = self._get_page_text_from_vectordb(
                citation.filename, citation.page, vectordb_table
            )
            
            # Step 4: Validate content match
            content_matches, match_score = self._validate_content_match(
                citation, pdf_text_sample
            )
            
            return CitationValidationResult(
                citation=citation,
                is_valid=True,
                pdf_accessible=True,
                content_matches=content_matches,
                match_score=match_score,
                pdf_text_sample=pdf_text_sample[:500],  # First 500 chars
                validation_time=time.time() - start_time
            )
            
        except Exception as e:
            return CitationValidationResult(
                citation=citation,
                is_valid=False,
                pdf_accessible=False,
                content_matches=False,
                match_score=0.0,
                pdf_text_sample="",
                error_message=str(e),
                validation_time=time.time() - start_time
            )
    
    def _find_source_url(self, filename: str, vectordb_table) -> Optional[str]:
        """Find the source URL for a filename from vectordb metadata."""
        try:
            # Clean filename for matching
            clean_filename = filename.replace('.pdf', '').replace('.PDF', '')
            
            # Query vectordb for matching documents
            df = vectordb_table.to_pandas()
            
            # Search in metadata
            for _, row in df.iterrows():
                metadata = row['metadata']
                if isinstance(metadata, dict):
                    source = metadata.get('source', '')
                    source_url = metadata.get('source_url', '')
                    
                    if (clean_filename in source or 
                        source in clean_filename or
                        clean_filename == source):
                        return source_url
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding source URL for {filename}: {e}")
            return None
    
    def _check_pdf_accessibility(self, url: str) -> bool:
        """Check if a PDF URL is accessible."""
        try:
            response = self.session.head(url, timeout=self.timeout)
            return response.status_code == 200
        except Exception:
            return False
    
    def _get_page_text_from_vectordb(self, filename: str, page: int, vectordb_table) -> str:
        """Get text content for a specific page from vectordb."""
        try:
            clean_filename = filename.replace('.pdf', '').replace('.PDF', '')
            
            df = vectordb_table.to_pandas()
            
            # Find chunks from the specific document and page
            page_texts = []
            for _, row in df.iterrows():
                metadata = row['metadata']
                if isinstance(metadata, dict):
                    source = metadata.get('source', '')
                    chunk_page = metadata.get('page', 0)
                    
                    if ((clean_filename in source or source in clean_filename) and 
                        chunk_page == page):
                        page_texts.append(row['text'])
            
            return ' '.join(page_texts)
            
        except Exception as e:
            logger.error(f"Error getting page text for {filename} page {page}: {e}")
            return ""
    
    def _validate_content_match(self, citation: Citation, pdf_text: str) -> Tuple[bool, float]:
        """Validate if citation context matches PDF content."""
        if not pdf_text:
            return False, 0.0
        
        try:
            # Extract key phrases from citation context
            context_words = set(citation.context.lower().split())
            pdf_words = set(pdf_text.lower().split())
            
            # Remove common words and citation markers
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
                         'cite', 'page', 'pdf', 'document', 'section', 'paragraph'}
            
            context_words = context_words - stop_words
            pdf_words = pdf_words - stop_words
            
            if not context_words:
                return False, 0.0
            
            # Calculate similarity score
            intersection = context_words.intersection(pdf_words)
            match_score = len(intersection) / len(context_words) if context_words else 0.0
            
            # Consider it a match if > 30% of words match
            content_matches = match_score > 0.3
            
            return content_matches, match_score
            
        except Exception as e:
            logger.error(f"Error validating content match: {e}")
            return False, 0.0


class TestDataGenerator:
    """Generates comprehensive test queries for different scenarios."""
    
    def __init__(self):
        self.test_queries = [
            # Factual questions
            TestQuery(
                question="What are the main objectives of proceeding R.22-07-005?",
                category="factual",
                expected_doc_types=["DECISION", "RULING"],
                complexity="simple",
                description="Basic factual query about proceeding objectives"
            ),
            TestQuery(
                question="What are the specific rate design requirements for multi-family housing mentioned in the decisions?",
                category="factual", 
                expected_doc_types=["DECISION", "COMPLIANCE FILING"],
                complexity="medium",
                description="Specific factual details about rate requirements"
            ),
            TestQuery(
                question="What compensation was granted to intervenors and for which decisions?",
                category="factual",
                expected_doc_types=["DECISION"],
                complexity="complex",
                description="Complex factual query requiring multiple document synthesis"
            ),
            
            # Procedural questions
            TestQuery(
                question="What is the current procedural status of this rulemaking?",
                category="procedural",
                expected_doc_types=["RULING", "DECISION"],
                complexity="simple",
                description="Basic procedural status query"
            ),
            TestQuery(
                question="What are the comment filing requirements and deadlines for the proposed decision?",
                category="procedural",
                expected_doc_types=["PROPOSED DECISION", "RULING"],
                complexity="medium",
                description="Specific procedural requirements"
            ),
            TestQuery(
                question="What ex parte communication rules apply and how have they been enforced?",
                category="procedural",
                expected_doc_types=["RULING", "EXPARTE"],
                complexity="complex",
                description="Complex procedural rules and enforcement"
            ),
            
            # Timeline-based questions
            TestQuery(
                question="When was the most recent decision issued in this proceeding?",
                category="timeline",
                expected_doc_types=["DECISION"],
                complexity="simple",
                description="Simple timeline query"
            ),
            TestQuery(
                question="What is the timeline for compliance filings related to Decision D.24-05-028?",
                category="timeline",
                expected_doc_types=["DECISION", "COMPLIANCE FILING"],
                complexity="medium",
                description="Specific decision compliance timeline"
            ),
            TestQuery(
                question="How has the statutory deadline been extended and what are the new milestones?",
                category="timeline",
                expected_doc_types=["DECISION", "RULING"],
                complexity="complex",
                description="Complex timeline changes and extensions"
            ),
            
            # Technical questions
            TestQuery(
                question="What are the technical requirements for demand flexibility mentioned in the documents?",
                category="technical",
                expected_doc_types=["DECISION", "COMMENTS"],
                complexity="medium",
                description="Technical requirements analysis"
            ),
            TestQuery(
                question="How do the utilities' joint reports address implementation working group recommendations?",
                category="technical",
                expected_doc_types=["COMPLIANCE FILING", "COMMENTS"],
                complexity="complex",
                description="Technical implementation analysis"
            ),
            
            # Edge cases
            TestQuery(
                question="What specific provisions apply to Bear Valley Electric Service's multi-family housing charges?",
                category="factual",
                expected_doc_types=["COMPLIANCE FILING"],
                complexity="medium",
                description="Edge case: specific utility requirements"
            ),
            TestQuery(
                question="What were the grounds for denying rehearing of Decision 24-05-028?",
                category="factual",
                expected_doc_types=["DECISION"],
                complexity="complex",
                description="Edge case: rehearing denial analysis"
            )
        ]
    
    def get_test_queries(self, category: Optional[str] = None, complexity: Optional[str] = None) -> List[TestQuery]:
        """Get test queries filtered by category and/or complexity."""
        queries = self.test_queries
        
        if category:
            queries = [q for q in queries if q.category == category]
        
        if complexity:
            queries = [q for q in queries if q.complexity == complexity]
        
        return queries
    
    def get_query_categories(self) -> List[str]:
        """Get all available query categories."""
        return list(set(q.category for q in self.test_queries))
    
    def get_query_complexities(self) -> List[str]:
        """Get all available complexity levels."""
        return list(set(q.complexity for q in self.test_queries))


class CitationAccuracyTester:
    """Main class that orchestrates citation accuracy testing."""
    
    def __init__(self, proceeding: str = "R2207005"):
        self.proceeding = proceeding
        self.rag_system = CPUCRAGSystem(current_proceeding=proceeding)
        self.citation_extractor = CitationExtractor()
        self.pdf_validator = PDFContentValidator()
        self.test_generator = TestDataGenerator()
        
        # Initialize LanceDB connection
        self.vectordb_table = None
        self._init_vectordb()
    
    def _init_vectordb(self):
        """Initialize vectordb connection."""
        try:
            db_path = f"local_lance_db/{self.proceeding}"
            db = lancedb.connect(db_path)
            table_name = f"{self.proceeding}_documents"
            
            if table_name in db.table_names():
                self.vectordb_table = db.open_table(table_name)
                logger.info(f"Connected to LanceDB table: {table_name}")
            else:
                logger.error(f"Table {table_name} not found in {db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize vectordb: {e}")
    
    def test_single_query(self, query: str, max_citations: int = 10) -> Dict[str, Any]:
        """Test citation accuracy for a single query."""
        try:
            logger.info(f"Testing query: {query}")
            
            # Get RAG response
            start_time = time.time()
            
            if not hasattr(self.rag_system, 'retriever') or self.rag_system.retriever is None:
                logger.error("RAG system retriever not properly initialized")
                return {
                    'query': query,
                    'error': 'RAG system not properly initialized',
                    'citations': [],
                    'metrics': {}
                }
            
            # Use the RAG system to get response (it returns a generator)
            response_generator = self.rag_system.query(query)
            
            # Collect all generator results
            response_parts = list(response_generator)
            response_time = time.time() - start_time
            
            # The last part should be the result payload
            if response_parts and isinstance(response_parts[-1], dict):
                result_payload = response_parts[-1]
                response = result_payload.get("answer", "")
                raw_response = result_payload.get("raw_part1_answer", "")
            else:
                response = ""
                raw_response = ""
            
            logger.info(f"Got RAG response in {response_time:.2f}s")
            
            # Extract citations from both the final answer and raw answer
            citations = self.citation_extractor.extract_citations(response)
            if not citations and raw_response:
                citations = self.citation_extractor.extract_citations(raw_response)
            logger.info(f"Extracted {len(citations)} citations")
            
            # Limit citations for testing
            citations = citations[:max_citations]
            
            # Validate each citation
            validation_results = []
            if self.vectordb_table is not None:
                for citation in citations:
                    result = self.pdf_validator.validate_citation(citation, self.vectordb_table)
                    validation_results.append(result)
                    logger.debug(f"Validated citation: {citation.filename} page {citation.page} -> {result.is_valid}")
            else:
                logger.warning("Vectordb table not available, skipping citation validation")
            
            # Calculate metrics for this query
            metrics = self._calculate_query_metrics(response, citations, validation_results)
            
            return {
                'query': query,
                'response': response,
                'response_time': response_time,
                'citations': [asdict(c) for c in citations],
                'validation_results': [asdict(r) for r in validation_results],
                'metrics': metrics,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error testing query '{query}': {e}")
            logger.error(traceback.format_exc())
            return {
                'query': query,
                'error': str(e),
                'citations': [],
                'validation_results': [],
                'metrics': {},
                'response_time': 0
            }
    
    def run_comprehensive_test(self, categories: Optional[List[str]] = None, 
                             complexities: Optional[List[str]] = None,
                             max_queries: Optional[int] = None) -> Dict[str, Any]:
        """Run comprehensive citation accuracy tests."""
        logger.info("Starting comprehensive citation accuracy test")
        
        # Get test queries
        test_queries = []
        if categories or complexities:
            for category in (categories or self.test_generator.get_query_categories()):
                for complexity in (complexities or self.test_generator.get_query_complexities()):
                    test_queries.extend(self.test_generator.get_test_queries(category, complexity))
        else:
            test_queries = self.test_generator.get_test_queries()
        
        # Remove duplicates
        seen_questions = set()
        unique_queries = []
        for q in test_queries:
            if q.question not in seen_questions:
                unique_queries.append(q)
                seen_questions.add(q.question)
        
        test_queries = unique_queries
        
        if max_queries:
            test_queries = test_queries[:max_queries]
        
        logger.info(f"Testing {len(test_queries)} queries")
        
        # Run tests
        test_results = []
        overall_start_time = time.time()
        
        for i, test_query in enumerate(test_queries, 1):
            logger.info(f"Running test {i}/{len(test_queries)}: {test_query.category} - {test_query.complexity}")
            
            result = self.test_single_query(test_query.question)
            result['test_query_info'] = asdict(test_query)
            test_results.append(result)
            
            # Brief pause between queries
            time.sleep(1)
        
        total_test_time = time.time() - overall_start_time
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(test_results)
        
        # Generate detailed report
        report = self._generate_detailed_report(test_results, overall_metrics, total_test_time)
        
        return {
            'test_results': test_results,
            'overall_metrics': overall_metrics,
            'report': report,
            'total_test_time': total_test_time,
            'total_queries': len(test_queries)
        }
    
    def _calculate_query_metrics(self, response: str, citations: List[Citation], 
                               validation_results: List[CitationValidationResult]) -> Dict[str, Any]:
        """Calculate metrics for a single query."""
        has_citations = len(citations) > 0
        
        if not validation_results:
            return {
                'has_citations': has_citations,
                'citation_count': len(citations),
                'validation_available': False
            }
        
        valid_citations = sum(1 for r in validation_results if r.is_valid)
        accessible_pdfs = sum(1 for r in validation_results if r.pdf_accessible)
        content_matches = sum(1 for r in validation_results if r.content_matches)
        
        avg_match_score = sum(r.match_score for r in validation_results) / len(validation_results) if validation_results else 0
        avg_validation_time = sum(r.validation_time for r in validation_results) / len(validation_results) if validation_results else 0
        
        return {
            'has_citations': has_citations,
            'citation_count': len(citations),
            'valid_citations': valid_citations,
            'accessible_pdfs': accessible_pdfs,
            'content_matches': content_matches,
            'validation_available': True,
            'accuracy_rate': (valid_citations / len(citations) * 100) if citations else 0,
            'precision_rate': (content_matches / len(citations) * 100) if citations else 0,
            'avg_match_score': avg_match_score,
            'avg_validation_time': avg_validation_time
        }
    
    def _calculate_overall_metrics(self, test_results: List[Dict[str, Any]]) -> CitationAccuracyMetrics:
        """Calculate overall metrics across all test results."""
        total_responses = len(test_results)
        responses_with_citations = sum(1 for r in test_results if r.get('metrics', {}).get('has_citations', False))
        total_citations = sum(r.get('metrics', {}).get('citation_count', 0) for r in test_results)
        valid_citations = sum(r.get('metrics', {}).get('valid_citations', 0) for r in test_results)
        accessible_pdfs = sum(r.get('metrics', {}).get('accessible_pdfs', 0) for r in test_results)
        content_matches = sum(r.get('metrics', {}).get('content_matches', 0) for r in test_results)
        
        metrics = CitationAccuracyMetrics(
            total_responses=total_responses,
            responses_with_citations=responses_with_citations,
            total_citations=total_citations,
            valid_citations=valid_citations,
            accessible_pdfs=accessible_pdfs,
            content_matches=content_matches
        )
        
        metrics.calculate_metrics()
        return metrics
    
    def _generate_detailed_report(self, test_results: List[Dict[str, Any]], 
                                overall_metrics: CitationAccuracyMetrics,
                                total_test_time: float) -> str:
        """Generate a detailed test report."""
        report_lines = [
            "=" * 80,
            "CITATION ACCURACY TEST REPORT - PROCEEDING R2207005",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Test Time: {total_test_time:.2f} seconds",
            "",
            "OVERALL METRICS:",
            "-" * 40,
            f"Total Responses: {overall_metrics.total_responses}",
            f"Responses with Citations: {overall_metrics.responses_with_citations}",
            f"Total Citations: {overall_metrics.total_citations}",
            f"Valid Citations: {overall_metrics.valid_citations}",
            f"Content Matches: {overall_metrics.content_matches}",
            "",
            f"Citation Coverage: {overall_metrics.citation_coverage:.1f}%",
            f"Citation Accuracy: {overall_metrics.citation_accuracy:.1f}%", 
            f"Citation Precision: {overall_metrics.citation_precision:.1f}%",
            f"False Citation Rate: {overall_metrics.false_citation_rate:.1f}%",
            "",
            "DETAILED RESULTS BY CATEGORY:",
            "-" * 40
        ]
        
        # Group results by category
        category_results = defaultdict(list)
        for result in test_results:
            if 'test_query_info' in result:
                category = result['test_query_info']['category']
                category_results[category].append(result)
        
        for category, results in category_results.items():
            report_lines.append(f"\n{category.upper()} Questions:")
            
            for result in results:
                metrics = result.get('metrics', {})
                query_info = result.get('test_query_info', {})
                
                status = "✅" if metrics.get('has_citations') else "❌"
                accuracy = f"{metrics.get('accuracy_rate', 0):.1f}%" if metrics.get('citation_count', 0) > 0 else "N/A"
                
                report_lines.append(f"  {status} {query_info.get('complexity', 'unknown').title()}: {accuracy} accuracy")
                report_lines.append(f"     Question: {result.get('query', 'Unknown')[:80]}...")
                
                if result.get('error'):
                    report_lines.append(f"     ❌ ERROR: {result['error']}")
                elif metrics.get('citation_count', 0) > 0:
                    report_lines.append(f"     Citations: {metrics['citation_count']}, Valid: {metrics.get('valid_citations', 0)}")
        
        # Add failure analysis
        report_lines.extend([
            "",
            "FAILURE ANALYSIS:",
            "-" * 40
        ])
        
        failed_citations = []
        for result in test_results:
            for validation in result.get('validation_results', []):
                if not validation.get('is_valid', False):
                    failed_citations.append(validation)
        
        if failed_citations:
            error_types = defaultdict(int)
            for failure in failed_citations:
                error_msg = failure.get('error_message', 'Unknown error')
                error_types[error_msg] += 1
            
            report_lines.append("Common failure types:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"  - {error_type}: {count} citations")
        else:
            report_lines.append("No citation validation failures detected.")
        
        # Add recommendations
        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
            "-" * 40
        ])
        
        if overall_metrics.citation_coverage < 80:
            report_lines.append("• LOW CITATION COVERAGE: Consider updating prompts to encourage more citations")
        
        if overall_metrics.citation_accuracy < 90:
            report_lines.append("• LOW CITATION ACCURACY: Review citation generation logic for hallucination issues")
        
        if overall_metrics.citation_precision < 70:
            report_lines.append("• LOW CITATION PRECISION: Improve content matching between citations and sources")
        
        if overall_metrics.false_citation_rate > 10:
            report_lines.append("• HIGH FALSE CITATION RATE: Implement citation validation in response generation")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_detailed_results(self, test_results: Dict[str, Any], output_dir: Path = None) -> Path:
        """Save detailed test results to JSON file."""
        if output_dir is None:
            output_dir = Path(".")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"citation_accuracy_test_{self.proceeding}_{timestamp}.json"
        output_path = output_dir / filename
        
        # Convert CitationAccuracyMetrics to dict for JSON serialization
        if 'overall_metrics' in test_results:
            test_results['overall_metrics'] = asdict(test_results['overall_metrics'])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to: {output_path}")
        return output_path


def main():
    """Main function to run citation accuracy tests."""
    print("🔍 Citation Accuracy Test Suite for R2207005")
    print("=" * 60)
    
    try:
        # Initialize tester
        tester = CitationAccuracyTester("R2207005")
        
        # Run comprehensive tests
        print("Starting comprehensive citation accuracy tests...")
        results = tester.run_comprehensive_test(max_queries=5)  # Limit for initial testing
        
        # Print report
        print(results['report'])
        
        # Save detailed results
        output_path = tester.save_detailed_results(results)
        print(f"\n📄 Detailed results saved to: {output_path}")
        
        # Quick summary
        metrics = results['overall_metrics']
        print(f"\n📊 QUICK SUMMARY:")
        print(f"   Citation Coverage: {metrics['citation_coverage']:.1f}%")
        print(f"   Citation Accuracy: {metrics['citation_accuracy']:.1f}%")
        print(f"   Citation Precision: {metrics['citation_precision']:.1f}%")
        print(f"   False Citation Rate: {metrics['false_citation_rate']:.1f}%")
        
        return results
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    main()