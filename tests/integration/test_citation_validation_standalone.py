#!/usr/bin/env python3
"""
Standalone Citation Validation Test

This script tests citation validation without requiring the full RAG system,
making it useful for quick validation and debugging of citation issues.

It can:
1. Parse citations from sample text
2. Validate citations against LanceDB content
3. Test citation format parsing
4. Analyze citation patterns in existing responses
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.append(str(Path(__file__).parent.resolve()))

import lancedb
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParsedCitation:
    """A parsed citation with validation results."""
    filename: str
    page: int
    line: Optional[int] = None
    raw_text: str = ""
    found_in_vectordb: bool = False
    source_url: str = ""
    vectordb_text_sample: str = ""
    context_words: List[str] = None
    match_score: float = 0.0


class CitationParser:
    """Parses and validates citation formats."""
    
    def __init__(self):
        self.patterns = [
            # Format: [CITE:filename.pdf,page_12,line_45]
            (r'\[CITE:\s*([^,]+?)\s*,\s*page_(\d+),\s*line_(\d+)\]', 'full_format'),
            # Format: [CITE:filename.pdf,page_12]
            (r'\[CITE:\s*([^,]+?)\s*,\s*page_(\d+)\s*\]', 'page_format'),
            # Format: [CITE:filename,p.12]
            (r'\[CITE:\s*([^,]+?)\s*,\s*p\.?\s*(\d+)\]', 'alt_format'),
            # Format: [filename.pdf:page_12]
            (r'\[([^:]+?):page_(\d+)\]', 'colon_format'),
        ]
    
    def parse_citations_from_text(self, text: str) -> List[ParsedCitation]:
        """Parse all citations from text."""
        citations = []
        
        for pattern, format_type in self.patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            
            for match in regex.finditer(text):
                groups = match.groups()
                filename = groups[0].strip()
                page = int(groups[1])
                line = int(groups[2]) if len(groups) > 2 and groups[2] else None
                
                citation = ParsedCitation(
                    filename=filename,
                    page=page,
                    line=line,
                    raw_text=match.group(0)
                )
                
                citations.append(citation)
                
                logger.debug(f"Parsed citation: {filename} page {page} (format: {format_type})")
        
        return citations
    
    def test_citation_formats(self) -> Dict[str, Any]:
        """Test various citation formats to ensure parser works correctly."""
        test_cases = [
            "[CITE:566911513.pdf,page_1]",
            "[CITE:571985189,page_5,line_23]", 
            "[CITE: Document_123.pdf, page_10 ]",
            "[CITE:report.pdf,p.15]",
            "[CITE:analysis.pdf, p 20]",
            "[document.pdf:page_8]",
            "[invalid format]",
            "No citations here",
            "[CITE:multi_word_doc.pdf,page_1] and [CITE:another.pdf,page_2]"
        ]
        
        results = {
            'test_cases': len(test_cases),
            'parsed_citations': 0,
            'successful_parses': [],
            'failed_parses': []
        }
        
        for i, test_text in enumerate(test_cases):
            citations = self.parse_citations_from_text(test_text)
            
            if citations:
                results['parsed_citations'] += len(citations)
                results['successful_parses'].append({
                    'input': test_text,
                    'citations': [asdict(c) for c in citations]
                })
            else:
                results['failed_parses'].append(test_text)
        
        return results


class VectorDBValidator:
    """Validates citations against vectordb content."""
    
    def __init__(self, proceeding: str = "R2207005"):
        self.proceeding = proceeding
        self.db = None
        self.table = None
        self._connect()
    
    def _connect(self):
        """Connect to LanceDB."""
        try:
            db_path = f"local_lance_db/{self.proceeding}"
            self.db = lancedb.connect(db_path)
            table_name = f"{self.proceeding}_documents"
            
            if table_name in self.db.table_names():
                self.table = self.db.open_table(table_name)
                logger.info(f"Connected to {table_name}")
            else:
                logger.error(f"Table {table_name} not found")
                
        except Exception as e:
            logger.error(f"Failed to connect to vectordb: {e}")
    
    def validate_citations(self, citations: List[ParsedCitation]) -> List[ParsedCitation]:
        """Validate citations against vectordb content."""
        if not self.table:
            logger.warning("VectorDB not available for validation")
            return citations
        
        # Load vectordb data once
        try:
            df = self.table.to_pandas()
            logger.info(f"Loaded {len(df)} chunks from vectordb")
        except Exception as e:
            logger.error(f"Failed to load vectordb data: {e}")
            return citations
        
        validated_citations = []
        
        for citation in citations:
            validated = self._validate_single_citation(citation, df)
            validated_citations.append(validated)
        
        return validated_citations
    
    def _validate_single_citation(self, citation: ParsedCitation, df: pd.DataFrame) -> ParsedCitation:
        """Validate a single citation."""
        # Clean filename for matching
        clean_filename = citation.filename.replace('.pdf', '').replace('.PDF', '')
        
        # Find matching documents
        matching_chunks = []
        
        for _, row in df.iterrows():
            metadata = row['metadata']
            if isinstance(metadata, dict):
                source = metadata.get('source', '')
                page = metadata.get('page', 0)
                source_url = metadata.get('source_url', '')
                
                # Check if this chunk matches our citation
                if ((clean_filename in source or source in clean_filename) and 
                    page == citation.page):
                    
                    matching_chunks.append({
                        'text': row['text'],
                        'source_url': source_url,
                        'metadata': metadata
                    })
        
        if matching_chunks:
            citation.found_in_vectordb = True
            citation.source_url = matching_chunks[0]['source_url']
            
            # Combine text from all matching chunks for this page
            combined_text = ' '.join(chunk['text'] for chunk in matching_chunks)
            citation.vectordb_text_sample = combined_text[:500]  # First 500 chars
            
            # Calculate match score if context is available
            if hasattr(citation, 'context_words') and citation.context_words:
                citation.match_score = self._calculate_match_score(
                    citation.context_words, combined_text
                )
        
        return citation
    
    def _calculate_match_score(self, context_words: List[str], vectordb_text: str) -> float:
        """Calculate similarity score between context and vectordb text."""
        if not context_words or not vectordb_text:
            return 0.0
        
        vectordb_words = set(vectordb_text.lower().split())
        context_words_set = set(word.lower() for word in context_words)
        
        # Remove common words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        context_words_set -= stop_words
        vectordb_words -= stop_words
        
        if not context_words_set:
            return 0.0
        
        intersection = context_words_set.intersection(vectordb_words)
        return len(intersection) / len(context_words_set)
    
    def get_vectordb_statistics(self) -> Dict[str, Any]:
        """Get statistics about the vectordb."""
        if not self.table:
            return {'error': 'VectorDB not available'}
        
        try:
            df = self.table.to_pandas()
            
            # Extract document statistics
            documents = {}
            for _, row in df.iterrows():
                metadata = row['metadata']
                if isinstance(metadata, dict):
                    source = metadata.get('source', 'unknown')
                    page = metadata.get('page', 0)
                    
                    if source not in documents:
                        documents[source] = {
                            'pages': set(),
                            'chunks': 0,
                            'source_url': metadata.get('source_url', '')
                        }
                    
                    documents[source]['pages'].add(page)
                    documents[source]['chunks'] += 1
            
            # Convert sets to lists for JSON serialization
            for doc_info in documents.values():
                doc_info['pages'] = sorted(list(doc_info['pages']))
                doc_info['page_count'] = len(doc_info['pages'])
                doc_info['max_page'] = max(doc_info['pages']) if doc_info['pages'] else 0
            
            return {
                'total_chunks': len(df),
                'total_documents': len(documents),
                'documents': documents,
                'avg_chunks_per_doc': len(df) / len(documents) if documents else 0
            }
            
        except Exception as e:
            return {'error': str(e)}


def analyze_sample_responses():
    """Analyze sample RAG responses for citation patterns."""
    sample_responses = [
        """
        According to the proceeding documents, the main objectives of R.22-07-005 include advancing demand flexibility through electric rates [CITE:571985189.pdf,page_1]. The decision addresses Assembly Bill 205 requirements for electric utilities [CITE:566911514.pdf,page_2,line_15]. Additionally, the implementation working group recommendations are detailed in the joint reports [CITE:566911513.pdf,page_5].
        """,
        """
        The proposed decision correcting errors in Decision 15-01-039 has specific comment filing requirements [CITE:566593612.pdf,page_3]. Opening comments must not exceed 15 pages and are due no later than June 9, 2025 [CITE:566593612.pdf,page_3,line_22].
        """,
        """
        Decision D2505026 grants compensation to Utility Consumers' Action Network for substantial contribution to Decision D.24-05-028 [CITE:564706741.pdf,page_1]. The compensation request was filed on July 15, 2024 [CITE:557609581.pdf,page_1].
        """,
        """
        No specific citations in this response, just general information about the proceeding.
        """,
        """
        Invalid citations: [CITE:nonexistent.pdf,page_999] and [CITE:another_fake.pdf,page_123,line_456]. These should be flagged as potential hallucinations.
        """
    ]
    
    parser = CitationParser()
    validator = VectorDBValidator()
    
    print("📊 CITATION ANALYSIS REPORT")
    print("=" * 60)
    
    all_citations = []
    
    for i, response in enumerate(sample_responses, 1):
        print(f"\nSample Response {i}:")
        print("-" * 30)
        
        citations = parser.parse_citations_from_text(response)
        print(f"Citations found: {len(citations)}")
        
        if citations:
            validated_citations = validator.validate_citations(citations)
            
            for citation in validated_citations:
                print(f"  📄 {citation.filename} page {citation.page}")
                print(f"     Found in VectorDB: {'✅' if citation.found_in_vectordb else '❌'}")
                if citation.source_url:
                    print(f"     Source URL: {citation.source_url[:60]}...")
                if citation.vectordb_text_sample:
                    print(f"     Text sample: {citation.vectordb_text_sample[:100]}...")
                print()
            
            all_citations.extend(validated_citations)
        else:
            print("  No citations found")
    
    # Overall statistics
    print(f"\n📈 OVERALL STATISTICS:")
    print("-" * 30)
    print(f"Total citations: {len(all_citations)}")
    print(f"Valid citations: {sum(1 for c in all_citations if c.found_in_vectordb)}")
    print(f"Invalid citations: {sum(1 for c in all_citations if not c.found_in_vectordb)}")
    
    if all_citations:
        accuracy = sum(1 for c in all_citations if c.found_in_vectordb) / len(all_citations) * 100
        print(f"Citation accuracy: {accuracy:.1f}%")
    
    # Show invalid citations
    invalid_citations = [c for c in all_citations if not c.found_in_vectordb]
    if invalid_citations:
        print(f"\n❌ INVALID CITATIONS ({len(invalid_citations)}):")
        for citation in invalid_citations:
            print(f"  {citation.filename} page {citation.page}")


def test_citation_format_parsing():
    """Test citation format parsing."""
    print("🧪 CITATION FORMAT PARSING TEST")
    print("=" * 50)
    
    parser = CitationParser()
    results = parser.test_citation_formats()
    
    print(f"Test cases: {results['test_cases']}")
    print(f"Parsed citations: {results['parsed_citations']}")
    print(f"Successful parses: {len(results['successful_parses'])}")
    print(f"Failed parses: {len(results['failed_parses'])}")
    
    print("\nSuccessful parses:")
    for parse in results['successful_parses']:
        print(f"  Input: {parse['input']}")
        for citation in parse['citations']:
            print(f"    → {citation['filename']} page {citation['page']}")
    
    if results['failed_parses']:
        print(f"\nFailed to parse:")
        for failed in results['failed_parses']:
            print(f"  {failed}")


def show_vectordb_info():
    """Show information about the vectordb."""
    print("📊 VECTORDB INFORMATION")
    print("=" * 40)
    
    validator = VectorDBValidator()
    stats = validator.get_vectordb_statistics()
    
    if 'error' in stats:
        print(f"❌ Error: {stats['error']}")
        return
    
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Average chunks per document: {stats['avg_chunks_per_doc']:.1f}")
    
    print(f"\nDocument details (showing first 10):")
    document_items = list(stats['documents'].items())[:10]
    
    for doc_name, info in document_items:
        print(f"  📄 {doc_name}")
        print(f"     Pages: {info['page_count']} (max: {info['max_page']})")
        print(f"     Chunks: {info['chunks']}")
        if info['source_url']:
            print(f"     URL: {info['source_url'][:60]}...")
        print()
    
    if len(stats['documents']) > 10:
        print(f"... and {len(stats['documents']) - 10} more documents")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone citation validation test")
    parser.add_argument('--test-parsing', action='store_true',
                       help='Test citation format parsing')
    parser.add_argument('--test-samples', action='store_true', 
                       help='Analyze sample responses for citations')
    parser.add_argument('--show-vectordb', action='store_true',
                       help='Show vectordb information')
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    
    args = parser.parse_args()
    
    if args.all or (not any([args.test_parsing, args.test_samples, args.show_vectordb])):
        # Run all tests if no specific test is requested
        test_citation_format_parsing()
        print()
        show_vectordb_info()
        print()
        analyze_sample_responses()
    else:
        if args.test_parsing:
            test_citation_format_parsing()
        
        if args.show_vectordb:
            show_vectordb_info()
        
        if args.test_samples:
            analyze_sample_responses()


if __name__ == "__main__":
    main()