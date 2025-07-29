#!/usr/bin/env python3
"""
Docling Performance Analysis Test Script

This script processes 5 representative PDF documents from proceeding R2207005 
using the standard Docling pipeline to collect detailed performance metrics.

Requirements:
- Process documents using extract_and_chunk_with_docling_url function
- Disable OCR and Chonkie fallbacks to test pure Docling
- Collect detailed performance metrics for each document
- Provide comprehensive analysis and recommendations

Author: Claude Code Analysis System
Date: 2025-07-26
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import statistics

# Import our existing data processing functions
from data_processing import extract_and_chunk_with_docling_url
import config

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DoclingPerformanceAnalyzer:
    """Comprehensive Docling performance analysis system."""
    
    def __init__(self):
        self.results = []
        self.test_start_time = datetime.now()
        
        # Selected representative documents from R2207005
        self.test_documents = [
            {
                "id": "doc1_final_decision",
                "url": "https://docs.cpuc.ca.gov/PublishedDocs/Published/G000/M571/K985/571985189.PDF",
                "title": "571985189 - Final Decision D2506047",
                "document_type": "Final Decision",
                "expected_complexity": "medium",
                "description": "Decision correcting errors in Decision 25-01-039"
            },
            {
                "id": "doc2_compliance_filing", 
                "url": "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M566/K911/566911513.PDF",
                "title": "566911513 - Joint Compliance Report",
                "document_type": "E-Filed: Compliance Filing",
                "expected_complexity": "high",
                "description": "Joint report in compliance with Ordering Paragraph 5 of D.24-05-028"
            },
            {
                "id": "doc3_proposed_decision",
                "url": "https://docs.cpuc.ca.gov/PublishedDocs/Efile/G000/M566/K593/566593612.PDF", 
                "title": "566593612 - Proposed Decision",
                "document_type": "E-Filed: Proposed Decision",
                "expected_complexity": "medium",
                "description": "Proposed Decision correcting errors in Decision 15-01-039"
            },
            {
                "id": "doc4_agenda_decision",
                "url": "https://docs.cpuc.ca.gov/PublishedDocs/Published/G000/M564/K706/564706741.PDF",
                "title": "564706741 - Agenda Decision D2505026", 
                "document_type": "Agenda Decision",
                "expected_complexity": "high",
                "description": "Decision granting compensation for substantial contribution"
            },
            {
                "id": "doc5_compensation_decision",
                "url": "https://docs.cpuc.ca.gov/PublishedDocs/Published/G000/M562/K527/562527349.PDF",
                "title": "562527349 - Compensation Decision D2504015",
                "document_type": "Final Decision", 
                "expected_complexity": "high",
                "description": "Decision granting compensation to California Environmental Justice Alliance"
            }
        ]
    
    def analyze_document_content(self, chunks: List, doc_info: Dict) -> Dict[str, Any]:
        """Analyze the content extracted from a document."""
        if not chunks:
            return {
                "total_chunks": 0,
                "total_text_length": 0,
                "content_types": [],
                "avg_chunk_length": 0,
                "content_quality_score": 0.0,
                "structure_preservation": False,
                "sample_content": "No content extracted"
            }
        
        # Basic statistics
        total_chunks = len(chunks)
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        total_text_length = sum(chunk_lengths)
        avg_chunk_length = statistics.mean(chunk_lengths) if chunk_lengths else 0
        
        # Content type analysis
        content_types = {}
        for chunk in chunks:
            content_type = chunk.metadata.get('content_type', 'unknown')
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        # Content quality assessment
        quality_score = self._assess_content_quality(chunks)
        
        # Structure preservation check
        has_tables = any('table' in chunk.metadata.get('content_type', '').lower() for chunk in chunks)
        has_structured_content = any(chunk.metadata.get('content_type') in ['title', 'section-header', 'list-item'] for chunk in chunks)
        structure_preservation = has_tables or has_structured_content
        
        # Sample content (first 300 characters from first chunk)
        sample_content = chunks[0].page_content[:300] + "..." if chunks and chunks[0].page_content else "No content available"
        
        return {
            "total_chunks": total_chunks,
            "total_text_length": total_text_length,
            "content_types": dict(content_types),
            "avg_chunk_length": avg_chunk_length,
            "content_quality_score": quality_score,
            "structure_preservation": structure_preservation,
            "sample_content": sample_content,
            "chunk_length_stats": {
                "min": min(chunk_lengths) if chunk_lengths else 0,
                "max": max(chunk_lengths) if chunk_lengths else 0,
                "median": statistics.median(chunk_lengths) if chunk_lengths else 0,
                "std_dev": statistics.stdev(chunk_lengths) if len(chunk_lengths) > 1 else 0
            }
        }
    
    def _assess_content_quality(self, chunks: List) -> float:
        """Assess the quality of extracted content on a 0-1 scale."""
        if not chunks:
            return 0.0
        
        quality_indicators = []
        
        for chunk in chunks:
            content = chunk.page_content
            if not content or not content.strip():
                quality_indicators.append(0.0)
                continue
                
            score = 0.0
            
            # Check for readable text (not garbled)
            readable_chars = sum(1 for c in content if c.isalnum() or c.isspace() or c in '.,!?;:-()[]{}')
            if len(content) > 0:
                readability = readable_chars / len(content)
                score += readability * 0.4
            
            # Check for proper sentence structure
            sentences = content.count('.') + content.count('!') + content.count('?')
            if len(content) > 50 and sentences > 0:
                score += min(sentences / (len(content) / 100), 1.0) * 0.3
            
            # Check for proper capitalization
            if content.strip():
                words = content.split()
                if words:
                    capitalized = sum(1 for word in words if word[0].isupper())
                    score += min(capitalized / len(words), 0.3) * 0.3
            
            quality_indicators.append(min(score, 1.0))
        
        return statistics.mean(quality_indicators) if quality_indicators else 0.0
    
    def process_document(self, doc_info: Dict) -> Dict[str, Any]:
        """Process a single document and collect detailed metrics."""
        logger.info(f"Processing document: {doc_info['title']}")
        
        start_time = time.time()
        processing_errors = []
        chunks = []
        
        try:
            # Process with Docling - DISABLE OCR and Chonkie fallbacks for pure Docling test
            chunks = extract_and_chunk_with_docling_url(
                pdf_url=doc_info["url"],
                document_title=doc_info["title"],
                proceeding="R2207005",
                enable_ocr_fallback=False,      # DISABLED for pure Docling test
                enable_chonkie_fallback=False   # DISABLED for pure Docling test
            )
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(error_msg)
            processing_errors.append(error_msg)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Analyze content
        content_analysis = self.analyze_document_content(chunks, doc_info)
        
        # Calculate processing efficiency
        efficiency_metrics = {
            "chunks_per_second": content_analysis["total_chunks"] / processing_time if processing_time > 0 else 0,
            "chars_per_second": content_analysis["total_text_length"] / processing_time if processing_time > 0 else 0,
            "processing_success": len(chunks) > 0,
            "error_count": len(processing_errors)
        }
        
        result = {
            "document_info": doc_info,
            "processing_time_seconds": round(processing_time, 3),
            "processing_errors": processing_errors,
            "content_analysis": content_analysis,
            "efficiency_metrics": efficiency_metrics,
            "timestamp": datetime.now().isoformat(),
            "docling_config": {
                "ocr_fallback_enabled": False,
                "chonkie_fallback_enabled": False,
                "fast_mode": getattr(config, 'DOCLING_FAST_MODE', False),
                "max_pages": getattr(config, 'DOCLING_MAX_PAGES', None),
                "threads": getattr(config, 'DOCLING_THREADS', 1)
            }
        }
        
        logger.info(f"Completed processing {doc_info['title']} in {processing_time:.3f}s - {content_analysis['total_chunks']} chunks extracted")
        return result
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run the complete Docling performance analysis."""
        logger.info("Starting Docling Performance Analysis")
        logger.info(f"Testing {len(self.test_documents)} documents with pure Docling processing")
        
        # Process each document
        for doc_info in self.test_documents:
            result = self.process_document(doc_info)
            self.results.append(result)
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report()
        
        # Save results
        self._save_results(report)
        
        logger.info("Docling Performance Analysis completed")
        return report
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        test_end_time = datetime.now()
        total_test_duration = (test_end_time - self.test_start_time).total_seconds()
        
        # Aggregate statistics
        successful_docs = [r for r in self.results if r["efficiency_metrics"]["processing_success"]]
        failed_docs = [r for r in self.results if not r["efficiency_metrics"]["processing_success"]]
        
        if successful_docs:
            processing_times = [r["processing_time_seconds"] for r in successful_docs]
            chunk_counts = [r["content_analysis"]["total_chunks"] for r in successful_docs]
            text_lengths = [r["content_analysis"]["total_text_length"] for r in successful_docs]
            quality_scores = [r["content_analysis"]["content_quality_score"] for r in successful_docs]
            
            aggregate_stats = {
                "total_documents_processed": len(self.results),
                "successful_extractions": len(successful_docs),
                "failed_extractions": len(failed_docs),
                "success_rate": len(successful_docs) / len(self.results) * 100,
                
                "processing_time_stats": {
                    "mean": statistics.mean(processing_times),
                    "median": statistics.median(processing_times),
                    "min": min(processing_times),
                    "max": max(processing_times),
                    "std_dev": statistics.stdev(processing_times) if len(processing_times) > 1 else 0
                },
                
                "chunk_extraction_stats": {
                    "total_chunks": sum(chunk_counts),
                    "mean_chunks_per_doc": statistics.mean(chunk_counts),
                    "median_chunks_per_doc": statistics.median(chunk_counts),
                    "min_chunks": min(chunk_counts),
                    "max_chunks": max(chunk_counts)
                },
                
                "text_extraction_stats": {
                    "total_characters": sum(text_lengths),
                    "mean_chars_per_doc": statistics.mean(text_lengths),
                    "median_chars_per_doc": statistics.median(text_lengths),
                    "min_chars": min(text_lengths),
                    "max_chars": max(text_lengths)
                },
                
                "content_quality_stats": {
                    "mean_quality_score": statistics.mean(quality_scores),
                    "median_quality_score": statistics.median(quality_scores),
                    "min_quality_score": min(quality_scores),
                    "max_quality_score": max(quality_scores)
                }
            }
        else:
            aggregate_stats = {
                "total_documents_processed": len(self.results),
                "successful_extractions": 0,
                "failed_extractions": len(failed_docs),
                "success_rate": 0.0,
                "processing_time_stats": {},
                "chunk_extraction_stats": {},
                "text_extraction_stats": {},
                "content_quality_stats": {}
            }
        
        # Content type analysis across all documents
        all_content_types = {}
        for result in successful_docs:
            for content_type, count in result["content_analysis"]["content_types"].items():
                all_content_types[content_type] = all_content_types.get(content_type, 0) + count
        
        # Processing efficiency analysis
        efficiency_analysis = {
            "overall_efficiency": {
                "total_test_duration": total_test_duration,
                "docs_per_minute": len(self.results) / (total_test_duration / 60) if total_test_duration > 0 else 0,
                "average_processing_speed": statistics.mean([r["processing_time_seconds"] for r in self.results]) if self.results else 0
            },
            "content_types_found": all_content_types,
            "structure_preservation_rate": sum(1 for r in successful_docs if r["content_analysis"]["structure_preservation"]) / len(successful_docs) * 100 if successful_docs else 0
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(aggregate_stats, efficiency_analysis)
        
        return {
            "test_metadata": {
                "test_start_time": self.test_start_time.isoformat(),
                "test_end_time": test_end_time.isoformat(),
                "total_duration_seconds": total_test_duration,
                "proceeding": "R2207005",
                "docling_mode": "pure_docling_no_fallbacks",
                "documents_tested": len(self.test_documents)
            },
            "individual_results": self.results,
            "aggregate_statistics": aggregate_stats,
            "efficiency_analysis": efficiency_analysis,
            "recommendations": recommendations
        }
    
    def _generate_recommendations(self, stats: Dict, efficiency: Dict) -> List[str]:
        """Generate actionable recommendations based on test results."""
        recommendations = []
        
        if stats.get("success_rate", 0) < 80:
            recommendations.append("Success rate below 80% indicates potential issues with document formats or Docling configuration")
        
        if stats.get("processing_time_stats", {}).get("mean", 0) > 30:
            recommendations.append("Average processing time exceeds 30 seconds - consider enabling DOCLING_FAST_MODE")
        
        if stats.get("content_quality_stats", {}).get("mean_quality_score", 0) < 0.7:
            recommendations.append("Content quality scores below 0.7 suggest text extraction issues - consider OCR fallback for scanned documents")
        
        if efficiency.get("structure_preservation_rate", 0) < 50:
            recommendations.append("Low structure preservation rate - review table extraction settings")
        
        content_types = efficiency.get("content_types_found", {})
        if not any("table" in ct.lower() for ct in content_types):
            recommendations.append("No table content detected - verify table extraction configuration")
        
        if stats.get("failed_extractions", 0) > 0:
            recommendations.append("Some documents failed extraction - consider implementing OCR and Chonkie fallbacks for production")
        
        return recommendations
    
    def _save_results(self, report: Dict[str, Any]):
        """Save the comprehensive analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON report
        results_file = Path(f"docling_performance_analysis_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Generate and save markdown report
        markdown_report = self._generate_markdown_report(report)
        markdown_file = Path(f"docling_performance_report_{timestamp}.md")
        with open(markdown_file, 'w') as f:
            f.write(markdown_report)
        
        logger.info(f"Markdown report saved to: {markdown_file}")
    
    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate a formatted markdown report."""
        stats = report["aggregate_statistics"]
        efficiency = report["efficiency_analysis"]
        
        markdown = f"""# Docling Performance Analysis Report
        
## Test Overview
- **Test Date**: {report["test_metadata"]["test_start_time"][:10]}
- **Duration**: {report["test_metadata"]["total_duration_seconds"]:.1f} seconds
- **Proceeding**: {report["test_metadata"]["proceeding"]}
- **Mode**: Pure Docling (OCR and Chonkie fallbacks disabled)
- **Documents Tested**: {report["test_metadata"]["documents_tested"]}

## Executive Summary
- **Success Rate**: {stats.get("success_rate", 0):.1f}%
- **Documents Processed**: {stats.get("successful_extractions", 0)}/{stats.get("total_documents_processed", 0)}
- **Total Chunks Extracted**: {stats.get("chunk_extraction_stats", {}).get("total_chunks", 0):,}
- **Total Text Extracted**: {stats.get("text_extraction_stats", {}).get("total_characters", 0):,} characters

## Processing Performance

### Timing Statistics
| Metric | Value |
|--------|-------|
| Mean Processing Time | {stats.get("processing_time_stats", {}).get("mean", 0):.2f}s |
| Median Processing Time | {stats.get("processing_time_stats", {}).get("median", 0):.2f}s |
| Fastest Document | {stats.get("processing_time_stats", {}).get("min", 0):.2f}s |
| Slowest Document | {stats.get("processing_time_stats", {}).get("max", 0):.2f}s |

### Content Extraction
| Metric | Value |
|--------|-------|
| Mean Chunks per Document | {stats.get("chunk_extraction_stats", {}).get("mean_chunks_per_doc", 0):.1f} |
| Mean Characters per Document | {stats.get("text_extraction_stats", {}).get("mean_chars_per_doc", 0):,.0f} |
| Content Quality Score | {stats.get("content_quality_stats", {}).get("mean_quality_score", 0):.2f}/1.0 |
| Structure Preservation Rate | {efficiency.get("structure_preservation_rate", 0):.1f}% |

## Individual Document Results

"""
        
        for result in report["individual_results"]:
            doc = result["document_info"]
            content = result["content_analysis"]
            markdown += f"""### {doc["title"]}
- **Type**: {doc["document_type"]}
- **Processing Time**: {result["processing_time_seconds"]}s
- **Chunks Extracted**: {content["total_chunks"]}
- **Text Length**: {content["total_text_length"]:,} characters
- **Quality Score**: {content["content_quality_score"]:.2f}/1.0
- **Success**: {'✅ Yes' if result["efficiency_metrics"]["processing_success"] else '❌ No'}

**Sample Content**: {content["sample_content"][:200]}...

**Content Types Found**: {', '.join(content["content_types"].keys()) if content["content_types"] else 'None'}

"""
        
        markdown += f"""## Content Type Analysis

"""
        content_types = efficiency.get("content_types_found", {})
        for content_type, count in sorted(content_types.items(), key=lambda x: x[1], reverse=True):
            markdown += f"- **{content_type}**: {count} chunks\n"
        
        markdown += f"""
## Recommendations

"""
        for i, rec in enumerate(report["recommendations"], 1):
            markdown += f"{i}. {rec}\n"
        
        markdown += f"""
## Technical Configuration
- **Docling Fast Mode**: {report["individual_results"][0]["docling_config"]["fast_mode"] if report["individual_results"] else "Unknown"}
- **Max Pages Limit**: {report["individual_results"][0]["docling_config"]["max_pages"] if report["individual_results"] else "None"}
- **Thread Count**: {report["individual_results"][0]["docling_config"]["threads"] if report["individual_results"] else "1"}
- **OCR Fallback**: Disabled (for pure Docling testing)
- **Chonkie Fallback**: Disabled (for pure Docling testing)

---
*Report generated by Docling Performance Analyzer on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
        
        return markdown

def main():
    """Main execution function."""
    print("🔬 Docling Performance Analysis Starting...")
    print("=" * 60)
    
    analyzer = DoclingPerformanceAnalyzer()
    
    try:
        report = analyzer.run_comprehensive_analysis()
        
        print("\n" + "=" * 60)
        print("📊 ANALYSIS COMPLETE - KEY FINDINGS:")
        print("=" * 60)
        
        stats = report["aggregate_statistics"]
        print(f"✅ Success Rate: {stats.get('success_rate', 0):.1f}%")
        print(f"📄 Documents Processed: {stats.get('successful_extractions', 0)}/{stats.get('total_documents_processed', 0)}")
        print(f"🧩 Total Chunks: {stats.get('chunk_extraction_stats', {}).get('total_chunks', 0):,}")
        print(f"📝 Total Characters: {stats.get('text_extraction_stats', {}).get('total_characters', 0):,}")
        print(f"⏱️  Average Processing Time: {stats.get('processing_time_stats', {}).get('mean', 0):.2f}s")
        print(f"🎯 Content Quality: {stats.get('content_quality_stats', {}).get('mean_quality_score', 0):.2f}/1.0")
        
        print(f"\n📋 TOP RECOMMENDATIONS:")
        for i, rec in enumerate(report["recommendations"][:3], 1):
            print(f"{i}. {rec}")
        
        print(f"\n📁 Detailed results saved to JSON and Markdown files")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()