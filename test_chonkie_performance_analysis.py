#!/usr/bin/env python3
"""
Chonkie Processing Performance Analysis - Direct Comparison with Docling

This script processes the exact same 5 PDF documents from proceeding R2207005 that were
analyzed with Docling, using pure Chonkie processing for head-to-head comparison.

Key Features:
- Forces pure Chonkie processing (bypasses Docling/OCR)
- Tests all 3 Chonkie strategies: recursive, sentence, token  
- Collects identical performance metrics as Docling analysis
- Processes same document URLs as Docling test
- Provides comprehensive performance comparison data

Usage:
    python test_chonkie_performance_analysis.py
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
import config
from data_processing import extract_text_from_url, safe_chunk_with_chonkie

def force_chonkie_processing(pdf_url: str, document_title: str, chunker_type: str = "recursive") -> Dict:
    """
    Force pure Chonkie processing on a PDF URL, bypassing Docling and OCR completely.
    
    This function replicates the Chonkie fallback logic but forces it as the primary method,
    providing direct comparison with Docling processing results.
    
    Args:
        pdf_url (str): URL of the PDF to process
        document_title (str): Title for identification
        chunker_type (str): Chonkie strategy - "recursive", "sentence", or "token"
        
    Returns:
        Dict: Processing results with metrics matching Docling analysis format
    """
    start_time = time.time()
    result = {
        "document_info": {
            "url": pdf_url,
            "title": document_title,
            "chunker_type": chunker_type
        },
        "processing_time_seconds": 0,
        "processing_errors": [],
        "content_analysis": {
            "total_chunks": 0,
            "total_text_length": 0,
            "avg_chunk_length": 0,
            "content_quality_score": 0.0,
            "sample_content": "",
            "chunk_length_stats": {
                "min": 0,
                "max": 0,
                "median": 0,
                "std_dev": 0
            }
        },
        "efficiency_metrics": {
            "chunks_per_second": 0,
            "chars_per_second": 0,
            "processing_success": False,
            "error_count": 0
        },
        "chonkie_config": {
            "chunker_type": chunker_type,
            "chunk_size": config.CHONKIE_CHUNK_SIZE,
            "chunk_overlap": config.CHONKIE_CHUNK_OVERLAP,
            "min_text_length": config.CHONKIE_MIN_TEXT_LENGTH,
            "use_pdfplumber": config.CHONKIE_USE_PDFPLUMBER,
            "use_pypdf2": config.CHONKIE_USE_PYPDF2
        },
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        logger.info(f"Starting Chonkie {chunker_type} processing for: {document_title}")
        
        # Step 1: Extract raw text from PDF
        text_extraction_start = time.time()
        raw_text = extract_text_from_url(pdf_url)
        text_extraction_time = time.time() - text_extraction_start
        
        if not raw_text or len(raw_text.strip()) < config.CHONKIE_MIN_TEXT_LENGTH:
            error_msg = f"Insufficient text extracted: {len(raw_text) if raw_text else 0} chars (min: {config.CHONKIE_MIN_TEXT_LENGTH})"
            result["processing_errors"].append(error_msg)
            result["efficiency_metrics"]["error_count"] = 1
            logger.warning(error_msg)
            return result
        
        logger.info(f"Text extraction completed in {text_extraction_time:.3f}s: {len(raw_text)} characters")
        
        # Step 2: Chunk text using specified Chonkie strategy  
        chunking_start = time.time()
        chunk_texts = safe_chunk_with_chonkie(raw_text, chunker_type)
        chunking_time = time.time() - chunking_start
        
        if not chunk_texts:
            error_msg = f"Chonkie {chunker_type} chunking failed - no chunks produced"
            result["processing_errors"].append(error_msg)
            result["efficiency_metrics"]["error_count"] = 1
            logger.warning(error_msg)
            return result
        
        logger.info(f"Chonkie {chunker_type} chunking completed in {chunking_time:.3f}s: {len(chunk_texts)} chunks")
        
        # Step 3: Calculate performance metrics
        processing_time = time.time() - start_time
        result["processing_time_seconds"] = processing_time
        
        # Content analysis
        result["content_analysis"]["total_chunks"] = len(chunk_texts)
        result["content_analysis"]["total_text_length"] = len(raw_text)
        result["content_analysis"]["avg_chunk_length"] = len(raw_text) / len(chunk_texts) if chunk_texts else 0
        
        # Calculate chunk length statistics
        chunk_lengths = [len(chunk) for chunk in chunk_texts]
        if chunk_lengths:
            import statistics
            result["content_analysis"]["chunk_length_stats"] = {
                "min": min(chunk_lengths),
                "max": max(chunk_lengths),
                "median": statistics.median(chunk_lengths),
                "std_dev": statistics.stdev(chunk_lengths) if len(chunk_lengths) > 1 else 0
            }
        
        # Sample content (first 300 characters)
        result["content_analysis"]["sample_content"] = chunk_texts[0][:300] + "..." if chunk_texts else ""
        
        # Content quality assessment (simplified for Chonkie)
        result["content_analysis"]["content_quality_score"] = _assess_content_quality(chunk_texts, raw_text)
        
        # Efficiency metrics
        result["efficiency_metrics"]["chunks_per_second"] = len(chunk_texts) / processing_time if processing_time > 0 else 0
        result["efficiency_metrics"]["chars_per_second"] = len(raw_text) / processing_time if processing_time > 0 else 0
        result["efficiency_metrics"]["processing_success"] = True
        
        logger.info(f"Chonkie {chunker_type} processing completed successfully: {len(chunk_texts)} chunks in {processing_time:.3f}s")
        
    except Exception as e:
        error_msg = f"Chonkie processing failed: {str(e)}"
        result["processing_errors"].append(error_msg)
        result["efficiency_metrics"]["error_count"] = 1
        result["processing_time_seconds"] = time.time() - start_time
        logger.error(error_msg, exc_info=True)
    
    return result

def _assess_content_quality(chunks: List[str], raw_text: str) -> float:
    """
    Assess content quality for Chonkie processing.
    
    This provides a simplified quality score focusing on text preservation
    and chunking effectiveness, comparable to Docling's quality assessment.
    """
    if not chunks or not raw_text:
        return 0.0
    
    try:
        # Calculate text preservation ratio
        total_chunk_text = sum(len(chunk) for chunk in chunks)
        preservation_ratio = min(1.0, total_chunk_text / len(raw_text))
        
        # Assess chunk distribution quality
        chunk_lengths = [len(chunk) for chunk in chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        
        # Penalize extremely short or long chunks
        length_quality = 1.0
        if avg_length < 100:  # Too short
            length_quality *= 0.7
        elif avg_length > 2000:  # Too long
            length_quality *= 0.8
            
        # Check for content variety (not just repeated text)
        unique_starts = set()
        for chunk in chunks[:10]:  # Sample first 10 chunks
            if len(chunk) > 50:
                unique_starts.add(chunk[:50].strip())
        
        variety_ratio = len(unique_starts) / min(10, len(chunks)) if chunks else 0
        
        # Combined quality score
        quality_score = (preservation_ratio * 0.4 + length_quality * 0.3 + variety_ratio * 0.3)
        
        return round(quality_score, 4)
        
    except Exception as e:
        logger.warning(f"Content quality assessment failed: {e}")
        return 0.5  # Default moderate score

def run_chonkie_analysis():
    """
    Run comprehensive Chonkie analysis on the same 5 documents from R2207005
    that were analyzed with Docling for direct performance comparison.
    """
    # Same 5 documents from Docling analysis
    test_documents = [
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
    
    # Test all 3 Chonkie strategies
    chonkie_strategies = ["recursive", "sentence", "token"]
    
    test_start_time = datetime.now()
    logger.info(f"Starting Chonkie performance analysis at {test_start_time}")
    logger.info(f"Testing {len(test_documents)} documents with {len(chonkie_strategies)} strategies each")
    
    results = {
        "test_metadata": {
            "test_start_time": test_start_time.isoformat(),
            "test_end_time": None,
            "total_duration_seconds": 0,
            "proceeding": "R2207005",
            "processing_mode": "pure_chonkie_all_strategies",
            "documents_tested": len(test_documents),
            "strategies_tested": chonkie_strategies,
            "total_combinations": len(test_documents) * len(chonkie_strategies)
        },
        "strategy_results": {strategy: [] for strategy in chonkie_strategies},
        "comparative_analysis": {},
        "aggregate_statistics": {}
    }
    
    # Process each document with each strategy
    total_combinations = 0
    successful_combinations = 0
    
    for doc in test_documents:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing document: {doc['title']}")
        logger.info(f"URL: {doc['url']}")
        logger.info(f"Expected complexity: {doc['expected_complexity']}")
        
        for strategy in chonkie_strategies:
            logger.info(f"\n  Testing strategy: {strategy}")
            
            result = force_chonkie_processing(
                pdf_url=doc["url"],
                document_title=doc["title"], 
                chunker_type=strategy
            )
            
            # Add document metadata to result
            result["document_info"].update(doc)
            
            # Store result
            results["strategy_results"][strategy].append(result)
            
            total_combinations += 1
            if result["efficiency_metrics"]["processing_success"]:
                successful_combinations += 1
                
            # Log summary
            success_status = "✅ SUCCESS" if result["efficiency_metrics"]["processing_success"] else "❌ FAILED"
            logger.info(f"    {success_status}: {result['content_analysis']['total_chunks']} chunks, "
                       f"{result['processing_time_seconds']:.2f}s, "
                       f"quality: {result['content_analysis']['content_quality_score']:.3f}")
    
    # Calculate final statistics
    test_end_time = datetime.now()
    total_duration = (test_end_time - test_start_time).total_seconds()
    
    results["test_metadata"]["test_end_time"] = test_end_time.isoformat()
    results["test_metadata"]["total_duration_seconds"] = total_duration
    results["test_metadata"]["success_rate"] = (successful_combinations / total_combinations) * 100
    
    # Generate comparative analysis between strategies
    results["comparative_analysis"] = _generate_strategy_comparison(results["strategy_results"])
    
    # Generate aggregate statistics
    results["aggregate_statistics"] = _generate_aggregate_statistics(results["strategy_results"])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"CHONKIE ANALYSIS COMPLETE")
    logger.info(f"Total duration: {total_duration:.1f} seconds")
    logger.info(f"Success rate: {results['test_metadata']['success_rate']:.1f}%")
    logger.info(f"Combinations processed: {successful_combinations}/{total_combinations}")
    
    # Save results
    timestamp = test_start_time.strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"chonkie_performance_analysis_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Generate summary report
    _generate_summary_report(results, results_file.with_suffix('.md'))
    
    return results

def _generate_strategy_comparison(strategy_results: Dict) -> Dict:
    """Generate detailed comparison between Chonkie strategies."""
    comparison = {}
    
    for strategy, results in strategy_results.items():
        if not results:
            continue
            
        successful_results = [r for r in results if r["efficiency_metrics"]["processing_success"]]
        
        if successful_results:
            processing_times = [r["processing_time_seconds"] for r in successful_results]
            chunk_counts = [r["content_analysis"]["total_chunks"] for r in successful_results]
            quality_scores = [r["content_analysis"]["content_quality_score"] for r in successful_results]
            
            comparison[strategy] = {
                "success_rate": len(successful_results) / len(results) * 100,
                "avg_processing_time": sum(processing_times) / len(processing_times),
                "avg_chunks_produced": sum(chunk_counts) / len(chunk_counts),
                "avg_quality_score": sum(quality_scores) / len(quality_scores),
                "total_successful": len(successful_results),
                "total_attempted": len(results)
            }
    
    # Rank strategies by combined performance
    for strategy, stats in comparison.items():
        # Combined score (lower processing time + higher quality + higher success rate)
        speed_score = 1.0 / (stats["avg_processing_time"] + 0.1)  # Avoid division by zero
        quality_score = stats["avg_quality_score"]
        success_score = stats["success_rate"] / 100
        
        stats["combined_performance_score"] = (speed_score * 0.3 + quality_score * 0.4 + success_score * 0.3)
    
    return comparison

def _generate_aggregate_statistics(strategy_results: Dict) -> Dict:
    """Generate aggregate statistics across all strategies."""
    all_results = []
    for results in strategy_results.values():
        all_results.extend(results)
    
    successful_results = [r for r in all_results if r["efficiency_metrics"]["processing_success"]]
    
    if not successful_results:
        return {"error": "No successful processing results to analyze"}
    
    processing_times = [r["processing_time_seconds"] for r in successful_results]
    chunk_counts = [r["content_analysis"]["total_chunks"] for r in successful_results]
    quality_scores = [r["content_analysis"]["content_quality_score"] for r in successful_results]
    text_lengths = [r["content_analysis"]["total_text_length"] for r in successful_results]
    
    import statistics
    
    return {
        "total_combinations_attempted": len(all_results),
        "successful_combinations": len(successful_results),
        "overall_success_rate": len(successful_results) / len(all_results) * 100,
        
        "processing_time_stats": {
            "mean": statistics.mean(processing_times),
            "median": statistics.median(processing_times),
            "min": min(processing_times),
            "max": max(processing_times),
            "std_dev": statistics.stdev(processing_times) if len(processing_times) > 1 else 0
        },
        
        "chunk_extraction_stats": {
            "total_chunks": sum(chunk_counts),
            "mean_chunks_per_combination": statistics.mean(chunk_counts),
            "median_chunks": statistics.median(chunk_counts),
            "min_chunks": min(chunk_counts),
            "max_chunks": max(chunk_counts)
        },
        
        "content_quality_stats": {
            "mean_quality_score": statistics.mean(quality_scores),
            "median_quality_score": statistics.median(quality_scores),
            "min_quality_score": min(quality_scores),
            "max_quality_score": max(quality_scores)
        },
        
        "text_extraction_stats": {
            "total_characters_extracted": sum(text_lengths),
            "mean_chars_per_document": statistics.mean(text_lengths),
            "median_chars": statistics.median(text_lengths),
            "min_chars": min(text_lengths),
            "max_chars": max(text_lengths)
        }
    }

def _generate_summary_report(results: Dict, report_file: Path):
    """Generate a comprehensive markdown summary report."""
    
    report_content = f"""# Chonkie Processing Performance Analysis Report
**CPUC Regulatory Document Processing System**

## Executive Summary

This comprehensive analysis evaluated Chonkie processing performance on 5 representative PDF documents from CPUC proceeding R2207005, testing all 3 chunking strategies (recursive, sentence, token) for direct comparison with Docling processing results.

### Key Results
- **{results['test_metadata']['success_rate']:.1f}% Overall Success Rate**: {results['aggregate_statistics']['successful_combinations']}/{results['test_metadata']['total_combinations']} combinations processed successfully
- **{results['aggregate_statistics']['chunk_extraction_stats']['total_chunks']} Total Chunks Extracted**: Average of {results['aggregate_statistics']['chunk_extraction_stats']['mean_chunks_per_combination']:.1f} chunks per combination
- **{results['aggregate_statistics']['text_extraction_stats']['total_characters_extracted']:,} Characters Extracted**: Average of {results['aggregate_statistics']['text_extraction_stats']['mean_chars_per_document']:,.0f} characters per document
- **{results['aggregate_statistics']['processing_time_stats']['mean']:.2f}s Average Processing Time**: Range from {results['aggregate_statistics']['processing_time_stats']['min']:.2f}s to {results['aggregate_statistics']['processing_time_stats']['max']:.2f}s
- **{results['aggregate_statistics']['content_quality_stats']['mean_quality_score']:.3f} Average Content Quality Score**: On 0-1 scale

## Test Configuration

### Technical Setup
- **Processing Mode**: Pure Chonkie (Docling and OCR bypassed)
- **Strategies Tested**: {', '.join(results['test_metadata']['strategies_tested'])}
- **Documents Tested**: {results['test_metadata']['documents_tested']} from R2207005
- **Total Combinations**: {results['test_metadata']['total_combinations']} (5 docs × 3 strategies)
- **Test Duration**: {results['test_metadata']['total_duration_seconds']:.2f} seconds

### Chonkie Configuration
- **Chunk Size**: {config.CHONKIE_CHUNK_SIZE}
- **Chunk Overlap**: {config.CHONKIE_CHUNK_OVERLAP}
- **Min Text Length**: {config.CHONKIE_MIN_TEXT_LENGTH}
- **PDF Extraction**: PDFplumber: {config.CHONKIE_USE_PDFPLUMBER}, PyPDF2: {config.CHONKIE_USE_PYPDF2}

## Strategy Performance Comparison

"""

    # Add strategy comparison table
    if results['comparative_analysis']:
        report_content += "| Strategy | Success Rate | Avg Time (s) | Avg Chunks | Avg Quality | Performance Score |\n"
        report_content += "|----------|-------------|-------------|------------|-------------|------------------|\n"
        
        # Sort strategies by performance score
        sorted_strategies = sorted(results['comparative_analysis'].items(), 
                                 key=lambda x: x[1]['combined_performance_score'], 
                                 reverse=True)
        
        for strategy, stats in sorted_strategies:
            report_content += f"| {strategy.title()} | {stats['success_rate']:.1f}% | {stats['avg_processing_time']:.2f}s | {stats['avg_chunks_produced']:.1f} | {stats['avg_quality_score']:.3f} | {stats['combined_performance_score']:.3f} |\n"

    report_content += f"""

## Individual Document Analysis

"""

    # Add individual document results
    for strategy in results['strategy_results']:
        report_content += f"### {strategy.title()} Strategy Results\n\n"
        
        for result in results['strategy_results'][strategy]:
            doc_info = result['document_info']
            success_icon = "✅" if result['efficiency_metrics']['processing_success'] else "❌"
            
            report_content += f"#### {success_icon} {doc_info['title']}\n"
            report_content += f"- **Processing Time**: {result['processing_time_seconds']:.2f}s\n"
            report_content += f"- **Chunks Extracted**: {result['content_analysis']['total_chunks']}\n"
            report_content += f"- **Text Length**: {result['content_analysis']['total_text_length']:,} characters\n"
            report_content += f"- **Quality Score**: {result['content_analysis']['content_quality_score']:.3f}/1.0\n"
            report_content += f"- **Avg Chunk Length**: {result['content_analysis']['avg_chunk_length']:.1f} chars\n"
            
            if result['processing_errors']:
                report_content += f"- **Errors**: {', '.join(result['processing_errors'])}\n"
            
            # Sample content
            if result['content_analysis']['sample_content']:
                sample = result['content_analysis']['sample_content'][:200] + "..."
                report_content += f"- **Sample Content**: {sample}\n"
                
            report_content += "\n"

    report_content += f"""

## Performance Analysis

### Processing Time Analysis
| Metric | Value | Assessment |
|--------|-------|------------|
| Mean Processing Time | {results['aggregate_statistics']['processing_time_stats']['mean']:.2f}s | Comparison with Docling needed |
| Median Processing Time | {results['aggregate_statistics']['processing_time_stats']['median']:.2f}s | Consistent performance |
| Fastest Processing | {results['aggregate_statistics']['processing_time_stats']['min']:.2f}s | Best case scenario |
| Slowest Processing | {results['aggregate_statistics']['processing_time_stats']['max']:.2f}s | Worst case scenario |
| Standard Deviation | {results['aggregate_statistics']['processing_time_stats']['std_dev']:.2f}s | Performance variability |

### Content Extraction Performance  
| Metric | Value | Assessment |
|--------|-------|------------|
| Total Chunks Extracted | {results['aggregate_statistics']['chunk_extraction_stats']['total_chunks']} | Total across all combinations |
| Mean Chunks per Combination | {results['aggregate_statistics']['chunk_extraction_stats']['mean_chunks_per_combination']:.1f} | Average chunking effectiveness |
| Text Extraction Rate | {results['aggregate_statistics']['text_extraction_stats']['mean_chars_per_document']:,.0f} chars/doc | Character extraction efficiency |
| Overall Success Rate | {results['aggregate_statistics']['overall_success_rate']:.1f}% | Processing reliability |

## Strategy Recommendations

Based on the performance analysis:

"""

    # Add strategy recommendations
    if results['comparative_analysis']:
        sorted_strategies = sorted(results['comparative_analysis'].items(), 
                                 key=lambda x: x[1]['combined_performance_score'], 
                                 reverse=True)
        
        best_strategy = sorted_strategies[0]
        report_content += f"1. **Recommended Primary Strategy**: {best_strategy[0].title()}\n"
        report_content += f"   - Success Rate: {best_strategy[1]['success_rate']:.1f}%\n"
        report_content += f"   - Average Processing Time: {best_strategy[1]['avg_processing_time']:.2f}s\n"
        report_content += f"   - Quality Score: {best_strategy[1]['avg_quality_score']:.3f}\n\n"
        
        if len(sorted_strategies) > 1:
            fallback_strategy = sorted_strategies[1] 
            report_content += f"2. **Recommended Fallback Strategy**: {fallback_strategy[0].title()}\n"
            report_content += f"   - Use when primary strategy fails\n"
            report_content += f"   - Success Rate: {fallback_strategy[1]['success_rate']:.1f}%\n\n"

    report_content += f"""

## Comparison with Docling Analysis

This analysis processes the identical 5 documents from R2207005 analyzed in the Docling performance study, enabling direct head-to-head comparison:

### Document Matching
- ✅ Same 5 PDF documents from R2207005
- ✅ Same document URLs and identifiers  
- ✅ Same complexity classifications
- ✅ Same performance metrics collected

### Key Comparison Points
1. **Processing Speed**: Chonkie avg {results['aggregate_statistics']['processing_time_stats']['mean']:.2f}s vs Docling avg 18.56s
2. **Success Rate**: Chonkie {results['aggregate_statistics']['overall_success_rate']:.1f}% vs Docling 100%
3. **Content Quality**: Chonkie avg {results['aggregate_statistics']['content_quality_stats']['mean_quality_score']:.3f} vs Docling avg 0.62
4. **Chunk Production**: Compare chunking strategies and effectiveness
5. **Structure Preservation**: Evaluate text vs structured content extraction

## Technical Recommendations

### Production Deployment
1. **Hybrid Strategy**: Use best-performing Chonkie strategy as Docling fallback
2. **Document Classification**: Route simple documents to Chonkie, complex to Docling
3. **Performance Optimization**: Configure chunk sizes based on document type
4. **Quality Assurance**: Monitor content quality scores for different strategies

### Future Analysis
1. **Extended Testing**: Test on larger document sets across multiple proceedings
2. **Content Quality Deep Dive**: Detailed comparison of extracted content accuracy
3. **Performance Optimization**: Fine-tune chunk sizes and overlap for optimal results
4. **Integration Testing**: Test Chonkie as seamless Docling fallback

---
*Technical Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}*  
*Analysis Duration: {results['test_metadata']['total_duration_seconds']:.1f} seconds*  
*Combinations Processed: {results['test_metadata']['total_combinations']} (5 documents × 3 strategies)*
"""

    # Write report
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Summary report generated: {report_file}")

if __name__ == "__main__":
    logger.info("Starting Chonkie Performance Analysis")
    logger.info("=" * 60)
    
    try:
        results = run_chonkie_analysis()
        logger.info("\n✅ Analysis completed successfully!")
        logger.info(f"Check the generated files for detailed results and comparison data.")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        exit(1)