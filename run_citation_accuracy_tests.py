#!/usr/bin/env python3
"""
Citation Accuracy Test Runner

This script provides a command-line interface for running citation accuracy tests
on proceeding R2207005 with various options and configurations.

Usage:
    python run_citation_accuracy_tests.py [options]

Examples:
    # Run all tests
    python run_citation_accuracy_tests.py
    
    # Run only factual questions
    python run_citation_accuracy_tests.py --category factual
    
    # Run simple complexity tests only
    python run_citation_accuracy_tests.py --complexity simple
    
    # Run limited number of tests for quick validation
    python run_citation_accuracy_tests.py --max-queries 3
    
    # Run specific test query
    python run_citation_accuracy_tests.py --single-query "What are the main objectives of proceeding R.22-07-005?"
    
    # Generate comparison report between different test runs
    python run_citation_accuracy_tests.py --compare-reports file1.json file2.json
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.resolve()))

from test_citation_accuracy_r2207005 import CitationAccuracyTester, TestQuery


def run_single_query_test(query: str, proceeding: str = "R2207005"):
    """Run a test on a single query."""
    print(f"🔍 Testing single query: {query}")
    print("-" * 60)
    
    tester = CitationAccuracyTester(proceeding)
    result = tester.test_single_query(query)
    
    # Print results
    if result.get('error'):
        print(f"❌ ERROR: {result['error']}")
        return
    
    print(f"📝 Response: {result['response'][:200]}...")
    print(f"⏱️  Response Time: {result['response_time']:.2f}s")
    print(f"📊 Citations Found: {len(result['citations'])}")
    
    if result['citations']:
        print("\nCitation Details:")
        for i, citation in enumerate(result['citations'], 1):
            print(f"  {i}. {citation['filename']} (page {citation['page']})")
    
    if result['validation_results']:
        print("\nValidation Results:")
        valid_count = sum(1 for v in result['validation_results'] if v['is_valid'])
        accessible_count = sum(1 for v in result['validation_results'] if v['pdf_accessible'])
        match_count = sum(1 for v in result['validation_results'] if v['content_matches'])
        
        print(f"  Valid Citations: {valid_count}/{len(result['validation_results'])}")
        print(f"  Accessible PDFs: {accessible_count}/{len(result['validation_results'])}")
        print(f"  Content Matches: {match_count}/{len(result['validation_results'])}")
        
        # Show failed validations
        failures = [v for v in result['validation_results'] if not v['is_valid']]
        if failures:
            print(f"\nFailures ({len(failures)}):")
            for failure in failures:
                citation = failure['citation']
                print(f"  ❌ {citation['filename']} page {citation['page']}: {failure['error_message']}")


def run_category_tests(categories: Optional[List[str]] = None, 
                      complexities: Optional[List[str]] = None,
                      max_queries: Optional[int] = None,
                      proceeding: str = "R2207005"):
    """Run tests filtered by categories and complexities."""
    print("🧪 Running Citation Accuracy Tests")
    print("=" * 60)
    
    if categories:
        print(f"Categories: {', '.join(categories)}")
    if complexities:
        print(f"Complexities: {', '.join(complexities)}")
    if max_queries:
        print(f"Max Queries: {max_queries}")
    print()
    
    tester = CitationAccuracyTester(proceeding)
    results = tester.run_comprehensive_test(
        categories=categories,
        complexities=complexities, 
        max_queries=max_queries
    )
    
    # Print report
    print(results['report'])
    
    # Save results
    output_path = tester.save_detailed_results(results)
    print(f"\n📄 Results saved to: {output_path}")
    
    return results, output_path


def compare_test_reports(file1: Path, file2: Path):
    """Compare two test reports and show differences."""
    print("📊 Comparing Citation Accuracy Test Reports")
    print("=" * 60)
    
    try:
        with open(file1, 'r') as f:
            report1 = json.load(f)
        with open(file2, 'r') as f:
            report2 = json.load(f)
        
        metrics1 = report1.get('overall_metrics', {})
        metrics2 = report2.get('overall_metrics', {})
        
        print(f"Report 1: {file1}")
        print(f"Report 2: {file2}")
        print()
        
        # Compare key metrics
        comparisons = [
            ('Citation Coverage', 'citation_coverage'),
            ('Citation Accuracy', 'citation_accuracy'),
            ('Citation Precision', 'citation_precision'),
            ('False Citation Rate', 'false_citation_rate')
        ]
        
        print("METRIC COMPARISON:")
        print("-" * 40)
        
        for name, key in comparisons:
            val1 = metrics1.get(key, 0)
            val2 = metrics2.get(key, 0)
            diff = val2 - val1
            
            direction = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            color = "🟢" if (key != 'false_citation_rate' and diff > 0) or (key == 'false_citation_rate' and diff < 0) else "🔴" if diff != 0 else "⚪"
            
            print(f"{name:20} {val1:6.1f}% → {val2:6.1f}% ({diff:+5.1f}%) {direction} {color}")
        
        # Test count comparison
        count1 = metrics1.get('total_responses', 0)
        count2 = metrics2.get('total_responses', 0)
        print(f"\nTest Count: {count1} → {count2}")
        
        # Time comparison if available
        time1 = report1.get('total_test_time', 0)
        time2 = report2.get('total_test_time', 0)
        if time1 and time2:
            print(f"Test Time: {time1:.1f}s → {time2:.1f}s")
    
    except Exception as e:
        print(f"❌ Error comparing reports: {e}")


def list_available_options():
    """List available test categories and complexities."""
    print("📋 Available Test Options")
    print("=" * 40)
    
    tester = CitationAccuracyTester("R2207005")
    
    categories = tester.test_generator.get_query_categories()
    complexities = tester.test_generator.get_query_complexities()
    
    print(f"Categories: {', '.join(categories)}")
    print(f"Complexities: {', '.join(complexities)}")
    print(f"Total Test Queries: {len(tester.test_generator.get_test_queries())}")
    
    print("\nSample Queries by Category:")
    for category in categories:
        queries = tester.test_generator.get_test_queries(category=category)
        sample = queries[0] if queries else None
        if sample:
            print(f"  {category}: {sample.question[:60]}...")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Run citation accuracy tests for CPUC RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Main test options
    parser.add_argument('--proceeding', default='R2207005',
                       help='Proceeding number to test (default: R2207005)')
    
    parser.add_argument('--category', action='append',
                       choices=['factual', 'procedural', 'timeline', 'technical'],
                       help='Test only specific categories (can specify multiple)')
    
    parser.add_argument('--complexity', action='append',
                       choices=['simple', 'medium', 'complex'],
                       help='Test only specific complexity levels (can specify multiple)')
    
    parser.add_argument('--max-queries', type=int,
                       help='Maximum number of queries to test')
    
    # Single query option
    parser.add_argument('--single-query', type=str,
                       help='Test a single specific query')
    
    # Comparison option
    parser.add_argument('--compare-reports', nargs=2, metavar=('FILE1', 'FILE2'),
                       help='Compare two test report JSON files')
    
    # Info options
    parser.add_argument('--list-options', action='store_true',
                       help='List available test categories and complexities')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Handle different modes
    if args.list_options:
        list_available_options()
        return
    
    if args.compare_reports:
        file1, file2 = args.compare_reports
        compare_test_reports(Path(file1), Path(file2))
        return
    
    if args.single_query:
        run_single_query_test(args.single_query, args.proceeding)
        return
    
    # Run comprehensive tests
    try:
        results, output_path = run_category_tests(
            categories=args.category,
            complexities=args.complexity,
            max_queries=args.max_queries,
            proceeding=args.proceeding
        )
        
        # Quick summary for CLI
        metrics = results['overall_metrics']
        print(f"\n🎯 SUMMARY:")
        print(f"   Tested {metrics['total_responses']} queries in {results['total_test_time']:.1f}s")
        print(f"   Citation Coverage: {metrics['citation_coverage']:.1f}%")
        print(f"   Citation Accuracy: {metrics['citation_accuracy']:.1f}%")
        
        # Success/failure indication
        if metrics['citation_accuracy'] >= 90:
            print("   ✅ EXCELLENT citation accuracy!")
        elif metrics['citation_accuracy'] >= 75:
            print("   ⚠️  GOOD citation accuracy, room for improvement")
        else:
            print("   ❌ LOW citation accuracy, needs attention")
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()