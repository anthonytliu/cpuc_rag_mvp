# Citation Accuracy Testing Guide for R2207005

This guide explains how to use the comprehensive citation accuracy test suite to detect and fix citation issues in the CPUC RAG system.

## Overview

The citation accuracy test suite validates that citations in RAG responses map correctly to actual PDF content, detects hallucinated citations, and provides detailed accuracy metrics.

## Test Suite Components

### 1. Main Test Suite (`test_citation_accuracy_r2207005.py`)

The comprehensive test suite that includes:
- **Citation Extraction**: Parses citations from RAG responses using multiple format patterns
- **PDF Content Validation**: Validates citations against actual vectordb content
- **Accuracy Metrics**: Calculates coverage, accuracy, precision, and false citation rates
- **Test Data Generation**: Creates diverse test queries across different categories
- **Automated Reporting**: Generates detailed reports with actionable insights

### 2. Test Runner (`run_citation_accuracy_tests.py`)

Command-line interface for running tests with various options:
- Filter by category (factual, procedural, timeline, technical)
- Filter by complexity (simple, medium, complex)
- Run single queries for debugging
- Compare test reports
- List available options

### 3. Standalone Validator (`test_citation_validation_standalone.py`)

Lightweight validation tool that works without the full RAG system:
- Test citation format parsing
- Validate citations against vectordb
- Analyze sample responses
- Debug citation issues

## Quick Start

### Run All Tests
```bash
python run_citation_accuracy_tests.py
```

### Run Limited Tests (for quick validation)
```bash
python run_citation_accuracy_tests.py --max-queries 5
```

### Run Specific Categories
```bash
# Test only factual questions
python run_citation_accuracy_tests.py --category factual

# Test only simple complexity
python run_citation_accuracy_tests.py --complexity simple

# Test multiple categories
python run_citation_accuracy_tests.py --category factual --category technical
```

### Test Single Query
```bash
python run_citation_accuracy_tests.py --single-query "What are the main objectives of proceeding R.22-07-005?"
```

### Standalone Validation
```bash
# Run all standalone tests
python test_citation_validation_standalone.py

# Test only citation parsing
python test_citation_validation_standalone.py --test-parsing

# Show vectordb information
python test_citation_validation_standalone.py --show-vectordb
```

## Understanding Test Results

### Citation Accuracy Metrics

1. **Citation Coverage**: Percentage of responses that include citations
   - Target: >80% (most responses should have citations)
   - Low coverage suggests prompt issues

2. **Citation Accuracy**: Percentage of citations that map to valid documents
   - Target: >90% (citations should reference real documents)
   - Low accuracy indicates hallucination issues

3. **Citation Precision**: Percentage of citations with matching content
   - Target: >70% (citations should match actual content)
   - Low precision suggests content mismatch

4. **False Citation Rate**: Percentage of citations that are incorrect
   - Target: <10% (minimize incorrect citations)
   - High rate indicates systematic issues

### Report Structure

The test report includes:
- **Overall Metrics**: Summary statistics across all tests
- **Category Breakdown**: Results by question type (factual, procedural, etc.)
- **Failure Analysis**: Common citation failure patterns
- **Recommendations**: Specific actions to improve citation accuracy

## Test Query Categories

### Factual Questions
Test basic fact retrieval with specific citations:
- "What are the main objectives of proceeding R.22-07-005?"
- "What compensation was granted to intervenors?"

### Procedural Questions  
Test procedural information and requirements:
- "What is the current procedural status?"
- "What are the comment filing requirements?"

### Timeline Questions
Test temporal information and deadlines:
- "When was the most recent decision issued?"
- "What is the timeline for compliance filings?"

### Technical Questions
Test complex technical content:
- "What are the technical requirements for demand flexibility?"
- "How do utilities address implementation recommendations?"

## Common Citation Issues

### 1. Hallucinated Citations
**Problem**: Citations reference non-existent documents or pages
**Detection**: Citations not found in vectordb
**Solution**: Improve prompt instructions, add validation

### 2. Wrong Page References
**Problem**: Citations point to incorrect pages
**Detection**: Content mismatch between citation context and page content
**Solution**: Improve page number extraction and validation

### 3. Missing Citations
**Problem**: Responses lack proper citations
**Detection**: Low citation coverage metrics
**Solution**: Update prompts to enforce citations

### 4. Inconsistent Citation Formats
**Problem**: Multiple citation formats used inconsistently
**Detection**: Parse failures in citation extraction
**Solution**: Standardize citation format in prompts

## Debugging Citation Issues

### Step 1: Run Standalone Validation
```bash
python test_citation_validation_standalone.py --test-samples
```
This shows which citations are valid/invalid without running full RAG queries.

### Step 2: Test Single Problematic Query
```bash
python run_citation_accuracy_tests.py --single-query "Your problematic query here"
```
This provides detailed validation for a specific query.

### Step 3: Check Vectordb Content
```bash
python test_citation_validation_standalone.py --show-vectordb
```
This shows available documents and pages in the vectordb.

### Step 4: Analyze Citation Patterns
Review the detailed JSON output to identify:
- Most common citation failures
- Documents with high failure rates
- Patterns in problematic citations

## Integration with Development Workflow

### Regular Testing
Run citation tests regularly during development:
```bash
# Quick check during development
python run_citation_accuracy_tests.py --max-queries 3

# Full validation before releases
python run_citation_accuracy_tests.py
```

### Continuous Monitoring
- Set up automated tests to run citation validation
- Monitor citation accuracy metrics over time
- Alert on significant accuracy degradation

### Comparison Testing
Compare citation accuracy before/after changes:
```bash
# Run baseline test
python run_citation_accuracy_tests.py > results_before.json

# Make changes to system

# Run comparison test
python run_citation_accuracy_tests.py > results_after.json

# Compare results
python run_citation_accuracy_tests.py --compare-reports results_before.json results_after.json
```

## Advanced Usage

### Custom Test Queries
Add custom test queries by modifying the `TestDataGenerator` class in `test_citation_accuracy_r2207005.py`:

```python
TestQuery(
    question="Your custom question here",
    category="factual",  # or procedural, timeline, technical
    expected_doc_types=["DECISION", "RULING"],
    complexity="medium",  # simple, medium, complex
    description="Description of what this tests"
)
```

### Custom Validation Logic
Extend the `PDFContentValidator` class to implement custom validation rules:
- Stricter content matching requirements
- Domain-specific validation logic
- Integration with external validation services

### Performance Optimization
For large-scale testing:
- Use `--max-queries` to limit test scope
- Focus on specific categories that matter most
- Run tests in parallel (modify test runner for parallel execution)

## Troubleshooting

### "VectorDB not available" Error
- Ensure R2207005 LanceDB exists: `local_lance_db/R2207005/`
- Check vectordb connectivity with standalone validator
- Verify proceeding data has been processed

### "RAG system not properly initialized" Error
- Ensure all dependencies are installed
- Check config.py settings
- Verify model availability

### High False Citation Rate
- Review prompt templates for citation instructions
- Check if documents are properly indexed
- Validate document metadata accuracy

### Low Citation Coverage
- Update prompts to require citations
- Ensure sufficient relevant documents exist
- Check if retrieval is finding appropriate content

## Output Files

The test suite generates several output files:

1. **Test Results JSON**: Detailed results with all validation data
   - Format: `citation_accuracy_test_R2207005_YYYYMMDD_HHMMSS.json`
   - Contains: All test data, validation results, metrics

2. **Console Report**: Human-readable summary displayed during test execution
   - Overall metrics
   - Category breakdown
   - Failure analysis
   - Recommendations

3. **Comparison Reports**: When comparing two test runs
   - Metric differences
   - Performance changes
   - Regression analysis

## Best Practices

1. **Start Small**: Begin with limited queries (`--max-queries 3`) for quick iteration
2. **Test Incrementally**: Test individual components before full system validation
3. **Monitor Trends**: Track citation accuracy over time to catch regressions
4. **Focus on Failures**: Prioritize fixing high-impact citation failures
5. **Document Changes**: Record what changes improve citation accuracy
6. **Validate Regularly**: Make citation testing part of your development process

## Future Enhancements

Potential improvements to the citation accuracy test suite:

1. **Real PDF Validation**: Download and validate against actual PDF content
2. **Semantic Matching**: Use embedding similarity for content validation
3. **Multi-Proceeding Support**: Extend testing to other proceedings
4. **Performance Benchmarking**: Track response time impact of citation validation
5. **Interactive Debugging**: Web interface for exploring citation issues
6. **Integration Testing**: Test citation accuracy in end-to-end workflows

---

For questions or issues with the citation accuracy testing suite, refer to the detailed code comments in the test files or create an issue in the project repository.