# Chonkie Processing Performance Analysis Report
**CPUC Regulatory Document Processing System**

## Executive Summary

This comprehensive analysis evaluated Chonkie processing performance on 5 representative PDF documents from CPUC proceeding R2207005, testing all 3 chunking strategies (recursive, sentence, token) for direct comparison with Docling processing results.

### Key Results
- **100.0% Overall Success Rate**: 15/15 combinations processed successfully
- **1017 Total Chunks Extracted**: Average of 67.8 chunks per combination
- **984,855 Characters Extracted**: Average of 65,657 characters per document
- **2.62s Average Processing Time**: Range from 0.90s to 5.57s
- **1.000 Average Content Quality Score**: On 0-1 scale

## Test Configuration

### Technical Setup
- **Processing Mode**: Pure Chonkie (Docling and OCR bypassed)
- **Strategies Tested**: recursive, sentence, token
- **Documents Tested**: 5 from R2207005
- **Total Combinations**: 15 (5 docs × 3 strategies)
- **Test Duration**: 39.37 seconds

### Chonkie Configuration
- **Chunk Size**: 1000
- **Chunk Overlap**: 100
- **Min Text Length**: 100
- **PDF Extraction**: PDFplumber: True, PyPDF2: True

## Strategy Performance Comparison

| Strategy | Success Rate | Avg Time (s) | Avg Chunks | Avg Quality | Performance Score |
|----------|-------------|-------------|------------|-------------|------------------|
| Token | 100.0% | 2.28s | 66.2 | 1.000 | 0.826 |
| Sentence | 100.0% | 2.77s | 68.4 | 1.000 | 0.804 |
| Recursive | 100.0% | 2.82s | 68.8 | 1.000 | 0.803 |


## Individual Document Analysis

### Recursive Strategy Results

#### ✅ 571985189 - Final Decision D2506047
- **Processing Time**: 1.02s
- **Chunks Extracted**: 17
- **Text Length**: 16,175 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 951.5 chars
- **Sample Content**: ALJ/CS8/nd3 Date of Issuance 7/3/2025
Decision 25-06-047 June 26, 2025
BEFORE THE PUBLIC UTILITIES COMMISSION OF THE STATE OF CALIFORNIA
Order Instituting Rulemaking to
Advance Demand Flexibility Thro...

#### ✅ 566911513 - Joint Compliance Report
- **Processing Time**: 5.57s
- **Chunks Extracted**: 201
- **Text Length**: 191,553 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 953.0 chars
- **Sample Content**: BEFORE THE PUBLIC UTILITIES COMMISSION
OF THE STATE OF CALIFORNIA
FILED
05/27/25
Order Instituting Rulemaking to Advance 04:59 PM
Demand Flexibility Through Electric Rates. R.22-07-005 R2207005
JOINT ...

#### ✅ 566593612 - Proposed Decision
- **Processing Time**: 0.97s
- **Chunks Extracted**: 19
- **Text Length**: 17,661 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 929.5 chars
- **Sample Content**: STATE OF CALIFORNIA GAVIN NEWSOM, Governor
PUBLIC UTILITIES COMMISSION FILED
505 VAN NESS AVENUE
05/20/25
SAN FRANCISCO, CA 94102-3298
10:33 AM
R2207005
May 20, 2025 Agenda ID #23513
Ratesetting
TO PA...

#### ✅ 564706741 - Agenda Decision D2505026
- **Processing Time**: 2.36s
- **Chunks Extracted**: 49
- **Text Length**: 47,423 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 967.8 chars
- **Sample Content**: ALJ/CS8/RM3/jds PROPOSED DECISION Agenda ID #23459
Ratesetting
Decision
BEFORE THE PUBLIC UTILITIES COMMISSION OF THE STATE OF CALIFORNIA
Order Instituting Rulemaking to Advance Demand
Flexibility Thr...

#### ✅ 562527349 - Compensation Decision D2504015
- **Processing Time**: 4.19s
- **Chunks Extracted**: 58
- **Text Length**: 55,473 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 956.4 chars
- **Sample Content**: ALJ/CS8/RM3/avs Date of Issuance 4/7/2025
Decision 25-04-015 April 3, 2025
BEFORE THE PUBLIC UTILITIES COMMISSION OF THE STATE OF CALIFORNIA
Order Instituting Rulemaking to
Advance Demand Flexibility ...

### Sentence Strategy Results

#### ✅ 571985189 - Final Decision D2506047
- **Processing Time**: 1.30s
- **Chunks Extracted**: 17
- **Text Length**: 16,175 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 951.5 chars
- **Sample Content**: ALJ/CS8/nd3 Date of Issuance 7/3/2025
Decision 25-06-047 June 26, 2025
BEFORE THE PUBLIC UTILITIES COMMISSION OF THE STATE OF CALIFORNIA
Order Instituting Rulemaking to
Advance Demand Flexibility Thro...

#### ✅ 566911513 - Joint Compliance Report
- **Processing Time**: 5.38s
- **Chunks Extracted**: 200
- **Text Length**: 191,553 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 957.8 chars
- **Sample Content**: BEFORE THE PUBLIC UTILITIES COMMISSION
OF THE STATE OF CALIFORNIA
FILED
05/27/25
Order Instituting Rulemaking to Advance 04:59 PM
Demand Flexibility Through Electric Rates. R.22-07-005 R2207005
JOINT ...

#### ✅ 566593612 - Proposed Decision
- **Processing Time**: 1.26s
- **Chunks Extracted**: 19
- **Text Length**: 17,661 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 929.5 chars
- **Sample Content**: STATE OF CALIFORNIA GAVIN NEWSOM, Governor
PUBLIC UTILITIES COMMISSION FILED
505 VAN NESS AVENUE
05/20/25
SAN FRANCISCO, CA 94102-3298
10:33 AM
R2207005
May 20, 2025 Agenda ID #23513
Ratesetting
TO PA...

#### ✅ 564706741 - Agenda Decision D2505026
- **Processing Time**: 2.73s
- **Chunks Extracted**: 49
- **Text Length**: 47,423 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 967.8 chars
- **Sample Content**: ALJ/CS8/RM3/jds PROPOSED DECISION Agenda ID #23459
Ratesetting
Decision
BEFORE THE PUBLIC UTILITIES COMMISSION OF THE STATE OF CALIFORNIA
Order Instituting Rulemaking to Advance Demand
Flexibility Thr...

#### ✅ 562527349 - Compensation Decision D2504015
- **Processing Time**: 3.19s
- **Chunks Extracted**: 57
- **Text Length**: 55,473 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 973.2 chars
- **Sample Content**: ALJ/CS8/RM3/avs Date of Issuance 4/7/2025
Decision 25-04-015 April 3, 2025
BEFORE THE PUBLIC UTILITIES COMMISSION OF THE STATE OF CALIFORNIA
Order Instituting Rulemaking to
Advance Demand Flexibility ...

### Token Strategy Results

#### ✅ 571985189 - Final Decision D2506047
- **Processing Time**: 1.22s
- **Chunks Extracted**: 17
- **Text Length**: 16,175 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 951.5 chars
- **Sample Content**: ALJ/CS8/nd3 Date of Issuance 7/3/2025
Decision 25-06-047 June 26, 2025
BEFORE THE PUBLIC UTILITIES COMMISSION OF THE STATE OF CALIFORNIA
Order Instituting Rulemaking to
Advance Demand Flexibility Thro...

#### ✅ 566911513 - Joint Compliance Report
- **Processing Time**: 4.05s
- **Chunks Extracted**: 192
- **Text Length**: 191,553 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 997.7 chars
- **Sample Content**: BEFORE THE PUBLIC UTILITIES COMMISSION
OF THE STATE OF CALIFORNIA
FILED
05/27/25
Order Instituting Rulemaking to Advance 04:59 PM
Demand Flexibility Through Electric Rates. R.22-07-005 R2207005
JOINT ...

#### ✅ 566593612 - Proposed Decision
- **Processing Time**: 0.90s
- **Chunks Extracted**: 18
- **Text Length**: 17,661 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 981.2 chars
- **Sample Content**: STATE OF CALIFORNIA GAVIN NEWSOM, Governor
PUBLIC UTILITIES COMMISSION FILED
505 VAN NESS AVENUE
05/20/25
SAN FRANCISCO, CA 94102-3298
10:33 AM
R2207005
May 20, 2025 Agenda ID #23513
Ratesetting
TO PA...

#### ✅ 564706741 - Agenda Decision D2505026
- **Processing Time**: 2.75s
- **Chunks Extracted**: 48
- **Text Length**: 47,423 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 988.0 chars
- **Sample Content**: ALJ/CS8/RM3/jds PROPOSED DECISION Agenda ID #23459
Ratesetting
Decision
BEFORE THE PUBLIC UTILITIES COMMISSION OF THE STATE OF CALIFORNIA
Order Instituting Rulemaking to Advance Demand
Flexibility Thr...

#### ✅ 562527349 - Compensation Decision D2504015
- **Processing Time**: 2.48s
- **Chunks Extracted**: 56
- **Text Length**: 55,473 characters
- **Quality Score**: 1.000/1.0
- **Avg Chunk Length**: 990.6 chars
- **Sample Content**: ALJ/CS8/RM3/avs Date of Issuance 4/7/2025
Decision 25-04-015 April 3, 2025
BEFORE THE PUBLIC UTILITIES COMMISSION OF THE STATE OF CALIFORNIA
Order Instituting Rulemaking to
Advance Demand Flexibility ...



## Performance Analysis

### Processing Time Analysis
| Metric | Value | Assessment |
|--------|-------|------------|
| Mean Processing Time | 2.62s | Comparison with Docling needed |
| Median Processing Time | 2.48s | Consistent performance |
| Fastest Processing | 0.90s | Best case scenario |
| Slowest Processing | 5.57s | Worst case scenario |
| Standard Deviation | 1.58s | Performance variability |

### Content Extraction Performance  
| Metric | Value | Assessment |
|--------|-------|------------|
| Total Chunks Extracted | 1017 | Total across all combinations |
| Mean Chunks per Combination | 67.8 | Average chunking effectiveness |
| Text Extraction Rate | 65,657 chars/doc | Character extraction efficiency |
| Overall Success Rate | 100.0% | Processing reliability |

## Strategy Recommendations

Based on the performance analysis:

1. **Recommended Primary Strategy**: Token
   - Success Rate: 100.0%
   - Average Processing Time: 2.28s
   - Quality Score: 1.000

2. **Recommended Fallback Strategy**: Sentence
   - Use when primary strategy fails
   - Success Rate: 100.0%



## Comparison with Docling Analysis

This analysis processes the identical 5 documents from R2207005 analyzed in the Docling performance study, enabling direct head-to-head comparison:

### Document Matching
- ✅ Same 5 PDF documents from R2207005
- ✅ Same document URLs and identifiers  
- ✅ Same complexity classifications
- ✅ Same performance metrics collected

### Key Comparison Points
1. **Processing Speed**: Chonkie avg 2.62s vs Docling avg 18.56s
2. **Success Rate**: Chonkie 100.0% vs Docling 100%
3. **Content Quality**: Chonkie avg 1.000 vs Docling avg 0.62
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
*Technical Report Generated: July 25, 2025 at 22:01:44*  
*Analysis Duration: 39.4 seconds*  
*Combinations Processed: 15 (5 documents × 3 strategies)*
