#!/usr/bin/env python3
"""
Investigation of Chonkie's date extraction capabilities for timeline building.

This script explores how we can use Chonkie to extract dates from PDFs
to build accurate timelines for CPUC proceedings.
"""

import sys
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config

def investigate_chonkie_date_features():
    """Investigate Chonkie's built-in date extraction capabilities."""
    print("🔍 INVESTIGATING CHONKIE DATE EXTRACTION CAPABILITIES")
    print("=" * 70)
    
    try:
        # Import Chonkie and explore its features
        from chonkie import TokenChunker, RecursiveChunker, SemanticChunker
        
        print("✅ Chonkie imported successfully")
        print(f"📦 Available chunkers: TokenChunker, RecursiveChunker, SemanticChunker")
        
        # Check if Chonkie has any built-in date extraction features
        chunkers_to_test = [TokenChunker, RecursiveChunker, SemanticChunker]
        
        for chunker_class in chunkers_to_test:
            print(f"\n🔧 Investigating {chunker_class.__name__}:")
            
            # Check class attributes and methods
            attributes = [attr for attr in dir(chunker_class) if not attr.startswith('_')]
            date_related = [attr for attr in attributes if any(term in attr.lower() 
                          for term in ['date', 'time', 'temporal', 'timeline', 'extract'])]
            
            if date_related:
                print(f"   📅 Date-related attributes: {date_related}")
            else:
                print(f"   ❌ No obvious date-related attributes found")
            
            # Check documentation or help
            try:
                doc = chunker_class.__doc__
                if doc and any(term in doc.lower() for term in ['date', 'time', 'temporal']):
                    print(f"   📝 Documentation mentions date/time features")
                else:
                    print(f"   📝 No date/time features mentioned in documentation")
            except:
                pass
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Chonkie: {e}")
        return False

def test_chonkie_with_date_content():
    """Test Chonkie chunking with date-rich content to see what metadata is preserved."""
    print("\n🧪 TESTING CHONKIE WITH DATE-RICH CONTENT")
    print("=" * 70)
    
    # Sample CPUC-style text with various date formats
    sample_text = """
    BEFORE THE PUBLIC UTILITIES COMMISSION OF THE STATE OF CALIFORNIA
    
    ORDER INSTITUTING RULEMAKING
    Rulemaking 22-07-005
    (Filed July 14, 2022)
    
    DECISION 23-06-019
    June 15, 2023
    
    PROCEDURAL SCHEDULE:
    - Initial comments due: August 15, 2022
    - Reply comments due: September 1, 2022
    - Workshop scheduled for: October 12, 2022
    - Proposed decision issued: March 30, 2023
    - Final decision adopted: June 15, 2023
    
    The Commission issued Resolution E-5252 on December 8, 2021, which established
    preliminary guidelines. Subsequently, on January 25, 2022, the Commission
    held a public workshop to discuss implementation details.
    
    This proceeding addresses issues related to demand response programs that
    were first identified in Decision 20-12-042, issued on December 17, 2020.
    The timeline for implementation spans from Q1 2023 through Q4 2025.
    
    Effective Date: July 1, 2023
    Implementation Deadline: December 31, 2025
    """
    
    try:
        from chonkie import RecursiveChunker
        
        # Test recursive chunker with date content
        chunker = RecursiveChunker()
        chunks = chunker.chunk(sample_text)
        
        print(f"📊 Generated {len(chunks)} chunks from date-rich text")
        
        # Analyze what information Chonkie preserves
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\n🔍 Chunk {i+1}:")
            print(f"   📝 Text: {chunk.text[:100]}...")
            
            # Check if chunk has any metadata
            if hasattr(chunk, '__dict__'):
                metadata = {k: v for k, v in chunk.__dict__.items() if k != 'text'}
                if metadata:
                    print(f"   📋 Metadata: {metadata}")
                else:
                    print(f"   📋 No additional metadata found")
            
            # Check for character positions (useful for date context)
            if hasattr(chunk, 'start_index') and hasattr(chunk, 'end_index'):
                print(f"   📍 Position: {chunk.start_index}-{chunk.end_index}")
            
        return chunks
        
    except Exception as e:
        print(f"❌ Failed to test Chonkie with date content: {e}")
        return None

def extract_dates_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract dates from text using regex patterns common in CPUC documents."""
    print("\n📅 EXTRACTING DATES WITH REGEX PATTERNS")
    print("=" * 70)
    
    date_patterns = [
        # Full date formats
        (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', 'full_month_name'),
        (r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},\s+\d{4}\b', 'abbreviated_month'),
        (r'\b\d{1,2}/\d{1,2}/\d{4}\b', 'mm_dd_yyyy'),
        (r'\b\d{1,2}-\d{1,2}-\d{4}\b', 'mm-dd-yyyy'),
        (r'\b\d{4}-\d{1,2}-\d{1,2}\b', 'yyyy-mm-dd'),
        
        # Decision and resolution patterns
        (r'Decision\s+\d{2}-\d{2}-\d{3}', 'decision_id'),
        (r'Resolution\s+[A-Z]-\d{4}', 'resolution_id'),
        (r'Rulemaking\s+\d{2}-\d{2}-\d{3}', 'rulemaking_id'),
        
        # Relative dates and quarters
        (r'\bQ[1-4]\s+\d{4}\b', 'quarter'),
        (r'\b\d{4}\s+Q[1-4]\b', 'year_quarter'),
        
        # Context-specific dates
        (r'(?:filed|issued|adopted|effective)\s+(?:on\s+)?([^,\n.]+(?:\d{4}))', 'action_date'),
        (r'(?:due|deadline|scheduled)\s+(?:for\s+)?([^,\n.]+(?:\d{4}))', 'deadline_date'),
    ]
    
    extracted_dates = []
    
    for pattern, date_type in date_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            extracted_dates.append({
                'text': match.group(),
                'type': date_type,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'context': text[max(0, match.start()-50):match.end()+50].strip()
            })
    
    # Sort by position in text
    extracted_dates.sort(key=lambda x: x['start_pos'])
    
    print(f"📊 Found {len(extracted_dates)} date references:")
    for date_info in extracted_dates[:10]:  # Show first 10
        print(f"   📅 '{date_info['text']}' ({date_info['type']}) at pos {date_info['start_pos']}")
        print(f"      Context: ...{date_info['context']}...")
    
    return extracted_dates

def design_chonkie_date_extraction_system():
    """Design a system that combines Chonkie chunking with date extraction."""
    print("\n🎯 DESIGNING CHONKIE + DATE EXTRACTION SYSTEM")
    print("=" * 70)
    
    design = {
        "approach": "Enhanced Chonkie Chunker with Date Metadata",
        "components": [
            {
                "name": "DateAwareChunker",
                "description": "Wrapper around Chonkie that adds date extraction",
                "features": [
                    "Uses Chonkie for intelligent text chunking",
                    "Extracts dates from each chunk using regex patterns",
                    "Adds temporal metadata to chunk objects",
                    "Preserves character positions for context",
                    "Identifies date types (filing, deadline, effective, etc.)"
                ]
            },
            {
                "name": "TemporalMetadata",
                "description": "Enhanced metadata structure for temporal information",
                "fields": [
                    "extracted_dates: List of dates found in chunk",
                    "primary_date: Most significant date in chunk", 
                    "date_types: Types of dates (filing, deadline, etc.)",
                    "temporal_context: Surrounding text for each date",
                    "chronological_order: Sequence position in document"
                ]
            },
            {
                "name": "TimelineBuilder", 
                "description": "Constructs timelines from date-enriched chunks",
                "capabilities": [
                    "Aggregates dates across all chunks",
                    "Identifies key procedural dates",
                    "Builds chronological sequences",
                    "Links dates to document sections",
                    "Generates timeline visualizations"
                ]
            }
        ],
        "integration_points": [
            "Integrates with existing Chonkie-based processing",
            "Compatible with current embedding pipeline", 
            "Adds temporal metadata to vector store",
            "Enables timeline-based search and retrieval"
        ]
    }
    
    for component in design["components"]:
        print(f"\n🔧 {component['name']}:")
        print(f"   📝 {component['description']}")
        
        if 'features' in component:
            print(f"   ✨ Features:")
            for feature in component['features']:
                print(f"      • {feature}")
        
        if 'fields' in component:
            print(f"   📋 Fields:")
            for field in component['fields']:
                print(f"      • {field}")
        
        if 'capabilities' in component:
            print(f"   🚀 Capabilities:")
            for capability in component['capabilities']:
                print(f"      • {capability}")
    
    print(f"\n🔗 Integration:")
    for integration in design["integration_points"]:
        print(f"   • {integration}")
    
    return design

def prototype_date_aware_chunker():
    """Create a prototype of a date-aware chunker using Chonkie."""
    print("\n🛠️ PROTOTYPING DATE-AWARE CHUNKER")
    print("=" * 70)
    
    class DateAwareChunk:
        """Enhanced chunk with date metadata."""
        def __init__(self, text: str, start_index: int = 0, end_index: int = 0):
            self.text = text
            self.start_index = start_index
            self.end_index = end_index
            self.extracted_dates = []
            self.primary_date = None
            self.temporal_context = {}
    
    class DateAwareChunker:
        """Chonkie wrapper with date extraction capabilities."""
        
        def __init__(self, base_chunker=None):
            if base_chunker is None:
                try:
                    from chonkie import RecursiveChunker
                    self.base_chunker = RecursiveChunker()
                except ImportError:
                    print("⚠️ Chonkie not available, using simple text splitting")
                    self.base_chunker = None
        
        def extract_dates_from_chunk(self, text: str, start_pos: int = 0) -> List[Dict]:
            """Extract dates from a text chunk."""
            date_patterns = [
                (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', 'full_date'),
                (r'\b\d{1,2}/\d{1,2}/\d{4}\b', 'short_date'),
                (r'Decision\s+\d{2}-\d{2}-\d{3}', 'decision'),
                (r'Resolution\s+[A-Z]-\d{4}', 'resolution'),
                (r'(?:filed|issued|adopted|effective)\s+(?:on\s+)?([^,\n.]+\d{4})', 'action_date'),
            ]
            
            dates = []
            for pattern, date_type in date_patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    dates.append({
                        'text': match.group(),
                        'type': date_type,
                        'chunk_pos': match.start(),
                        'global_pos': start_pos + match.start(),
                        'context': text[max(0, match.start()-30):match.end()+30].strip()
                    })
            
            return dates
        
        def chunk_with_dates(self, text: str) -> List[DateAwareChunk]:
            """Chunk text and extract dates from each chunk."""
            
            if self.base_chunker:
                # Use Chonkie for intelligent chunking
                base_chunks = self.base_chunker.chunk(text)
                chunks = []
                
                for chunk in base_chunks:
                    # Create enhanced chunk
                    start_idx = getattr(chunk, 'start_index', 0)
                    end_idx = getattr(chunk, 'end_index', len(chunk.text))
                    
                    date_chunk = DateAwareChunk(chunk.text, start_idx, end_idx)
                    
                    # Extract dates from this chunk
                    dates = self.extract_dates_from_chunk(chunk.text, start_idx)
                    date_chunk.extracted_dates = dates
                    
                    # Identify primary date (first significant date)
                    primary_dates = [d for d in dates if d['type'] in ['full_date', 'decision', 'resolution']]
                    if primary_dates:
                        date_chunk.primary_date = primary_dates[0]
                    
                    chunks.append(date_chunk)
                
            else:
                # Fallback: simple sentence-based chunking
                sentences = text.split('. ')
                chunks = []
                pos = 0
                
                for sentence in sentences:
                    if sentence.strip():
                        chunk_text = sentence.strip() + '.'
                        date_chunk = DateAwareChunk(chunk_text, pos, pos + len(chunk_text))
                        
                        # Extract dates
                        dates = self.extract_dates_from_chunk(chunk_text, pos)
                        date_chunk.extracted_dates = dates
                        
                        if dates:
                            date_chunk.primary_date = dates[0]
                        
                        chunks.append(date_chunk)
                        pos += len(chunk_text) + 1
            
            return chunks
    
    # Test the prototype
    sample_text = """
    DECISION 23-06-019 was issued on June 15, 2023, following extensive review.
    The initial rulemaking R.22-07-005 was filed on July 14, 2022.
    Comments were due on August 15, 2022, with a workshop scheduled for October 12, 2022.
    The effective date is July 1, 2023, with implementation required by December 31, 2025.
    """
    
    print("🧪 Testing DateAwareChunker prototype:")
    chunker = DateAwareChunker()
    date_chunks = chunker.chunk_with_dates(sample_text)
    
    print(f"📊 Generated {len(date_chunks)} date-aware chunks:")
    
    for i, chunk in enumerate(date_chunks):
        print(f"\n📄 Chunk {i+1}:")
        print(f"   📝 Text: {chunk.text[:80]}...")
        print(f"   📍 Position: {chunk.start_index}-{chunk.end_index}")
        print(f"   📅 Dates found: {len(chunk.extracted_dates)}")
        
        for date_info in chunk.extracted_dates:
            print(f"      • {date_info['text']} ({date_info['type']})")
        
        if chunk.primary_date:
            print(f"   🎯 Primary date: {chunk.primary_date['text']}")
    
    return date_chunks

def analyze_cpuc_date_patterns():
    """Analyze common date patterns in CPUC documents."""
    print("\n📋 ANALYZING CPUC-SPECIFIC DATE PATTERNS")
    print("=" * 70)
    
    cpuc_patterns = {
        "Procedural Dates": [
            "Filed: [date]",
            "Issued: [date]", 
            "Adopted: [date]",
            "Effective: [date]",
            "Due: [date]",
            "Scheduled for: [date]"
        ],
        "Document References": [
            "Decision XX-XX-XXX",
            "Resolution X-XXXX", 
            "Rulemaking XX-XX-XXX",
            "Advice Letter XXXX-X"
        ],
        "Timeline Markers": [
            "Q1/Q2/Q3/Q4 YYYY",
            "Phase 1/2/3 implementation",
            "Initial/Reply comments",
            "Workshop/Hearing dates"
        ],
        "Deadline Types": [
            "Comment deadlines",
            "Implementation deadlines",
            "Compliance deadlines",
            "Reporting deadlines"
        ]
    }
    
    for category, patterns in cpuc_patterns.items():
        print(f"\n📂 {category}:")
        for pattern in patterns:
            print(f"   • {pattern}")
    
    print(f"\n💡 Key Insights:")
    print(f"   • CPUC documents have highly structured date patterns")
    print(f"   • Dates are often associated with specific actions/events")
    print(f"   • Timeline building requires understanding document structure")
    print(f"   • Context around dates is crucial for interpretation")
    
    return cpuc_patterns

if __name__ == "__main__":
    print("🔍 CHONKIE DATE EXTRACTION INVESTIGATION")
    print("Exploring how to leverage Chonkie for timeline building")
    print("=" * 80)
    
    # Step 1: Investigate Chonkie's built-in capabilities
    chonkie_available = investigate_chonkie_date_features()
    
    if chonkie_available:
        # Step 2: Test with date-rich content
        chunks = test_chonkie_with_date_content()
    
    # Step 3: Show regex-based date extraction
    sample_text = """
    DECISION 23-06-019 was issued on June 15, 2023. The rulemaking R.22-07-005 
    was filed July 14, 2022. Comments due August 15, 2022. Effective July 1, 2023.
    """
    dates = extract_dates_from_text(sample_text)
    
    # Step 4: Design comprehensive system
    design = design_chonkie_date_extraction_system()
    
    # Step 5: Create working prototype
    prototype_chunks = prototype_date_aware_chunker()
    
    # Step 6: Analyze CPUC-specific patterns
    cpuc_patterns = analyze_cpuc_date_patterns()
    
    print(f"\n{'='*80}")
    print("🎯 CONCLUSIONS:")
    print("✅ Chonkie provides excellent chunking but no built-in date extraction")
    print("✅ We can enhance Chonkie with date-aware metadata")
    print("✅ CPUC documents have predictable date patterns")
    print("✅ Timeline building is achievable with enhanced chunking")
    print("🚀 Ready to implement DateAwareChunker for timeline features!")