#!/usr/bin/env python3
"""
Date-Aware Chunker for CPUC Timeline Building

This module enhances Chonkie's chunking capabilities with comprehensive
date extraction and temporal metadata for building accurate timelines
from CPUC proceedings.

Features:
- Leverages Chonkie's intelligent text chunking
- Extracts and categorizes dates from each chunk
- Adds rich temporal metadata to chunks
- Supports timeline construction and temporal search
- Handles CPUC-specific document patterns

Author: Claude Code
"""

import re
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class DateType(Enum):
    """Types of dates found in CPUC documents."""
    FILING_DATE = "filing_date"
    ISSUE_DATE = "issue_date" 
    ADOPTION_DATE = "adoption_date"
    EFFECTIVE_DATE = "effective_date"
    DEADLINE = "deadline"
    COMMENT_DEADLINE = "comment_deadline"
    WORKSHOP_DATE = "workshop_date"
    HEARING_DATE = "hearing_date"
    DECISION_REFERENCE = "decision_reference"
    RESOLUTION_REFERENCE = "resolution_reference"
    RULEMAKING_REFERENCE = "rulemaking_reference"
    IMPLEMENTATION_DATE = "implementation_date"
    COMPLIANCE_DATE = "compliance_date"
    QUARTER_REFERENCE = "quarter_reference"
    GENERAL_DATE = "general_date"

@dataclass
class ExtractedDate:
    """Represents a date extracted from text with context."""
    text: str                    # Original date text (e.g., "June 15, 2023")
    parsed_date: Optional[date]  # Parsed date object (if parseable)
    date_type: DateType         # Category of date
    confidence: float           # Confidence in extraction (0.0-1.0)
    chunk_position: int         # Position within chunk
    global_position: int        # Position in full document
    context_before: str         # Text before the date
    context_after: str          # Text after the date
    full_context: str          # Full context around date
    action_verb: Optional[str]  # Associated action (filed, issued, etc.)
    document_reference: Optional[str]  # Associated document ID
    
    def __str__(self) -> str:
        return f"{self.text} ({self.date_type.value})"

@dataclass
class DateAwareChunk:
    """Enhanced chunk with temporal metadata."""
    text: str
    start_index: int = 0
    end_index: int = 0
    
    # Original Chonkie metadata
    token_count: int = 0
    level: int = 0
    
    # Enhanced temporal metadata
    extracted_dates: List[ExtractedDate] = field(default_factory=list)
    primary_date: Optional[ExtractedDate] = None
    chronological_order: Optional[int] = None
    temporal_significance: float = 0.0  # How important this chunk is for timeline
    
    # Document structure metadata
    contains_decision: bool = False
    contains_resolution: bool = False
    contains_rulemaking: bool = False
    procedural_significance: float = 0.0
    
    def get_dates_by_type(self, date_type: DateType) -> List[ExtractedDate]:
        """Get all dates of a specific type from this chunk."""
        return [d for d in self.extracted_dates if d.date_type == date_type]
    
    def get_earliest_date(self) -> Optional[ExtractedDate]:
        """Get the earliest date in this chunk."""
        valid_dates = [d for d in self.extracted_dates if d.parsed_date]
        if not valid_dates:
            return None
        return min(valid_dates, key=lambda x: x.parsed_date)
    
    def get_latest_date(self) -> Optional[ExtractedDate]:
        """Get the latest date in this chunk.""" 
        valid_dates = [d for d in self.extracted_dates if d.parsed_date]
        if not valid_dates:
            return None
        return max(valid_dates, key=lambda x: x.parsed_date)

class DateAwareChunker:
    """Enhanced chunker that combines Chonkie's intelligence with date extraction."""
    
    def __init__(self, chunker_type: str = "recursive", enable_date_parsing: bool = True):
        """
        Initialize the date-aware chunker.
        
        Args:
            chunker_type: Type of Chonkie chunker to use ('recursive', 'token', 'semantic')
            enable_date_parsing: Whether to parse dates into datetime objects
        """
        self.chunker_type = chunker_type
        self.enable_date_parsing = enable_date_parsing
        self.base_chunker = self._initialize_base_chunker()
        
        # CPUC-specific date patterns with associated metadata
        self.date_patterns = self._initialize_date_patterns()
        
    def _initialize_base_chunker(self):
        """Initialize the underlying Chonkie chunker."""
        try:
            if self.chunker_type == "recursive":
                from chonkie import RecursiveChunker
                return RecursiveChunker()
            elif self.chunker_type == "token":
                from chonkie import TokenChunker
                return TokenChunker()
            elif self.chunker_type == "semantic":
                from chonkie import SemanticChunker
                return SemanticChunker()
            else:
                logger.warning(f"Unknown chunker type: {self.chunker_type}, using recursive")
                from chonkie import RecursiveChunker
                return RecursiveChunker()
        except ImportError as e:
            logger.warning(f"Chonkie not available: {e}. Using fallback chunker.")
            return None
    
    def _initialize_date_patterns(self) -> List[Tuple[str, DateType, float, Optional[str]]]:
        """Initialize regex patterns for date extraction with metadata."""
        return [
            # High-confidence procedural dates with action verbs
            (r'(?:filed|submitted)\s+(?:on\s+)?(\w+\s+\d{1,2},\s+\d{4})', DateType.FILING_DATE, 0.95, 'filed'),
            (r'(?:issued|published)\s+(?:on\s+)?(\w+\s+\d{1,2},\s+\d{4})', DateType.ISSUE_DATE, 0.95, 'issued'),
            (r'(?:adopted|approved)\s+(?:on\s+)?(\w+\s+\d{1,2},\s+\d{4})', DateType.ADOPTION_DATE, 0.95, 'adopted'),
            (r'(?:effective|takes\s+effect)\s+(?:on\s+)?(\w+\s+\d{1,2},\s+\d{4})', DateType.EFFECTIVE_DATE, 0.95, 'effective'),
            
            # Comment and deadline patterns
            (r'(?:comments?\s+(?:are\s+)?due|deadline\s+for\s+comments?)\s+(?:on\s+)?(\w+\s+\d{1,2},\s+\d{4})', DateType.COMMENT_DEADLINE, 0.9, 'due'),
            (r'(?:reply\s+comments?\s+(?:are\s+)?due)\s+(?:on\s+)?(\w+\s+\d{1,2},\s+\d{4})', DateType.COMMENT_DEADLINE, 0.9, 'due'),
            (r'(?:deadline|due\s+date)\s+(?:is\s+)?(?:on\s+)?(\w+\s+\d{1,2},\s+\d{4})', DateType.DEADLINE, 0.85, 'due'),
            
            # Workshop and hearing dates
            (r'(?:workshop|meeting)\s+(?:is\s+)?(?:scheduled\s+(?:for\s+)?)?(?:on\s+)?(\w+\s+\d{1,2},\s+\d{4})', DateType.WORKSHOP_DATE, 0.9, 'scheduled'),
            (r'(?:hearing|proceeding)\s+(?:is\s+)?(?:scheduled\s+(?:for\s+)?)?(?:on\s+)?(\w+\s+\d{1,2},\s+\d{4})', DateType.HEARING_DATE, 0.9, 'scheduled'),
            
            # Implementation and compliance dates
            (r'(?:implement(?:ation)?|shall\s+be\s+implemented)\s+(?:by\s+)?(?:on\s+)?(\w+\s+\d{1,2},\s+\d{4})', DateType.IMPLEMENTATION_DATE, 0.85, 'implementation'),
            (r'(?:complia(?:nce|nt)\s+(?:with|date)|shall\s+comply)\s+(?:by\s+)?(?:on\s+)?(\w+\s+\d{1,2},\s+\d{4})', DateType.COMPLIANCE_DATE, 0.85, 'compliance'),
            
            # Document references with dates
            (r'(Decision\s+\d{2}-\d{2}-\d{3})', DateType.DECISION_REFERENCE, 0.95, None),
            (r'(Resolution\s+[A-Z]-\d{4})', DateType.RESOLUTION_REFERENCE, 0.95, None),
            (r'(Rulemaking\s+\d{2}-\d{2}-\d{3})', DateType.RULEMAKING_REFERENCE, 0.95, None),
            
            # Quarter references
            (r'(Q[1-4]\s+\d{4})', DateType.QUARTER_REFERENCE, 0.8, None),
            (r'(\d{4}\s+Q[1-4])', DateType.QUARTER_REFERENCE, 0.8, None),
            
            # General date formats (lower confidence)
            (r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})\b', DateType.GENERAL_DATE, 0.7, None),
            (r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},\s+\d{4})\b', DateType.GENERAL_DATE, 0.6, None),
            (r'\b(\d{1,2}/\d{1,2}/\d{4})\b', DateType.GENERAL_DATE, 0.6, None),
            (r'\b(\d{1,2}-\d{1,2}-\d{4})\b', DateType.GENERAL_DATE, 0.6, None),
            (r'\b(\d{4}-\d{1,2}-\d{1,2})\b', DateType.GENERAL_DATE, 0.6, None),
        ]
    
    def _parse_date_string(self, date_str: str) -> Optional[date]:
        """Parse a date string into a date object."""
        if not self.enable_date_parsing:
            return None
            
        # Common date formats in CPUC documents
        formats = [
            '%B %d, %Y',      # January 15, 2023
            '%b %d, %Y',      # Jan 15, 2023
            '%b. %d, %Y',     # Jan. 15, 2023
            '%m/%d/%Y',       # 01/15/2023
            '%m-%d-%Y',       # 01-15-2023
            '%Y-%m-%d',       # 2023-01-15
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt).date()
            except ValueError:
                continue
        
        logger.debug(f"Could not parse date: {date_str}")
        return None
    
    def _extract_dates_from_text(self, text: str, global_offset: int = 0) -> List[ExtractedDate]:
        """Extract dates from text using CPUC-specific patterns."""
        extracted_dates = []
        
        for pattern, date_type, confidence, action_verb in self.date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                date_text = match.group(1) if match.groups() else match.group()
                
                # Extract context around the date
                context_start = max(0, match.start() - 50)
                context_end = min(len(text), match.end() + 50)
                full_context = text[context_start:context_end].strip()
                
                context_before = text[context_start:match.start()].strip()
                context_after = text[match.end():context_end].strip()
                
                # Try to parse the date
                parsed_date = self._parse_date_string(date_text)
                
                # Look for document references in context
                doc_ref = None
                doc_patterns = [
                    r'(Decision\s+\d{2}-\d{2}-\d{3})',
                    r'(Resolution\s+[A-Z]-\d{4})',
                    r'(Rulemaking\s+\d{2}-\d{2}-\d{3})'
                ]
                for doc_pattern in doc_patterns:
                    doc_match = re.search(doc_pattern, full_context, re.IGNORECASE)
                    if doc_match:
                        doc_ref = doc_match.group(1)
                        break
                
                extracted_date = ExtractedDate(
                    text=date_text,
                    parsed_date=parsed_date,
                    date_type=date_type,
                    confidence=confidence,
                    chunk_position=match.start(),
                    global_position=global_offset + match.start(),
                    context_before=context_before,
                    context_after=context_after,
                    full_context=full_context,
                    action_verb=action_verb,
                    document_reference=doc_ref
                )
                
                extracted_dates.append(extracted_date)
        
        # Sort by position and remove duplicates
        extracted_dates.sort(key=lambda x: x.chunk_position)
        
        # Remove overlapping matches (keep highest confidence)
        filtered_dates = []
        for date_obj in extracted_dates:
            overlapping = [d for d in filtered_dates 
                         if abs(d.chunk_position - date_obj.chunk_position) < 10]
            if not overlapping:
                filtered_dates.append(date_obj)
            elif date_obj.confidence > max(d.confidence for d in overlapping):
                # Remove lower confidence overlapping dates
                for d in overlapping:
                    filtered_dates.remove(d)
                filtered_dates.append(date_obj)
        
        return filtered_dates
    
    def _calculate_temporal_significance(self, chunk: DateAwareChunk) -> float:
        """Calculate how temporally significant a chunk is for timeline building."""
        significance = 0.0
        
        # Base significance from number of dates
        significance += len(chunk.extracted_dates) * 0.1
        
        # Higher significance for high-confidence dates
        for date_obj in chunk.extracted_dates:
            significance += date_obj.confidence * 0.2
        
        # Extra significance for procedural dates
        procedural_types = {
            DateType.FILING_DATE, DateType.ISSUE_DATE, DateType.ADOPTION_DATE,
            DateType.EFFECTIVE_DATE, DateType.IMPLEMENTATION_DATE
        }
        
        procedural_dates = [d for d in chunk.extracted_dates if d.date_type in procedural_types]
        significance += len(procedural_dates) * 0.3
        
        # Extra significance for document references
        if chunk.contains_decision or chunk.contains_resolution or chunk.contains_rulemaking:
            significance += 0.2
        
        return min(1.0, significance)  # Cap at 1.0
    
    def _identify_primary_date(self, chunk: DateAwareChunk) -> Optional[ExtractedDate]:
        """Identify the most important date in a chunk."""
        if not chunk.extracted_dates:
            return None
        
        # Priority order for date types
        priority_order = [
            DateType.FILING_DATE,
            DateType.ISSUE_DATE,
            DateType.ADOPTION_DATE,
            DateType.EFFECTIVE_DATE,
            DateType.DECISION_REFERENCE,
            DateType.RESOLUTION_REFERENCE,
            DateType.RULEMAKING_REFERENCE,
            DateType.IMPLEMENTATION_DATE,
            DateType.COMPLIANCE_DATE,
            DateType.COMMENT_DEADLINE,
            DateType.WORKSHOP_DATE,
            DateType.HEARING_DATE,
            DateType.DEADLINE,
            DateType.QUARTER_REFERENCE,
            DateType.GENERAL_DATE
        ]
        
        # Find highest priority date type present
        for date_type in priority_order:
            dates_of_type = [d for d in chunk.extracted_dates if d.date_type == date_type]
            if dates_of_type:
                # Return highest confidence date of this type
                return max(dates_of_type, key=lambda x: x.confidence)
        
        # Fallback to highest confidence date
        return max(chunk.extracted_dates, key=lambda x: x.confidence)
    
    def chunk_with_dates(self, text: str) -> List[DateAwareChunk]:
        """
        Chunk text using Chonkie and enhance with date extraction.
        
        Args:
            text: Input text to chunk and analyze
            
        Returns:
            List of DateAwareChunk objects with temporal metadata
        """
        enhanced_chunks = []
        
        if self.base_chunker:
            # Use Chonkie for intelligent chunking
            try:
                base_chunks = self.base_chunker.chunk(text)
                
                for chunk in base_chunks:
                    # Create enhanced chunk from Chonkie chunk
                    enhanced_chunk = DateAwareChunk(
                        text=chunk.text,
                        start_index=getattr(chunk, 'start_index', 0),
                        end_index=getattr(chunk, 'end_index', len(chunk.text)),
                        token_count=getattr(chunk, 'token_count', len(chunk.text.split())),
                        level=getattr(chunk, 'level', 0)
                    )
                    
                    # Extract dates from this chunk
                    dates = self._extract_dates_from_text(
                        chunk.text, 
                        enhanced_chunk.start_index
                    )
                    enhanced_chunk.extracted_dates = dates
                    
                    # Identify document structure elements
                    chunk_lower = chunk.text.lower()
                    enhanced_chunk.contains_decision = bool(re.search(r'decision\s+\d{2}-\d{2}-\d{3}', chunk_lower))
                    enhanced_chunk.contains_resolution = bool(re.search(r'resolution\s+[a-z]-\d{4}', chunk_lower))
                    enhanced_chunk.contains_rulemaking = bool(re.search(r'rulemaking\s+\d{2}-\d{2}-\d{3}', chunk_lower))
                    
                    # Calculate temporal significance
                    enhanced_chunk.temporal_significance = self._calculate_temporal_significance(enhanced_chunk)
                    
                    # Identify primary date
                    enhanced_chunk.primary_date = self._identify_primary_date(enhanced_chunk)
                    
                    enhanced_chunks.append(enhanced_chunk)
                    
            except Exception as e:
                logger.error(f"Error using Chonkie chunker: {e}")
                # Fall back to simple chunking
                enhanced_chunks = self._fallback_chunk_with_dates(text)
        else:
            # Use fallback chunking
            enhanced_chunks = self._fallback_chunk_with_dates(text)
        
        # Assign chronological order based on primary dates
        self._assign_chronological_order(enhanced_chunks)
        
        return enhanced_chunks
    
    def _fallback_chunk_with_dates(self, text: str) -> List[DateAwareChunk]:
        """Fallback chunking when Chonkie is not available."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_pos = 0
        
        for sentence in sentences:
            if sentence.strip():
                chunk = DateAwareChunk(
                    text=sentence.strip(),
                    start_index=current_pos,
                    end_index=current_pos + len(sentence),
                    token_count=len(sentence.split())
                )
                
                # Extract dates
                dates = self._extract_dates_from_text(sentence, current_pos)
                chunk.extracted_dates = dates
                
                # Calculate significance and identify primary date
                chunk.temporal_significance = self._calculate_temporal_significance(chunk)
                chunk.primary_date = self._identify_primary_date(chunk)
                
                chunks.append(chunk)
                current_pos += len(sentence) + 1
        
        return chunks
    
    def _assign_chronological_order(self, chunks: List[DateAwareChunk]):
        """Assign chronological order to chunks based on their primary dates."""
        # Sort chunks with valid primary dates by date
        dated_chunks = [(chunk, chunk.primary_date.parsed_date) 
                       for chunk in chunks 
                       if chunk.primary_date and chunk.primary_date.parsed_date]
        
        dated_chunks.sort(key=lambda x: x[1])
        
        # Assign chronological order
        for order, (chunk, _) in enumerate(dated_chunks):
            chunk.chronological_order = order
    
    def build_timeline(self, chunks: List[DateAwareChunk]) -> Dict[str, Any]:
        """
        Build a comprehensive timeline from date-aware chunks.
        
        Args:
            chunks: List of DateAwareChunk objects
            
        Returns:
            Dictionary containing timeline data and analysis
        """
        timeline_events = []
        date_type_counts = {}
        
        # Extract all dates from all chunks
        for chunk in chunks:
            for date_obj in chunk.extracted_dates:
                if date_obj.parsed_date:  # Only include parseable dates
                    event = {
                        'date': date_obj.parsed_date,
                        'date_text': date_obj.text,
                        'type': date_obj.date_type.value,
                        'confidence': date_obj.confidence,
                        'action': date_obj.action_verb,
                        'document_ref': date_obj.document_reference,
                        'context': date_obj.full_context,
                        'chunk_text': chunk.text[:200] + '...' if len(chunk.text) > 200 else chunk.text
                    }
                    timeline_events.append(event)
                    
                    # Count date types
                    date_type = date_obj.date_type.value
                    date_type_counts[date_type] = date_type_counts.get(date_type, 0) + 1
        
        # Sort timeline by date
        timeline_events.sort(key=lambda x: x['date'])
        
        # Calculate timeline statistics
        timeline_stats = {
            'total_events': len(timeline_events),
            'date_range': None,
            'most_common_type': None,
            'type_distribution': date_type_counts,
            'high_confidence_events': len([e for e in timeline_events if e['confidence'] > 0.8])
        }
        
        if timeline_events:
            timeline_stats['date_range'] = {
                'start': timeline_events[0]['date'],
                'end': timeline_events[-1]['date']
            }
            
            if date_type_counts:
                timeline_stats['most_common_type'] = max(date_type_counts.items(), key=lambda x: x[1])
        
        return {
            'events': timeline_events,
            'statistics': timeline_stats,
            'chunks_analyzed': len(chunks),
            'temporally_significant_chunks': len([c for c in chunks if c.temporal_significance > 0.5])
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the DateAwareChunker with sample CPUC text
    sample_cpuc_text = """
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
    
    print("🧪 TESTING DATE-AWARE CHUNKER")
    print("=" * 60)
    
    # Initialize chunker
    chunker = DateAwareChunker(chunker_type="recursive")
    
    # Process text
    chunks = chunker.chunk_with_dates(sample_cpuc_text)
    
    print(f"📊 Generated {len(chunks)} date-aware chunks:")
    
    for i, chunk in enumerate(chunks):
        print(f"\n📄 Chunk {i+1}:")
        print(f"   📝 Text: {chunk.text[:100]}...")
        print(f"   📅 Dates: {len(chunk.extracted_dates)}")
        print(f"   🎯 Primary: {chunk.primary_date}")
        print(f"   ⭐ Significance: {chunk.temporal_significance:.2f}")
        
        if chunk.chronological_order is not None:
            print(f"   📈 Chronological order: {chunk.chronological_order}")
        
        for date_obj in chunk.extracted_dates[:3]:  # Show first 3 dates
            print(f"      • {date_obj}")
    
    # Build timeline
    timeline = chunker.build_timeline(chunks)
    
    print(f"\n📈 TIMELINE ANALYSIS:")
    print(f"   📊 Total events: {timeline['statistics']['total_events']}")
    print(f"   📅 Date range: {timeline['statistics']['date_range']}")
    print(f"   🏆 Most common type: {timeline['statistics']['most_common_type']}")
    print(f"   ⭐ High confidence events: {timeline['statistics']['high_confidence_events']}")
    
    print(f"\n🎯 First 5 timeline events:")
    for event in timeline['events'][:5]:
        print(f"   📅 {event['date']}: {event['date_text']} ({event['type']})")
        if event['action']:
            print(f"      Action: {event['action']}")
        if event['document_ref']:
            print(f"      Document: {event['document_ref']}")