#!/usr/bin/env python3
"""
Enhanced Data Processing with Date-Aware Chunking

This module integrates the DateAwareChunker with the existing data processing
pipeline to add temporal metadata to document chunks for timeline building.

Features:
- Integrates DateAwareChunker with existing processing
- Adds temporal metadata to LangChain Document objects
- Maintains compatibility with current embedding pipeline
- Enables timeline-based search and retrieval

Author: Claude Code
"""

import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from date_aware_chunker import DateAwareChunker, DateAwareChunk, ExtractedDate

logger = logging.getLogger(__name__)

class EnhancedChunkProcessor:
    """Enhanced processor that adds temporal metadata to document chunks."""
    
    def __init__(self, chunker_type: str = "recursive", enable_timeline_metadata: bool = True):
        """
        Initialize the enhanced chunk processor.
        
        Args:
            chunker_type: Type of Chonkie chunker to use
            enable_timeline_metadata: Whether to add timeline metadata to chunks
        """
        self.chunker_type = chunker_type
        self.enable_timeline_metadata = enable_timeline_metadata
        self.date_chunker = DateAwareChunker(chunker_type=chunker_type) if enable_timeline_metadata else None
    
    def process_text_with_temporal_metadata(self, text: str, base_metadata: Dict[str, Any] = None) -> List[Document]:
        """
        Process text into Document objects with enhanced temporal metadata.
        
        Args:
            text: Input text to process
            base_metadata: Base metadata to include in all chunks (source, proceeding, etc.)
            
        Returns:
            List of LangChain Document objects with temporal metadata
        """
        if base_metadata is None:
            base_metadata = {}
        
        documents = []
        
        if self.enable_timeline_metadata and self.date_chunker:
            # Use date-aware chunking
            date_chunks = self.date_chunker.chunk_with_dates(text)
            
            for chunk in date_chunks:
                # Create enhanced metadata
                enhanced_metadata = base_metadata.copy()
                
                # Add temporal metadata
                enhanced_metadata.update(self._create_temporal_metadata(chunk))
                
                # Add standard Chonkie metadata
                enhanced_metadata.update({
                    'chunk_start_index': chunk.start_index,
                    'chunk_end_index': chunk.end_index,
                    'chunk_token_count': chunk.token_count,
                    'chunk_level': chunk.level,
                    'temporal_significance': chunk.temporal_significance,
                    'chronological_order': chunk.chronological_order
                })
                
                # Create Document object
                doc = Document(
                    page_content=chunk.text,
                    metadata=enhanced_metadata
                )
                
                documents.append(doc)
        
        else:
            # Fallback to simple chunking without temporal metadata
            documents = self._simple_chunk_without_dates(text, base_metadata)
        
        return documents
    
    def _create_temporal_metadata(self, chunk: DateAwareChunk) -> Dict[str, Any]:
        """Create temporal metadata dictionary from a DateAwareChunk."""
        metadata = {}
        
        # Primary date information
        if chunk.primary_date:
            metadata['primary_date_text'] = chunk.primary_date.text
            metadata['primary_date_type'] = chunk.primary_date.date_type.value
            metadata['primary_date_confidence'] = chunk.primary_date.confidence
            
            if chunk.primary_date.parsed_date:
                metadata['primary_date_parsed'] = chunk.primary_date.parsed_date.isoformat()
                metadata['primary_date_year'] = chunk.primary_date.parsed_date.year
                metadata['primary_date_month'] = chunk.primary_date.parsed_date.month
                metadata['primary_date_day'] = chunk.primary_date.parsed_date.day
            
            if chunk.primary_date.action_verb:
                metadata['primary_date_action'] = chunk.primary_date.action_verb
            if chunk.primary_date.document_reference:
                metadata['primary_date_document'] = chunk.primary_date.document_reference
        
        # All extracted dates
        if chunk.extracted_dates:
            metadata['extracted_dates_count'] = len(chunk.extracted_dates)
            metadata['extracted_dates_texts'] = [d.text for d in chunk.extracted_dates]
            metadata['extracted_dates_types'] = [d.date_type.value for d in chunk.extracted_dates]
            metadata['extracted_dates_confidences'] = [d.confidence for d in chunk.extracted_dates]
            
            # Parsed dates for timeline construction
            parsed_dates = [d.parsed_date.isoformat() for d in chunk.extracted_dates if d.parsed_date]
            if parsed_dates:
                metadata['extracted_dates_parsed'] = parsed_dates
                metadata['earliest_date'] = min(parsed_dates)
                metadata['latest_date'] = max(parsed_dates)
        
        # Document structure indicators
        metadata['contains_decision'] = chunk.contains_decision
        metadata['contains_resolution'] = chunk.contains_resolution
        metadata['contains_rulemaking'] = chunk.contains_rulemaking
        metadata['procedural_significance'] = chunk.procedural_significance
        
        # Date type categorization for filtering
        date_categories = {
            'filing_dates': [],
            'deadline_dates': [],
            'procedural_dates': [],
            'document_references': []
        }
        
        for date_obj in chunk.extracted_dates:
            if date_obj.date_type.value in ['filing_date', 'issue_date', 'adoption_date']:
                date_categories['filing_dates'].append(date_obj.text)
            elif date_obj.date_type.value in ['deadline', 'comment_deadline', 'implementation_date', 'compliance_date']:
                date_categories['deadline_dates'].append(date_obj.text)
            elif date_obj.date_type.value in ['workshop_date', 'hearing_date', 'effective_date']:
                date_categories['procedural_dates'].append(date_obj.text)
            elif date_obj.date_type.value in ['decision_reference', 'resolution_reference', 'rulemaking_reference']:
                date_categories['document_references'].append(date_obj.text)
        
        # Add non-empty categories
        for category, dates in date_categories.items():
            if dates:
                metadata[category] = dates
        
        return metadata
    
    def _simple_chunk_without_dates(self, text: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """Fallback simple chunking without date extraction."""
        # Simple sentence-based chunking
        sentences = text.split('. ')
        documents = []
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                chunk_text = sentence.strip() + ('.' if not sentence.endswith('.') else '')
                
                # Create metadata
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'simple_chunking': True,
                    'temporal_metadata_enabled': False
                })
                
                doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
                documents.append(doc)
        
        return documents
    
    def build_timeline_from_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Build a timeline from documents with temporal metadata.
        
        Args:
            documents: List of Document objects with temporal metadata
            
        Returns:
            Timeline data structure
        """
        timeline_events = []
        
        for doc in documents:
            metadata = doc.metadata
            
            # Extract timeline events from metadata
            if 'primary_date_parsed' in metadata:
                event = {
                    'date': metadata['primary_date_parsed'],
                    'date_text': metadata['primary_date_text'],
                    'type': metadata['primary_date_type'],
                    'confidence': metadata['primary_date_confidence'],
                    'chunk_text': doc.page_content[:150] + '...' if len(doc.page_content) > 150 else doc.page_content,
                    'source': metadata.get('source', 'Unknown'),
                    'proceeding': metadata.get('proceeding', 'Unknown'),
                    'page': metadata.get('page', 'Unknown')
                }
                
                if 'primary_date_action' in metadata:
                    event['action'] = metadata['primary_date_action']
                if 'primary_date_document' in metadata:
                    event['document_reference'] = metadata['primary_date_document']
                
                timeline_events.append(event)
            
            # Also include all extracted dates for comprehensive timeline
            if 'extracted_dates_parsed' in metadata:
                for i, date_str in enumerate(metadata['extracted_dates_parsed']):
                    if i == 0:  # Skip primary date (already added)
                        continue
                    
                    texts = metadata.get('extracted_dates_texts', [])
                    types = metadata.get('extracted_dates_types', [])
                    confidences = metadata.get('extracted_dates_confidences', [])
                    
                    if i < len(texts) and i < len(types) and i < len(confidences):
                        event = {
                            'date': date_str,
                            'date_text': texts[i],
                            'type': types[i],
                            'confidence': confidences[i],
                            'chunk_text': doc.page_content[:150] + '...' if len(doc.page_content) > 150 else doc.page_content,
                            'source': metadata.get('source', 'Unknown'),
                            'proceeding': metadata.get('proceeding', 'Unknown'),
                            'page': metadata.get('page', 'Unknown')
                        }
                        timeline_events.append(event)
        
        # Sort by date
        timeline_events.sort(key=lambda x: x['date'])
        
        # Generate statistics
        stats = {
            'total_events': len(timeline_events),
            'unique_sources': len(set(e['source'] for e in timeline_events)),
            'date_range': None,
            'type_distribution': {}
        }
        
        if timeline_events:
            stats['date_range'] = {
                'start': timeline_events[0]['date'],
                'end': timeline_events[-1]['date']
            }
        
        # Count event types
        for event in timeline_events:
            event_type = event['type']
            stats['type_distribution'][event_type] = stats['type_distribution'].get(event_type, 0) + 1
        
        return {
            'events': timeline_events,
            'statistics': stats
        }

def enhance_existing_processing_with_dates(pdf_url: str, document_title: str = None, 
                                         proceeding: str = None, enable_timeline: bool = True) -> List[Document]:
    """
    Enhanced version of existing processing that adds temporal metadata.
    
    This function can be used as a drop-in replacement for existing processing
    functions to add timeline capabilities.
    
    Args:
        pdf_url: URL to the PDF document
        document_title: Title of the document
        proceeding: Proceeding number
        enable_timeline: Whether to enable timeline metadata extraction
        
    Returns:
        List of Document objects with enhanced temporal metadata
    """
    # Import existing processing function
    try:
        from data_processing import _process_with_hybrid_evaluation
        
        # Process using existing pipeline
        base_documents = _process_with_hybrid_evaluation(
            pdf_url=pdf_url,
            document_title=document_title, 
            proceeding=proceeding
        )
        
        if not base_documents or not enable_timeline:
            return base_documents
        
        # Enhance with temporal metadata
        enhanced_processor = EnhancedChunkProcessor(enable_timeline_metadata=True)
        enhanced_documents = []
        
        for doc in base_documents:
            # Get base metadata from original document
            base_metadata = doc.metadata.copy()
            
            # Process the text with temporal enhancement
            temporal_docs = enhanced_processor.process_text_with_temporal_metadata(
                text=doc.page_content,
                base_metadata=base_metadata
            )
            
            enhanced_documents.extend(temporal_docs)
        
        logger.info(f"Enhanced {len(base_documents)} base documents into {len(enhanced_documents)} temporal documents")
        return enhanced_documents
        
    except ImportError as e:
        logger.error(f"Could not import existing processing: {e}")
        return []
    except Exception as e:
        logger.error(f"Error in enhanced processing: {e}")
        return []

# Example usage and testing
if __name__ == "__main__":
    print("🧪 TESTING ENHANCED DATA PROCESSING WITH TEMPORAL METADATA")
    print("=" * 80)
    
    # Sample CPUC document text
    sample_text = """
    BEFORE THE PUBLIC UTILITIES COMMISSION OF THE STATE OF CALIFORNIA
    
    DECISION 23-06-019
    June 15, 2023
    
    This decision addresses the rulemaking R.22-07-005, which was filed on July 14, 2022.
    Initial comments were due on August 15, 2022, and a workshop was scheduled for 
    October 12, 2022. The proposed decision was issued on March 30, 2023.
    
    This proceeding builds upon Decision 20-12-042, issued on December 17, 2020,
    which established the framework for demand response programs.
    
    The effective date of this decision is July 1, 2023, with implementation
    required by December 31, 2025.
    """
    
    # Test enhanced processing
    processor = EnhancedChunkProcessor(enable_timeline_metadata=True)
    
    base_metadata = {
        'source': 'Sample CPUC Decision',
        'proceeding': 'R.22-07-005',
        'page': 1,
        'document_type': 'decision'
    }
    
    documents = processor.process_text_with_temporal_metadata(sample_text, base_metadata)
    
    print(f"📊 Generated {len(documents)} enhanced documents:")
    
    for i, doc in enumerate(documents):
        print(f"\n📄 Document {i+1}:")
        print(f"   📝 Content: {doc.page_content[:100]}...")
        
        # Show temporal metadata
        temporal_fields = [k for k in doc.metadata.keys() if any(term in k.lower() 
                          for term in ['date', 'temporal', 'chronological'])]
        
        if temporal_fields:
            print(f"   📅 Temporal metadata:")
            for field in temporal_fields[:5]:  # Show first 5 temporal fields
                print(f"      • {field}: {doc.metadata[field]}")
        
        # Show document structure metadata
        structure_fields = ['contains_decision', 'contains_resolution', 'contains_rulemaking']
        structure_data = {k: doc.metadata[k] for k in structure_fields if k in doc.metadata and doc.metadata[k]}
        if structure_data:
            print(f"   🏛️  Document structure: {structure_data}")
    
    # Build timeline
    timeline = processor.build_timeline_from_documents(documents)
    
    print(f"\n📈 TIMELINE ANALYSIS:")
    print(f"   📊 Total events: {timeline['statistics']['total_events']}")
    print(f"   📅 Date range: {timeline['statistics']['date_range']}")
    print(f"   📋 Event types: {list(timeline['statistics']['type_distribution'].keys())}")
    
    print(f"\n🎯 Timeline events:")
    for event in timeline['events'][:5]:  # Show first 5 events
        print(f"   📅 {event['date']}: {event['date_text']} ({event['type']})")
        if 'action' in event:
            print(f"      Action: {event['action']}")
        if 'document_reference' in event:
            print(f"      Document: {event['document_reference']}")
        print(f"      Context: {event['chunk_text'][:80]}...")
    
    print(f"\n✅ Enhanced data processing with temporal metadata is working!")
    print(f"🚀 Ready for integration with existing pipeline!")