"""
Timeline Data Processing for CPUC RAG System

This module handles the extraction and processing of timeline data from the vector store,
including event identification, classification, and timeline generation.

Author: Claude Code
"""

import logging
import re
from datetime import datetime
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import json

from timeline_models import (
    TimelineEvent, TimelineData, EventType, EventClassifier, TimelineFilter
)

logger = logging.getLogger(__name__)


class TimelineProcessor:
    """Main processor for timeline data extraction and management"""
    
    def __init__(self, rag_system=None):
        """Initialize timeline processor with RAG system"""
        self.rag_system = rag_system
        self.classifier = EventClassifier()
        self._cache = {}
        self._last_update = None
    
    def extract_timeline_events(self, proceeding_number: str = None, 
                              refresh_cache: bool = False) -> TimelineData:
        """
        Extract timeline events from vector store
        
        Args:
            proceeding_number: Specific proceeding to extract (None for all)
            refresh_cache: Force refresh of cached data
            
        Returns:
            TimelineData object with extracted events
        """
        cache_key = f"timeline_{proceeding_number or 'all'}"
        
        if not refresh_cache and cache_key in self._cache:
            logger.info(f"Using cached timeline data for {proceeding_number or 'all proceedings'}")
            return self._cache[cache_key]
        
        logger.info(f"Extracting timeline events for {proceeding_number or 'all proceedings'}")
        
        if not self.rag_system or not hasattr(self.rag_system, 'vectordb') or self.rag_system.vectordb is None:
            logger.error("No RAG system or vector database available")
            return TimelineData()
        
        # Get all chunks from vector store
        try:
            all_chunks = self.rag_system.vectordb.get()
            logger.info(f"Retrieved {len(all_chunks['ids'])} chunks from vector store")
        except Exception as e:
            logger.error(f"Failed to retrieve chunks from vector store: {e}")
            return TimelineData()
        
        # Process chunks to extract events
        events = []
        processed_documents = set()
        
        for chunk_id, metadata in zip(all_chunks['ids'], all_chunks['metadatas']):
            try:
                # Skip if we've already processed this document
                doc_key = f"{metadata.get('source', '')}-{metadata.get('document_date', '')}"
                if doc_key in processed_documents:
                    continue
                
                # Filter by proceeding if specified
                if proceeding_number and metadata.get('proceeding_number') != proceeding_number:
                    continue
                
                # Extract event from metadata
                event = self._extract_event_from_metadata(metadata)
                if event:
                    events.append(event)
                    processed_documents.add(doc_key)
                    
            except Exception as e:
                logger.warning(f"Error processing chunk {chunk_id}: {e}")
                continue
        
        # Create timeline data
        timeline_data = TimelineData(events=events)
        
        # Cache the result
        self._cache[cache_key] = timeline_data
        self._last_update = datetime.now()
        
        logger.info(f"Extracted {len(events)} timeline events")
        return timeline_data
    
    def _extract_event_from_metadata(self, metadata: Dict) -> Optional[TimelineEvent]:
        """Extract timeline event from chunk metadata"""
        try:
            # Required fields
            source = metadata.get('source', '')
            if not source:
                return None
            
            # Extract date - prefer publication_date over document_date
            publication_date = metadata.get('publication_date')
            document_date = metadata.get('document_date')
            
            if publication_date:
                event_date = datetime.fromisoformat(publication_date)
            elif document_date:
                event_date = datetime.fromisoformat(document_date)
            else:
                # Try to extract date from filename
                event_date = self._extract_date_from_filename(source)
                if not event_date:
                    logger.warning(f"No date found for document: {source}")
                    return None
            
            # Classify event type
            doc_type = metadata.get('document_type', '')
            event_type = self.classifier.classify_document_type(doc_type, source)
            
            # Generate title and description
            title = self._generate_event_title(source, event_type, event_date)
            description = self._generate_event_description(metadata, source, 
                                                         publication_date is not None)
            
            # Calculate importance score
            is_final = 'final' in source.lower() or 'decision' in doc_type.lower()
            has_decision_number = bool(re.search(r'D\d{7}', source))
            importance_score = self.classifier.calculate_importance_score(
                event_type, is_final, has_decision_number
            )
            
            # Extract participants
            participants = self.classifier.extract_participants(source)
            
            # Get source URL
            source_url = metadata.get('source_url', '')
            
            # Extract confidence number if available
            confidence_number = self._extract_confidence_number(source)
            
            # Create event
            event = TimelineEvent(
                event_date=event_date,
                event_type=event_type,
                title=title,
                description=description,
                importance_score=importance_score,
                source_document=source,
                source_url=source_url,
                proceeding_number=metadata.get('proceeding_number', ''),
                participants=participants,
                document_id=metadata.get('chunk_id', ''),
                confidence_number=confidence_number,
                filing_organization=participants[0] if participants else '',
                keywords=self._extract_keywords(source, description)
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Error extracting event from metadata: {e}")
            return None
    
    def _extract_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract date from filename using various patterns"""
        # Common date patterns in CPUC filenames
        patterns = [
            r'(\d{2}_\d{2}_\d{4})',  # MM_DD_YYYY
            r'(\d{1,2}/\d{1,2}/\d{4})',  # M/D/YYYY
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(D\d{4})',  # Decision year (D2024)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                date_str = match.group(1)
                
                try:
                    # Handle MM_DD_YYYY format
                    if '_' in date_str:
                        month, day, year = date_str.split('_')
                        return datetime(int(year), int(month), int(day))
                    
                    # Handle M/D/YYYY format
                    elif '/' in date_str:
                        month, day, year = date_str.split('/')
                        return datetime(int(year), int(month), int(day))
                    
                    # Handle YYYY-MM-DD format
                    elif '-' in date_str:
                        return datetime.fromisoformat(date_str)
                    
                    # Handle decision year (assume middle of year)
                    elif date_str.startswith('D'):
                        year = int(date_str[1:])
                        return datetime(year, 6, 1)
                        
                except (ValueError, IndexError):
                    continue
        
        return None
    
    def _generate_event_title(self, filename: str, event_type: EventType, event_date: datetime) -> str:
        """Generate human-readable title for event"""
        # Extract organization from filename
        org_match = re.search(r'filed by ([^on]+) on', filename, re.IGNORECASE)
        organization = org_match.group(1).strip() if org_match else "Unknown"
        
        # Clean up organization name
        organization = organization.replace('_', ' ').title()
        
        # Format date
        date_str = event_date.strftime("%B %d, %Y")
        
        # Generate title based on event type
        if event_type == EventType.DECISION:
            if 'D' in filename and re.search(r'D\d{7}', filename):
                decision_num = re.search(r'D\d{7}', filename).group()
                return f"Decision {decision_num} - {date_str}"
            else:
                return f"Decision by {organization} - {date_str}"
        
        elif event_type == EventType.RULING:
            return f"Ruling by {organization} - {date_str}"
        
        elif event_type == EventType.COMMENT:
            return f"Comments by {organization} - {date_str}"
        
        elif event_type == EventType.MOTION:
            return f"Motion by {organization} - {date_str}"
        
        elif event_type == EventType.BRIEF:
            return f"Brief by {organization} - {date_str}"
        
        else:
            return f"{event_type.value} by {organization} - {date_str}"
    
    def _generate_event_description(self, metadata: Dict, filename: str, 
                                  has_publication_date: bool = False) -> str:
        """Generate event description from metadata and filename"""
        description_parts = []
        
        # Add document type if available
        doc_type = metadata.get('document_type', '')
        if doc_type:
            description_parts.append(f"Document Type: {doc_type.title()}")
        
        # Add proceeding number
        proceeding = metadata.get('proceeding_number', '')
        if proceeding:
            description_parts.append(f"Proceeding: {proceeding}")
        
        # Add confidence number if available
        conf_number = self._extract_confidence_number(filename)
        if conf_number:
            description_parts.append(f"Confidence #: {conf_number}")
        
        # Add date source information
        if has_publication_date:
            description_parts.append("Date: Official Publication Date")
        else:
            description_parts.append("Date: Extracted from Document")
        
        # Add file source
        description_parts.append(f"Source: {filename}")
        
        return " | ".join(description_parts)
    
    def _extract_confidence_number(self, filename: str) -> str:
        """Extract confidence number from filename"""
        match = re.search(r'Conf#\s*(\d+)', filename, re.IGNORECASE)
        return match.group(1) if match else ""
    
    def _extract_keywords(self, filename: str, description: str) -> List[str]:
        """Extract keywords from filename and description"""
        keywords = []
        
        # Common CPUC keywords
        keyword_patterns = [
            r'microgrid', r'energy storage', r'solar', r'renewable',
            r'rate design', r'tariff', r'demand response', r'grid',
            r'electric vehicle', r'ev', r'transportation electrification',
            r'distributed energy', r'der', r'interconnection',
            r'reliability', r'outage', r'emergency', r'wildfire',
            r'environmental justice', r'low income', r'disadvantaged',
            r'utility', r'investor owned', r'iou', r'community choice'
        ]
        
        text = f"{filename} {description}".lower()
        
        for pattern in keyword_patterns:
            if re.search(pattern, text):
                keywords.append(pattern.replace(r'\\', ''))
        
        return keywords
    
    def get_timeline_summary(self, timeline_data: TimelineData) -> Dict:
        """Generate summary statistics for timeline data"""
        if not timeline_data.events:
            return {"total_events": 0, "date_range": None, "event_types": {}, "participants": {}}
        
        # Event type distribution
        event_types = defaultdict(int)
        for event in timeline_data.events:
            event_types[event.event_type.value] += 1
        
        # Participant distribution
        participants = defaultdict(int)
        for event in timeline_data.events:
            for participant in event.participants:
                participants[participant] += 1
        
        # Importance distribution
        importance_distribution = defaultdict(int)
        for event in timeline_data.events:
            importance_distribution[event.importance_score] += 1
        
        return {
            "total_events": len(timeline_data.events),
            "date_range": timeline_data.date_range,
            "event_types": dict(event_types),
            "participants": dict(participants),
            "importance_distribution": dict(importance_distribution),
            "proceedings": list(timeline_data.proceeding_numbers)
        }
    
    def filter_timeline_events(self, timeline_data: TimelineData, 
                             filter_criteria: TimelineFilter) -> TimelineData:
        """Filter timeline events based on criteria"""
        filtered_events = timeline_data.filter_events(filter_criteria)
        return TimelineData(events=filtered_events)
    
    def get_major_events(self, timeline_data: TimelineData, 
                        min_importance: int = 7) -> List[TimelineEvent]:
        """Get major events above importance threshold"""
        return timeline_data.get_events_by_importance(min_importance, 10)
    
    def get_events_by_date_range(self, timeline_data: TimelineData,
                                start_date: datetime, end_date: datetime) -> List[TimelineEvent]:
        """Get events within specific date range"""
        return timeline_data.get_events_by_date_range(start_date, end_date)
    
    def export_timeline_data(self, timeline_data: TimelineData, 
                           format: str = "json") -> str:
        """Export timeline data in specified format"""
        if format.lower() == "json":
            return json.dumps(timeline_data.to_dict(), indent=2)
        
        elif format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "Date", "Type", "Title", "Description", "Importance",
                "Organization", "Proceeding", "Source URL"
            ])
            
            # Write events
            for event in timeline_data.sort_events():
                writer.writerow([
                    event.event_date.strftime("%Y-%m-%d"),
                    event.event_type.value,
                    event.title,
                    event.description,
                    event.importance_score,
                    event.filing_organization,
                    event.proceeding_number,
                    event.source_url
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_cache(self):
        """Clear cached timeline data"""
        self._cache.clear()
        self._last_update = None
        logger.info("Timeline cache cleared")


def create_timeline_processor(rag_system=None) -> TimelineProcessor:
    """Factory function to create timeline processor"""
    return TimelineProcessor(rag_system=rag_system)