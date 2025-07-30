"""
Timeline Data Models for CPUC RAG System

This module defines the data structures and models used for timeline functionality,
including event classification, importance scoring, and data validation.

Author: Claude Code
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Set
from enum import Enum
import json


class EventType(Enum):
    """Classification of CPUC proceeding events"""
    DECISION = "Decision"
    RULING = "Ruling"
    PROPOSED_DECISION = "Proposed Decision"
    MOTION = "Motion"
    COMMENT = "Comment"
    REPLY = "Reply"
    BRIEF = "Brief"
    TESTIMONY = "Testimony"
    NOTICE = "Notice"
    APPLICATION = "Application"
    AMENDMENT = "Amendment"
    EXPARTE = "Ex Parte"
    COMPLIANCE = "Compliance Filing"
    RESPONSE = "Response"
    STATEMENT = "Statement"
    REQUEST = "Request"
    OTHER = "Other"


class ImportanceLevel(Enum):
    """Importance levels for timeline events"""
    CRITICAL = 10      # Final Decisions, Major Rulings
    HIGH = 8           # Proposed Decisions, Significant Rulings
    MEDIUM = 6         # Motions, Major Comments, Testimony
    LOW = 4            # Standard Comments, Replies
    INFORMATIONAL = 2  # Administrative filings, Notices


@dataclass
class TimelineEvent:
    """Core timeline event structure"""
    event_date: datetime
    event_type: EventType
    title: str
    description: str
    importance_score: int
    source_document: str
    source_url: str
    proceeding_number: str
    participants: List[str] = field(default_factory=list)
    page_references: List[int] = field(default_factory=list)
    document_id: str = ""
    confidence_number: str = ""
    filing_organization: str = ""
    keywords: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and normalize event data"""
        if not isinstance(self.event_date, datetime):
            raise ValueError("event_date must be a datetime object")
        
        if self.importance_score < 1 or self.importance_score > 10:
            raise ValueError("importance_score must be between 1 and 10")
        
        # Normalize title and description
        self.title = self.title.strip()
        self.description = self.description.strip()
        
        # Extract organization from participants if not set
        if not self.filing_organization and self.participants:
            self.filing_organization = self.participants[0]
    
    def to_dict(self) -> Dict:
        """Convert event to dictionary for JSON serialization"""
        return {
            "event_date": self.event_date.isoformat(),
            "event_type": self.event_type.value,
            "title": self.title,
            "description": self.description,
            "importance_score": self.importance_score,
            "source_document": self.source_document,
            "source_url": self.source_url,
            "proceeding_number": self.proceeding_number,
            "participants": self.participants,
            "page_references": self.page_references,
            "document_id": self.document_id,
            "confidence_number": self.confidence_number,
            "filing_organization": self.filing_organization,
            "keywords": self.keywords
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TimelineEvent':
        """Create event from dictionary"""
        event_data = data.copy()
        event_data['event_date'] = datetime.fromisoformat(event_data['event_date'])
        event_data['event_type'] = EventType(event_data['event_type'])
        return cls(**event_data)


@dataclass
class TimelineFilter:
    """Filter parameters for timeline queries"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    event_types: Set[EventType] = field(default_factory=set)
    min_importance: int = 1
    max_importance: int = 10
    proceeding_numbers: Set[str] = field(default_factory=set)
    participants: Set[str] = field(default_factory=set)
    keywords: Set[str] = field(default_factory=set)
    
    def matches(self, event: TimelineEvent) -> bool:
        """Check if event matches filter criteria"""
        # Date range check
        if self.start_date and event.event_date < self.start_date:
            return False
        if self.end_date and event.event_date > self.end_date:
            return False
        
        # Event type check
        if self.event_types and event.event_type not in self.event_types:
            return False
        
        # Importance check
        if not (self.min_importance <= event.importance_score <= self.max_importance):
            return False
        
        # Proceeding check
        if self.proceeding_numbers and event.proceeding_number not in self.proceeding_numbers:
            return False
        
        # Participants check
        if self.participants:
            if not any(participant in self.participants for participant in event.participants):
                return False
        
        # Keywords check
        if self.keywords:
            event_text = f"{event.title} {event.description}".lower()
            if not any(keyword.lower() in event_text for keyword in self.keywords):
                return False
        
        return True


@dataclass
class TimelineData:
    """Container for timeline data and metadata"""
    events: List[TimelineEvent] = field(default_factory=list)
    proceeding_numbers: Set[str] = field(default_factory=set)
    date_range: tuple = field(default_factory=lambda: (None, None))
    participants: Set[str] = field(default_factory=set)
    event_types: Set[EventType] = field(default_factory=set)
    total_events: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Update metadata based on events"""
        self.update_metadata()
    
    def update_metadata(self):
        """Update metadata from current events"""
        if not self.events:
            return
        
        self.total_events = len(self.events)
        
        # Update proceeding numbers
        self.proceeding_numbers = {event.proceeding_number for event in self.events if event.proceeding_number}
        
        # Update date range
        dates = [event.event_date for event in self.events]
        self.date_range = (min(dates), max(dates))
        
        # Update participants
        self.participants = set()
        for event in self.events:
            self.participants.update(event.participants)
        
        # Update event types
        self.event_types = {event.event_type for event in self.events}
        
        self.last_updated = datetime.now()
    
    def add_event(self, event: TimelineEvent):
        """Add event and update metadata"""
        self.events.append(event)
        self.update_metadata()
    
    def filter_events(self, filter_criteria: TimelineFilter) -> List[TimelineEvent]:
        """Filter events based on criteria"""
        return [event for event in self.events if filter_criteria.matches(event)]
    
    def get_events_by_type(self, event_type: EventType) -> List[TimelineEvent]:
        """Get all events of a specific type"""
        return [event for event in self.events if event.event_type == event_type]
    
    def get_events_by_importance(self, min_importance: int = 1, max_importance: int = 10) -> List[TimelineEvent]:
        """Get events within importance range"""
        return [event for event in self.events if min_importance <= event.importance_score <= max_importance]
    
    def get_events_by_date_range(self, start_date: datetime, end_date: datetime) -> List[TimelineEvent]:
        """Get events within date range"""
        return [event for event in self.events if start_date <= event.event_date <= end_date]
    
    def sort_events(self, by_date: bool = True, ascending: bool = True) -> List[TimelineEvent]:
        """Sort events by date or importance"""
        if by_date:
            return sorted(self.events, key=lambda e: e.event_date, reverse=not ascending)
        else:
            return sorted(self.events, key=lambda e: e.importance_score, reverse=not ascending)
    
    def to_dict(self) -> Dict:
        """Convert timeline data to dictionary"""
        return {
            "events": [event.to_dict() for event in self.events],
            "proceeding_numbers": list(self.proceeding_numbers),
            "date_range": [d.isoformat() if d else None for d in self.date_range],
            "participants": list(self.participants),
            "event_types": [et.value for et in self.event_types],
            "total_events": self.total_events,
            "last_updated": self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TimelineData':
        """Create timeline data from dictionary"""
        events = [TimelineEvent.from_dict(event_data) for event_data in data.get('events', [])]
        timeline_data = cls(events=events)
        timeline_data.last_updated = datetime.fromisoformat(data.get('last_updated', datetime.now().isoformat()))
        return timeline_data


class EventClassifier:
    """Utility class for event classification and scoring"""
    
    # Document type to event type mapping
    TYPE_MAPPING = {
        'decision': EventType.DECISION,
        'ruling': EventType.RULING,
        'proposed decision': EventType.PROPOSED_DECISION,
        'motion': EventType.MOTION,
        'comment': EventType.COMMENT,
        'comments': EventType.COMMENT,
        'reply': EventType.REPLY,
        'brief': EventType.BRIEF,
        'testimony': EventType.TESTIMONY,
        'notice': EventType.NOTICE,
        'application': EventType.APPLICATION,
        'amendment': EventType.AMENDMENT,
        'exparte': EventType.EXPARTE,
        'ex parte': EventType.EXPARTE,
        'compliance': EventType.COMPLIANCE,
        'compliance filing': EventType.COMPLIANCE,
        'response': EventType.RESPONSE,
        'statement': EventType.STATEMENT,
        'request': EventType.REQUEST,
    }
    
    # Importance scoring rules
    IMPORTANCE_RULES = {
        EventType.DECISION: ImportanceLevel.CRITICAL,
        EventType.RULING: ImportanceLevel.HIGH,
        EventType.PROPOSED_DECISION: ImportanceLevel.HIGH,
        EventType.MOTION: ImportanceLevel.MEDIUM,
        EventType.BRIEF: ImportanceLevel.MEDIUM,
        EventType.TESTIMONY: ImportanceLevel.MEDIUM,
        EventType.COMMENT: ImportanceLevel.LOW,
        EventType.REPLY: ImportanceLevel.LOW,
        EventType.NOTICE: ImportanceLevel.INFORMATIONAL,
        EventType.APPLICATION: ImportanceLevel.MEDIUM,
        EventType.AMENDMENT: ImportanceLevel.MEDIUM,
        EventType.EXPARTE: ImportanceLevel.MEDIUM,
        EventType.COMPLIANCE: ImportanceLevel.INFORMATIONAL,
        EventType.RESPONSE: ImportanceLevel.LOW,
        EventType.STATEMENT: ImportanceLevel.INFORMATIONAL,
        EventType.REQUEST: ImportanceLevel.LOW,
        EventType.OTHER: ImportanceLevel.INFORMATIONAL,
    }
    
    @classmethod
    def classify_document_type(cls, document_type: str, filename: str = "") -> EventType:
        """Classify document type into event type"""
        if not document_type:
            document_type = ""
        
        # Check document type first
        doc_type_lower = document_type.lower().strip()
        if doc_type_lower in cls.TYPE_MAPPING:
            return cls.TYPE_MAPPING[doc_type_lower]
        
        # Check filename for type hints
        filename_lower = filename.lower()
        for type_key, event_type in cls.TYPE_MAPPING.items():
            if type_key in filename_lower:
                return event_type
        
        return EventType.OTHER
    
    @classmethod
    def calculate_importance_score(cls, event_type: EventType, is_final: bool = False, 
                                 has_decision_number: bool = False) -> int:
        """Calculate importance score for event"""
        base_score = cls.IMPORTANCE_RULES.get(event_type, ImportanceLevel.INFORMATIONAL).value
        
        # Boost for final decisions
        if is_final and event_type == EventType.DECISION:
            base_score = ImportanceLevel.CRITICAL.value
        
        # Boost for numbered decisions/rulings
        if has_decision_number and event_type in [EventType.DECISION, EventType.RULING]:
            base_score = min(base_score + 1, 10)
        
        return base_score
    
    @classmethod
    def extract_participants(cls, filename: str) -> List[str]:
        """Extract participant organizations from filename"""
        import re
        
        participants = []
        
        # CPUC filename patterns: "Type filed by ORGANIZATION on DATE Conf# NUMBER.pdf"
        # Extract the organization name from the "filed by" pattern
        filed_by_pattern = r'filed by (.+?) on \d{2}_\d{2}_\d{4}'
        match = re.search(filed_by_pattern, filename, re.IGNORECASE)
        
        if match:
            org_name = match.group(1).strip()
            # Clean up common patterns
            org_name = org_name.replace('_', ' ')
            org_name = cls._normalize_organization_name(org_name)
            participants.append(org_name)
            return participants
        
        # ALJ/CPUC staff patterns
        alj_pattern = r'(ALJ_\w+_CPUC|CALJ_\w+_CPUC)'
        match = re.search(alj_pattern, filename, re.IGNORECASE)
        if match:
            alj_name = match.group(1).replace('_', ' ')
            participants.append(alj_name)
            return participants
        
        # Common organization patterns in filenames (fallback)
        common_orgs = [
            'Pacific Gas and Electric Company', 'PG&E', 'PACIFIC GAS AND ELECTRIC',
            'Southern California Edison', 'SCE', 'SOUTHERN CALIFORNIA EDISON',
            'San Diego Gas & Electric', 'SDG&E', 'SAN DIEGO GAS',
            'California Community Choice Association', 'CalCCA', 'CALIFORNIA COMMUNITY CHOICE',
            'Sierra Club', 'SIERRA CLUB',
            'Solar Energy Industries Association', 'SEIA', 'SOLAR ENERGY INDUSTRIES',
            'California Environmental Justice Alliance', 'CEJA',
            'The Utility Reform Network', 'TURN', 'UTILITY REFORM NETWORK',
            'Cal Advocates', 'CAL ADVOCATES', 'CALIFORNIA ADVOCATES',
            'Clean Coalition', 'CLEAN COALITION',
            'Center for Accessible Technology', 'CFORTECH',
            'Bear Valley Electric Service', 'BEAR VALLEY',
            'California Energy Storage Alliance', 'CESA',
            'Natural Resources Defense Council', 'NRDC',
            'Vehicle-Grid Integration Council', 'VGIC',
            'Microgrid Resources Coalition', 'MICROGRID RESOURCES',
            'Coalition of California Utility Employees', 'CCUE',
            'California Large Energy Consumers Association', 'CLECA',
            'California Independent System Operator', 'CAISO',
            'Utility Consumers Action Network', 'UCAN',
            'City of San José', 'SAN JOSE',
            'AVA Community Energy', 'AVA COMMUNITY',
            'PearlX Infrastructure LLC', 'PEARLX',
            'Weave Grid Inc', 'WEAVE GRID',
            'LIBERTY UTILITIES', 'LIBERTY'
        ]
        
        filename_upper = filename.upper()
        for org in common_orgs:
            if org.upper() in filename_upper:
                participants.append(org)
                break  # Take first match to avoid duplicates
        
        return participants
    
    @classmethod
    def _normalize_organization_name(cls, org_name: str) -> str:
        """Normalize organization name for consistency"""
        # Common normalization rules
        normalizations = {
            'PACIFIC GAS AND ELECTRIC COMPANY': 'Pacific Gas and Electric Company',
            'PG&E': 'Pacific Gas and Electric Company',
            'SOUTHERN CALIFORNIA EDISON COMPANY': 'Southern California Edison Company',
            'SCE': 'Southern California Edison Company',
            'SAN DIEGO GAS & ELECTRIC COMPANY': 'San Diego Gas & Electric Company',
            'SDG&E': 'San Diego Gas & Electric Company',
            'THE UTILITY REFORM NETWORK': 'The Utility Reform Network',
            'UTILITY REFORM NETWORK': 'The Utility Reform Network',
            'CALIFORNIA ADVOCATES FOR NURSING HOME REFORM': 'Cal Advocates',
            'CAL ADVOCATES': 'Cal Advocates',
            'CALIFORNIA COMMUNITY CHOICE ASSOCIATION': 'California Community Choice Association',
            'CALCCA': 'California Community Choice Association',
            'CALIFORNIA ENVIRONMENTAL JUSTICE ALLIANCE': 'California Environmental Justice Alliance',
            'CEJA': 'California Environmental Justice Alliance',
            'SOLAR ENERGY INDUSTRIES ASSOCIATION': 'Solar Energy Industries Association',
            'SEIA': 'Solar Energy Industries Association',
            'NATURAL RESOURCES DEFENSE COUNCIL': 'Natural Resources Defense Council',
            'NRDC': 'Natural Resources Defense Council',
            'CALIFORNIA ENERGY STORAGE ALLIANCE': 'California Energy Storage Alliance',
            'CESA': 'California Energy Storage Alliance',
            'CALIFORNIA INDEPENDENT SYSTEM OPERATOR CORPORATION': 'California Independent System Operator',
            'CAISO': 'California Independent System Operator',
            'UTILITY CONSUMERS ACTION NETWORK': 'Utility Consumers\' Action Network',
            'UCAN': 'Utility Consumers\' Action Network'
        }
        
        org_upper = org_name.upper()
        for key, value in normalizations.items():
            if key in org_upper:
                return value
        
        # Title case for unknown organizations
        return org_name.title()