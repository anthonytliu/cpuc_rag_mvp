#!/usr/bin/env python3
"""
Simple test of timeline functionality
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve()))

from timeline_models import EventType, EventClassifier, TimelineEvent
from datetime import datetime

def test_timeline_components():
    """Test individual timeline components"""
    print("🔍 Testing timeline components...")
    
    # Test event classification
    classifier = EventClassifier()
    
    # Test document type classification
    test_cases = [
        ("decision", "Decision filed by CPUC.pdf"),
        ("ruling", "Ruling filed by ALJ_WANG_CPUC.pdf"),
        ("comment", "Comments filed by Sierra Club.pdf"),
        ("brief", "Brief filed by Cal Advocates.pdf"),
        ("motion", "Motion filed by PG&E.pdf"),
    ]
    
    print("\n📋 Testing event classification:")
    for doc_type, filename in test_cases:
        event_type = classifier.classify_document_type(doc_type, filename)
        importance = classifier.calculate_importance_score(event_type)
        participants = classifier.extract_participants(filename)
        
        print(f"  {doc_type} -> {event_type.value} (importance: {importance}) - {participants}")
    
    # Test timeline event creation
    print("\n🎯 Testing timeline event creation:")
    test_event = TimelineEvent(
        event_date=datetime(2024, 1, 15),
        event_type=EventType.DECISION,
        title="Test Decision - January 15, 2024",
        description="Test decision description",
        importance_score=9,
        source_document="test_decision.pdf",
        source_url="https://docs.cpuc.ca.gov/test",
        proceeding_number="R.22-07-005",
        participants=["CPUC"],
        filing_organization="CPUC"
    )
    
    print(f"  Created event: {test_event.title}")
    print(f"  Event type: {test_event.event_type.value}")
    print(f"  Importance: {test_event.importance_score}/10")
    print(f"  Date: {test_event.event_date.strftime('%Y-%m-%d')}")
    
    # Test event serialization
    event_dict = test_event.to_dict()
    recreated_event = TimelineEvent.from_dict(event_dict)
    print(f"  Serialization test: {'✅ PASSED' if recreated_event.title == test_event.title else '❌ FAILED'}")
    
    print("\n✅ Timeline component tests completed!")

if __name__ == "__main__":
    test_timeline_components()